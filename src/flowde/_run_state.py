"""Persist one pipeline run in its output directory."""

import json
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from PIL import Image

from flowde._run_settings import file_digest
from flowde.usage import RequestUsage, UsageTotals

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

ExistingRun = Literal["error", "resume", "overwrite"]
RUN_STATE_VERSION = 2


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode(
        "utf-8"
    )


def atomic_write(path: Path, content: bytes) -> None:
    """Replace a file only after its complete contents have reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".flowde-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(save_dir: Path) -> Iterator[None]:
    """Prevent concurrent writers; process exit automatically releases the lock."""
    metadata = save_dir / ".flowde"
    lock_path = metadata / "run.lock"
    if any(path.is_symlink() for path in (save_dir, metadata, lock_path)):
        msg = "The output directory, .flowde, and run.lock must not be symbolic links."
        raise ValueError(msg)
    metadata.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if sys.platform == "win32":
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                msg = f"Another run is using {save_dir}."
                raise RuntimeError(msg) from error
        else:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                msg = f"Another run is using {save_dir}."
                raise RuntimeError(msg) from error
        try:
            yield
        finally:
            if sys.platform == "win32":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def protect_inputs(save_dir: Path, paths: list[Path]) -> None:
    root = save_dir.resolve()
    for path in paths:
        if path.resolve().is_relative_to(root):
            msg = (
                f"Output directory {save_dir} contains an input file: {path}. "
                "Use a dedicated output directory."
            )
            raise ValueError(msg)


class RunState:
    """
    Store run settings and input progress in separate files under `.flowde`.

    `run_metadata.state` records the format version, pipeline stage, settings
    and input paths. `input_records/<stem>.state` stores each input's result,
    reported usage, error, completion status and output fingerprints.
    `shared_output_fingerprints.state` tracks combined classification or rotation
    JSONs. Input filename stems must be unique ignoring case across the run.

    Progress updates replace only the affected input record. The run metadata
    changes when inputs are first registered or added on resume. Classification
    and rotation still rewrite their combined output JSON and its fingerprints.
    Resume and overwrite require every expected record in the current format.

    Callers must hold output_lock() while loading or changing run state.
    """

    def __init__(
        self,
        save_dir: Path,
        kind: str,
        settings: dict[str, Any],
        on_existing: ExistingRun,
    ) -> None:
        self.root = save_dir.resolve()
        self.metadata_path = self.root / ".flowde" / "run_metadata.state"
        self.input_records_path = self.metadata_path.parent / "input_records"
        self.shared_path = (
            self.metadata_path.parent / "shared_output_fingerprints.state"
        )
        if on_existing not in {"error", "resume", "overwrite"}:
            msg = "on_existing must be 'error', 'resume', or 'overwrite'."
            raise ValueError(msg)

        occupied = any(path.name != ".flowde" for path in self.root.iterdir()) or any(
            path.name != "run.lock" for path in self.metadata_path.parent.iterdir()
        )
        if on_existing == "resume" or (occupied and on_existing == "overwrite"):
            self.data = self._load()
            if on_existing == "resume":
                if self.data["kind"] != kind:
                    msg = "This output directory belongs to a different kind of run."
                    raise ValueError(msg)
                if self.data["settings"] != settings:
                    msg = (
                        "Run settings have changed. Use a new output directory "
                        "or on_existing='overwrite'."
                    )
                    raise ValueError(msg)
                return
            self._clear_previous_run()
        elif occupied:
            msg = (
                f"Output directory {save_dir} already contains work. "
                "Choose on_existing='resume' or 'overwrite'."
            )
            raise FileExistsError(msg)
        self.data = {
            "version": RUN_STATE_VERSION,
            "kind": kind,
            "settings": settings,
            "input_records": {},
            "shared_output_fingerprints": {},
        }

    @staticmethod
    def _invalid_record(path: Path) -> ValueError:
        return ValueError(
            f"Missing or invalid saved run record: {path}. "
            "Use a new output directory; existing files have not been changed."
        )

    def record_path(self, key: str) -> Path:
        return self.input_records_path / f"{Path(key).stem}.state"

    def _read_record(self, path: Path) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise self._invalid_record(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise self._invalid_record(path) from error
        if not isinstance(value, dict):
            raise self._invalid_record(path)
        return value

    def _load(self) -> dict[str, Any]:
        run_metadata = self._read_record(self.metadata_path)
        if run_metadata.get("version") != RUN_STATE_VERSION:
            msg = (
                f"Unsupported saved run version in {self.metadata_path}. "
                "Old runs cannot be resumed or overwritten by this version; "
                "use a new output directory or the previous package version."
            )
            raise ValueError(msg)
        input_paths = run_metadata.get("input_paths")
        if (
            run_metadata.get("kind")
            not in ("classification", "parsing", "rotation", "extraction")
            or not isinstance(run_metadata.get("settings"), dict)
            or not isinstance(input_paths, list)
            or any(not isinstance(key, str) for key in input_paths)
            or len({Path(key).stem.casefold() for key in input_paths})
            != len(input_paths)
            or self.input_records_path.is_symlink()
            or not self.input_records_path.is_dir()
        ):
            raise self._invalid_record(self.metadata_path)
        input_records = {}
        for key in input_paths:
            path = self.record_path(key)
            record = self._read_record(path)
            if (
                record.get("key") != key
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("input"), dict)
                or not isinstance(record["input"].get("file_digest"), str)
                or type(record.get("has_result")) is not bool
                or type(record.get("completed")) is not bool
                or (record["completed"] and not record["has_result"])
                or "result" not in record
                or (record["has_result"] and record["result"] is None)
                or "error" not in record
                or (
                    record["error"] is not None
                    and (
                        not isinstance(record["error"], dict)
                        or not isinstance(record["error"].get("type"), str)
                        or not isinstance(record["error"].get("message"), str)
                    )
                )
                or not isinstance(record.get("usage"), list)
            ):
                raise self._invalid_record(path)
            for usage in record["usage"]:
                if (
                    not isinstance(usage, dict)
                    or not isinstance(usage.get("model"), str)
                    or not isinstance(usage.get("provider"), str)
                    or any(
                        value is not None and not isinstance(value, (int, float))
                        for name, value in usage.items()
                        if name not in ("model", "provider")
                    )
                ):
                    raise self._invalid_record(path)
                try:
                    RequestUsage(**usage)
                except TypeError as error:
                    raise self._invalid_record(path) from error
            input_records[key] = record
        data = {
            "version": run_metadata["version"],
            "kind": run_metadata["kind"],
            "settings": run_metadata["settings"],
            "input_records": input_records,
            "shared_output_fingerprints": self._read_record(self.shared_path),
        }
        self._recorded_output_paths(data)
        return data

    @staticmethod
    def _output_name(name: Any) -> bool:
        return (
            isinstance(name, str)
            and name not in ("", ".", "..", ".flowde")
            and Path(name).name == name
            and "/" not in name
            and "\\" not in name
        )

    def _validate_output_fingerprints(
        self, output_fingerprints: Any, expected: set[str], path: Path
    ) -> set[str]:
        if not isinstance(output_fingerprints, dict) or any(
            name not in expected
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in output_fingerprints.items()
        ):
            raise self._invalid_record(path)
        return set(output_fingerprints)

    def _recorded_output_paths(self, data: dict[str, Any] | None = None) -> set[Path]:
        """Validate ownership without trusting paths from edited records."""
        data = self.data if data is None else data
        combined = {
            "classification": "classifications.json",
            "rotation": "rotations.json",
        }.get(data["kind"])
        answered = any(
            record["has_result"] for record in data["input_records"].values()
        )
        outputs = self._validate_output_fingerprints(
            data["shared_output_fingerprints"],
            {combined} if combined and answered else set(),
            self.shared_path,
        )
        if (
            combined
            and any(record["completed"] for record in data["input_records"].values())
            and combined not in outputs
        ):
            raise self._invalid_record(self.shared_path)
        for key, record in data["input_records"].items():
            path = self.record_path(key)
            output_names = record.get("outputs")
            if (
                not isinstance(output_names, list)
                or any(not isinstance(name, str) for name in output_names)
                or len(set(output_names)) != len(output_names)
            ):
                raise self._invalid_record(path)
            expected = set(output_names)
            if data["kind"] == "extraction":
                if any(
                    not self._output_name(output) or not output.endswith(".png")
                    for output in output_names
                ):
                    raise self._invalid_record(path)
                if record["has_result"]:
                    self._validate_output_fingerprints(record["result"], expected, path)
            else:
                image = Path(record["path"])
                if not self._output_name(image.name):
                    raise self._invalid_record(path)
                planned = []
                if data["kind"] == "parsing":
                    planned = [f"{image.stem}.json"]
                elif data["kind"] == "rotation":
                    planned = [f"rotated_images/{image.name}"]
                elif record["has_result"]:
                    positives = data["settings"].get("positive_classes")
                    if positives is not None and record["result"] in positives:
                        planned = [f"positive_images/{image.name}"]
                if output_names != planned:
                    raise self._invalid_record(path)
                if not record["has_result"]:
                    expected = set()
            owned = self._validate_output_fingerprints(
                record.get("output_fingerprints"), expected, path
            )
            if outputs & owned or (record["completed"] and owned != expected):
                raise self._invalid_record(path)
            outputs.update(owned)
        return {self.root / name for name in outputs}

    def _clear_previous_run(self) -> None:
        outputs = self._recorded_output_paths()
        metadata = self.metadata_path.parent
        record_files = {self.record_path(key) for key in self.input_records}
        allowed_files = (
            outputs
            | record_files
            | {self.metadata_path, self.shared_path, metadata / "run.lock"}
        )
        allowed_directories = {metadata, self.input_records_path} | {
            path.parent for path in outputs
        }

        # Inspect everything before deleting anything. Never follow symlinks or
        # assume that an unrecorded JSON belongs to Flowde because of its name.
        pending = [self.root]
        while pending:
            for path in pending.pop().iterdir():
                if not path.is_symlink():
                    if path in allowed_directories and path.is_dir():
                        pending.append(path)
                        continue
                    if path in allowed_files and path.is_file():
                        continue
                msg = (
                    f"Cannot overwrite {self.root}: unexpected file or directory "
                    f"{path.relative_to(self.root)}. No files have been deleted. "
                    "Move it elsewhere or use a new output directory."
                )
                raise ValueError(msg)

        for path in sorted(outputs):
            path.unlink(missing_ok=True)
        for directory in allowed_directories - {
            metadata,
            self.root,
            self.input_records_path,
        }:
            if directory.exists():
                directory.rmdir()
        # Retain the records until public-output cleanup succeeds. Keep the
        # lock's inode throughout so another writer cannot enter during cleanup.
        for path in record_files:
            path.unlink()
        self.input_records_path.rmdir()
        self.shared_path.unlink()
        self.metadata_path.unlink()

    @property
    def input_records(self) -> dict[str, Any]:
        return self.data["input_records"]

    def save(self, key: str) -> None:
        """Save this input's record without changing other records or run metadata."""
        atomic_write(self.record_path(key), json_bytes(self.input_records[key]))

    def prepare(self, inputs: list[dict[str, Any]]) -> None:
        input_name = "PDF" if self.data["kind"] == "extraction" else "image"
        # Supplied names determine public outputs; resolved names determine records.
        for field in ("path", "key"):
            stems = {
                Path(record[field]).stem.casefold(): key
                for key, record in self.input_records.items()
            }
            for item in inputs:
                key = item["key"]
                stem = Path(item[field]).stem.casefold()
                if stem in stems and stems[stem] != key:
                    msg = (
                        "Input filename stems must be unique ignoring case "
                        "across the run: "
                        f"{stems[stem]} and {key}."
                    )
                    raise ValueError(msg)
                stems[stem] = key
        changed = []
        added = False
        for item in inputs:
            key = item["key"]
            previous = self.input_records.get(key)
            if previous and previous["has_result"]:
                if (
                    previous["input"] != item["input"]
                    or Path(previous["path"]).name != Path(item["path"]).name
                ):
                    subject = (
                        "PDF" if input_name == "PDF" else "image or partial flowchart"
                    )
                    msg = f"Previously processed {subject} has changed: {item['path']}."
                    raise ValueError(msg)
            else:
                self.input_records[key] = {
                    **item,
                    "has_result": False,
                    "completed": False,
                    "result": None,
                    "error": None,
                    "usage": previous["usage"] if previous else [],
                    "output_fingerprints": {},
                }
                changed.append(key)
                added = added or previous is None
        self.input_records_path.mkdir(exist_ok=True)
        # Write initial records before registering them in the run metadata.
        # No work starts until the complete inventory has been saved.
        for key in changed:
            self.save(key)
        if not self.metadata_path.exists():
            atomic_write(
                self.shared_path, json_bytes(self.data["shared_output_fingerprints"])
            )
        if added or not self.metadata_path.exists():
            run_metadata = {
                name: self.data[name] for name in ("version", "kind", "settings")
            }
            run_metadata["input_paths"] = list(self.input_records)
            atomic_write(self.metadata_path, json_bytes(run_metadata))

    def totals(self) -> UsageTotals:
        totals = UsageTotals()
        for index, record in enumerate(self.input_records.values()):
            for usage in record["usage"]:
                totals.add(index, RequestUsage(**usage))
            if record["completed"]:
                totals.finished_items.add(index)
        return totals

    def can_restore(self, key: str) -> bool:
        """Whether the saved result can be reused without executing the function."""
        return bool(self.input_records[key]["has_result"])

    def add_usage(self, key: str, usage: RequestUsage) -> None:
        self.input_records[key]["usage"].append(asdict(usage))
        self.save(key)

    def failure(self, key: str, error: BaseException) -> None:
        self.input_records[key]["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self.save(key)

    def result(self, key: str, result: Any) -> None:
        # Reject an unserialisable answer before putting it in the run record.
        json_bytes(result)
        record = self.input_records[key]
        record.update(result=result, has_result=True, error=None)
        if self.data["kind"] == "classification":
            positives = self.data["settings"]["positive_classes"]
            record["outputs"] = (
                [f"positive_images/{Path(record['path']).name}"]
                if positives is not None and result in positives
                else []
            )
        # Persist the answer first: a failed copy/write must not repeat the request.
        self.save(key)

    def write_artifact(
        self, relative_path: str, content: bytes, *, key: str | None = None
    ) -> None:
        output_fingerprints = (
            self.data["shared_output_fingerprints"]
            if key is None
            else self.input_records[key]["output_fingerprints"]
        )
        path = self.root / relative_path
        if not path.resolve().is_relative_to(self.root):
            msg = f"Output path escapes the run directory: {path}."
            raise ValueError(msg)
        if (
            path.is_file()
            and path.read_bytes() != content
            and output_fingerprints.get(relative_path) != file_digest(path)
        ):
            msg = (
                f"Saved output has been modified: {path}. "
                "Use a new directory or overwrite."
            )
            raise ValueError(msg)
        atomic_write(path, content)
        output_fingerprints[relative_path] = file_digest(path)
        if key is None:
            atomic_write(self.shared_path, json_bytes(output_fingerprints))

    def publish(self, key: str) -> None:
        record = self.input_records[key]
        if self.data["kind"] == "parsing":
            self.write_artifact(
                record["outputs"][0], json_bytes(record["result"]), key=key
            )
        else:
            classifications = [
                {"img_path": entry["path"], "label": entry["result"]}
                for _, entry in sorted(self.input_records.items())
                if entry["has_result"]
            ]
            if self.data["kind"] == "rotation":
                self.write_artifact("rotations.json", json_bytes(classifications))
                self._publish_rotation(key)
            else:
                self.write_artifact("classifications.json", json_bytes(classifications))
                if record["outputs"]:
                    image = Path(key)
                    if file_digest(image) != record["input"]["file_digest"]:
                        msg = f"Input image changed before copying: {image}."
                        raise ValueError(msg)
                    self.write_artifact(
                        record["outputs"][0], image.read_bytes(), key=key
                    )
        record.update(completed=True, error=None)
        self.save(key)

    def _publish_rotation(self, key: str) -> None:
        record = self.input_records[key]
        image = Path(key)
        content = image.read_bytes()
        if sha256(content).hexdigest() != record["input"]["file_digest"]:
            msg = f"Input image changed before rotating: {image}."
            raise ValueError(msg)
        # Always derive the copy from the unchanged source, including on resume.
        # Encode fully before atomic replacement so an interrupted save cannot
        # leave a half-written image at the public output path.
        with Image.open(BytesIO(content)) as original, BytesIO() as buffer:
            corrected = original.rotate(-record["result"], expand=True)
            corrected.save(buffer, format=original.format)
            self.write_artifact(record["outputs"][0], buffer.getvalue(), key=key)
