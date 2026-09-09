"""Persist one classification or parsing run inside its dedicated output directory."""

import json
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from flowde._run_settings import file_digest
from flowde.usage import RequestUsage, UsageTotals

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

ExistingRun = Literal["error", "resume", "overwrite"]


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
    def __init__(
        self,
        save_dir: Path,
        kind: str,
        settings: dict[str, Any],
        on_existing: ExistingRun,
    ) -> None:
        self.root = save_dir.resolve()
        self.path = self.root / ".flowde" / "run.state"
        if on_existing not in {"error", "resume", "overwrite"}:
            msg = "on_existing must be 'error', 'resume', or 'overwrite'."
            raise ValueError(msg)

        occupied = any(path.name != ".flowde" for path in self.root.iterdir()) or any(
            path.name != "run.lock" for path in self.path.parent.iterdir()
        )
        if on_existing == "resume":
            if not self.path.is_file():
                msg = f"No saved run exists in {save_dir}; cannot resume."
                raise ValueError(msg)
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("version") != 1 or self.data.get("kind") != kind:
                msg = (
                    "This output directory belongs to a different kind "
                    "or version of run."
                )
                raise ValueError(msg)
            if self.data["settings"] != settings:
                msg = (
                    "Run settings have changed. Use a new output directory "
                    "or on_existing='overwrite'."
                )
                raise ValueError(msg)
        else:
            if occupied and on_existing == "error":
                msg = (
                    f"Output directory {save_dir} already contains work. "
                    "Choose on_existing='resume' or 'overwrite'."
                )
                raise FileExistsError(msg)
            if occupied and on_existing == "overwrite":
                self._clear_previous_run()
            self.data = {
                "version": 1,
                "kind": kind,
                "settings": settings,
                "items": {},
                "artifacts": {},
            }

    def _recorded_output_paths(self) -> set[Path]:
        """Read the previous run's file inventory without trusting arbitrary paths."""
        msg = (
            f"Cannot overwrite {self.root}: a valid Flowde .flowde/run.state "
            "is required. Use a new output directory."
        )
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError(msg)
        try:
            previous = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(msg) from error
        if (
            not isinstance(previous, dict)
            or previous.get("version") != 1
            or previous.get("kind") not in ("classification", "parsing")
            or not isinstance(previous.get("settings"), dict)
            or not isinstance(previous.get("items"), dict)
            or not isinstance(previous.get("artifacts"), dict)
        ):
            raise ValueError(msg)

        # Only accept names associated with answered images, not arbitrary paths
        # supplied by a damaged or edited inventory.
        expected = set()
        for record in previous["items"].values():
            if not isinstance(record, dict):
                raise ValueError(msg)  # noqa: TRY004 - Invalid saved data, not an argument type.
            name = record.get("output")
            if (
                not isinstance(name, str)
                or name in ("", ".", "..", ".flowde")
                or Path(name).name != name
                or "/" in name
                or "\\" in name
                or not isinstance(record.get("has_result"), bool)
            ):
                raise ValueError(msg)
            if record["has_result"]:
                if previous["kind"] == "parsing":
                    expected.add(name)
                else:
                    expected.update(("classifications.json", f"positive_images/{name}"))

        for name, digest in previous["artifacts"].items():
            if (
                name not in expected
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(msg)
        return {self.root / name for name in previous["artifacts"]}

    def _clear_previous_run(self) -> None:
        outputs = self._recorded_output_paths()
        metadata = self.path.parent
        allowed_files = outputs | {self.path, metadata / "run.lock"}
        allowed_directories = {metadata} | {path.parent for path in outputs}

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
        for directory in allowed_directories - {metadata, self.root}:
            if directory.exists():
                # rmdir refuses nonempty directories, including if a new file
                # appeared after inspection. No recursive deletion is used.
                directory.rmdir()
        # Keep the old record until cleanup succeeds, so interrupted cleanup can
        # be retried. Keep run.lock's inode throughout to exclude other writers.
        self.path.unlink()

    @property
    def items(self) -> dict[str, Any]:
        return self.data["items"]

    def save(self) -> None:
        atomic_write(self.path, json_bytes(self.data))

    def prepare(self, inputs: list[dict[str, Any]]) -> None:
        outputs = {record["output"]: key for key, record in self.items.items()}
        for item in inputs:
            key = item["key"]
            previous = self.items.get(key)
            if item["output"] in outputs and outputs[item["output"]] != key:
                msg = (
                    f"Output {item['output']} is already associated "
                    "with a different image."
                )
                raise ValueError(msg)
            outputs[item["output"]] = key
            if previous and previous["has_result"]:
                if (
                    previous["input"] != item["input"]
                    or previous["output"] != item["output"]
                ):
                    msg = (
                        "Previously processed image or partial flowchart has changed: "
                        f"{item['path']}."
                    )
                    raise ValueError(msg)
            else:
                self.items[key] = {
                    **item,
                    "has_result": False,
                    "completed": False,
                    "result": None,
                    "error": None,
                    "usage": previous["usage"] if previous else [],
                }
        self.save()

    def totals(self) -> UsageTotals:
        totals = UsageTotals()
        for index, record in enumerate(self.items.values()):
            for usage in record["usage"]:
                totals.add(index, RequestUsage(**usage))
            if record["completed"]:
                totals.finished_items.add(index)
        return totals

    def add_usage(self, key: str, usage: RequestUsage) -> None:
        self.items[key]["usage"].append(asdict(usage))
        self.save()

    def failure(self, key: str, error: BaseException) -> None:
        self.items[key]["error"] = {"type": type(error).__name__, "message": str(error)}
        self.save()

    def result(self, key: str, result: Any) -> None:
        # Reject an unserialisable answer before putting it in the run record.
        json_bytes(result)
        self.items[key].update(result=result, has_result=True, error=None)
        # Persist the answer first: a failed copy/write must not repeat the request.
        self.save()

    def write_artifact(self, relative_path: str, content: bytes) -> None:
        path = self.root / relative_path
        if not path.resolve().is_relative_to(self.root):
            msg = f"Output path escapes the run directory: {path}."
            raise ValueError(msg)
        if (
            path.is_file()
            and path.read_bytes() != content
            and self.data["artifacts"].get(relative_path) != file_digest(path)
        ):
            msg = (
                f"Saved output has been modified: {path}. "
                "Use a new directory or overwrite."
            )
            raise ValueError(msg)
        atomic_write(path, content)
        self.data["artifacts"][relative_path] = file_digest(path)

    def publish(self, key: str) -> None:
        record = self.items[key]
        if self.data["kind"] == "parsing":
            self.write_artifact(record["output"], json_bytes(record["result"]))
        else:
            classifications = [
                {"img_path": entry["path"], "label": entry["result"]}
                for _, entry in sorted(self.items.items())
                if entry["has_result"]
            ]
            self.write_artifact("classifications.json", json_bytes(classifications))
            positives = self.data["settings"]["positive_classes"]
            if positives is not None and record["result"] in positives:
                image = Path(key)
                if file_digest(image) != record["input"]["image"]:
                    msg = f"Input image changed before copying: {image}."
                    raise ValueError(msg)
                self.write_artifact(f"positive_images/{image.name}", image.read_bytes())
        record.update(completed=True, error=None)
        self.save()
