"""Publish complete PDF extractions and recover interrupted image sets."""

from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

from flowde._run_settings import file_digest
from flowde._run_state import ExistingRun, RunState


def staging_directory(root: Path, key: str) -> Path:
    return root / sha256(key.encode()).hexdigest()


def image_manifest(directory: Path) -> dict[str, str]:
    """Validate the extractor's output before publishing any of its files."""
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".png":
            msg = f"Extractors must save only top-level PNG files; found {path.name}."
            raise ValueError(msg)
        with Image.open(path) as image:
            if image.format != "PNG":
                msg = f"Extracted image is not a PNG: {path.name}."
                raise ValueError(msg)
            image.verify()
        result[path.name] = file_digest(path)
    return result


class ExtractionRunState(RunState):
    def __init__(
        self,
        save_dir: Path,
        kind: str,
        settings: dict[str, Any],
        on_existing: ExistingRun,
        *,
        staging_root: Path,
    ) -> None:
        super().__init__(save_dir, kind, settings, on_existing)
        self.staging_root = staging_root

    def prepare(self, inputs: list[dict[str, Any]]) -> None:
        # Validate the persisted inventory before using any saved output paths.
        if self.metadata_path.exists():
            self._recorded_output_paths()
        super().prepare(inputs)
        owners: dict[str, str] = {}
        for key, record in self.input_records.items():
            for name in record["outputs"]:
                other = owners.setdefault(name.casefold(), key)
                if other != key:
                    msg = f"Extracted image {name} belongs to more than one PDF."
                    raise ValueError(msg)
            self._check_existing(key)

    def _check_existing(self, key: str) -> None:
        record = self.input_records[key]
        for name in record["outputs"]:
            path = self.root / name
            # During interrupted publication a file may still contain the old
            # extraction or may already contain the newly recorded extraction.
            allowed = {
                record["output_fingerprints"].get(name),
                (record.get("result") or {}).get(name),
            }
            if path.is_symlink() or (
                path.exists()
                and (not path.is_file() or file_digest(path) not in allowed)
            ):
                msg = (
                    f"Saved output has been modified: {path}. "
                    "Use a new directory or overwrite."
                )
                raise ValueError(msg)

    def can_restore(self, key: str) -> bool:
        record = self.input_records[key]
        reusable = bool(record["has_result"]) and (
            set(record["outputs"]) == set(record["result"])
            and all(
                (self.root / name).is_file() and file_digest(self.root / name) == digest
                for name, digest in record["result"].items()
            )
        )
        if not reusable:
            record["completed"] = False
        return reusable

    def result(self, key: str, result: Any) -> None:
        record = self.input_records[key]
        if file_digest(Path(key)) != record["input"]["file_digest"]:
            msg = f"Input PDF changed during extraction: {record['path']}."
            raise ValueError(msg)
        self._check_existing(key)
        other_names = {
            name.casefold()
            for other, other_record in self.input_records.items()
            if other != key
            for name in other_record["outputs"]
        }
        own_names = set(record["outputs"])
        existing_names = {path.name.casefold() for path in self.root.iterdir()}
        staged = image_manifest(staging_directory(self.staging_root, key))
        if result != staged:
            msg = "Staged extraction changed before it could be saved."
            raise ValueError(msg)
        seen = set()
        for name in result:
            folded = name.casefold()
            if (
                folded in seen
                or folded in other_names
                or (folded in existing_names and name not in own_names)
            ):
                msg = (
                    f"Extracted image filename conflicts with existing output: {name}."
                )
                raise ValueError(msg)
            seen.add(folded)
        # Record ownership BEFORE any public writes. After a forced stop, resume
        # can recognise both old files and files published from this new set.
        for name in own_names:
            path = self.root / name
            if path.is_file():
                record["output_fingerprints"][name] = file_digest(path)
        record["outputs"] = sorted(own_names | result.keys())
        for name, digest in result.items():
            record["output_fingerprints"].setdefault(name, digest)
        super().result(key, result)

    def publish(self, key: str) -> None:
        record = self.input_records[key]
        self._check_existing(key)
        stage = staging_directory(self.staging_root, key)
        if stage.exists():
            if image_manifest(stage) != record["result"]:
                msg = "Staged extraction changed before it could be published."
                raise ValueError(msg)
            for name in record["result"]:
                self.write_artifact(name, (stage / name).read_bytes(), key=key)
            for name in set(record["outputs"]) - record["result"].keys():
                (self.root / name).unlink(missing_ok=True)
                record["output_fingerprints"].pop(name, None)
            record["outputs"] = sorted(record["result"])
        record.update(completed=True, error=None)
        self.save(key)
        if stage.exists():
            for path in stage.iterdir():
                path.unlink()
            stage.rmdir()
