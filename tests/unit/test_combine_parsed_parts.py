import json
import re
from itertools import product
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from flowde.combine_parsed_parts import combine_parsed_parts


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in directory.rglob("*")
    }


def output_record(save_dir):
    return json.loads((save_dir / ".flowde" / "combined.state").read_text())


def symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"Symbolic links are unavailable: {error}")


@pytest.fixture
def part_dirs(tmp_path):
    directories = {
        part: tmp_path / "parts" / part
        for part in ("nodes_dir", "labels_dir", "flow_dir", "additional_texts_dir")
    }
    # Deliberately create b before a, and vary node order between the files.
    for name in ("b", "a"):
        write_json(
            directories["nodes_dir"] / f"{name}.json",
            {
                "nodes": [
                    {"node_number": 2, "text": f"{name}: analysed — 45"},
                    {"node_number": 1, "text": f"{name}: allocated — 50"},
                ],
            },
        )
        write_json(
            directories["labels_dir"] / f"{name}.json",
            {
                "nodes": [
                    {"node_number": 1, "labels": ["Allocation"]},
                    {"node_number": 2, "labels": []},
                ],
            },
        )
        write_json(
            directories["flow_dir"] / f"{name}.json",
            {
                "nodes": [
                    {"node_number": 2, "points_to": []},
                    {"node_number": 1, "points_to": [2]},
                ],
            },
        )
        write_json(
            directories["additional_texts_dir"] / f"{name}.json",
            {"additional_texts": [f"Figure {name}"]},
        )
    return directories


@pytest.mark.parametrize(
    ("include_labels", "include_flow", "include_additional_texts"),
    list(product((False, True), repeat=3)),
)
def test_combines_optional_parts_and_returns_models_in_filename_order(
    tmp_path, part_dirs, include_labels, include_flow, include_additional_texts
):
    save_dir = tmp_path / "nested" / "combined"
    before = snapshot(tmp_path / "parts")
    results = combine_parsed_parts(
        nodes_dir=part_dirs["nodes_dir"],
        save_dir=save_dir,
        labels_dir=part_dirs["labels_dir"] if include_labels else None,
        flow_dir=part_dirs["flow_dir"] if include_flow else None,
        additional_texts_dir=(
            part_dirs["additional_texts_dir"] if include_additional_texts else None
        ),
    )

    assert len(results) == 2
    assert all(isinstance(result, BaseModel) for result in results)
    assert sorted(path.name for path in save_dir.iterdir()) == [
        ".flowde",
        "a.json",
        "b.json",
    ]
    assert output_record(save_dir) == {
        "version": 1,
        "kind": "combined_parsed_parts",
        "outputs": ["a.json", "b.json"],
    }
    assert sorted(path.name for path in (save_dir / ".flowde").iterdir()) == [
        "combined.state"
    ]
    for name, result in zip(("a", "b"), results, strict=True):
        expected = {
            "nodes": [
                {"node_number": 1, "text": f"{name}: allocated — 50"},
                {"node_number": 2, "text": f"{name}: analysed — 45"},
            ],
        }
        if include_labels:
            expected["nodes"][0]["labels"] = ["Allocation"]
            expected["nodes"][1]["labels"] = []
        if include_flow:
            expected["nodes"][0]["points_to"] = [2]
            expected["nodes"][1]["points_to"] = []
        if include_additional_texts:
            expected["additional_texts"] = [f"Figure {name}"]
        assert result.model_dump() == expected
        output_text = (save_dir / f"{name}.json").read_text(encoding="utf-8")
        assert json.loads(output_text) == expected
        assert "—" in output_text
    assert snapshot(tmp_path / "parts") == before


def test_ignores_metadata_non_json_files_and_subdirectories(tmp_path, part_dirs):
    for directory in part_dirs.values():
        metadata = directory / ".flowde"
        metadata.mkdir()
        (metadata / "run_metadata.state").write_text(
            "invalid state, intentionally ignored"
        )
        write_json(directory / "nested" / "extra.json", {"invalid": True})
        (directory / "notes.txt").write_text("not a parsing result")
        (directory / "directory.json").mkdir()
    before = snapshot(tmp_path / "parts")
    save_dir = tmp_path / "combined"

    results = combine_parsed_parts(**part_dirs, save_dir=save_dir)

    assert len(results) == 2
    assert sorted(path.name for path in save_dir.iterdir()) == [
        ".flowde",
        "a.json",
        "b.json",
    ]
    assert snapshot(tmp_path / "parts") == before


def test_requires_node_text_directory(tmp_path):
    save_dir = tmp_path / "combined"
    with pytest.raises(ValueError, match="input directory is required"):
        combine_parsed_parts(nodes_dir=None, save_dir=save_dir)
    assert not save_dir.exists()


@pytest.mark.parametrize(
    "parameter", ["nodes_dir", "labels_dir", "flow_dir", "additional_texts_dir"]
)
@pytest.mark.parametrize("invalid_kind", ["missing", "file"])
def test_rejects_missing_or_non_directory_inputs(
    tmp_path, part_dirs, parameter, invalid_kind
):
    invalid_path = tmp_path / "invalid-input"
    if invalid_kind == "file":
        invalid_path.write_text("a file, not a directory")
    part_dirs[parameter] = invalid_path
    save_dir = tmp_path / "combined"

    with pytest.raises(NotADirectoryError, match=re.escape(str(invalid_path))):
        combine_parsed_parts(**part_dirs, save_dir=save_dir)

    assert not save_dir.exists()


@pytest.mark.parametrize(
    "parameter", ["nodes_dir", "labels_dir", "flow_dir", "additional_texts_dir"]
)
def test_rejects_an_empty_supplied_directory(tmp_path, part_dirs, parameter):
    part_dirs[parameter] = tmp_path / "empty-part"
    part_dirs[parameter].mkdir()
    save_dir = tmp_path / "combined"

    with pytest.raises(ValueError, match=r"No JSON files|same length"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir)

    assert not save_dir.exists()


@pytest.mark.parametrize(
    "parameter", ["labels_dir", "flow_dir", "additional_texts_dir"]
)
@pytest.mark.parametrize("mismatch", ["missing", "extra", "different-stem"])
def test_rejects_unmatched_part_files_without_changing_output(
    tmp_path, part_dirs, parameter, mismatch
):
    directory = part_dirs[parameter]
    if mismatch == "missing":
        (directory / "b.json").unlink()
    elif mismatch == "extra":
        (directory / "c.json").write_bytes((directory / "b.json").read_bytes())
    else:
        (directory / "b.json").rename(directory / "c.json")
    save_dir = tmp_path / "combined"
    write_json(save_dir / "old.json", {"previous": "result"})
    before = snapshot(save_dir)

    with pytest.raises(ValueError, match=r"same length|Mismatched stems"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert snapshot(save_dir) == before


@pytest.mark.parametrize("existing_output", [False, True])
@pytest.mark.parametrize(
    ("parameter", "invalid_content", "exception"),
    [
        pytest.param("nodes_dir", b"not JSON", ValidationError, id="invalid-json"),
        pytest.param("nodes_dir", {"nodes": []}, ValueError, id="empty-nodes"),
        pytest.param(
            "nodes_dir",
            {"nodes": [{"node_number": 2, "text": "Wrong start"}]},
            ValueError,
            id="nonconsecutive-numbers",
        ),
        pytest.param(
            "labels_dir",
            {"nodes": [{"node_number": 1, "labels": []}]},
            ValueError,
            id="different-node-numbers",
        ),
        pytest.param(
            "labels_dir",
            {"nodes": [{"node_number": 1}]},
            ValidationError,
            id="missing-required-field",
        ),
        pytest.param(
            "flow_dir",
            {"nodes": [{"node_number": 1, "points_to": [2], "text": "extra"}]},
            ValidationError,
            id="extra-part-field",
        ),
        pytest.param(
            "additional_texts_dir",
            {"additional_texts": [{"invalid": "not a string"}]},
            ValidationError,
            id="invalid-additional-text",
        ),
        pytest.param(
            "nodes_dir", {"options": []}, ValidationError, id="ground-truth-wrapper"
        ),
        pytest.param("nodes_dir", b"\xff", UnicodeDecodeError, id="invalid-utf8"),
    ],
)
def test_validates_every_input_before_creating_or_changing_output(
    tmp_path, part_dirs, existing_output, parameter, invalid_content, exception
):
    save_dir = tmp_path / "combined"
    if existing_output:
        combine_parsed_parts(**part_dirs, save_dir=save_dir)
    # a is valid and sorts first; b must fail before a can be written.
    invalid_path = part_dirs[parameter] / "b.json"
    if isinstance(invalid_content, bytes):
        invalid_path.write_bytes(invalid_content)
    else:
        write_json(invalid_path, invalid_content)
    before = snapshot(save_dir)
    inputs_before = snapshot(tmp_path / "parts")

    with pytest.raises(exception):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert save_dir.exists() == existing_output
    assert snapshot(save_dir) == before
    assert snapshot(tmp_path / "parts") == inputs_before


@pytest.mark.parametrize("contents", ["json", "text", "directory"])
def test_default_rejects_occupied_output_without_changing_files(
    tmp_path, part_dirs, contents
):
    save_dir = tmp_path / "combined"
    save_dir.mkdir()
    if contents == "directory":
        (save_dir / "notes").mkdir()
    else:
        (save_dir / f"existing.{contents}").write_text("keep this")
    before = snapshot(save_dir)

    with pytest.raises(FileExistsError, match="on_existing='overwrite'"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir)

    assert snapshot(save_dir) == before


@pytest.mark.parametrize("create_output", [False, True])
@pytest.mark.parametrize("on_existing", ["error", "overwrite"])
def test_accepts_missing_or_empty_output(
    tmp_path, part_dirs, create_output, on_existing
):
    save_dir = tmp_path / "combined"
    if create_output:
        save_dir.mkdir()

    results = combine_parsed_parts(
        **part_dirs, save_dir=save_dir, on_existing=on_existing
    )

    assert len(results) == 2
    assert sorted(path.name for path in save_dir.iterdir()) == [
        ".flowde",
        "a.json",
        "b.json",
    ]


def test_overwrite_replaces_results_and_removes_only_recorded_obsolete_jsons(
    tmp_path, part_dirs
):
    save_dir = tmp_path / "combined"
    combine_parsed_parts(**part_dirs, save_dir=save_dir)
    for directory in part_dirs.values():
        (directory / "b.json").unlink()
    node_file = part_dirs["nodes_dir"] / "a.json"
    changed = json.loads(node_file.read_text(encoding="utf-8"))
    changed["nodes"][0]["text"] = "Updated analysis"
    write_json(node_file, changed)
    inputs_before = snapshot(tmp_path / "parts")

    results = combine_parsed_parts(
        **part_dirs, save_dir=save_dir, on_existing="overwrite"
    )

    assert len(results) == 1
    assert results[0].model_dump()["nodes"][1]["text"] == "Updated analysis"
    assert (
        json.loads((save_dir / "a.json").read_text(encoding="utf-8"))
        == results[0].model_dump()
    )
    assert not (save_dir / "b.json").exists()
    assert sorted(path.name for path in save_dir.iterdir()) == [".flowde", "a.json"]
    assert output_record(save_dir)["outputs"] == ["a.json"]
    assert snapshot(tmp_path / "parts") == inputs_before


@pytest.mark.parametrize(
    "extra",
    [
        "notes.json",
        "notes.txt",
        "nested/old.json",
        "empty",
        "directory.json",
        ".flowde/notes.json",
    ],
)
def test_overwrite_refuses_unrecorded_entries_without_changing_any_files(
    tmp_path, part_dirs, extra
):
    save_dir = tmp_path / "combined"
    combine_parsed_parts(**part_dirs, save_dir=save_dir)
    path = save_dir / extra
    if extra in {"empty", "directory.json"}:
        path.mkdir()
    else:
        write_json(path, {"notes": "User-owned data"})
    before = snapshot(save_dir)

    with pytest.raises(ValueError, match="unexpected file or directory"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert snapshot(save_dir) == before


def test_overwrite_refuses_unrecorded_file_matching_a_new_output(tmp_path, part_dirs):
    save_dir = tmp_path / "combined"
    combine_parsed_parts(**part_dirs, save_dir=save_dir)
    for directory in part_dirs.values():
        (directory / "c.json").write_bytes((directory / "b.json").read_bytes())
    write_json(save_dir / "c.json", {"notes": "Not a combined result"})
    before = snapshot(save_dir)

    with pytest.raises(ValueError, match=r"unexpected file or directory c\.json"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert snapshot(save_dir) == before
    assert output_record(save_dir)["outputs"] == ["a.json", "b.json"]


@pytest.mark.parametrize("filename", ["a.json", "notes.json"])
def test_overwrite_refuses_existing_json_without_an_output_record(
    tmp_path, part_dirs, filename
):
    save_dir = tmp_path / "combined"
    write_json(save_dir / filename, {"keep": "User-owned data"})
    before = snapshot(save_dir)

    with pytest.raises(ValueError, match=r"valid Flowde \.flowde/combined\.state"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert snapshot(save_dir) == before


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(None, id="missing"),
        pytest.param(b"{", id="invalid-json"),
        pytest.param(b"\xff", id="invalid-utf8"),
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(b"{}", id="missing-fields"),
        pytest.param({"version": 2}, id="wrong-version"),
        pytest.param({"version": True}, id="boolean-version"),
        pytest.param({"kind": "parsing"}, id="wrong-kind"),
        pytest.param({"outputs": {}}, id="not-a-list"),
        pytest.param({"outputs": [42]}, id="not-a-string"),
        pytest.param({"outputs": ["a.json", "a.json"]}, id="duplicate-name"),
        pytest.param({"outputs": ["../outside.json"]}, id="parent-path"),
        pytest.param({"outputs": ["nested/file.json"]}, id="nested-path"),
        pytest.param({"outputs": ["nested\\file.json"]}, id="backslash-path"),
        pytest.param({"outputs": ["notes.txt"]}, id="not-a-json-name"),
        pytest.param("absolute", id="absolute-path"),
    ],
)
def test_overwrite_rejects_invalid_output_records(tmp_path, part_dirs, invalid):
    save_dir = tmp_path / "combined"
    combine_parsed_parts(**part_dirs, save_dir=save_dir)
    outside = tmp_path / "outside.json"
    write_json(outside, {"keep": "Outside the combined directory"})
    outside_before = outside.read_bytes()
    record_path = save_dir / ".flowde" / "combined.state"
    if invalid is None:
        record_path.unlink()
    elif isinstance(invalid, bytes):
        record_path.write_bytes(invalid)
    else:
        value = output_record(save_dir)
        value.update({"outputs": [str(outside)]} if invalid == "absolute" else invalid)
        write_json(record_path, value)
    before = snapshot(save_dir)

    with pytest.raises(ValueError, match=r"valid Flowde \.flowde/combined\.state"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert snapshot(save_dir) == before
    assert outside.read_bytes() == outside_before


@pytest.mark.parametrize("change", ["missing", "edited"])
def test_overwrite_recreates_missing_or_edited_recorded_results(
    tmp_path, part_dirs, change
):
    save_dir = tmp_path / "combined"
    expected = combine_parsed_parts(**part_dirs, save_dir=save_dir)
    if change == "missing":
        (save_dir / "a.json").unlink()
    else:
        write_json(save_dir / "a.json", {"manually": "edited"})

    results = combine_parsed_parts(
        **part_dirs, save_dir=save_dir, on_existing="overwrite"
    )

    assert [result.model_dump() for result in results] == [
        result.model_dump() for result in expected
    ]
    assert output_record(save_dir)["outputs"] == ["a.json", "b.json"]
    for name, result in zip(("a", "b"), results, strict=True):
        assert (
            json.loads((save_dir / f"{name}.json").read_text(encoding="utf-8"))
            == result.model_dump()
        )


@pytest.mark.parametrize("location", ["metadata-link", "record-link", "metadata-file"])
def test_rejects_unsafe_metadata_without_changing_protected_files(
    tmp_path, part_dirs, location
):
    save_dir = tmp_path / "combined"
    combine_parsed_parts(**part_dirs, save_dir=save_dir)
    metadata = save_dir / ".flowde"
    target = tmp_path / "outside"
    if location == "metadata-link":
        metadata.rename(target)
        symlink_or_skip(metadata, target, directory=True)
    elif location == "record-link":
        record_path = metadata / "combined.state"
        record_path.rename(target)
        symlink_or_skip(record_path, target)
    else:
        metadata.rename(target)
        metadata.write_text("User-owned file")
    protected = snapshot(target) if target.is_dir() else target.read_bytes()
    before = snapshot(save_dir)

    with pytest.raises(
        ValueError, match=r"metadata must be an ordinary directory|valid Flowde"
    ):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert snapshot(save_dir) == before
    assert (snapshot(target) if target.is_dir() else target.read_bytes()) == protected


@pytest.mark.parametrize(
    "location",
    ["nodes_dir", "labels_dir", "flow_dir", "additional_texts_dir", "parent"],
)
def test_rejects_output_that_contains_input_directories(tmp_path, part_dirs, location):
    save_dir = tmp_path / "parts" if location == "parent" else part_dirs[location]
    before = snapshot(tmp_path / "parts")

    with pytest.raises(ValueError, match="contains input directory"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert snapshot(tmp_path / "parts") == before


@pytest.mark.parametrize("on_existing", ["resume", "replace", ""])
def test_rejects_unsupported_existing_output_options(tmp_path, part_dirs, on_existing):
    save_dir = tmp_path / "combined"
    with pytest.raises(ValueError, match="'error' or 'overwrite'"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing=on_existing)
    assert not save_dir.exists()


def test_rejects_output_path_that_is_a_file(tmp_path, part_dirs):
    save_dir = tmp_path / "combined"
    save_dir.write_text("keep this file")

    with pytest.raises(NotADirectoryError, match="Output path is not a directory"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert save_dir.read_text() == "keep this file"


def test_rejects_a_directory_at_an_output_filename_before_writing(tmp_path, part_dirs):
    save_dir = tmp_path / "combined"
    combine_parsed_parts(**part_dirs, save_dir=save_dir)
    (save_dir / "b.json").unlink()
    (save_dir / "b.json").mkdir()
    before = snapshot(save_dir)

    with pytest.raises(IsADirectoryError, match="required output filename"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    assert snapshot(save_dir) == before


@pytest.mark.parametrize("link_kind", ["output-directory", "output-json", "input-json"])
def test_rejects_symbolic_links_that_could_overwrite_other_data(
    tmp_path, part_dirs, link_kind
):
    save_dir = tmp_path / "combined"
    save_dir.mkdir()
    if link_kind == "output-directory":
        target = tmp_path / "real-output"
        target.mkdir()
        write_json(target / "a.json", {"keep": "original"})
        save_dir.rmdir()
        symlink_or_skip(save_dir, target, directory=True)
        protected = target
    elif link_kind == "output-json":
        target = tmp_path / "external.json"
        write_json(target, {"keep": "original"})
        symlink_or_skip(save_dir / "b.json", target)
        protected = target
    else:
        target = save_dir / "a.json"
        original = part_dirs["nodes_dir"] / "a.json"
        target.write_bytes(original.read_bytes())
        original.unlink()
        symlink_or_skip(original, target)
        protected = target
    before = snapshot(protected) if protected.is_dir() else protected.read_bytes()

    with pytest.raises(ValueError, match=r"symbolic link|contains an input file"):
        combine_parsed_parts(**part_dirs, save_dir=save_dir, on_existing="overwrite")

    after = snapshot(protected) if protected.is_dir() else protected.read_bytes()
    assert after == before


@pytest.mark.parametrize("existing_output", [False, True])
def test_failed_write_preserves_complete_files_and_can_be_rerun(
    tmp_path, part_dirs, monkeypatch, existing_output
):
    save_dir = tmp_path / "combined"
    if existing_output:
        for directory in part_dirs.values():
            (directory / "obsolete.json").write_bytes(
                (directory / "a.json").read_bytes()
            )
        combine_parsed_parts(**part_dirs, save_dir=save_dir)
        for directory in part_dirs.values():
            (directory / "obsolete.json").unlink()
        previous_b = (save_dir / "b.json").read_bytes()
        previous_obsolete = (save_dir / "obsolete.json").read_bytes()
    for name in ("a", "b"):
        path = part_dirs["nodes_dir"] / f"{name}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["nodes"][1]["text"] = f"Updated {name}"
        write_json(path, value)
    original_replace = Path.replace

    def fail_second_replace(path, target):
        if Path(target) == save_dir / "b.json":
            msg = "Simulated write failure"
            raise OSError(msg)
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_second_replace)
        with pytest.raises(OSError, match="Simulated write failure"):
            combine_parsed_parts(
                **part_dirs, save_dir=save_dir, on_existing="overwrite"
            )

    first_result = json.loads((save_dir / "a.json").read_text(encoding="utf-8"))
    assert first_result["nodes"][0]["text"] == "Updated a"
    if existing_output:
        assert (save_dir / "b.json").read_bytes() == previous_b
        assert (save_dir / "obsolete.json").read_bytes() == previous_obsolete
        assert output_record(save_dir)["outputs"] == [
            "a.json",
            "b.json",
            "obsolete.json",
        ]
    else:
        assert not (save_dir / "b.json").exists()
        assert output_record(save_dir)["outputs"] == ["a.json", "b.json"]
    assert not list(save_dir.rglob(".flowde-*"))

    results = combine_parsed_parts(
        **part_dirs, save_dir=save_dir, on_existing="overwrite"
    )

    assert sorted(path.name for path in save_dir.iterdir()) == [
        ".flowde",
        "a.json",
        "b.json",
    ]
    assert output_record(save_dir)["outputs"] == ["a.json", "b.json"]
    for name, result in zip(("a", "b"), results, strict=True):
        assert (
            json.loads((save_dir / f"{name}.json").read_text(encoding="utf-8"))
            == result.model_dump()
        )


@pytest.mark.parametrize("existing_output", [False, True])
def test_failed_initial_record_write_leaves_results_unchanged_and_can_be_retried(
    tmp_path, part_dirs, monkeypatch, existing_output
):
    save_dir = tmp_path / "combined"
    if existing_output:
        combine_parsed_parts(**part_dirs, save_dir=save_dir)
    before = snapshot(save_dir)
    record_path = save_dir / ".flowde" / "combined.state"
    original_replace = Path.replace

    def fail_record_replace(path, target):
        if Path(target) == record_path:
            msg = "Simulated record failure"
            raise OSError(msg)
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_record_replace)
        with pytest.raises(OSError, match="Simulated record failure"):
            combine_parsed_parts(
                **part_dirs, save_dir=save_dir, on_existing="overwrite"
            )

    if existing_output:
        assert snapshot(save_dir) == before
    else:
        assert list(save_dir.iterdir()) == [save_dir / ".flowde"]
        assert not list((save_dir / ".flowde").iterdir())
    assert not list(save_dir.rglob(".flowde-*"))

    results = combine_parsed_parts(
        **part_dirs, save_dir=save_dir, on_existing="overwrite"
    )

    assert output_record(save_dir)["outputs"] == ["a.json", "b.json"]
    for name, result in zip(("a", "b"), results, strict=True):
        assert (
            json.loads((save_dir / f"{name}.json").read_text(encoding="utf-8"))
            == result.model_dump()
        )


@pytest.mark.parametrize("failure", ["obsolete-removal", "final-record-write"])
def test_failed_cleanup_retains_output_ownership_and_can_be_retried(
    tmp_path, part_dirs, monkeypatch, failure
):
    save_dir = tmp_path / "combined"
    combine_parsed_parts(**part_dirs, save_dir=save_dir)
    for directory in part_dirs.values():
        (directory / "c.json").write_bytes((directory / "b.json").read_bytes())
        (directory / "b.json").unlink()
    previous_b = (save_dir / "b.json").read_bytes()
    original_replace = Path.replace
    original_unlink = Path.unlink
    record_writes = 0

    def fail_final_record_replace(path, target):
        nonlocal record_writes
        if Path(target) == save_dir / ".flowde" / "combined.state":
            record_writes += 1
            if record_writes == 2:
                msg = "Simulated cleanup failure"
                raise OSError(msg)
        return original_replace(path, target)

    def fail_obsolete_unlink(path, *args, **kwargs):
        if path == save_dir / "b.json":
            msg = "Simulated cleanup failure"
            raise OSError(msg)
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        if failure == "obsolete-removal":
            patch.setattr(Path, "unlink", fail_obsolete_unlink)
        else:
            patch.setattr(Path, "replace", fail_final_record_replace)
        with pytest.raises(OSError, match="Simulated cleanup failure"):
            combine_parsed_parts(
                **part_dirs, save_dir=save_dir, on_existing="overwrite"
            )

    expected = {
        "a": json.loads((save_dir / "a.json").read_text(encoding="utf-8")),
        "c": json.loads((save_dir / "c.json").read_text(encoding="utf-8")),
    }
    assert expected["a"]["nodes"][0]["text"] == "a: allocated — 50"
    assert expected["c"]["nodes"][0]["text"] == "b: allocated — 50"
    assert output_record(save_dir)["outputs"] == ["a.json", "b.json", "c.json"]
    if failure == "obsolete-removal":
        assert (save_dir / "b.json").read_bytes() == previous_b
    else:
        assert not (save_dir / "b.json").exists()
    assert not list(save_dir.rglob(".flowde-*"))

    results = combine_parsed_parts(
        **part_dirs, save_dir=save_dir, on_existing="overwrite"
    )

    assert sorted(path.name for path in save_dir.iterdir()) == [
        ".flowde",
        "a.json",
        "c.json",
    ]
    assert output_record(save_dir)["outputs"] == ["a.json", "c.json"]
    for name, result in zip(("a", "c"), results, strict=True):
        assert (
            json.loads((save_dir / f"{name}.json").read_text(encoding="utf-8"))
            == expected[name]
            == result.model_dump()
        )
