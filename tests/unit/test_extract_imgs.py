import signal
import time
from unittest.mock import Mock

import pytest
from PIL import Image

from flowde import _run_state, model_function
from flowde._extraction_state import ExtractionRunState
from flowde.extract_imgs import extract_imgs, extract_imgs_pdf_list


@pytest.fixture(params=["directory", "paths"])
def api(request):
    return request.param


@pytest.fixture
def pdf_dir(tmp_path):
    directory = tmp_path / "pdfs"
    directory.mkdir()
    for name in ("c", "a", "b"):
        (directory / f"{name}.pdf").write_bytes(f"PDF {name}".encode())
    return directory


def png(path, colour=10):
    Image.new("RGB", (3, 2), (colour, 0, 0)).save(path)


def extractor(action=None, count=2):
    def extract(pdf_path, save_dir):
        if action is not None:
            action(pdf_path, save_dir)
        else:
            for index in range(count):
                png(save_dir / f"{pdf_path.stem}_{index}.png", ord(pdf_path.stem[0]))

    return model_function(extract, version=1)


def run(api, pdf_dir, output, fn=None, **options):
    fn = extractor() if fn is None else fn
    if api == "directory":
        return extract_imgs(pdf_dir, output, fn, **options)
    return extract_imgs_pdf_list(sorted(pdf_dir.glob("*.pdf")), output, fn, **options)


def image_names(output):
    return sorted(path.name for path in output.glob("*.png"))


def contents(output):
    return {
        str(p.relative_to(output)): p.read_bytes()
        for p in output.rglob("*")
        if p.is_file()
    }


def test_each_pdf_is_published_only_after_success_and_staging_is_cleaned(
    api, pdf_dir, tmp_path, saved_state
):
    output = tmp_path / "output"
    calls = []
    stages = []
    original_pdfs = contents(pdf_dir)

    def extract(pdf, stage):
        calls.append(pdf.stem)
        stages.append(stage)
        assert stage != output
        png(stage / f"{pdf.stem}_0.png")
        assert not list(output.glob(f"{pdf.stem}_*.png"))
        if pdf.stem == "b":
            assert image_names(output) == ["a_0.png", "a_1.png"]
        png(stage / f"{pdf.stem}_1.png")

    assert run(api, pdf_dir, output, extractor(extract)) is None
    assert calls == ["a", "b", "c"]
    assert image_names(output) == [
        f"{name}_{i}.png" for name in "abc" for i in range(2)
    ]
    assert all(not stage.exists() for stage in stages)
    assert contents(pdf_dir) == original_pdfs
    assert {p.name for p in (output / ".flowde").iterdir()} == {
        "run_metadata.state",
        "run.lock",
        "input_records",
        "shared_output_fingerprints.state",
    }
    records = saved_state(output)["input_records"]
    for name in "abc":
        record = records[str((pdf_dir / f"{name}.pdf").resolve())]
        assert record["outputs"] == [f"{name}_0.png", f"{name}_1.png"]
        assert "output" not in record


@pytest.mark.parametrize("n_jobs", [1, 2, 3])
def test_workers_save_distinct_complete_image_sets(api, pdf_dir, tmp_path, n_jobs):
    output = tmp_path / "output"
    run(api, pdf_dir, output, n_jobs=n_jobs)
    assert len(image_names(output)) == 6
    for path in output.glob("*.png"):
        with Image.open(path) as image:
            assert image.size == (3, 2)
            assert image.getpixel((0, 0)) == (ord(path.name[0]), 0, 0)
    never = model_function(
        Mock(spec=[], side_effect=AssertionError("Must reuse completed PDFs")),
        version=1,
    )
    run(api, pdf_dir, output, never, on_existing="resume", n_jobs=1)
    never.assert_not_called()


def test_partial_failure_restarts_entire_pdf_and_keeps_completed_pdfs(
    api, pdf_dir, tmp_path, saved_state
):
    output = tmp_path / "output"
    calls = []
    stages = []

    def fail(pdf, stage):
        calls.append(pdf.stem)
        stages.append(stage)
        png(stage / f"{pdf.stem}_0.png")
        if pdf.stem == "b":
            msg = "page 5 failed"
            raise RuntimeError(msg)
        png(stage / f"{pdf.stem}_1.png")

    with pytest.raises(RuntimeError, match="page 5 failed"):
        run(api, pdf_dir, output, extractor(fail))
    assert calls == ["a", "b"]
    assert image_names(output) == ["a_0.png", "a_1.png"]
    assert all(not stage.exists() for stage in stages)
    state = saved_state(output)
    assert (
        state["input_records"][str((pdf_dir / "b.pdf").resolve())]["error"]["message"]
        == "page 5 failed"
    )
    # Failed PDFs can be corrected, and new PDFs can be added to a resumed run.
    (pdf_dir / "b.pdf").write_text("repaired PDF")
    (pdf_dir / "d.pdf").write_text("new PDF")
    calls.clear()

    def succeed(pdf, stage):
        calls.append(pdf.stem)
        png(stage / f"{pdf.stem}_0.png")

    run(api, pdf_dir, output, extractor(succeed), on_existing="resume")
    assert calls == ["b", "c", "d"]
    assert image_names(output) == [
        "a_0.png",
        "a_1.png",
        "b_0.png",
        "c_0.png",
        "d_0.png",
    ]


def test_zero_images_is_a_completed_pdf(api, pdf_dir, tmp_path, saved_state):
    output = tmp_path / "output"
    run(api, pdf_dir, output, extractor(count=0))
    never = model_function(Mock(spec=[]), version=1)
    run(api, pdf_dir, output, never, on_existing="resume")
    never.assert_not_called()
    assert image_names(output) == []
    state = saved_state(output)
    assert all(record["completed"] for record in state["input_records"].values())
    assert all(record["outputs"] == [] for record in state["input_records"].values())


def test_existing_run_settings_and_completed_pdf_changes_are_rejected(
    api, pdf_dir, tmp_path
):
    output = tmp_path / "output"
    run(api, pdf_dir, output)
    before = contents(output)
    with pytest.raises(FileExistsError, match="already contains work"):
        run(api, pdf_dir, output)
    assert contents(output) == before
    changed = model_function(Mock(spec=[]), version=2)
    with pytest.raises(ValueError, match="settings have changed"):
        run(api, pdf_dir, output, changed, on_existing="resume")
    changed.assert_not_called()
    (pdf_dir / "a.pdf").write_text("changed completed PDF")
    never = model_function(Mock(spec=[]), version=1)
    with pytest.raises(ValueError, match="Previously processed PDF"):
        run(api, pdf_dir, output, never, on_existing="resume")
    never.assert_not_called()
    assert contents(output) == before


def test_missing_png_reextracts_its_pdf_and_removes_stale_images(
    api, pdf_dir, tmp_path
):
    output = tmp_path / "output"
    run(api, pdf_dir, output, extractor(count=3))
    (output / "a_0.png").unlink()
    calls = []

    def fewer(pdf, stage):
        calls.append(pdf.stem)
        png(stage / f"{pdf.stem}_0.png", 50)

    run(api, pdf_dir, output, extractor(fewer), on_existing="resume")
    assert calls == ["a"]
    assert image_names(output) == [
        "a_0.png",
        *[f"{name}_{i}.png" for name in "bc" for i in range(3)],
    ]


def test_missing_image_does_not_allow_an_edited_neighbour_to_be_overwritten(
    api, pdf_dir, tmp_path
):
    output = tmp_path / "output"
    run(api, pdf_dir, output)
    (output / "a_0.png").unlink()
    png(output / "a_1.png", 90)
    before = contents(output)
    never = model_function(Mock(spec=[]), version=1)
    with pytest.raises(ValueError, match="Saved output has been modified"):
        run(api, pdf_dir, output, never, on_existing="resume")
    never.assert_not_called()
    assert contents(output) == before


def test_overwrite_removes_old_outputs_and_accepts_new_settings(api, pdf_dir, tmp_path):
    output = tmp_path / "output"
    run(api, pdf_dir, output, extractor(count=3))
    (output / "a_0.png").unlink()
    (output / "b_0.png").write_text("edited")
    (pdf_dir / "a.pdf").unlink()
    fn = model_function(extractor(count=1), version=2)
    run(api, pdf_dir, output, fn, on_existing="overwrite")
    assert image_names(output) == ["b_0.png", "c_0.png"]


@pytest.mark.parametrize(
    "extra", ["notes.json", "notes.txt", ".flowde/notes.txt", "unrelated"]
)
def test_overwrite_preserves_unrecorded_files(api, pdf_dir, tmp_path, extra):
    output = tmp_path / "output"
    run(api, pdf_dir, output)
    path = output / extra
    if path.suffix:
        path.write_text("user work")
    else:
        path.mkdir()
    before = contents(output)
    with pytest.raises(ValueError, match="unexpected file or directory"):
        run(api, pdf_dir, output, on_existing="overwrite")
    assert contents(output) == before
    assert path.exists()


@pytest.mark.parametrize("mode", ["error", "overwrite"])
def test_fresh_empty_directory_accepts_start_or_overwrite(api, pdf_dir, tmp_path, mode):
    output = tmp_path / "output"
    output.mkdir()
    run(api, pdf_dir, output, on_existing=mode)
    assert len(image_names(output)) == 6


def test_resume_and_overwrite_refuse_untracked_existing_images(api, pdf_dir, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    png(output / "old.png")
    with pytest.raises(ValueError, match="Missing or invalid saved run record"):
        run(api, pdf_dir, output, on_existing="resume")
    with pytest.raises(ValueError, match="Missing or invalid saved run record"):
        run(api, pdf_dir, output, on_existing="overwrite")
    assert image_names(output) == ["old.png"]


@pytest.mark.parametrize("interrupts", [1, 2])
def test_interrupt_finishes_active_pdf_or_forces_stop(
    api, pdf_dir, tmp_path, interrupts
):
    output = tmp_path / "output"
    calls = []

    def stop(pdf, stage):
        calls.append(pdf.stem)
        png(stage / f"{pdf.stem}_0.png")
        handler = signal.getsignal(signal.SIGINT)
        for _ in range(interrupts):
            handler(signal.SIGINT, None)
        png(stage / f"{pdf.stem}_1.png")

    expected = "Stopped after saving active PDFs" if interrupts == 1 else "Forced stop"
    with pytest.raises(KeyboardInterrupt, match=expected):
        run(api, pdf_dir, output, extractor(stop))
    assert calls == ["a"]
    assert image_names(output) == (["a_0.png", "a_1.png"] if interrupts == 1 else [])
    run(api, pdf_dir, output, on_existing="resume")
    assert len(image_names(output)) == 6


def wait_for(path):
    deadline = time.monotonic() + 20
    while not path.exists():
        if time.monotonic() > deadline:
            msg = f"Worker did not reach {path.name}"
            raise AssertionError(msg)
        time.sleep(0.01)


def test_parallel_failure_drains_active_pdf_and_starts_no_more(
    api, pdf_dir, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    started = tmp_path / "b-started"
    observed = tmp_path / "failure-observed"
    unexpected = tmp_path / "c-started"
    original_failure = ExtractionRunState.failure

    def record_failure(state, key, error):
        original_failure(state, key, error)
        observed.touch()

    monkeypatch.setattr(ExtractionRunState, "failure", record_failure)

    def extract(pdf, stage):
        png(stage / f"{pdf.stem}_0.png")
        if pdf.stem == "a":
            wait_for(started)
            msg = "PDF A failed"
            raise RuntimeError(msg)
        if pdf.stem == "b":
            started.touch()
            wait_for(observed)
            png(stage / "b_1.png")
        if pdf.stem == "c":
            unexpected.touch()

    with pytest.raises(RuntimeError, match="PDF A failed"):
        run(api, pdf_dir, output, extractor(extract), n_jobs=2)
    assert image_names(output) == ["b_0.png", "b_1.png"]
    assert not unexpected.exists()
    run(api, pdf_dir, output, on_existing="resume", n_jobs=2)
    assert len(image_names(output)) == 6


@pytest.mark.parametrize("after_write", [False, True])
def test_interrupted_publication_can_resume_or_overwrite(
    api, pdf_dir, tmp_path, monkeypatch, after_write
):
    output = tmp_path / "output"
    original_write = _run_state.atomic_write

    def fail_write(path, data):
        if path.name == "a_1.png":
            if after_write:
                original_write(path, data)
            msg = "saving failed"
            raise OSError(msg)
        original_write(path, data)

    monkeypatch.setattr(_run_state, "atomic_write", fail_write)
    with pytest.raises(OSError, match="saving failed"):
        run(api, pdf_dir, output)
    assert (output / "a_0.png").is_file()
    monkeypatch.setattr(_run_state, "atomic_write", original_write)
    run(api, pdf_dir, output, on_existing="resume")
    assert len(image_names(output)) == 6
    run(api, pdf_dir, output, extractor(count=1), on_existing="overwrite")
    assert image_names(output) == ["a_0.png", "b_0.png", "c_0.png"]


def test_repeated_recovery_handles_new_crops_after_partial_publication(
    pdf_dir, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    run("directory", pdf_dir, output)
    (output / "a_1.png").unlink()
    original_write = _run_state.atomic_write

    def fail_write(path, data):
        if path.name == "a_1.png":
            msg = "saving failed"
            raise OSError(msg)
        original_write(path, data)

    monkeypatch.setattr(_run_state, "atomic_write", fail_write)
    for colour in (30, 40):

        def different_crops(pdf, stage, colour=colour):
            for index in range(2):
                png(stage / f"{pdf.stem}_{index}.png", colour)

        with pytest.raises(OSError, match="saving failed"):
            run(
                "directory",
                pdf_dir,
                output,
                extractor(different_crops),
                on_existing="resume",
            )
    monkeypatch.setattr(_run_state, "atomic_write", original_write)
    run("directory", pdf_dir, output, on_existing="resume")
    assert len(image_names(output)) == 6


def test_output_lock_and_input_protection(api, pdf_dir, tmp_path):
    output = tmp_path / "output"
    with (
        _run_state.output_lock(output),
        pytest.raises(RuntimeError, match="Another run"),
    ):
        run(api, pdf_dir, output)
    before = contents(pdf_dir)
    with pytest.raises(ValueError, match="contains an input file"):
        run(api, pdf_dir, pdf_dir, on_existing="overwrite")
    assert contents(pdf_dir) == before


def test_conflicting_custom_output_names_do_not_overwrite_another_pdf(
    pdf_dir, tmp_path
):
    output = tmp_path / "output"

    def conflicting(pdf, stage):
        png(stage / "same.png", ord(pdf.stem))

    with pytest.raises(ValueError, match="filename conflicts"):
        run("directory", pdf_dir, output, extractor(conflicting))
    with Image.open(output / "same.png") as image:
        assert image.getpixel((0, 0)) == (ord("a"), 0, 0)


@pytest.mark.parametrize(
    "bad_output", ["invalid-png", "text", "subdirectory", "symlink"]
)
def test_invalid_extractor_outputs_never_reach_final_folder(
    pdf_dir, tmp_path, bad_output
):
    output = tmp_path / "output"

    def invalid(pdf, stage):
        png(stage / "good.png")
        if bad_output == "invalid-png":
            (stage / "bad.png").write_text("broken PNG")
        elif bad_output == "text":
            (stage / "notes.txt").write_text("not an image")
        elif bad_output == "subdirectory":
            (stage / "nested").mkdir()
        else:
            (stage / "linked.png").symlink_to(stage / "good.png")

    with pytest.raises((ValueError, OSError)):
        run("directory", pdf_dir, output, extractor(invalid))
    assert image_names(output) == []


def test_changed_pdf_during_extraction_is_not_published(pdf_dir, tmp_path):
    output = tmp_path / "output"

    def changing(pdf, stage):
        png(stage / f"{pdf.stem}_0.png")
        pdf.write_text("changed during extraction")

    with pytest.raises(ValueError, match="Input PDF changed during extraction"):
        run("directory", pdf_dir, output, extractor(changing))
    assert image_names(output) == []


def test_explicit_paths_preserve_order_and_resume_overlapping_selections(
    pdf_dir, tmp_path
):
    output = tmp_path / "output"
    calls = []

    def extract(pdf, stage):
        calls.append(pdf.stem)
        png(stage / f"{pdf.stem}_0.png")

    fn = extractor(extract)
    extract_imgs_pdf_list([pdf_dir / "b.pdf", pdf_dir / "a.pdf"], output, fn)
    assert calls == ["b", "a"]
    calls.clear()
    extract_imgs_pdf_list(
        [pdf_dir / "b.pdf", pdf_dir / "c.pdf"], output, fn, on_existing="resume"
    )
    assert calls == ["c"]
    assert image_names(output) == ["a_0.png", "b_0.png", "c_0.png"]


def test_input_validation_happens_before_extraction(pdf_dir, tmp_path):
    output = tmp_path / "output"
    never = model_function(Mock(spec=[]), version=1)
    for invalid in (tmp_path / "missing", pdf_dir / "a.pdf"):
        with pytest.raises(ValueError, match="does not exist or is not a directory"):
            extract_imgs(invalid, output, never)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a PDF")
    with pytest.raises(ValueError, match="No PDF files"):
        extract_imgs(empty, output, never)
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_imgs_pdf_list([], output, never)
    with pytest.raises(ValueError, match="Duplicate file stem"):
        extract_imgs_pdf_list([pdf_dir / "a.pdf"] * 2, output, never)
    with pytest.raises(ValueError, match="declare their settings"):
        extract_imgs(pdf_dir, output, lambda **kwargs: None)
    assert not output.exists()
    never.assert_not_called()


def test_directory_ignores_nested_pdfs_and_non_pdf_files(pdf_dir, tmp_path):
    (pdf_dir / "notes.txt").write_text("notes")
    nested = pdf_dir / "nested"
    nested.mkdir()
    (nested / "d.pdf").write_text("nested PDF")
    output = tmp_path / "output"
    extract_imgs(pdf_dir, output, extractor(count=1))
    assert image_names(output) == ["a_0.png", "b_0.png", "c_0.png"]


def test_pdf_named_like_metadata_can_still_resume(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / ".flowde.pdf").write_bytes(b"PDF")
    output = tmp_path / "output"
    extract_imgs(pdf_dir, output, extractor(count=1))
    extract_imgs(pdf_dir, output, extractor(count=1), on_existing="resume")
    assert image_names(output) == [".flowde_0.png"]
