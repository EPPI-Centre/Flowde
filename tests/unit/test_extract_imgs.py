from pathlib import Path

import pytest

import flowde.extract_imgs as extract_imgs_module


def create_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake pdf content")


def create_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake png content")


def relative_png_paths(base_dir: Path) -> list[Path]:
    return sorted(path.relative_to(base_dir) for path in base_dir.rglob("*.png"))


def fake_extract_fn(pdf_path: Path, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{pdf_path.stem}_0.png"
    output_path.write_text(f"extracted from {pdf_path.name}")


def extract_should_not_be_called(pdf_path: Path, save_dir: Path) -> None:
    raise AssertionError("extract_fn should not be called")


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_extract_imgs_pdf_list_extracts_images(tmp_path: Path, n_jobs: int) -> None:
    pdf_dirs = [
        tmp_path / "test_dataset" / "paper_1",
        tmp_path / "experiment_data" / "extraction" / "test_dataset" / "paper_2",
    ]
    pdf_paths = [
        pdf_dirs[0] / "paper_1.pdf",
        pdf_dirs[1] / "paper_2.pdf",
    ]
    save_dirs = [
        pdf_dirs[0] / "method_1",
        pdf_dirs[1] / "method_1",
    ]

    for pdf_path in pdf_paths:
        create_pdf(pdf_path)

    result = extract_imgs_module.extract_imgs_pdf_list(
        pdf_paths=pdf_paths,
        save_dirs=save_dirs,
        extract_fn=fake_extract_fn,
        n_jobs=n_jobs,
    )

    assert result is None

    expected_png_paths = sorted(
        [
            (save_dirs[0] / "paper_1_0.png").relative_to(tmp_path),
            (save_dirs[1] / "paper_2_0.png").relative_to(tmp_path),
        ]
    )

    assert relative_png_paths(tmp_path) == expected_png_paths
    assert (save_dirs[0] / "paper_1_0.png").read_text() == "extracted from paper_1.pdf"
    assert (save_dirs[1] / "paper_2_0.png").read_text() == "extracted from paper_2.pdf"


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_extract_imgs_pdf_list_raises_if_save_dirs_length_does_not_match_pdf_paths(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    pdf_dirs = [
        tmp_path / "experiment_data" / "extraction" / "test_dataset" / "paper_1",
        tmp_path / "experiment_data" / "extraction" / "test_dataset" / "paper_2",
    ]
    pdf_paths = [
        pdf_dirs[0] / "paper_1.pdf",
        pdf_dirs[1] / "paper_2.pdf",
    ]
    save_dirs = [
        pdf_dirs[0] / "method_1",
    ]

    for pdf_path in pdf_paths:
        create_pdf(pdf_path)

    with pytest.raises(ValueError) as exc_info:
        extract_imgs_module.extract_imgs_pdf_list(
            pdf_paths=pdf_paths,
            save_dirs=save_dirs,
            extract_fn=extract_should_not_be_called,
            n_jobs=n_jobs,
        )

    assert str(exc_info.value) == "Length of save_dirs must match length of pdf_paths"
    assert not save_dirs[0].exists()
    assert relative_png_paths(tmp_path) == []


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_extract_imgs_extracts_all_pdfs_into_save_dir(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    save_dir = tmp_path / "imgs"

    pdf_paths = [
        pdf_dir / "paper_b.pdf",
        pdf_dir / "paper_a.pdf",
    ]

    for pdf_path in pdf_paths:
        create_pdf(pdf_path)

    result = extract_imgs_module.extract_imgs(
        pdf_dir=pdf_dir,
        save_dir=save_dir,
        extract_fn=fake_extract_fn,
        n_jobs=n_jobs,
    )

    assert result is None

    assert relative_png_paths(tmp_path) == [
        Path("imgs/paper_a_0.png"),
        Path("imgs/paper_b_0.png"),
    ]
    assert (save_dir / "paper_a_0.png").read_text() == "extracted from paper_a.pdf"
    assert (save_dir / "paper_b_0.png").read_text() == "extracted from paper_b.pdf"


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["sequential", "parallel"])
def test_extract_imgs_ignores_non_pdf_files(tmp_path: Path, n_jobs: int) -> None:
    pdf_dir = tmp_path / "pdfs"
    save_dir = tmp_path / "imgs"

    create_pdf(pdf_dir / "paper_1.pdf")
    create_png(pdf_dir / "not_a_pdf.png")
    (pdf_dir / "notes.txt").write_text("not a pdf")

    extract_imgs_module.extract_imgs(
        pdf_dir=pdf_dir,
        save_dir=save_dir,
        extract_fn=fake_extract_fn,
        n_jobs=n_jobs,
    )

    assert relative_png_paths(tmp_path) == [
        Path("imgs/paper_1_0.png"),
        Path("pdfs/not_a_pdf.png"),
    ]
    assert (save_dir / "paper_1_0.png").read_text() == "extracted from paper_1.pdf"
    assert not (save_dir / "not_a_pdf_0.png").exists()
    assert not (save_dir / "notes_0.png").exists()


def test_extract_imgs_raises_if_pdf_dir_does_not_exist(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "missing_pdfs"
    save_dir = tmp_path / "imgs"

    with pytest.raises(ValueError) as exc_info:
        extract_imgs_module.extract_imgs(
            pdf_dir=pdf_dir,
            save_dir=save_dir,
            extract_fn=extract_should_not_be_called,
        )

    assert str(exc_info.value) == (
        f"PDF directory {pdf_dir} does not exist or is not a directory."
    )
    assert not save_dir.exists()
    assert relative_png_paths(tmp_path) == []


def test_extract_imgs_raises_if_pdf_dir_is_a_file(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "not_a_dir.pdf"
    save_dir = tmp_path / "imgs"
    create_pdf(pdf_dir)

    with pytest.raises(ValueError) as exc_info:
        extract_imgs_module.extract_imgs(
            pdf_dir=pdf_dir,
            save_dir=save_dir,
            extract_fn=extract_should_not_be_called,
        )

    assert str(exc_info.value) == (
        f"PDF directory {pdf_dir} does not exist or is not a directory."
    )
    assert not save_dir.exists()
    assert relative_png_paths(tmp_path) == []


def test_extract_imgs_raises_if_no_pdfs_found(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    save_dir = tmp_path / "imgs"

    pdf_dir.mkdir()
    create_png(pdf_dir / "image.png")
    (pdf_dir / "notes.txt").write_text("not a pdf")

    with pytest.raises(ValueError) as exc_info:
        extract_imgs_module.extract_imgs(
            pdf_dir=pdf_dir,
            save_dir=save_dir,
            extract_fn=extract_should_not_be_called,
        )

    assert str(exc_info.value) == f"No PDF files found in directory {pdf_dir}."
    assert not save_dir.exists()
    assert relative_png_paths(tmp_path) == [Path("pdfs/image.png")]


def test_extract_imgs_sorts_pdf_paths_before_extracting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    save_dir = tmp_path / "imgs"

    create_pdf(pdf_dir / "paper_b.pdf")
    create_pdf(pdf_dir / "paper_a.pdf")
    create_pdf(pdf_dir / "paper_c.pdf")

    captured_pdf_paths: list[Path] = []
    captured_save_dirs: list[Path] = []
    captured_n_jobs: list[int] = []

    def fake_extract_imgs_pdf_list(
        pdf_paths: list[Path],
        save_dirs: list[Path],
        extract_fn,
        n_jobs: int = 1,
    ) -> None:
        captured_pdf_paths.extend(pdf_paths)
        captured_save_dirs.extend(save_dirs)
        captured_n_jobs.append(n_jobs)

    monkeypatch.setattr(
        extract_imgs_module,
        "extract_imgs_pdf_list",
        fake_extract_imgs_pdf_list,
    )

    extract_imgs_module.extract_imgs(
        pdf_dir=pdf_dir,
        save_dir=save_dir,
        extract_fn=extract_should_not_be_called,
        n_jobs=3,
    )

    assert captured_pdf_paths == [
        pdf_dir / "paper_a.pdf",
        pdf_dir / "paper_b.pdf",
        pdf_dir / "paper_c.pdf",
    ]
    assert captured_save_dirs == [save_dir, save_dir, save_dir]
    assert captured_n_jobs == [3]
    assert save_dir.exists()

    # This test monkeypatches the extraction call, so no images should be created.
    assert relative_png_paths(tmp_path) == []
