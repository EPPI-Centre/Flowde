from pathlib import Path
from typing import Protocol


class ExtractImgsFunction(Protocol):
    """
    Describe a callable that extracts PNG images from one PDF.

    Pass an extractor with this interface as `extract_fn` to
    [`extract_imgs()`][flowde.extract_imgs.extract_imgs]. Flowde supplies one
    PDF and a temporary output directory per call. The extractor writes the
    PNG files; Flowde validates the files and publishes the completed PDF's
    images in the run's output directory.

    Notes
    -----
    This protocol describes an interface; it does not implement extraction
    or perform runtime validation. A function or callable object can satisfy
    the interface without inheriting from `ExtractImgsFunction`.

    Custom extractors must also declare their settings with
    [`model_function()`][flowde.model_function]. The resulting `run_settings`
    attribute lets Flowde check compatibility when resuming a saved run.
    This requirement applies in addition to the call signature defined here.
    Built-in extractor factories declare their settings automatically.

    """

    def __call__(
        self,
        *,
        pdf_path: Path,
        save_dir: Path,
    ) -> None:
        """
        Extract images from one PDF into the supplied temporary directory.

        Parameters
        ----------
        pdf_path : Path
            Path to the input PDF. Flowde supplies this argument by keyword.
            The extractor must leave the input PDF unchanged.
        save_dir : Path
            Existing, empty temporary directory for this PDF. Flowde supplies
            this argument by keyword. This directory is separate from the
            final `save_dir` passed to
            [`extract_imgs()`][flowde.extract_imgs.extract_imgs].

            Save only valid PNG files with the lowercase `.png` extension
            directly inside this directory. Do not create subdirectories,
            symbolic links or other files. Filenames must be unique across
            all images in the run, including when compared without letter
            case. Including the PDF's filename stem in each PNG's name helps
            avoid collisions between PDFs.

        Returns
        -------
        None
            The saved PNG files are the extraction results. Finish writing
            and close every output file before returning. Leaving `save_dir`
            empty is valid when the PDF contains no images to extract.

        Notes
        -----
        Raise an exception if extraction fails. Flowde publishes a PDF's
        PNG files only after the extractor returns successfully and the PNGs
        pass validation. An extractor must not leave background work writing
        files after returning.

        """
        ...
