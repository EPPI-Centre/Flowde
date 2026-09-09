"""Describe the settings that must stay fixed when resuming a run."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

Function = TypeVar("Function", bound=Callable[..., Any])


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def settings_value(value: Any) -> Any:  # noqa: PLR0911 - One case per supported settings type.
    """Make settings comparable, including the contents of referenced files."""
    if isinstance(value, Path):
        return {"path": str(value.resolve()), "sha256": file_digest(value)}
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value.model_json_schema()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: settings_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: settings_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [settings_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((settings_value(item) for item in value), key=repr)
    return value


def model_function(
    fn: Function,
    *,
    result_structure: type[BaseModel] | None = None,
    **run_settings: Any,
) -> Function:
    """
    Attach declared settings and an optional result schema to a callable.

    Parameters
    ----------
    fn
        Classification or parsing callable. Its arguments and behavior are unchanged.
    result_structure
        Pydantic result class. Required by parsing batches unless already attached
        to `fn`; optional for custom classifiers.
    **run_settings
        Settings that affect the answers, such as a prompt, model, threshold, or
        algorithm version. These describe the callable; they do not configure it.
        File inputs should be `Path` objects so their contents can be checked.

    Returns
    -------
    Function
        The same callable with its `run_settings` attribute set and, when supplied,
        its `result_structure` attribute set.

    """
    configured = cast("Any", fn)
    if result_structure is not None:
        configured.result_structure = result_structure
    if (
        "few_shot_examples" in run_settings
        and run_settings["few_shot_examples"] is None
    ):
        run_settings["few_shot_examples"] = []
    configured.run_settings = run_settings
    return fn


def function_settings(fn: Callable[..., Any]) -> dict[str, Any]:
    declared = getattr(fn, "run_settings", None)
    if declared is None:
        msg = "Custom functions must declare their settings with model_function()."
        raise ValueError(msg)
    settings = {"function": declared}
    structure = getattr(fn, "result_structure", None)
    if structure is not None:
        settings["result_structure"] = structure
    # Round-trip now so unsupported settings fail before starting any requests.
    return cast(
        "dict[str, Any]",
        json.loads(json.dumps(settings_value(settings), allow_nan=False)),
    )


def referenced_files(settings: Any) -> list[Path]:
    """Find file inputs included in a settings snapshot before clearing outputs."""
    if isinstance(settings, dict):
        paths = []
        if isinstance(settings.get("path"), str) and "sha256" in settings:
            paths.append(Path(settings["path"]))
        for value in settings.values():
            paths.extend(referenced_files(value))
        return paths
    if isinstance(settings, list):
        return [path for value in settings for path in referenced_files(value)]
    return []
