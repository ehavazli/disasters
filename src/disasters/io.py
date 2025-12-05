from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

__all__ = [
    "ensure_directory",
    "build_output_path",
    "write_json",
    "read_json",
    "write_text",
    "read_text",
]


def ensure_directory(path: str | Path, create_parents: bool = True) -> Path:
    """Ensure that *path* exists as a directory and return it as a Path.

    - If the path does not exist, it is created.
    - If the path exists but is not a directory, NotADirectoryError is raised.
    - If the path exists and is a directory, it is returned unchanged.

    Parameters
    ----------
    path:
        Directory path to ensure. Can be a string or a Path.
    create_parents:
        If True (default), create parent directories as needed
        (equivalent to ``mkdir(parents=True)``). If False, only create the
        final directory, and let ``mkdir`` raise if parents are missing.

    Returns
    -------
    pathlib.Path
        The directory path as a Path object.
    """
    directory = Path(path)

    if directory.exists():
        if not directory.is_dir():
            raise NotADirectoryError(f"Path exists and is not a directory: {directory}")
        return directory

    if create_parents:
        directory.mkdir(parents=True, exist_ok=True)
    else:
        directory.mkdir()

    log.debug("Created directory: %s", directory)
    return directory


def build_output_path(
    root: str | Path,
    *parts: str | Path,
    suffix: str | None = None,
    create_parents: bool = True,
) -> Path:
    """Build an output path under *root* from the given *parts*.

    Examples
    --------
    >>> build_output_path("work", "flood", "mosaic.tif")
    PosixPath('work/flood/mosaic.tif')

    If *suffix* is given and the final component has no suffix, it is added:

    >>> build_output_path("work", "summary", suffix=".json")
    PosixPath('work/summary.json')

    Parameters
    ----------
    root:
        Root directory under which the path is constructed.
    parts:
        Additional path components (subdirectories and/or filename).
    suffix:
        Optional file suffix to enforce if the final component has no suffix.
        Should include the leading dot, e.g. ".json".
    create_parents:
        If True (default), create the parent directory (and its parents)
        if it does not exist.

    Returns
    -------
    pathlib.Path
        The full output path.
    """
    root_path = ensure_directory(root, create_parents=create_parents)
    path = root_path.joinpath(*map(Path, parts))

    if suffix is not None and path.suffix == "":
        path = path.with_suffix(suffix)

    # Ensure parent exists (root is already handled, but subfolders may be new)
    ensure_directory(path.parent, create_parents=True)
    return path


def write_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write *data* as JSON to *path* and return the Path.

    The parent directory is created if needed.
    The file is written using UTF-8 encoding.

    Parameters
    ----------
    data:
        JSON-serializable Python object.
    path:
        Target file path.
    indent:
        Indentation level for pretty-printing (default: 2).

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    target = Path(path)
    ensure_directory(target.parent, create_parents=True)

    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=True)

    log.debug("Wrote JSON file: %s", target)
    return target


def read_json(path: str | Path) -> Any:
    """Read JSON from *path* and return the parsed object."""
    source = Path(path)

    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)

    log.debug("Read JSON file: %s", source)
    return data


def write_text(lines: str | Iterable[str], path: str | Path) -> Path:
    """Write text or lines of text to *path* and return the Path.

    Parameters
    ----------
    lines:
        Either a single string (written as-is) or an iterable of strings,
        each written as a separate line (newline appended if missing).
    path:
        Target file path.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    target = Path(path)
    ensure_directory(target.parent, create_parents=True)

    if isinstance(lines, str):
        text = lines
    else:
        # Normalize to lines with newlines, then join
        normalized = []
        for line in lines:
            if line.endswith("\n"):
                normalized.append(line)
            else:
                normalized.append(f"{line}\n")
        text = "".join(normalized)

    with target.open("w", encoding="utf-8") as f:
        f.write(text)

    log.debug("Wrote text file: %s", target)
    return target


def read_text(path: str | Path) -> str:
    """Read entire file contents from *path* and return as a string."""
    source = Path(path)

    with source.open("r", encoding="utf-8") as f:
        text = f.read()

    log.debug("Read text file: %s", source)
    return text
