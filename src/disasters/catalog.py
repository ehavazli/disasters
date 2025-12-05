from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd

log = logging.getLogger(__name__)


def get_metadata_csv_path(output_dir: str | Path) -> Path:
    """Return the expected path to the OPERA products metadata CSV.

    This mirrors the assumption in the original `read_opera_metadata_csv`
    implementation where the file is always named
    ``opera_products_metadata.csv`` inside the next_pass output directory.
    """
    return Path(output_dir) / "opera_products_metadata.csv"


def read_opera_metadata_csv(output_dir: str | Path) -> pd.DataFrame:
    """Read the OPERA products metadata CSV file and parse ``Start Time``.

    This is a direct refactor of the original ``read_opera_metadata_csv``
    function in ``disaster.py``. Behavior is preserved:

    - The CSV is expected to be named ``opera_products_metadata.csv``
      in ``output_dir``.
    - The ``Start Time`` column is parsed using two possible formats:
      * ``%Y-%m-%dT%H:%M:%S.%fZ`` (non-RTC products)
      * ``%Y-%m-%dT%H:%M:%SZ`` (RTC products)
    - The two parsed series are combined using ``combine_first`` to
      produce a single datetime64[ns] column named ``Start Time``.

    Parameters
    ----------
    output_dir
        Directory containing ``opera_products_metadata.csv``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a parsed ``Start Time`` column (datetime64[ns]).

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    csv_path = get_metadata_csv_path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Keep the same two-format parsing you had before
    format_microseconds = "%Y-%m-%dT%H:%M:%S.%fZ"  # Non-RTC data
    format_seconds_only = "%Y-%m-%dT%H:%M:%SZ"  # RTC data

    df_temp1 = pd.to_datetime(
        df["Start Time"],
        format=format_microseconds,
        errors="coerce",
    )

    df_temp2 = pd.to_datetime(
        df["Start Time"],
        format=format_seconds_only,
        errors="coerce",
    )

    df["Start Time"] = df_temp1.combine_first(df_temp2)

    log.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def add_start_date_column(
    df: pd.DataFrame, *, column_name: str = "Start Date"
) -> pd.DataFrame:
    """Add a ``Start Date`` column derived from ``Start Time``.

    This wraps the logic used in ``generate_products``:

    .. code-block:: python

        df_opera["Start Date"] = df_opera["Start Time"].dt.date.astype(str)

    Parameters
    ----------
    df
        Input DataFrame containing a ``Start Time`` datetime-like column.
    column_name
        Name of the date column to create (default: "Start Date").

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with an added/updated date column.
    """
    if "Start Time" not in df.columns:
        raise KeyError("Expected 'Start Time' column in metadata DataFrame")

    # This is exactly what you had in generate_products
    df[column_name] = df["Start Time"].dt.date.astype(str)
    return df


def get_unique_start_dates(
    df: pd.DataFrame,
    *,
    date_column: str = "Start Date",
) -> List[str]:
    """Return sorted unique date strings from the metadata DataFrame.

    This mirrors what ``generate_products`` does:

    .. code-block:: python

        df_opera["Start Date"] = df_opera["Start Time"].dt.date.astype(str)
        unique_dates = df_opera["Start Date"].dropna().unique()
        unique_dates.sort()

    Parameters
    ----------
    df
        Metadata DataFrame containing ``date_column``.
    date_column
        Name of the date column (default: "Start Date").

    Returns
    -------
    list of str
        Sorted unique date strings.
    """
    if date_column not in df.columns:
        raise KeyError(f"Expected '{date_column}' column in metadata DataFrame")

    unique_dates = df[date_column].dropna().unique()
    unique_dates.sort()
    return list(unique_dates)
