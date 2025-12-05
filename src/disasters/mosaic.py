from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from .auth import authenticate

log = logging.getLogger(__name__)

__all__ = [
    "compile_and_load_data",
    "same_utm_zone",
]


def compile_and_load_data(
    data_layer_links,
    mode,
    conf_layer_links=None,
    date_layer_links=None,
):
    """
    Compile and load data from the provided layer links for mosaicking. This modified version
    loads all datasets regardless of their CRS, but filters to the most common Sentinel-1
    unit (S1A or S1C) if applicable.

    This is a direct refactor of the function from `disaster.py`, with the same behavior.

    Args:
        data_layer_links (list): List of URLs corresponding to the OPERA data layers to mosaic.
        mode (str): Mode of operation, e.g., "flood", "fire", "landslide", "earthquake".
        conf_layer_links (list, optional): List of URLs for additional layers to filter false positives.
        date_layer_links (list, optional): List of URLs for date layers to filter by date.

    Returns:
        DS: List of rioxarray datasets loaded from the provided links (in granule order).
        conf_DS: List of rioxarray datasets for confidence layers (if applicable, in granule order).
        date_DS: List of rioxarray datasets for date layers (if applicable, in granule order).

        The exact tuple structure matches the original implementation:
        - flood: (DS, conf_DS)
        - fire/landslide: (DS, date_DS, conf_DS)
        - else: DS
    Raises:
        Exception: If there is an error loading any of the datasets.
    """
    import rioxarray
    from opera_utils.disp._remote import open_file

    # Preserve original behavior: suppress detailed logging from lower-level libs
    logging.getLogger().setLevel(logging.ERROR)

    # Authenticate to get username and password (Earthdata/ASF)
    username, password = authenticate()

    # Ensure only S1A or S1C are used (not both) for a single date
    satellite_counts = Counter()
    for link in data_layer_links:
        if "S1A" in link:
            satellite_counts["S1A"] += 1
        elif "S1C" in link:
            satellite_counts["S1C"] += 1

    if satellite_counts:
        # Get the satellite type with the highest count
        most_common_satellite, _ = satellite_counts.most_common(1)[0]
        print(
            f"[INFO] Most common satellite type: {most_common_satellite}, "
            f"keeping only those links."
        )

        # Create a boolean mask to filter all lists consistently
        is_most_common = [most_common_satellite in link for link in data_layer_links]

        data_layer_links = [
            link for i, link in enumerate(data_layer_links) if is_most_common[i]
        ]

        # Filter auxiliary links consistently if they exist
        if conf_layer_links:
            conf_layer_links = [
                link for i, link in enumerate(conf_layer_links) if is_most_common[i]
            ]
        if date_layer_links:
            date_layer_links = [
                link for i, link in enumerate(date_layer_links) if is_most_common[i]
            ]

    # Helper to load datasets
    def load_datasets(links):
        datasets = []
        for link in links:
            try:
                datasets.append(rioxarray.open_rasterio(link, masked=False))
            except Exception:
                # Fallback: use open_file helper with Earthdata credentials
                f = open_file(
                    link,
                    earthdata_username=username,
                    earthdata_password=password,
                )
                datasets.append(rioxarray.open_rasterio(f, masked=False))
        return datasets

    # Load the primary data layer (DS)
    DS = load_datasets(data_layer_links)

    # If conf_layer_links AND mode == 'flood' compile and load layers to use in filtering
    if conf_layer_links and mode == "flood":
        conf_DS = load_datasets(conf_layer_links)
        return DS, conf_DS

    # If conf_layer_links AND date_layer_links AND mode == 'fire' or mode == 'landslide'
    # compile and load layers to use in filtering
    if (
        conf_layer_links
        and date_layer_links
        and (mode == "fire" or mode == "landslide")
    ):
        conf_DS = load_datasets(conf_layer_links)
        date_DS = load_datasets(date_layer_links)
        return DS, date_DS, conf_DS
    else:
        return DS


def same_utm_zone(crs_a, crs_b) -> bool:
    """
    Returns True if both are projected UTM with the same EPSG (e.g., 32611 vs 32611).

    This is directly lifted from `disaster.py`. It does not require rasterio
    to be imported here as a type annotation; we only rely on the objects
    having a `.to_epsg()` method, as before.
    """
    try:
        if not crs_a or not crs_b:
            return False
        epsg_a = crs_a.to_epsg()
        epsg_b = crs_b.to_epsg()
        if epsg_a is None or epsg_b is None:
            return False
        # UTM EPSGs are typically 326xx (north) / 327xx (south)
        return epsg_a == epsg_b and (
            str(epsg_a).startswith("326") or str(epsg_a).startswith("327")
        )
    except Exception:
        return False
