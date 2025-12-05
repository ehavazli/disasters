from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

import numpy as np
import xarray as xr

__all__ = [
    "reclassify_snow_ice_as_water",
    "filter_by_date_and_confidence",
    "compute_date_threshold",
]


def reclassify_snow_ice_as_water(DS, conf_DS):
    """
    Reclassify false snow/ice positives (value 252) as water (value 1) based on the confidence layers. Only applicable for DSWx-HLS.

    Args:
        DS (list): List of rioxarray datasets (BWTR layers).
        conf_DS (list): List of rioxarray datasets (CONF layers).

    Returns:
        list: List of updated rioxarray datasets with 252 reclassified as 1.
        colormap: Colormap from the original datasets (if available).
    """
    from . import opera_mosaic  # keep same behavior, just relative import

    if conf_DS is None:
        raise ValueError("conf_DS must not be None when reclassifying snow/ice.")

    if len(DS) != len(conf_DS):
        raise ValueError("DS and conf_DS must be the same length.")

    values_to_reclassify = [1, 3, 4, 21, 23, 24]

    try:
        colormap = opera_mosaic.get_image_colormap(DS[0])
        print(
            f"[INFO] Colormap successfully retrieved and will be used in reclassified output"
        )
    except Exception:
        print("[INFO] Unable to get colormap")
        colormap = None

    updated_list = []

    for da_data, da_conf in zip(DS, conf_DS):
        # Get the original data values
        data_values = da_data.values.copy()
        conf_values = da_conf.values

        # Identify locations where DS == 252 and conf layer indicates water
        condition = (data_values == 252) & np.isin(conf_values, values_to_reclassify)

        # Reclassify those pixels to 1 (Water)
        data_values[condition] = 1

        # Create updated DataArray
        updated = xr.DataArray(
            data_values, coords=da_data.coords, dims=da_data.dims, attrs=da_data.attrs
        )

        # Preserve spatial metadata
        if hasattr(da_data, "rio"):
            updated = (
                updated.rio.write_nodata(da_data.rio.nodata)
                .rio.write_crs(da_data.rio.crs)
                .rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
                .rio.write_transform(da_data.rio.transform())
            )

        updated_list.append(updated)

    return updated_list, colormap


def filter_by_date_and_confidence(
    DS,
    DS_dates,
    date_threshold,
    DS_conf=None,
    confidence_threshold=None,
    fill_value=None,
):
    """
    Filters each data xarray in `DS` based on:
      - date threshold from `DS_dates`
      - optional confidence threshold from `DS_conf`

    Pixels not meeting the criteria are set to `fill_value`.
    If `fill_value` is None, defaults to da_data.rio.nodata or NaN.

    Parameters
    ----------
    DS : list of xr.DataArray
        List of data granules (e.g., VEG-DIST-STATUS tiles).
    DS_dates : list of xr.DataArray
        List of corresponding date granules.
    date_threshold : int or datetime-like
        Pixels with dates >= this value are retained.
    DS_conf : list of xr.DataArray, optional
        List of confidence rasters corresponding to `DS`. Default is None.
    confidence_threshold : float or int, optional
        Pixels with confidence >= this value are retained.
    fill_value : number, optional
        Value to fill where condition is not met. If None, uses nodata or NaN.

    Returns
    -------
    filtered_list: list of xr.DataArray filtered data granules.
    colormap: Colormap from the original datasets (if available).
    """
    from . import opera_mosaic  # relative import, same logic

    assert len(DS) == len(DS_dates), "DS and DS_dates must be same length"
    if DS_conf is not None:
        assert len(DS_conf) == len(DS), "DS_conf must match DS in length"

    try:
        colormap = opera_mosaic.get_image_colormap(DS[0])
        print(
            "[INFO] Colormap successfully retrieved and will be used in reclassified output"
        )
    except Exception:
        print("[INFO] Unable to get colormap")
        colormap = None

    filtered_list = []

    for i, (da_data, da_date) in enumerate(zip(DS, DS_dates)):
        print(f"[INFO] Filtering granule {i + 1}/{len(DS)}")

        # Date mask
        date_mask = da_date >= date_threshold

        # Optional confidence mask
        if DS_conf is not None and confidence_threshold is not None:
            conf_layer = DS_conf[i]
            print(f"[INFO] Confidence layer shape: {conf_layer.shape}")
            total_pixels = conf_layer.size

            # Construct confidence mask based on confidence_threshold
            conf_mask = conf_layer >= confidence_threshold

            retained_pixels = conf_mask.sum().item()
            print(
                f"[INFO] Confidence retained: {retained_pixels} / {total_pixels} "
                f"({retained_pixels / total_pixels:.2%})"
            )

            max_retained_conf = conf_layer.where(conf_mask).max().item()
            print(f"[INFO] Max confidence among retained pixels: {max_retained_conf}")

            combined_mask = date_mask & conf_mask
        else:
            combined_mask = date_mask

        # Determine fill value
        default_nodata = (
            da_data.rio.nodata
            if hasattr(da_data, "rio") and da_data.rio.nodata is not None
            else da_data.attrs.get("_FillValue", np.nan)
        )
        replacement = fill_value if fill_value is not None else default_nodata

        # Apply mask
        filtered = xr.where(combined_mask, da_data, replacement)

        # Preserve metadata
        filtered.attrs.update(da_data.attrs)

        if hasattr(da_data, "rio"):
            filtered = (
                filtered.rio.write_nodata(replacement)
                .rio.write_crs(da_data.rio.crs)
                .rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
                .rio.write_transform(da_data.rio.transform())
            )

        filtered_list.append(filtered)

    return filtered_list, colormap


def compute_date_threshold(date_str: str) -> int:
    """
    Compute the date threshold in days from a reference date (2020-12-31).

    Args:
        date_str (str): Date string in the format YYYY-MM-DD.

    Returns:
        date_threshold (int): Number of days from the reference date to the target date.
    """
    # Define the reference date and the target date
    reference_date = datetime(2020, 12, 31)
    target_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Calculate the difference between the two dates
    delta = target_date - reference_date

    # Get the number of days from the timedelta object
    date_threshold = delta.days

    return date_threshold
