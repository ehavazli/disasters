from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from osgeo import gdal

from .io import ensure_directory

__all__ = ["compute_and_write_difference", "save_gtiff_as_cog"]


def compute_and_write_difference(
    earlier_path: Path,
    later_path: Path,
    out_path: Path,
    nodata_value: float | int | None = None,
    log: bool = False,
) -> None:
    """Create a difference raster for either 'flood' or 'landslide' mode.

    This is a refactor of the original ``compute_and_write_difference`` from
    ``disaster.py`` with the same behavior:

    - For DSWx (``log=False``): compute a categorical transition code map,
      with codes C = L * 4 + E for classes in {0, 1, 2, 3}.
    - For RTC (``log=True``): compute a 10 * log10(Later / Earlier) difference.
    - If either input pixel is nodata, the output pixel is nodata.
    - Writes a Cloud-Optimized GeoTIFF (COG) to ``out_path`` via
      :func:`save_gtiff_as_cog`.

    Parameters
    ----------
    earlier_path
        Path to the "earlier" raster (GeoTIFF/COG).
    later_path
        Path to the "later" raster.
    out_path
        Output path for the difference COG.
    nodata_value
        Optional explicit nodata value to use. If None, attempts to inherit
        from inputs.
    log
        If True, compute a log-ratio difference (RTC-S1 case).
        If False, compute categorical transition codes (DSWx case).
    """
    import rioxarray
    import xarray as xr

    # Ensure parent directory for out_path exists (paranoid but cheap)
    ensure_directory(out_path.parent)

    # Open rasters and apply mask for nodata handling
    da_later = rioxarray.open_rasterio(later_path, masked=True)
    da_early = rioxarray.open_rasterio(earlier_path, masked=True)

    # Determine the nodata value to use
    nd = nodata_value
    if nd is None:
        nd = da_later.rio.nodata
        if nd is None:
            nd = da_early.rio.nodata

    # Difference calculation (based on 'log' argument)
    if log:
        diff = 10 * np.log10(da_later / da_early)
        diff = diff.astype("float32")
        print("[INFO] Computed log difference for RTC-S1.")
        # Clear existing metadata and set a description
        diff.attrs.clear()
        diff.attrs["DESCRIPTION"] = "Log Ratio Difference (Later / Earlier) for RTC-S1"
        metadata_to_save: dict[str, str] = {}
    else:
        print("[INFO] Computed categorical transition codes for DSWx.")
        VALID_CLASSES = [0, 1, 2, 3]
        MAX_CLASS_VALUE = 4

        # Define the transition codes and their descriptions (L * 4 + E)
        TRANSITION_DESCRIPTIONS = {
            # E (Earlier) -> L (Later)
            # WATER CHANGE (LOSS/RECESSION) (Later < Earlier)
            1: "Loss: Open Water (1) -> Not Water (0)",
            2: "Loss: Partial Water (2) -> Not Water (0)",
            3: "Loss: Inundated Veg (3) -> Not Water (0)",
            9: "Loss/Change: Partial Water (2) -> Open Water (1)",
            13: "Loss/Change: Inundated Veg (3) -> Open Water (1)",
            14: "Loss/Change: Inundated Veg (3) -> Partial Water (2)",
            # WATER CHANGE (GAIN/INUNDATION) (Later > Earlier)
            4: "Inundation: Not Water (0) -> Open Water (1)",
            8: "Inundation: Not Water (0) -> Partial Water (2)",
            12: "Inundation: Not Water (0) -> Inundated Veg (3)",
            6: "Change/Gain: Open Water (1) -> Partial Water (2)",
            7: "Change/Gain: Open Water (1) -> Inundated Veg (3)",
            11: "Change/Gain: Partial Water (2) -> Inundated Veg (3)",
            # NO CHANGE (Later = Earlier)
            0: "No Change: Not Water (0) -> Not Water (0)",
            5: "No Change: Open Water (1) -> Open Water (1)",
            10: "No Change: Partial Water (2) -> Partial Water (2)",
            15: "No Change: Inundated Veg (3) -> Inundated Veg (3)",
        }

        # Compute the transition code: Code = L * MAX_CLASS_VALUE + E
        L = da_later.fillna(0).astype(int)
        E = da_early.fillna(0).astype(int)
        transition_code = L * MAX_CLASS_VALUE + E

        # Create and apply a mask for invalid classes (25x values)
        invalid_class_mask = ~(np.isin(L, VALID_CLASSES) & np.isin(E, VALID_CLASSES))
        diff = xr.where(invalid_class_mask, np.nan, transition_code).astype(np.float32)

        # Prepare metadata for saving
        diff.attrs.clear()
        diff.attrs["DESCRIPTION"] = (
            "Categorical Transition Map (DSWx-HLS/S1 Water Products)"
        )

        # Convert the Python dict to GDAL metadata keys (KEY=VALUE) for GIS software
        metadata_to_save = {
            f"TRANSITION_CODE_{code}": desc
            for code, desc in TRANSITION_DESCRIPTIONS.items()
        }
        metadata_to_save["CODING_SCHEME"] = (
            "Transition Code (C) = L * 4 + E, where E, L are classes (0-3) in Earlier "
            "and Later rasters."
        )

    # Compute a mask for any location that was nodata in either input
    input_nodata_mask = xr.where(
        xr.ufuncs.isnan(da_later) | xr.ufuncs.isnan(da_early),
        True,
        False,
    )

    # Remove any Inf or NaN values that may have resulted from the difference calculation
    artifact_mask = xr.where(
        xr.ufuncs.isinf(diff) | xr.ufuncs.isnan(diff),
        True,
        False,
    )

    # Combine the masks
    final_nodata_mask = input_nodata_mask | artifact_mask

    # Apply the nodata value to masked pixels and set metadata
    if nd is not None:
        # Wherever the mask is True (input was nodata), set the difference to 'nd'
        diff = xr.where(final_nodata_mask, nd, diff)

        # Write nodata value and CRS metadata
        diff.rio.write_nodata(nd, encoded=True, inplace=True)
        diff.rio.write_crs(da_later.rio.crs, inplace=True)

        # Write the resulting difference array to a temporary GeoTIFF
        tmp_gtiff = out_path.with_suffix(".tmp.tif")
        ensure_directory(tmp_gtiff.parent)

        # Save with metadata
        diff.rio.to_raster(
            tmp_gtiff,
            compress="DEFLATE",
            tiled=True,
            dtype="float32",
            **{"GDAL_METADATA": metadata_to_save},
        )

        # Convert the temporary GeoTIFF to a Cloud Optimized GeoTIFF (COG)
        save_gtiff_as_cog(tmp_gtiff, out_path)

        # Clean up the temporary file
        try:
            tmp_gtiff.unlink(missing_ok=True)
        except Exception:
            pass


def save_gtiff_as_cog(src_path: Path, dst_path: Path | None = None) -> None:
    """Translate a GeoTIFF to a Cloud Optimized GeoTIFF (COG).

    This is a direct copy of the `save_gtiff_as_cog` helper from `disaster.py`,
    preserved as-is so you can later route all diff / mosaic outputs through
    this utility from the pipeline.

    Parameters
    ----------
    src_path
        Source GeoTIFF path.
    dst_path
        Destination path. If None or equal to ``src_path``, the translation is
        done in-place via a temporary file + rename.
    """
    if dst_path is None or src_path == dst_path:
        tmp_path = src_path.with_suffix(".cog.tmp.tif")
        dst_path = tmp_path
        in_place = True
    else:
        in_place = False

    # Ensure parent directory exists for destination (paranoid guard)
    ensure_directory(Path(dst_path).parent)

    ds = gdal.Open(str(src_path))
    if ds is None:
        raise RuntimeError(f"Could not open {src_path} for COG translation")

    creation_opts = [
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "BLOCKSIZE=512",
        "OVERVIEWS=IGNORE_EXISTING",
        "LEVEL=9",
        "BIGTIFF=IF_SAFER",
        "SPARSE_OK=YES",
        "RESAMPLING=AVERAGE",
    ]
    gdal.Translate(str(dst_path), ds, format="COG", creationOptions=creation_opts)

    if in_place:
        os.replace(dst_path, src_path)
