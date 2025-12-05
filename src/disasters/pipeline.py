from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import next_pass

from .catalog import read_opera_metadata_csv
from .diff import compute_and_write_difference, save_gtiff_as_cog
from .filters import (
    compute_date_threshold,
    filter_by_date_and_confidence,
    reclassify_snow_ice_as_water,
)
from .io import ensure_directory, write_json
from .layouts import make_layout, make_map
from .mosaic import compile_and_load_data, same_utm_zone


@dataclass
class PipelineConfig:
    """Configuration for running the OPERA disaster pipeline.

    This mirrors the CLI arguments in the original `disaster.py`:

    - bbox, zoom_bbox
    - output_dir
    - short_name, layer_name (currently unused, kept for future filtering)
    - date, number_of_dates
    - mode, functionality
    - layout_title
    - filter_date
    - reclassify_snow_ice
    """

    bbox: Sequence[float]
    output_dir: Path
    layout_title: str

    zoom_bbox: Sequence[float] | None = None
    short_name: str | None = None
    layer_name: str | None = None
    date: str | None = None
    number_of_dates: int = 5
    mode: str = "flood"  # 'flood', 'fire', 'landslide', 'earthquake'
    functionality: str = "opera_search"  # 'opera_search' or 'both'
    filter_date: str | None = None
    reclassify_snow_ice: bool = False


def run_pipeline(config: PipelineConfig) -> Path | None:
    """Run the end-to-end disaster pipeline (CLI-independent).

    This is a refactor of the original `main()` in `disaster.py`, but driven by
    a `PipelineConfig` instead of `argparse.Namespace`. It:

    1. Optionally exits early for `earthquake` mode (as before).
    2. Calls `next_pass.run_next_pass` to produce the OPERA metadata + URLs.
    3. Moves the `next_pass` output directory under `config.output_dir`.
    4. Reads and parses the OPERA metadata CSV.
    5. Creates a per-mode directory.
    6. Calls `generate_products` to build mosaics, maps, and layouts.

    Returns
    -------
    Path or None
        The mode directory path (e.g., `<output_dir>/flood`) if the pipeline ran,
        or None if exited early (e.g., earthquake mode).
    """
    # Terminate if user selects 'earthquake' mode, for now
    if config.mode == "earthquake":
        print("[INFO] Earthquake mode coming soon. Exiting...")
        return None

    # 1) Run next_pass
    output_dir = next_pass.run_next_pass(
        bbox=config.bbox,
        number_of_dates=config.number_of_dates,
        date=config.date,
        functionality=config.functionality,
    )
    output_dir = Path(output_dir)

    # 2) Ensure root output directory exists (via io.ensure_directory)
    ensure_directory(config.output_dir)
    print(f"[INFO] Created or reused output directory: {config.output_dir}")

    # 3) Move next_pass output under root
    dest = config.output_dir / output_dir.name
    output_dir.rename(dest)
    print(f"[INFO] Moved next_pass output directory to {dest}")

    # 4) Read metadata CSV
    df_opera = read_opera_metadata_csv(dest)

    # 5) Make a new directory with the mode name
    mode_dir = config.output_dir / config.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created mode directory: {mode_dir}")

    # 6) Generate products
    generate_products(
        df_opera=df_opera,
        mode=config.mode,
        mode_dir=mode_dir,
        layout_title=config.layout_title,
        bbox=list(config.bbox),
        zoom_bbox=list(config.zoom_bbox) if config.zoom_bbox is not None else None,
        filter_date=config.filter_date,
        reclassify_snow_ice=config.reclassify_snow_ice,
    )

    return mode_dir


def generate_products(
    df_opera,
    mode,
    mode_dir: Path,
    layout_title: str,
    bbox: list[float],
    zoom_bbox: list[float] | None,
    filter_date: str | None = None,
    reclassify_snow_ice: bool = False,
) -> None:
    """Generate mosaicked products, maps, and layouts for the given mode.

    This is a direct refactor of the original `generate_products` from
    `disaster.py`, but wired to use helper modules:

    - `compile_and_load_data` & `same_utm_zone` from `disasters.mosaic`
    - `reclassify_snow_ice_as_water`, `filter_by_date_and_confidence`,
      `compute_date_threshold` from `disasters.filters`
    - `compute_and_write_difference`, `save_gtiff_as_cog` from `disasters.diff`
    - `make_map`, `make_layout` from `disasters.layouts`

    Behavior and outputs are preserved.
    """
    import json
    import re
    from collections import defaultdict

    import rasterio
    from osgeo import gdal
    from rasterio.shutil import copy

    import opera_mosaic

    # Create data/maps/layouts directories using io.ensure_directory
    data_dir = ensure_directory(mode_dir / "data")
    print(f"[INFO] Created or reused output directory: {data_dir}")

    maps_dir = ensure_directory(mode_dir / "maps")
    print(f"[INFO] Created or reused output directory: {maps_dir}")

    layouts_dir = ensure_directory(mode_dir / "layouts")
    print(f"[INFO] Created or reused output directory: {layouts_dir}")

    # Mode-specific product + layer sets
    if mode == "flood":
        short_names = ["OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"]
        layer_names = ["WTR", "BWTR"]
    elif mode == "fire":
        short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ALERT-S1_V1"]
        layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS"]
    elif mode == "landslide":
        short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L2_RTC-S1_V1"]
        layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS", "RTC-VV", "RTC-VH"]
    elif mode == "earthquake":
        print("[INFO] Earthquake mode coming soon. Exiting...")
        return
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Extract and find unique dates, sort them
    df_opera["Start Date"] = df_opera["Start Time"].dt.date.astype(str)
    unique_dates = df_opera["Start Date"].dropna().unique()
    unique_dates.sort()

    # Create an index of mosaics created for use in pair-wise differencing
    mosaic_index = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for date in unique_dates:
        df_on_date = df_opera[df_opera["Start Date"] == date]

        for short_name in short_names:
            df_sn = df_on_date[df_on_date["Dataset"] == short_name]

            if df_sn.empty:
                continue

            for layer in layer_names:
                url_column = f"Download URL {layer}"
                if url_column not in df_sn.columns:
                    continue

                urls = df_sn[url_column].dropna().tolist()
                if not urls:
                    continue

                print(f"[INFO] Processing {short_name} - {layer} on {date}")
                print(f"[INFO] Found {len(urls)} URLs")

                layout_date = ""
                DS, conf_DS, date_DS = None, None, None

                if mode == "fire":
                    date_column = "Download URL VEG-DIST-DATE"
                    conf_column = "Download URL VEG-DIST-CONF"
                    date_layer_links = (
                        df_sn[date_column].dropna().tolist()
                        if date_column in df_sn.columns
                        else []
                    )
                    conf_layer_links = (
                        df_sn[conf_column].dropna().tolist()
                        if conf_column in df_sn.columns
                        else []
                    )
                    if not date_layer_links:
                        print(
                            f"[WARN] No VEG-DIST-DATE URLs found for {short_name} on {date}"
                        )
                    else:
                        print(
                            f"[INFO] Found {len(date_layer_links)} VEG-DIST-DATE URLs"
                        )
                    if not conf_layer_links:
                        print(
                            f"[WARN] No VEG-DIST-CONF URLs found for {short_name} on {date}"
                        )
                    else:
                        print(f"[INFO] Found {len(conf_layer_links)} CONF URLs")

                    DS, date_DS, conf_DS = compile_and_load_data(
                        urls,
                        mode,
                        conf_layer_links=conf_layer_links,
                        date_layer_links=date_layer_links,
                    )
                    if filter_date:
                        date_threshold = compute_date_threshold(filter_date)
                        layout_date = str(filter_date)
                        print(
                            f"[INFO] date_threshold set to {date_threshold} "
                            f"for filter_date {filter_date}"
                        )
                    else:
                        date_threshold = 0
                        layout_date = "All Dates"
                        print(
                            f"[INFO] date_threshold set to {date_threshold} "
                            f"for filter_date {filter_date}"
                        )

                elif mode == "landslide":
                    if short_name == "OPERA_L3_DIST-ALERT-HLS_V1":
                        date_column = "Download URL VEG-DIST-DATE"
                        conf_column = "Download URL VEG-DIST-CONF"
                        date_layer_links = (
                            df_sn[date_column].dropna().tolist()
                            if date_column in df_sn.columns
                            else []
                        )
                        conf_layer_links = (
                            df_sn[conf_column].dropna().tolist()
                            if conf_column in df_sn.columns
                            else []
                        )
                        if not date_layer_links:
                            print(
                                f"[WARN] No VEG-DIST-DATE URLs found for {short_name} "
                                f"on {date}"
                            )
                        else:
                            print(
                                f"[INFO] Found {len(date_layer_links)} VEG-DIST-DATE URLs"
                            )
                        if not conf_layer_links:
                            print(
                                f"[WARN] No VEG-DIST-CONF URLs found for {short_name} "
                                f"on {date}"
                            )
                        else:
                            print(f"[INFO] Found {len(conf_layer_links)} CONF URLs")

                        DS, date_DS, conf_DS = compile_and_load_data(
                            urls,
                            mode,
                            conf_layer_links=conf_layer_links,
                            date_layer_links=date_layer_links,
                        )

                        if filter_date:
                            date_threshold = compute_date_threshold(filter_date)
                            layout_date = str(filter_date)
                            print(
                                f"[INFO] date_threshold set to {date_threshold} "
                                f"for filter_date {filter_date}"
                            )
                        else:
                            date_threshold = 0
                            layout_date = "All Dates"
                            print(
                                f"[INFO] date_threshold set to {date_threshold} "
                                f"for filter_date {filter_date}"
                            )

                    elif short_name == "OPERA_L2_RTC-S1_V1":
                        DS = compile_and_load_data(urls, mode)

                elif mode == "flood":
                    conf_column = "Download URL CONF"
                    conf_layer_links = (
                        df_sn[conf_column].dropna().tolist()
                        if conf_column in df_sn.columns
                        else []
                    )
                    if not conf_layer_links:
                        print(f"[WARN] No CONF URLs found for {short_name} on {date}")
                        conf_DS = None
                    else:
                        print(f"[INFO] Found {len(conf_layer_links)} CONF URLs")

                    DS, conf_DS = compile_and_load_data(
                        urls, mode, conf_layer_links=conf_layer_links
                    )

                # Group loaded DataArrays by CRS (UTM Zone)
                crs_groups = defaultdict(list)
                conf_groups = defaultdict(list)
                date_groups = defaultdict(list)

                # Ensure all lists are non-empty before zipping
                if not DS:
                    print(
                        f"[WARN] No datasets loaded for {short_name} - {layer} on {date}. "
                        f"Skipping."
                    )
                    continue

                # Determine auxiliary list lengths for zipping
                aux_lists = []
                if conf_DS is not None and mode == "flood":
                    aux_lists.append(conf_DS)
                elif conf_DS is not None and mode in ["fire", "landslide"]:
                    aux_lists.extend([date_DS, conf_DS])

                if aux_lists:
                    # Zip DS with auxiliary layers (conf_DS, date_DS)
                    for i, (da_data, *aux_data) in enumerate(zip(DS, *aux_lists)):
                        try:
                            crs_str = str(da_data.rio.crs)
                        except AttributeError:
                            print(f"[WARN] Granule {i} missing CRS metadata. Skipping.")
                            continue

                        crs_groups[crs_str].append(da_data)
                        if mode == "flood":
                            conf_groups[crs_str].append(aux_data[0])
                        elif mode in ["fire", "landslide"] and short_name.startswith(
                            "OPERA_L3_DIST"
                        ):
                            date_groups[crs_str].append(aux_data[0])
                            conf_groups[crs_str].append(aux_data[1])
                else:
                    # Only DS is present (e.g., RTC-S1)
                    for i, da_data in enumerate(DS):
                        try:
                            crs_str = str(da_data.rio.crs)
                        except AttributeError:
                            print(f"[WARN] Granule {i} missing CRS metadata. Skipping.")
                            continue
                        crs_groups[crs_str].append(da_data)

                # Iterate through each CRS group to process and mosaic
                for crs_str, ds_group in crs_groups.items():

                    # Generate a descriptive UTM suffix (e.g., _18N, _17S)
                    utm_suffix = ""

                    # Try to extract the UTM Zone number and Hemisphere (N/S) from the CRS string
                    utm_match = re.search(r"UTM Zone (\d{1,2})([NS])", crs_str)

                    if utm_match:
                        zone_number = utm_match.group(1)
                        hemisphere = utm_match.group(2)
                        utm_suffix = f"_{zone_number}{hemisphere}"
                        print(f"[INFO] Detected UTM Zone: {utm_suffix}")
                    else:
                        # Fallback to EPSG if UTM info isn't explicitly in the WKT string
                        crs_match = re.search(r"EPSG:(\d+)", crs_str)
                        if crs_match:
                            epsg_code = int(crs_match.group(1))

                            # Standard UTM EPSG codes are 326xx (North) or 327xx (South)
                            if 32600 <= epsg_code <= 32660:
                                zone_number = epsg_code - 32600
                                utm_suffix = f"_{zone_number}N"
                                print(
                                    f"[INFO] Deduced UTM Zone from N-EPSG: {utm_suffix}"
                                )
                            elif 32700 <= epsg_code <= 32760:
                                zone_number = epsg_code - 32700
                                utm_suffix = f"_{zone_number}S"
                                print(
                                    f"[INFO] Deduced UTM Zone from S-EPSG: {utm_suffix}"
                                )
                            else:
                                # Fallback for non-standard or unknown EPSG
                                utm_suffix = f"_EPSG{epsg_code}"
                                print(
                                    f"[WARN] Non-UTM EPSG code {epsg_code}. "
                                    f"Using suffix: {utm_suffix}"
                                )
                        else:
                            # Last resort: use a hash of the full CRS string
                            print(
                                f"[WARN] Could not parse UTM or EPSG from CRS string: "
                                f"{crs_str}. Using raw hash."
                            )
                            utm_suffix = f"_Hash{hash(crs_str) % 10000}"

                    current_conf_DS = conf_groups.get(crs_str)
                    current_date_DS = date_groups.get(crs_str)

                    colormap = None  # Initialize colormap

                    # Filtering/Reclassification (Per CRS Group)
                    if mode == "fire" or (
                        mode == "landslide" and short_name.startswith("OPERA_L3_DIST")
                    ):
                        # Filter DIST layers by date and confidence
                        ds_group, colormap = filter_by_date_and_confidence(
                            ds_group,
                            current_date_DS,
                            date_threshold,
                            DS_conf=current_conf_DS,
                            confidence_threshold=0,
                            fill_value=None,
                        )

                    elif mode == "flood":
                        if (
                            reclassify_snow_ice is True
                            and short_name == "OPERA_L3_DSWX-HLS_V1"
                            and layer in ["BWTR", "WTR"]
                        ):
                            # Reclassify false snow/ice positives in DSWX-HLS only
                            if current_conf_DS is None:
                                print(
                                    f"[WARN] CONF layers not available; skipping "
                                    f"snow/ice reclassification for {short_name} "
                                    f"on {date}"
                                )
                            else:
                                print(
                                    f"[INFO] Reclassifying false snow/ice positives "
                                    f"as water based on CONF layers for CRS {utm_suffix}"
                                )
                                ds_group, colormap = reclassify_snow_ice_as_water(
                                    ds_group, current_conf_DS
                                )
                        else:
                            if (
                                reclassify_snow_ice is True
                                and short_name != "OPERA_L3_DSWX-HLS_V1"
                            ):
                                print(
                                    "[INFO] Snow/ice reclassification is only "
                                    "applicable to DSWx-HLS. Skipping for "
                                    f"{short_name}."
                                )
                            else:
                                print(
                                    "[INFO] Snow/ice reclassification not requested; "
                                    "proceeding without reclassification."
                                )

                    # Use pre-determined colormap or make another attempt to get it.
                    if colormap is None:
                        try:
                            colormap = opera_mosaic.get_image_colormap(ds_group[0])
                        except Exception:
                            colormap = None

                    # Mosaic the datasets using the appropriate method/rule
                    mosaic, _, nodata = opera_mosaic.mosaic_opera(
                        ds_group, product=short_name, merge_args={}
                    )
                    image = opera_mosaic.array_to_image(
                        mosaic, colormap=colormap, nodata=nodata
                    )

                    # Create filename and full paths
                    mosaic_name = f"{short_name}_{layer}_{date}{utm_suffix}_mosaic.tif"
                    mosaic_path = data_dir / mosaic_name
                    tmp_path = data_dir / f"tmp_{mosaic_name}"

                    # Save the mosaic to a temporary GeoTIFF
                    copy(image, tmp_path, driver="GTiff")

                    warp_args: dict = {
                        "xRes": 30,
                        "yRes": 30,
                        "creationOptions": ["COMPRESS=DEFLATE"],
                    }

                    if short_name.startswith("OPERA_L2_RTC"):
                        warp_args["outputType"] = gdal.GDT_Float32

                    # Reproject/compress using GDAL directly into the final GeoTIFF
                    gdal.Warp(mosaic_path, tmp_path, **warp_args)

                    # Convert to COG (writes back into mosaic_path)
                    save_gtiff_as_cog(mosaic_path, mosaic_path)

                    print(f"[INFO] Mosaic written as COG: {mosaic_path}")

                    # Clean up tmp file
                    if tmp_path.exists():
                        tmp_path.unlink()

                    # Add info to the mosiac index for pair-wise differencing
                    with rasterio.open(mosaic_path) as ds:
                        mosaic_crs = ds.crs

                    # UPDATED STRUCTURE: [short_name][layer][date] = {utm_suffix: info_dict}
                    mosaic_index[short_name][layer][str(date)][utm_suffix] = {
                        "path": mosaic_path,
                        "crs": mosaic_crs,
                    }
                    # Make a map with PyGMT
                    map_name = make_map(
                        maps_dir,
                        mosaic_path,
                        short_name,
                        layer,
                        date,
                        bbox,
                        zoom_bbox,
                        is_difference=False,
                        utm_suffix=utm_suffix,
                    )

                    # Make a PDF layout
                    make_layout(
                        layouts_dir,
                        map_name,
                        short_name,
                        layer,
                        date,
                        layout_date,
                        layout_title,
                        reclassify_snow_ice,
                        utm_suffix=utm_suffix,
                    )

    # Pair-wise differencing for 'flood' mode
    if mode == "flood":
        print("[INFO] Computing pairwise differences between water products...")
        skipped = []

        for short_name_k, layers_dict in mosaic_index.items():
            for layer_k, date_map in layers_dict.items():

                # Restructure by UTM zone for differencing
                utm_date_map = defaultdict(lambda: defaultdict(dict))
                for d, utm_dict in date_map.items():
                    for utm, info in utm_dict.items():
                        utm_date_map[utm][d] = info

                for utm_suffix_k, utm_map in utm_date_map.items():
                    dates = sorted(utm_map.keys())

                    # Generate difference for all possible pairs within this UTM zone
                    for i in range(len(dates)):
                        for j in range(i + 1, len(dates)):
                            d_early = dates[i]
                            d_later = dates[j]

                            early_info = utm_map[d_early]
                            later_info = utm_map[d_later]

                            crs_a = early_info["crs"]
                            crs_b = later_info["crs"]

                            # The check will always pass if utm_map was created correctly.
                            if not same_utm_zone(crs_a, crs_b):
                                # This block should rarely be hit if the grouping is correct
                                skipped.append(
                                    {
                                        "short_name": short_name_k,
                                        "layer": layer_k,
                                        "date_earlier": d_early,
                                        "date_later": d_later,
                                        "utm_suffix": utm_suffix_k,
                                        "crs_earlier": (
                                            crs_a.to_string() if crs_a else None
                                        ),
                                        "crs_later": (
                                            crs_b.to_string() if crs_b else None
                                        ),
                                        "reason": "internal CRS mismatch after grouping (error)",
                                    }
                                )
                                continue

                            # Name and path: {short}_{layer}_{LATER}_{EARLIER}{UTM}_diff.tif
                            diff_name = (
                                f"{short_name_k}_{layer_k}_{d_later}_{d_early}"
                                f"{utm_suffix_k}_diff.tif"
                            )
                            diff_path = (mode_dir / "data") / diff_name

                            try:
                                compute_and_write_difference(
                                    earlier_path=early_info["path"],
                                    later_path=later_info["path"],
                                    out_path=diff_path,
                                    nodata_value=None,
                                    log=False,
                                )
                                print(f"[INFO] Wrote diff COG: {diff_path}")

                                # Make a map with PyGMT
                                diff_date_str = f"{d_later}_{d_early}"
                                map_name = make_map(
                                    maps_dir,
                                    diff_path,
                                    short_name_k,
                                    layer_k,
                                    diff_date_str,
                                    bbox,
                                    zoom_bbox,
                                    is_difference=True,
                                    utm_suffix=utm_suffix_k,
                                )

                                # Make a PDF layout
                                if map_name:
                                    diff_date_str_layout = f"{d_early}, {d_later}"
                                    make_layout(
                                        layouts_dir,
                                        map_name,
                                        short_name_k,
                                        layer_k,
                                        diff_date_str,
                                        diff_date_str_layout,
                                        layout_title,
                                        reclassify_snow_ice,
                                        utm_suffix=utm_suffix_k,
                                    )

                            except Exception as e:  # noqa: PERF203
                                skipped.append(
                                    {
                                        "short_name": short_name_k,
                                        "layer": layer_k,
                                        "date_earlier": d_early,
                                        "date_later": d_later,
                                        "utm_suffix": utm_suffix_k,
                                        "error": str(e),
                                        "reason": (
                                            "no overlapping data values; both rasters "
                                            "contain only nodata in the overlap region."
                                        ),
                                    }
                                )

        # Report skipped pairs due to CRS/UTM differences or errors
        report_path = (mode_dir / "data") / "difference_skipped_pairs.json"
        write_json(skipped, report_path)
        print(f"[INFO] Difference skip report: {report_path} ({len(skipped)} skipped)")

    # Pair-wise differencing for 'landslide' mode (RTC-S1 log difference)
    if mode == "landslide":
        print(
            "[INFO] Computing pairwise log difference between RTC backscatter products..."
        )
        skipped = []

        for short_name_k, layers_dict in mosaic_index.items():
            for layer_k, date_map in layers_dict.items():

                utm_date_map = defaultdict(lambda: defaultdict(dict))
                for d, utm_dict in date_map.items():
                    for utm, info in utm_dict.items():
                        utm_date_map[utm][d] = info

                for utm_suffix_k, utm_map in utm_date_map.items():
                    # Only compute log-diff for RTC products
                    if short_name_k != "OPERA_L2_RTC-S1_V1":
                        continue

                    dates = sorted(utm_map.keys())

                    # Generate difference for all possible pairs within this UTM zone
                    for i in range(len(dates)):
                        for j in range(i + 1, len(dates)):
                            d_early = dates[i]
                            d_later = dates[j]

                            early_info = utm_map[d_early]
                            later_info = utm_map[d_later]

                            # The UTM/CRS check is now mostly redundant since they are grouped.
                            crs_a = early_info["crs"]
                            crs_b = later_info["crs"]
                            if not same_utm_zone(crs_a, crs_b):
                                skipped.append(
                                    {
                                        "short_name": short_name_k,
                                        "layer": layer_k,
                                        "date_earlier": d_early,
                                        "date_later": d_later,
                                        "utm_suffix": utm_suffix_k,
                                        "crs_earlier": (
                                            crs_a.to_string() if crs_a else None
                                        ),
                                        "crs_later": (
                                            crs_b.to_string() if crs_b else None
                                        ),
                                        "reason": "internal CRS mismatch after grouping (error)",
                                    }
                                )
                                continue

                            # Name and path: {short}_{layer}_{LATER}_{EARLIER}{UTM}_log_diff.tif
                            diff_name = (
                                f"{short_name_k}_{layer_k}_{d_later}_{d_early}"
                                f"{utm_suffix_k}_log-diff.tif"
                            )
                            diff_path = (mode_dir / "data") / diff_name

                            try:
                                compute_and_write_difference(
                                    earlier_path=early_info["path"],
                                    later_path=later_info["path"],
                                    out_path=diff_path,
                                    nodata_value=None,
                                    log=True,
                                )
                                print(f"[INFO] Wrote diff COG: {diff_path}")

                                # Make a map with PyGMT
                                diff_date_str = f"{d_later}_{d_early}"
                                map_name = make_map(
                                    maps_dir,
                                    diff_path,
                                    short_name_k,
                                    layer_k,
                                    diff_date_str,
                                    bbox,
                                    zoom_bbox,
                                    is_difference=True,
                                    utm_suffix=utm_suffix_k,
                                )

                                # Make a PDF layout
                                if map_name:
                                    diff_date_str_layout = f"{d_early}, {d_later}"
                                    make_layout(
                                        layouts_dir,
                                        map_name,
                                        short_name_k,
                                        layer_k,
                                        diff_date_str,
                                        diff_date_str_layout,
                                        layout_title,
                                        reclassify_snow_ice,
                                        utm_suffix=utm_suffix_k,
                                    )

                            except Exception as e:  # noqa: PERF203
                                skipped.append(
                                    {
                                        "short_name": short_name_k,
                                        "layer": layer_k,
                                        "date_earlier": d_early,
                                        "date_later": d_later,
                                        "utm_suffix": utm_suffix_k,
                                        "error": str(e),
                                        "reason": (
                                            "no overlapping data values; both rasters "
                                            "contain only nodata in the overlap region."
                                        ),
                                    }
                                )

        # Report skipped pairs due to CRS/UTM differences or errors
        report_path = (mode_dir / "data") / "log-difference_skipped_pairs.json"
        write_json(skipped, report_path)
        print(
            f"[INFO] Log-difference skip report: {report_path} ({len(skipped)} skipped)"
        )
