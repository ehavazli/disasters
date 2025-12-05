from __future__ import annotations

from pathlib import Path

import numpy as np
from osgeo import gdal

from .io import build_output_path, ensure_directory

__all__ = [
    "expand_region",
    "expand_region_to_aspect",
    "make_map",
    "make_layout",
]


def expand_region(region, width_deg=15, height_deg=10):
    """
    Return a new region [xmin, xmax, ymin, ymax] of fixed size,
    centered on the centroid of the input region, with coordinates rounded
    to 0 decimal places.

    Args:
        region (list): Input region in the form [xmin, xmax, ymin, ymax].
        width_deg (float): Desired width in degrees.
        height_deg (float): Desired height in degrees.

    Returns:
        list: New region with fixed size, centered on the input region.
    Raises:
        ValueError: If the input region is not in the correct format.
    """
    xmin, xmax, ymin, ymax = region
    center_lon = (xmin + xmax) / 2
    center_lat = (ymin + ymax) / 2

    half_width = width_deg / 2
    half_height = height_deg / 2

    expanded_region = [
        round(center_lon - half_width),
        round(center_lon + half_width),
        round(center_lat - half_height),
        round(center_lat + half_height),
    ]
    return expanded_region


def expand_region_to_aspect(region, target_aspect):
    """
    Expand the input region to match a target aspect ratio.

    Args:
        region (list): Input region in the form [xmin, xmax, ymin, ymax].
        target_aspect (float): Desired aspect ratio (width / height).

    Returns:
        list: New region with adjusted aspect ratio.
    Raises:
        ValueError: If the input region is not in the correct format.
    """
    xmin, xmax, ymin, ymax = map(float, region)
    width = xmax - xmin
    height = ymax - ymin
    current_aspect = width / height

    if current_aspect > target_aspect:
        # Too wide → expand height
        new_height = width / target_aspect
        pad = (new_height - height) / 2
        ymin -= pad
        ymax += pad
    else:
        # Too tall → expand width
        new_width = height * target_aspect
        pad = (new_width - width) / 2
        xmin -= pad
        xmax += pad

    return [xmin, xmax, ymin, ymax]


def make_map(
    maps_dir,
    mosaic_path,
    short_name,
    layer,
    date,
    bbox,
    zoom_bbox=None,
    is_difference=False,
    utm_suffix="",
):
    """
    Create a map using PyGMT from the provided mosaic path.

    Args:
        maps_dir (Path): Directory where the map will be saved.
        mosaic_path (Path): Path to the mosaic file.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the map.
        date (str): Date string in the format YYYY-MM-DD.
        bbox (list): Bounding box in the form [South, North, West, East].
        zoom_bbox (list, optional): Bounding box for the zoom-in inset map,
            in the form [South, North, West, East].
        is_difference (bool, optional): Flag to indicate if the mosaic is a
            difference product. Defaults to False.
        utm_suffix (str, optional): UTM zone suffix (e.g., '_18N') for unique
            naming. Defaults to "".

    Returns:
        map_name (Path): Path to the saved map image, or None if skipped.
    Raises:
        ImportError: If required libraries are not installed.
    """
    import math
    import os
    import re

    import pygmt
    import rioxarray
    from pygmt.params import Box
    from pyproj import Geod

    # Ensure maps_dir exists (defensive; pipeline should already do this)
    maps_dir = ensure_directory(maps_dir)

    # Check whether the product is a difference product
    if is_difference:
        match = re.search(
            r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(\_\d+N|\_\d+S|\_EPSG\d+|\_Hash\d+)?(?:log-)?diff",
            str(mosaic_path),
        )
        if match:
            date_later = match.group(1)
            date_earlier = match.group(2)
            date_str = f"{date_later}_{date_earlier}_diff"
        else:
            date_str = date
    else:
        date_str = date

    # Create a temporary path for the WGS84 reprojected file
    mosaic_wgs84 = Path(str(mosaic_path).replace(".tif", f"_WGS84_TMP{utm_suffix}.tif"))

    def cleanup_temp_file(filepath):
        """Helper to safely remove the temporary file."""
        if filepath.exists():
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"[WARN] Failed to clean up temporary WGS84 file {filepath}: {e}")

    try:
        # Reproject to WGS84 (into the temp file)
        gdal.Warp(
            mosaic_wgs84,
            mosaic_path,
            dstSRS="EPSG:4326",
            resampleAlg="near",
            creationOptions=["COMPRESS=DEFLATE"],
        )

        # Load mosaic from the temporary file
        grd = rioxarray.open_rasterio(mosaic_wgs84).squeeze()

        try:
            nodata_value = grd.rio.nodata
        except AttributeError:
            print("[WARN] 'nodata' attribute not found. Defaulting to 255.")
            nodata_value = 255
        if nodata_value is not None:
            grd = grd.where(grd != nodata_value)
        else:
            print("[WARN] No nodata value found or set. Skipping nodata removal.")

        # Define region
        region = [bbox[2], bbox[3], bbox[0], bbox[1]]  # [xmin, xmax, ymin, ymax]

        # Define target aspect ratio
        target_aspect = 60 / 100

        # Pad region to match target aspect ratio
        region_padded = expand_region_to_aspect(region, target_aspect)

        # Define projection
        center_lon = (region_padded[0] + region_padded[1]) / 2
        projection_width_cm = 15
        projection = f"M{center_lon}/{projection_width_cm}c"

        # Create PyGMT figure
        fig = pygmt.Figure()

        # Control map annotation font size
        pygmt.config(FONT_ANNOT="6p")

        # Base coast layer (optional)
        fig.coast(
            region=region_padded,
            projection=projection,
            borders="2/thin",
            shorelines="thin",
            land="grey",
            water="lightblue",
        )

        # Make the map for difference products (if applicable)
        if is_difference:
            data_values = grd.values[~np.isnan(grd.values)]
            if len(data_values) == 0:
                print(
                    f"[WARN] Difference product {mosaic_path} has no valid data "
                    f"after nodata removal. Skipping map generation."
                )
                cleanup_temp_file(mosaic_wgs84)  # Cleanup on exit point
                return None  # Skip map generation

            # Calculate the 2nd and 98th percentiles
            p2, p98 = np.percentile(data_values, [2, 98])

            # For difference products, ensure the color scale is symmetric around zero
            symmetric_limit = max(abs(p2), abs(p98))
            p_min = -symmetric_limit
            p_max = symmetric_limit

            # Calculate increment for 1000 steps
            inc = (p_max - p_min) / 1000.0

            cpt_name = "difference_cpt"

            # Use a good diverging colormap (e.g., 'vik' or 'balance')
            pygmt.makecpt(
                cmap="vik", series=[p_min, p_max, inc], output=cpt_name, continuous=True
            )

            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=cpt_name,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )

            # Set the colorbar label based on the type of difference
            if "_log-diff.tif" in str(mosaic_path):
                cbar_label = "Normalized backscatter (@~g@~@-0@-) difference (dB)"
            else:  # Regular difference
                cbar_label = "Difference in water extent (unitless)"

            fig.colorbar(cmap=cpt_name, frame=[f"x+l{cbar_label}"])

        # Add grid image (based on product/layer)
        elif short_name == "OPERA_L3_DSWX-HLS_V1" and layer == "WTR":
            color_palette = "palettes/DSWx-HLS_WTR.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif short_name == "OPERA_L3_DSWX-HLS_V1" and layer == "BWTR":
            color_palette = "palettes/DSWx-HLS_BWTR.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif short_name == "OPERA_L3_DSWX-S1_V1" and layer == "WTR":
            color_palette = "palettes/DSWx-S1_WTR.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif short_name == "OPERA_L3_DSWX-S1_V1" and layer == "BWTR":
            color_palette = "palettes/DSWx-S1_BWTR.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif layer == "VEG-ANOM-MAX":
            color_palette = "palettes/VEG-ANOM-MAX.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(
                cmap=color_palette,
                frame="xaf+lVEG-ANOM-MAX(%)",
            )

        elif layer == "VEG-DIST-STATUS":
            color_palette = "palettes/VEG-DIST-STATUS.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif short_name.startswith("OPERA_L2_RTC"):

            data_values = grd.values[~np.isnan(grd.values)]

            # Calculate the 2nd and 98th percentiles
            p2, p98 = np.percentile(data_values, [2, 98])

            # Ensure min is less than max
            if p2 >= p98:
                p2 -= 0.01
                p98 += 0.01

            # Calculate increment for 1000 steps
            inc = (p98 - p2) / 1000.0

            cpt_name = "rtc_grayscale"

            pygmt.makecpt(
                cmap="gray", series=[p2, p98, inc], output=cpt_name, continuous=True
            )

            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=cpt_name,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )

            fig.colorbar(
                cmap=cpt_name, frame=["x+lNormalized backscatter (@~g@~@-0@-)"]
            )

        # Add scalebar and compass rose
        xmin, xmax, ymin, ymax = region_padded
        center_lat = (ymin + ymax) / 2
        geod = Geod(ellps="WGS84")
        _, _, distance_m = geod.inv(xmin, center_lat, xmax, center_lat)

        # Set scalebar to ~25% of region width
        raw_length_km = distance_m * 0.25 / 1000
        exponent = math.floor(math.log10(raw_length_km))
        base = 10**exponent

        for factor in [1, 2, 5, 10]:
            scalebar_length_km = base * factor
            if scalebar_length_km >= raw_length_km:
                break

        fig.basemap(
            map_scale=f"jBR+o1c/0.6c+c-7+w{scalebar_length_km:.0f}k+f+lkm+ar",
            box=Box(fill="white@30", pen="0.5p,gray30,solid", radius="3p"),
        )

        fig.basemap(
            rose="jTR+o0.6c/0.2c+w1.5c",
            box=Box(fill="white@30", pen="0.5p,gray30,solid", radius="3p"),
        )

        bounds = grd.rio.bounds()
        region = [bounds[0], bounds[2], bounds[1], bounds[3]]
        region_expanded_main = expand_region(region, width_deg=25, height_deg=15)

        # Add inset map (regional context)
        with fig.inset(
            position="jBL+o0.2c/0.2c",
            box="+pblack",
            region=region_expanded_main,
            projection="M5c",
        ):
            # Use a plotting method to create a figure inside the inset.
            fig.coast(
                land="gray",
                borders=[1, 2],
                shorelines="1/thin",
                water="white",
            )
            # Coordinates for rectangular outline of the main region
            xmin_r, xmax_r, ymin_r, ymax_r = region_padded
            rectangle = [
                [xmin_r, ymin_r],
                [xmax_r, ymin_r],
                [xmax_r, ymax_r],
                [xmin_r, ymax_r],
                [xmin_r, ymin_r],
            ]

            # Plot the rectangle on the inset
            fig.plot(
                x=[pt[0] for pt in rectangle],
                y=[pt[1] for pt in rectangle],
                pen="2p,red",
            )

        # Optional inset for the zoom-in bbox (bottom right, include a scalebar)
        if zoom_bbox:
            zoom_region = [zoom_bbox[2], zoom_bbox[3], zoom_bbox[0], zoom_bbox[1]]

            # Calculate scale bar length for the zoom inset
            xmin_z, xmax_z, ymin_z, ymax_z = zoom_region
            center_lat_z = (ymin_z + ymax_z) / 2

            _, _, distance_m_z = geod.inv(xmin_z, center_lat_z, xmax_z, center_lat_z)
            raw_length_km_z = distance_m_z * 0.25 / 1000  # 25% of inset width in km

            scalebar_length_km_z = 1  # Default fallback
            if raw_length_km_z > 0:
                exponent_z = math.floor(math.log10(raw_length_km_z))
                base_z = 10**exponent_z

                for factor in [1, 2, 5, 10]:
                    scalebar_length_z = base_z * factor
                    if scalebar_length_z >= raw_length_km_z:
                        scalebar_length_km_z = scalebar_length_z
                        break

            with fig.inset(
                position="jBR+o0.5c/1.5c",
                box="+p1p,magenta",
                region=zoom_region,
                projection="M5c",
            ):

                # Add coastline to inset
                fig.coast(
                    region=zoom_region,
                    projection="M5c",
                    borders="1/thin",
                    shorelines="thin",
                    land="grey",
                    water="lightblue",
                )

                # Re-plot the data for the inset map
                if short_name == "OPERA_L3_DSWX-HLS_V1" and layer == "WTR":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif short_name == "OPERA_L3_DSWX-HLS_V1" and layer == "BWTR":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif short_name == "OPERA_L3_DSWX-S1_V1" and layer == "WTR":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif short_name == "OPERA_L3_DSWX-S1_V1" and layer == "BWTR":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif layer == "VEG-ANOM-MAX":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif layer == "VEG-DIST-STATUS":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif short_name.startswith("OPERA_L2_RTC"):
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=cpt_name,
                        nan_transparent=True,
                    )

                # Add scale bar to the inset map. Use Bottom-Left (jBL) inside the inset frame.
                fig.basemap(
                    map_scale=f"jBL+o-0.5c/-0.5c+c-7+w{scalebar_length_km_z:.0f}k+f+lkm+ar",
                    box=Box(fill="white@30", pen="0.5p,gray30,solid", radius="3p"),
                )

            # Plot a rectangle on the main map to show the zoom area
            fig.plot(
                x=[
                    zoom_region[0],
                    zoom_region[1],
                    zoom_region[1],
                    zoom_region[0],
                    zoom_region[0],
                ],
                y=[
                    zoom_region[2],
                    zoom_region[2],
                    zoom_region[3],
                    zoom_region[3],
                    zoom_region[2],
                ],
                pen="1p,magenta",
            )

        # Export map
        map_name = build_output_path(
            maps_dir,
            f"{short_name}_{layer}_{date}{utm_suffix}_map.png",
        )
        fig.savefig(map_name, dpi=900)
        cleanup_temp_file(mosaic_wgs84)

        return map_name

    except Exception as e:
        cleanup_temp_file(mosaic_wgs84)
        print(f"[ERROR] An error occurred during map generation for {mosaic_path}: {e}")
        raise


def make_layout(
    layout_dir,
    map_name,
    short_name,
    layer,
    date,
    layout_date,
    layout_title,
    reclassify_snow_ice=False,
    utm_suffix="",
):
    """
    Create a layout using matplotlib for the provided map.

    Args:
        layout_dir (Path): Directory where the layout will be saved.
        map_name (Path): Path to the map image.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the layout.
        date (str): Date string in the format YYYY-MM-DD.
        layout_date (str): Date threshold in the format YYYY-MM-DD.
        layout_title (str): Title for the layout.
        reclassify_snow_ice (bool, optional): Flag indicating if snow/ice
            reclassification was applied. Defaults to False.
    """
    import textwrap

    import matplotlib.image as mpimg
    import matplotlib.patches as patches  # noqa: F401  # kept for parity
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: F401

    # Ensure layout_dir exists (defensive; pipeline should already do this)
    layout_dir = ensure_directory(layout_dir)

    # Create blank figure
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_axis_off()

    # Set background
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add main map ===
    map_img = mpimg.imread(map_name)
    ax.imshow(map_img, extent=[0, 60, 0, 100])  # Main map on left 60% of layout

    # Add OPERA/ARIA logos
    logo_opera = imread("logos/OPERA_logo.png")
    logo_new = imread("logos/ARIA_logo.png")

    # Create a new axes for logos in the bottom-right corner
    logo_ax = fig.add_axes([0.82, 0.02, 0.06, 0.08], anchor="SE", zorder=10)
    logo_ax.imshow(logo_opera)
    logo_ax.axis("off")

    logo_ax2 = fig.add_axes([0.89, 0.02, 0.06, 0.08], anchor="SE", zorder=10)
    logo_ax2.imshow(logo_new)
    logo_ax2.axis("off")

    # Layout control
    x_pos = 0.65  # Just right of the map
    line_spacing = 0.04  # vertical spacing between blocks

    # Define wrap width (in characters)
    wrap_width = 50

    # Map text elements
    if short_name == "OPERA_L3_DSWX-S1_V1":
        subtitle = "OPERA Dynamic Surface Water eXtent from Sentinel-1 (DSWx-S1)"
        map_information = (
            "The ARIA/OPERA water extent map is derived from an OPERA DSWx-S1 "
            "mosaicked product from Copernicus Sentinel-1 data."
            "This map depicts regions of full surface water and inundated surface water. "
        )
        data_source = "Copernicus Sentinel-1"

    elif short_name == "OPERA_L3_DSWX-HLS_V1":
        subtitle = "OPERA Dynamic Surface Water eXtent from HLS (DSWx-HLS)"
        if reclassify_snow_ice is True:
            map_information = textwrap.dedent(
                """\
                The ARIA/OPERA water extent map is derived from an OPERA DSWx-HLS mosaicked 
                product from Harmonized Landsat and Sentinel-2 data.

                Note: Cloud/cloud shadow and snow/ice layers are derived from HLS Fmask 
                quality assurance (QA) data, which sometimes misclassifies sediment-rich water as snow/ice. 
                Snow/ice pixels were reclassified to open water to capture the full inundated extent.
            """
            )
        else:
            map_information = (
                "The ARIA/OPERA water extent map is derived from an OPERA DSWx-HLS "
                "mosaicked product from Harmonized Landsat and Sentinel-2 data."
                "This map depicts regions of full surface water and inundated surface water. "
            )
        data_source = "Copernicus Harmonized Landsat and Sentinel-2"

    elif short_name == "OPERA_L3_DIST-ALERT-S1_V1":
        subtitle = "OPERA Surface Disturbance Alert from Sentinel-1 (DIST-ALERT-S1)"
        map_information = (
            "The ARIA/OPERA surface disturbance alert map is derived from an OPERA "
            "DIST-ALERT-S1 mosaicked product from Copernicus Sentinel-1 data."
            "This map depicts regions of surface disturbance since " + layout_date + "."
        )
        data_source = "Copernicus Sentinel-1"

    elif short_name == "OPERA_L3_DIST-ALERT-HLS_V1":
        subtitle = (
            "OPERA Surface Disturbance Alert from Harmonized Landsat and "
            "Sentinel-2 (DIST-ALERT-HLS)"
        )
        map_information = (
            "The ARIA/OPERA surface disturbance alert map is derived from an OPERA "
            "DIST-ALERT-HLS mosaicked product from Harmonized Landsat and "
            "Sentinel-2 data. "
            "This map depicts regions of vegetation disturbance since "
            + layout_date
            + "."
        )
        data_source = "Copernicus Harmonized Landsat and Sentinel-2"

    elif short_name == "OPERA_L2_RTC-S1_V1":
        subtitle = (
            "OPERA Radiometrically Terrain Corrected Backscatter "
            "from Sentinel-1 (RTC-S1)"
        )
        map_information = (
            "The ARIA/OPERA backscatter map is derived from an OPERA RTC-S1 "
            "mosaicked product from Copernicus Sentinel-1 data."
            "This map depicts the radar backscatter intensity, which can be used "
            "to identify surface features and changes."
        )
        data_source = "Copernicus Sentinel-1"

    else:
        # Fallbacks if new products are introduced but this function is reused unchanged
        subtitle = short_name
        map_information = ""
        data_source = ""

    acquisitions = f"{date}"

    data_sources = textwrap.dedent(
        f"""\
        Product: {short_name}

        Layer: {layer}

        Data Source: {data_source}

        Resolution: 30 meters
    """
    )

    data_availability = textwrap.dedent(
        """\
        This product is available at: https://aria-share.jpl.nasa.gov/

        Visit the OPERA website: https://www.jpl.nasa.gov/go/opera/
    """
    )

    disclaimer = (
        "The results posted here are preliminary and unvalidated, "
        "intended to aid field response and provide a first look at the "
        "disaster-affected region."
    )

    # Wrapping text
    title_wrp = textwrap.fill(layout_title, width=40)
    subtitle_wrp = textwrap.fill(subtitle, width=wrap_width)
    acquisitions_wrp = textwrap.fill(acquisitions, width=wrap_width)
    map_information_wrp = textwrap.fill(map_information, width=wrap_width)
    data_sources_wrp = textwrap.fill(data_sources, width=wrap_width)
    data_availability_wrp = textwrap.fill(data_availability, width=wrap_width)
    disclaimer_wrp = textwrap.fill(disclaimer, width=wrap_width)

    # Starting y-position (top of the figure)
    y_start = 0.98

    # Define title
    ax.text(
        x_pos,
        y_start,
        title_wrp,
        fontsize=14,
        weight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Define subtitle
    ax.text(
        x_pos,
        y_start - line_spacing * 1,
        subtitle_wrp,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Acquisition heading
    ax.text(
        x_pos,
        y_start - line_spacing * 3.5,
        "Data Acquisitions:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Acquisition dates
    ax.text(
        x_pos,
        y_start - line_spacing * 4,
        acquisitions_wrp,
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Map information heading
    ax.text(
        x_pos,
        y_start - line_spacing * 6,
        "Map Information:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Map information text
    ax.text(
        x_pos,
        y_start - line_spacing * 6.5,
        map_information_wrp,
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Data sources heading
    ax.text(
        x_pos,
        y_start - line_spacing * (10 + 1.5),
        "Data Sources:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Data sources text
    ax.text(
        x_pos,
        y_start - line_spacing * (10.5 + 1.5),
        data_sources_wrp,
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
        linespacing=1,
        wrap=True,
    )

    # Data availability heading
    ax.text(
        x_pos,
        y_start - line_spacing * (15 + 1.5),
        "Product Availability:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Data availability text
    ax.text(
        x_pos,
        y_start - line_spacing * (15.5 + 1.5),
        data_availability_wrp,
        fontsize=8,
        ha="left",
        va="top",
        linespacing=1,
        transform=ax.transAxes,
        wrap=True,
    )

    # Disclaimer heading
    ax.text(
        x_pos,
        y_start - line_spacing * (18 + 1.5),
        "Disclaimer:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Disclaimer
    ax.text(
        x_pos,
        y_start - line_spacing * (18.5 + 1.5),
        disclaimer_wrp,
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    plt.tight_layout()

    layout_name = build_output_path(
        layout_dir,
        f"{short_name}_{layer}_{date}{utm_suffix}_layout.pdf",
    )
    plt.savefig(layout_name, format="pdf", bbox_inches="tight", dpi=400)
    return
