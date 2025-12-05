from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

log = logging.getLogger(__name__)


def _get_podaac_s3_session():
    """Return a boto3.Session configured with temporary S3 credentials from PODAAC."""
    import boto3
    import earthaccess

    temp_creds_req = earthaccess.get_s3_credentials(daac="PODAAC")

    session = boto3.Session(
        aws_access_key_id=temp_creds_req["accessKeyId"],
        aws_secret_access_key=temp_creds_req["secretAccessKey"],
        aws_session_token=temp_creds_req["sessionToken"],
        region_name="us-west-2",
    )

    log.debug("Obtained PODAAC S3 temporary credentials and created boto3.Session")
    return session


def _get_rasterio_aws_env(session=None):
    """Create a rasterio.Env configured for AWS S3 access via the given boto3 session.

    This mirrors the environment configuration in the original `authenticate` function.
    The caller is responsible for entering/exiting the context (or calling __enter__()).
    """
    import rasterio
    from rasterio.session import AWSSession

    if session is None:
        session = _get_podaac_s3_session()

    rio_env = rasterio.Env(
        AWSSession(session),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF, TIFF",
        GDAL_HTTP_COOKIEFILE=os.path.expanduser("~/cookies.txt"),
        GDAL_HTTP_COOKIEJAR=os.path.expanduser("~/cookies.txt"),
    )

    log.debug("Created rasterio.Env with AWS session and GDAL HTTP cookie settings")
    return rio_env


def _get_earthdata_credentials_from_netrc(
    machine: str = "urs.earthdata.nasa.gov",
    netrc_path: Path | None = None,
) -> Tuple[str, str]:
    """Read Earthdata (URS) credentials from a .netrc file.

    This reproduces the behavior of the original `authenticate` function:
    - Default .netrc is `$HOME/.netrc` if `netrc_path` is not given.
    - Uses the `urs.earthdata.nasa.gov` machine entry by default.

    Parameters
    ----------
    machine
        Machine name in the .netrc file to query.
    netrc_path
        Optional explicit path to a .netrc file. If None, the default
        resolution of `netrc.netrc()` is used.

    Returns
    -------
    (username, password)

    Raises
    ------
    FileNotFoundError
        If the .netrc file does not exist (when explicitly provided).
    KeyError
        If the given machine is not found in the .netrc.
    """
    import netrc as netrc_mod

    if netrc_path is None:
        netrc_file = Path.home() / ".netrc"
    else:
        netrc_file = Path(netrc_path)

    auths = netrc_mod.netrc(netrc_file)
    username, _, password = auths.authenticators(machine)

    if username is None or password is None:
        raise KeyError(
            f"Missing login or password for machine '{machine}' "
            f"in netrc file {netrc_file}"
        )

    log.debug(
        "Loaded Earthdata credentials from %s for machine %s", netrc_file, machine
    )
    return username, password


def authenticate() -> Tuple[str, str]:
    """Authenticate with Earthdata and ASF for data access.

    This replicates the behavior of the original `disaster.authenticate()`:

    1. Request temporary S3 credentials for PODAAC via `earthaccess.get_s3_credentials`.
    2. Create a boto3.Session with those credentials.
    3. Create a `rasterio.Env` configured for AWS S3 access and GDAL HTTP cookies,
       and immediately enter it (leaving the global rasterio environment active).
    4. Read Earthdata/URS username and password from `~/.netrc` for
       `urs.earthdata.nasa.gov`.

    Returns
    -------
    (username, password)
        Username and password for Earthdata/ASF access, as used in `open_file(...)`
        calls and similar.

    Notes
    -----
    - The rasterio.Env is entered and *not* exited here, matching the original
      script behavior. Any code that relies on this global GDAL/rasterio
      environment will continue to work unchanged.
    """
    # Set up PODAAC S3 / AWS / rasterio environment (global, like original code)
    session = _get_podaac_s3_session()
    rio_env = _get_rasterio_aws_env(session=session)
    rio_env.__enter__()  # keep behavior identical to original authenticate()

    # Parse credentials from the netrc file for ASF / URS access
    username, password = _get_earthdata_credentials_from_netrc(
        machine="urs.earthdata.nasa.gov"
    )

    log.info("Authentication environment initialized; Earthdata username loaded")
    return username, password
