from pathlib import Path
import os
import zipfile

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_MASTER_FILE = (
    ROOT
    / "data"
    / "processed"
    / "microtops_raw_master.csv"
)

AUX_DIR = (
    ROOT
    / "data"
    / "auxiliary"
)

OUTPUT_FILE = (
    AUX_DIR
    / "ozone_cams.csv"
)

DOWNLOAD_DIR = (
    AUX_DIR
    / "cams_downloads"
)


# ============================================================
# SETTINGS
# ============================================================

DATASET = (
    "cams-global-atmospheric-composition-forecasts"
)

VARIABLE = "total_column_ozone"

ANALYSIS_TIMES = [
    "00:00",
    "06:00",
    "12:00",
    "18:00",
]

# Small region around the Patras measurements.
# Order required by ADS:
# North, West, South, East
AREA = [
    38.6,
    21.4,
    38.0,
    22.2,
]


# 1 Dobson Unit corresponds to approximately
# 2.1415e-5 kg m^-2 of ozone.
KG_M2_PER_DU = 2.1415e-5


# ============================================================
# HELPERS
# ============================================================

def load_api_credentials():
    """
    Read the GitHub secret containing the ADS API configuration
    and write a temporary .cdsapirc file.
    """

    secret = os.environ.get(
        "COPERNICUS_ADS_API"
    )

    if not secret:
        raise RuntimeError(
            "COPERNICUS_ADS_API environment variable "
            "was not provided."
        )

    cdsapirc = Path.home() / ".cdsapirc"

    cdsapirc.write_text(
        secret.strip() + "\n",
        encoding="utf-8",
    )

    print(
        f"Temporary ADS configuration written to "
        f"{cdsapirc}"
    )


def read_measurement_dates():
    """
    Read unique Microtops observation dates from the
    processed raw master dataset.
    """

    if not RAW_MASTER_FILE.is_file():
        raise FileNotFoundError(
            f"Missing raw master file: "
            f"{RAW_MASTER_FILE}"
        )

    df = pd.read_csv(
        RAW_MASTER_FILE,
        dtype=str,
    )

    if "DATE" not in df.columns:
        raise ValueError(
            "Raw master dataset does not contain DATE."
        )

    dates = pd.to_datetime(
        df["DATE"],
        dayfirst=True,
        errors="coerce",
    )

    dates = sorted(
        dates.dropna()
        .dt.date
        .unique()
    )

    if not dates:
        raise ValueError(
            "No valid Microtops dates were found."
        )

    return dates


def download_cams_ozone(
    client,
    date_value,
):
    """
    Download CAMS total-column ozone for one date.
    """

    date_text = date_value.strftime(
        "%Y-%m-%d"
    )

    target = (
        DOWNLOAD_DIR
        / f"cams_ozone_{date_text}.netcdf_zip"
    )

    print()
    print(
        f"Requesting CAMS ozone: {date_text}"
    )

    request = {
        "date": date_text,
        "time": ANALYSIS_TIMES,
        "leadtime_hour": "0",
        "type": "analysis",
        "variable": VARIABLE,
        "area": AREA,
        "format": "netcdf_zip",
    }

    client.retrieve(
        DATASET,
        request,
        str(target),
    )

    return target


def extract_netcdf(zip_path):
    """
    Extract NetCDF file(s) from an ADS zipped NetCDF response.
    """

    extract_dir = (
        zip_path.parent
        / zip_path.stem
    )

    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        archive.extractall(
            extract_dir
        )

    files = list(
        extract_dir.glob("*.nc")
    )

    if not files:
        raise FileNotFoundError(
            f"No NetCDF file found inside "
            f"{zip_path}"
        )

    return files


def detect_ozone_variable(dataset):
    """
    Find the total-column ozone variable in the
    returned CAMS NetCDF dataset.
    """

    candidates = [
        "gtco3",
        "tco3",
        "total_column_ozone",
    ]

    for name in candidates:
        if name in dataset.data_vars:
            return name

    raise KeyError(
        "Could not identify total-column ozone "
        f"variable. Available variables: "
        f"{list(dataset.data_vars)}"
    )


def extract_rows(
    netcdf_files,
):
    """
    Convert CAMS NetCDF output into a small tabular dataset.
    """

    output_rows = []

    for file_path in netcdf_files:

        print(
            f"Reading CAMS file: "
            f"{file_path.name}"
        )

        ds = xr.open_dataset(
            file_path
        )

        ozone_var = (
            detect_ozone_variable(ds)
        )

        ozone = ds[
            ozone_var
        ]

        # Determine coordinate names.
        lat_name = (
            "latitude"
            if "latitude" in ozone.coords
            else "lat"
        )

        lon_name = (
            "longitude"
            if "longitude" in ozone.coords
            else "lon"
        )

        # Use the centre of the Microtops measurement
        # region for nearest-grid-cell extraction.
        target_lat = 38.283
        target_lon = 21.783

        point = ozone.sel(
            {
                lat_name: target_lat,
                lon_name: target_lon,
            },
            method="nearest",
        )

        # Identify available time coordinate.
        time_name = None

        for candidate in [
            "valid_time",
            "time",
        ]:

            if candidate in point.coords:
                time_name = candidate
                break

        if time_name is None:
            raise KeyError(
                "No CAMS time coordinate found."
            )

        for index in range(
            point.sizes.get(
                time_name,
                1,
            )
        ):

            item = (
                point.isel(
                    {
                        time_name: index
                    }
                )
                if time_name
                in point.dims
                else point
            )

            ozone_kg_m2 = float(
                item.values
            )

            ozone_du = (
                ozone_kg_m2
                / KG_M2_PER_DU
            )

            timestamp = pd.to_datetime(
                item[
                    time_name
                ].values
            )

            lat_used = float(
                item[
                    lat_name
                ].values
            )

            lon_used = float(
                item[
                    lon_name
                ].values
            )

            output_rows.append(
                {
                    "datetime_utc":
                        timestamp,
                    "ozone_kg_m2":
                        ozone_kg_m2,
                    "ozone_DU":
                        ozone_du,
                    "grid_latitude":
                        lat_used,
                    "grid_longitude":
                        lon_used,
                    "source":
                        "CAMS global atmospheric "
                        "composition analysis",
                }
            )

        ds.close()

    return output_rows


# ============================================================
# MAIN
# ============================================================

def main():

    AUX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    load_api_credentials()

    dates = read_measurement_dates()

    print()
    print(
        "Microtops dates requiring ozone:"
    )

    for date_value in dates:
        print(
            f"  {date_value}"
        )

    client = cdsapi.Client()

    all_rows = []

    for date_value in dates:

        downloaded = (
            download_cams_ozone(
                client,
                date_value,
            )
        )

        netcdf_files = (
            extract_netcdf(
                downloaded
            )
        )

        rows = extract_rows(
            netcdf_files
        )

        all_rows.extend(
            rows
        )

    result = pd.DataFrame(
        all_rows
    )

    result[
        "datetime_utc"
    ] = pd.to_datetime(
        result[
            "datetime_utc"
        ],
        utc=True,
        errors="coerce",
    )

    result = (
        result
        .drop_duplicates(
            subset=[
                "datetime_utc",
                "grid_latitude",
                "grid_longitude",
            ]
        )
        .sort_values(
            "datetime_utc"
        )
        .reset_index(
            drop=True
        )
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.6f",
    )

    print()
    print(
        "CAMS ozone dataset written to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
