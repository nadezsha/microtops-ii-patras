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

# Small region around Patras.
# ADS order:
# North, West, South, East
AREA = [
    38.6,
    21.4,
    38.0,
    22.2,
]

TARGET_LAT = 38.283
TARGET_LON = 21.783

# 1 DU ≈ 2.1415e-5 kg m^-2
KG_M2_PER_DU = 2.1415e-5


# ============================================================
# HELPERS
# ============================================================

def load_api_credentials():
    """
    Read ADS API configuration from GitHub secret and
    write a temporary ~/.cdsapirc file.
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

    Microtops DATE format is DD/MM/YYYY.
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
        format="%d/%m/%Y",
        errors="coerce",
    )

    invalid_n = int(
        dates.isna().sum()
    )

    if invalid_n > 0:
        print(
            f"Warning: {invalid_n} rows contain "
            f"unparseable DATE values."
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
        / f"cams_ozone_{date_text}.zip"
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
    Extract NetCDF file(s) from ADS zipped NetCDF output.
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

    files = sorted(
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
    Find the total-column ozone variable in CAMS output.
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


def detect_coordinate_name(
    data_array,
    candidates,
):
    """
    Return the first available coordinate name.
    """

    for name in candidates:
        if (
            name in data_array.coords
            or name in data_array.dims
        ):
            return name

    return None


def extract_rows(
    netcdf_files,
):
    """
    Extract ozone at the nearest CAMS grid point
    to Patras.
    """

    output_rows = []

    for file_path in netcdf_files:

        print(
            f"Reading CAMS file: "
            f"{file_path.name}"
        )

        with xr.open_dataset(
            file_path
        ) as ds:

            ozone_var = (
                detect_ozone_variable(ds)
            )

            ozone = ds[
                ozone_var
            ]

            lat_name = detect_coordinate_name(
                ozone,
                [
                    "latitude",
                    "lat",
                ],
            )

            lon_name = detect_coordinate_name(
                ozone,
                [
                    "longitude",
                    "lon",
                ],
            )

            if lat_name is None:
                raise KeyError(
                    "Could not identify CAMS latitude "
                    f"coordinate. Coordinates: "
                    f"{list(ozone.coords)}"
                )

            if lon_name is None:
                raise KeyError(
                    "Could not identify CAMS longitude "
                    f"coordinate. Coordinates: "
                    f"{list(ozone.coords)}"
                )

            point = ozone.sel(
                {
                    lat_name: TARGET_LAT,
                    lon_name: TARGET_LON,
                },
                method="nearest",
            )

            time_name = detect_coordinate_name(
                point,
                [
                    "valid_time",
                    "time",
                ],
            )

            if time_name is None:
                raise KeyError(
                    "No CAMS time coordinate found. "
                    f"Coordinates: {list(point.coords)}; "
                    f"dimensions: {point.dims}"
                )

            # ------------------------------------------------
            # Convert to a simple 1-D time series.
            # CAMS files can retain additional singleton
            # dimensions such as step/reference time.
            # ------------------------------------------------

            point = point.squeeze(
                drop=True
            )

            print(
                f"  Ozone variable: {ozone_var}"
            )
            print(
                f"  Point dimensions after squeeze: "
                f"{point.dims}"
            )
            print(
                f"  Point shape after squeeze: "
                f"{point.shape}"
            )

            if time_name in point.dims:

                n_times = point.sizes[
                    time_name
                ]

                for index in range(
                    n_times
                ):

                    item = point.isel(
                        {
                            time_name: index
                        }
                    )

                    item = item.squeeze(
                        drop=True
                    )

                    values = np.asarray(
                        item.values
                    )

                    if values.size != 1:
                        raise ValueError(
                            "Expected one CAMS ozone value "
                            "after spatial/time selection, "
                            f"but found shape {values.shape} "
                            f"with {values.size} values. "
                            f"Remaining dimensions: "
                            f"{item.dims}"
                        )

                    ozone_kg_m2 = float(
                        values.reshape(-1)[0]
                    )

                    ozone_du = (
                        ozone_kg_m2
                        / KG_M2_PER_DU
                    )

                    timestamp = (
                        pd.to_datetime(
                            point[
                                time_name
                            ].isel(
                                {
                                    time_name:
                                        index
                                }
                            ).values,
                            utc=True,
                        )
                    )

                    lat_used = float(
                        point[
                            lat_name
                        ].values
                    )

                    lon_used = float(
                        point[
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
                                (
                                    "CAMS global atmospheric "
                                    "composition analysis"
                                ),
                        }
                    )

            else:

                # In case the time coordinate exists only
                # as a scalar coordinate.
                values = np.asarray(
                    point.values
                )

                if values.size != 1:
                    raise ValueError(
                        "Expected one CAMS ozone value "
                        "for scalar-time output, but found "
                        f"shape {values.shape} with "
                        f"{values.size} values. "
                        f"Dimensions: {point.dims}"
                    )

                ozone_kg_m2 = float(
                    values.reshape(-1)[0]
                )

                ozone_du = (
                    ozone_kg_m2
                    / KG_M2_PER_DU
                )

                timestamp = pd.to_datetime(
                    point[
                        time_name
                    ].values,
                    utc=True,
                )

                lat_used = float(
                    point[
                        lat_name
                    ].values
                )

                lon_used = float(
                    point[
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
                            (
                                "CAMS global atmospheric "
                                "composition analysis"
                            ),
                    }
                )

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

    if not all_rows:
        raise ValueError(
            "No CAMS ozone rows were extracted."
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
        .dropna(
            subset=[
                "datetime_utc",
                "ozone_DU",
            ]
        )
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
