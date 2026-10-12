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

# 1 DU ≈ 2.1415 × 10^-5 kg m^-2
KG_M2_PER_DU = 2.1415e-5


# ============================================================
# API
# ============================================================

def load_api_credentials():

    secret = os.environ.get(
        "COPERNICUS_ADS_API"
    )

    if not secret:
        raise RuntimeError(
            "COPERNICUS_ADS_API environment variable "
            "was not provided."
        )

    cdsapirc = (
        Path.home()
        / ".cdsapirc"
    )

    cdsapirc.write_text(
        secret.strip() + "\n",
        encoding="utf-8",
    )

    print(
        "Temporary ADS configuration written to "
        f"{cdsapirc}"
    )


# ============================================================
# MICROTOPS DATES
# ============================================================

def read_measurement_dates():
    """
    Read unique Microtops dates.

    IMPORTANT:
    Microtops DATE is MM/DD/YYYY.
    """

    if not RAW_MASTER_FILE.is_file():

        raise FileNotFoundError(
            "Missing raw master dataset: "
            f"{RAW_MASTER_FILE}"
        )

    df = pd.read_csv(
        RAW_MASTER_FILE,
        dtype=str,
    )

    if "DATE" not in df.columns:

        raise ValueError(
            "Raw master dataset does not "
            "contain DATE."
        )

    # Microtops software format:
    # MM/DD/YYYY
    dates = pd.to_datetime(
        df["DATE"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    invalid_n = int(
        dates.isna().sum()
    )

    if invalid_n:

        print(
            f"WARNING: {invalid_n} invalid "
            "Microtops DATE values."
        )

    dates = sorted(
        dates
        .dropna()
        .dt.date
        .unique()
    )

    if not dates:

        raise ValueError(
            "No valid Microtops dates found."
        )

    return dates


# ============================================================
# CAMS DOWNLOAD
# ============================================================

def download_cams_ozone(
    client,
    date_value,
):

    date_text = (
        date_value.strftime(
            "%Y-%m-%d"
        )
    )

    target = (
        DOWNLOAD_DIR
        / f"cams_ozone_{date_text}.zip"
    )

    print()
    print(
        f"Requesting CAMS ozone: "
        f"{date_text}"
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


# ============================================================
# NETCDF EXTRACTION
# ============================================================

def extract_netcdf(
    zip_path,
):

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
        extract_dir.glob(
            "*.nc"
        )
    )

    if not files:

        raise FileNotFoundError(
            "No NetCDF files found in "
            f"{zip_path}"
        )

    return files


def detect_ozone_variable(
    dataset,
):

    candidates = [
        "gtco3",
        "tco3",
        "total_column_ozone",
    ]

    for name in candidates:

        if name in dataset.data_vars:

            return name

    raise KeyError(
        "Could not identify total-column "
        "ozone variable. "
        "Available variables: "
        f"{list(dataset.data_vars)}"
    )


def detect_coordinate(
    array,
    candidates,
):

    for candidate in candidates:

        if (
            candidate in array.coords
            or candidate in array.dims
        ):

            return candidate

    return None


# ============================================================
# EXTRACT PATRAS OZONE
# ============================================================

def extract_rows(
    netcdf_files,
):

    rows = []

    for file_path in netcdf_files:

        print()
        print(
            f"Reading CAMS file: "
            f"{file_path.name}"
        )

        with xr.open_dataset(
            file_path
        ) as ds:

            ozone_name = (
                detect_ozone_variable(
                    ds
                )
            )

            ozone = ds[
                ozone_name
            ]

            print(
                f"  Ozone variable: "
                f"{ozone_name}"
            )

            lat_name = detect_coordinate(
                ozone,
                [
                    "latitude",
                    "lat",
                ],
            )

            lon_name = detect_coordinate(
                ozone,
                [
                    "longitude",
                    "lon",
                ],
            )

            if lat_name is None:

                raise KeyError(
                    "Latitude coordinate "
                    "not found."
                )

            if lon_name is None:

                raise KeyError(
                    "Longitude coordinate "
                    "not found."
                )

            # --------------------------------------------
            # Select nearest CAMS grid cell to Patras
            # --------------------------------------------

            point = ozone.sel(
                {
                    lat_name:
                        TARGET_LAT,

                    lon_name:
                        TARGET_LON,
                },
                method="nearest",
            )

            point = point.squeeze(
                drop=True
            )

            print(
                "  Point dimensions: "
                f"{point.dims}"
            )

            print(
                "  Point shape: "
                f"{point.shape}"
            )

            # --------------------------------------------
            # CAMS currently returns our four analysis
            # times along forecast_reference_time.
            #
            # valid_time may be a coordinate rather than
            # the actual dimension.
            # --------------------------------------------

            time_dim = None

            preferred_dims = [
                "valid_time",
                "time",
                "forecast_reference_time",
            ]

            for candidate in preferred_dims:

                if candidate in point.dims:

                    time_dim = candidate
                    break

            if time_dim is None:

                if len(point.dims) == 1:

                    time_dim = (
                        point.dims[0]
                    )

                elif len(point.dims) == 0:

                    time_dim = None

                else:

                    raise ValueError(
                        "Could not determine CAMS "
                        "time dimension. "
                        f"Dimensions: {point.dims}"
                    )

            # --------------------------------------------
            # Multiple CAMS times
            # --------------------------------------------

            if time_dim is not None:

                n_times = (
                    point.sizes[
                        time_dim
                    ]
                )

                print(
                    f"  Number of CAMS times: "
                    f"{n_times}"
                )

                for index in range(
                    n_times
                ):

                    item = point.isel(
                        {
                            time_dim:
                                index
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
                            "Expected one ozone "
                            "value after selecting "
                            "time/grid point, but "
                            f"found shape "
                            f"{values.shape}."
                        )

                    ozone_kg_m2 = float(
                        values.reshape(
                            -1
                        )[0]
                    )

                    ozone_du = (
                        ozone_kg_m2
                        / KG_M2_PER_DU
                    )

                    # ----------------------------
                    # Determine actual valid time
                    # ----------------------------

                    timestamp = None

                    for time_coord in [
                        "valid_time",
                        "time",
                        "forecast_reference_time",
                    ]:

                        if (
                            time_coord
                            not in point.coords
                        ):
                            continue

                        coord = point[
                            time_coord
                        ]

                        if (
                            time_dim
                            in coord.dims
                        ):

                            coord = coord.isel(
                                {
                                    time_dim:
                                        index
                                }
                            )

                        coord_values = (
                            np.asarray(
                                coord.values
                            )
                        )

                        if (
                            coord_values.size
                            == 1
                        ):

                            timestamp = (
                                pd.to_datetime(
                                    coord_values
                                    .reshape(-1)[0],
                                    utc=True,
                                    errors="coerce",
                                )
                            )

                            if (
                                not pd.isna(
                                    timestamp
                                )
                            ):

                                break

                    if timestamp is None:

                        raise ValueError(
                            "Could not determine "
                            "CAMS valid time."
                        )

                    lat_used = float(
                        np.asarray(
                            point[
                                lat_name
                            ].values
                        ).reshape(-1)[0]
                    )

                    lon_used = float(
                        np.asarray(
                            point[
                                lon_name
                            ].values
                        ).reshape(-1)[0]
                    )

                    rows.append(
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
                                    "CAMS global "
                                    "atmospheric "
                                    "composition "
                                    "analysis"
                                ),
                        }
                    )

            # --------------------------------------------
            # Single CAMS value
            # --------------------------------------------

            else:

                # CAMS may return only one available analysis
                # for the current day. After squeeze(), this
                # becomes a scalar with no remaining dimension.
                values = np.asarray(
                    point.values
                )

                if values.size != 1:

                    raise ValueError(
                        "Expected one scalar CAMS "
                        "ozone value, but found "
                        f"shape {values.shape}."
                    )

                ozone_kg_m2 = float(
                    values.reshape(-1)[0]
                )

                ozone_du = (
                    ozone_kg_m2
                    / KG_M2_PER_DU
                )

                # Recover the scalar analysis timestamp.
                timestamp = None

                for time_coord in [
                    "valid_time",
                    "time",
                    "forecast_reference_time",
                ]:

                    if time_coord not in point.coords:
                        continue

                    coord_values = np.asarray(
                        point[
                            time_coord
                        ].values
                    )

                    if coord_values.size != 1:
                        continue

                    candidate = pd.to_datetime(
                        coord_values.reshape(-1)[0],
                        utc=True,
                        errors="coerce",
                    )

                    if not pd.isna(candidate):
                        timestamp = candidate
                        break

                if timestamp is None:

                    raise ValueError(
                        "CAMS returned scalar ozone, "
                        "but its timestamp could not "
                        "be determined."
                    )

                lat_used = float(
                    np.asarray(
                        point[
                            lat_name
                        ].values
                    ).reshape(-1)[0]
                )

                lon_used = float(
                    np.asarray(
                        point[
                            lon_name
                        ].values
                    ).reshape(-1)[0]
                )

                rows.append(
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
                                "CAMS global "
                                "atmospheric "
                                "composition "
                                "analysis"
                            ),
                    }
                )

    return rows


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

    dates = (
        read_measurement_dates()
    )

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

        extracted = (
            extract_rows(
                netcdf_files
            )
        )

        all_rows.extend(
            extracted
        )

    if not all_rows:

        raise ValueError(
            "No CAMS ozone observations "
            "were extracted."
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
        "=" * 72
    )

    print(
        "CAMS OZONE RETRIEVAL COMPLETE"
    )

    print(
        "=" * 72
    )

    print()
    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    print()
    print(
        f"Rows: {len(result)}"
    )

    print()
    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
