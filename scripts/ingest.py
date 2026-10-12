from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "microtops_raw_master.csv"


# ============================================================
# SETTINGS
# ============================================================

KNOWN_INSTRUMENTS = {
    20765,
    26126,
}


# ============================================================
# HELPERS
# ============================================================

def read_microtops_csv(path):
    """
    Read one Microtops CSV export.

    Keeps all original columns and adds the source filename.
    """

    print(f"Reading: {path.name}")

    # Microtops files contain:
    #   - one metadata line before the header
    #   - sometimes one format-marker line at the end
    df = pd.read_csv(
        path,
        skiprows=1,
        dtype=str,
        engine="python",
    )

    # Remove completely empty rows
    df = df.dropna(how="all").copy()

    # Remove trailing format-marker row if present.
    # This row may contain text such as:
    # "[MicrotopsII Database file format]"
    marker_mask = df.astype(str).apply(
        lambda row: row.str.contains(
            "MicrotopsII Database file format",
            case=False,
            na=False,
        ).any(),
        axis=1,
    )

    if marker_mask.any():
        print(
            f"  Removing {marker_mask.sum()} "
            f"format-marker row(s)"
        )
        df = df.loc[~marker_mask].copy()

    # Keep provenance
    df["source_file"] = path.name

    return df


def standardise_instrument_column(df):
    """
    Convert SN to numeric instrument serial number
    and validate against known instruments.
    """

    if "SN" not in df.columns:
        raise ValueError(
            "Required column 'SN' was not found."
        )

    df["SN"] = pd.to_numeric(
        df["SN"],
        errors="coerce",
    )

    if df["SN"].isna().any():
        bad_rows = df.loc[
            df["SN"].isna(),
            ["source_file"],
        ]

        raise ValueError(
            "Some rows have an invalid or missing SN:\n"
            + bad_rows.to_string(index=False)
        )

    df["SN"] = df["SN"].astype(int)

    unknown = sorted(
        set(df["SN"].unique())
        - KNOWN_INSTRUMENTS
    )

    if unknown:
        raise ValueError(
            "Unknown Microtops serial number(s) found: "
            + ", ".join(map(str, unknown))
        )

    return df


def remove_duplicate_measurements(df):
    """
    Remove measurements duplicated across overlapping
    Microtops exports.

    Uses instrument + DATE + TIME as the observation key.
    """

    required = {
        "SN",
        "DATE",
        "TIME",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Cannot identify duplicates because these "
            "columns are missing: "
            + ", ".join(missing)
        )

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "SN",
            "DATE",
            "TIME",
        ],
        keep="first",
    ).copy()

    removed = before - len(df)

    print(
        f"Duplicate rows removed: {removed}"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not raw_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    frames = []

    for path in raw_files:

        df = read_microtops_csv(path)

        df = standardise_instrument_column(
            df
        )

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    print()
    print(
        f"Rows before deduplication: "
        f"{len(combined)}"
    )

    combined = remove_duplicate_measurements(
        combined
    )

    # Sort chronologically where possible
    if (
        "DATE" in combined.columns
        and "TIME" in combined.columns
    ):

        combined["_datetime_sort"] = (
            pd.to_datetime(
                combined["DATE"].astype(str)
                + " "
                + combined["TIME"].astype(str),
                errors="coerce",
            )
        )

        combined = (
            combined
            .sort_values(
                [
                    "_datetime_sort",
                    "SN",
                ]
            )
            .drop(
                columns="_datetime_sort"
            )
            .reset_index(
                drop=True
            )
        )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Master raw dataset written to:"
    )
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Rows by instrument:"
    )
    print(
        combined["SN"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        f"Total rows: {len(combined)}"
    )

    print()
    print(
        f"Total columns: "
        f"{len(combined.columns)}"
    )


if __name__ == "__main__":
    main()
