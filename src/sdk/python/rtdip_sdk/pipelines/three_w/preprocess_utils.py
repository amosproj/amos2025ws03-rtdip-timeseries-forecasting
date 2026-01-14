import os
import pandas as pd
from typing import List, Optional


# ============================================================
# 1. Dataset loading
# ============================================================

def load_3w_dataset(
    root_path: str,
    scenarios: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load 3W dataset CSV files from given root path.
    """
    all_frames = []

    scenario_folders = (
        scenarios if scenarios is not None else sorted(os.listdir(root_path))
    )

    print(f"📂 Loading 3W dataset from: {root_path}")

    for scenario in scenario_folders:
        scenario_path = os.path.join(root_path, scenario)

        if not os.path.isdir(scenario_path):
            continue

        files = [f for f in os.listdir(scenario_path) if f.endswith(".csv")]
        print(f"➡️ Scenario {scenario}: {len(files)} files")

        for file in files:
            file_path = os.path.join(scenario_path, file)
            df = pd.read_csv(file_path)

            df["scenario_id"] = scenario
            df["well_file"] = file

            all_frames.append(df)

    if not all_frames:
        raise RuntimeError("❌ No CSV files loaded from 3W dataset")

    return pd.concat(all_frames, ignore_index=True)


# ============================================================
# 2. Timestamp standardization
# ============================================================

def standardize_timestamps(
    df: pd.DataFrame,
    column: str = "timestamp"
) -> pd.DataFrame:
    """
    Convert timestamp column to datetime and sort chronologically.
    """
    df[column] = pd.to_datetime(df[column], errors="coerce")
    df = df.dropna(subset=[column])
    df = df.sort_values(column).reset_index(drop=True)
    return df


# ============================================================
# 3. Missing value handling
# ============================================================

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "ffill"
) -> pd.DataFrame:
    """
    Handle missing values using a chosen strategy.
    """
    if strategy == "ffill":
        df = df.ffill()
    elif strategy == "bfill":
        df = df.bfill()
    elif strategy == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")

    return df


# ============================================================
# 4. Schema validation
# ============================================================

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate expected columns for 3W dataset.
    """
    required_columns = {
        "timestamp",
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "QGL",
        "class",
        "scenario_id",
        "well_file",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    return df


# ============================================================
# 5. Memory optimization
# ============================================================

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory footprint.
    """
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


# ============================================================
# 6. Final preparation for anomaly detection
# ============================================================

def prepare_for_anomaly_detection(
    df: pd.DataFrame,
    drop_label: bool = False
) -> pd.DataFrame:
    """
    Prepare dataframe for anomaly detection models.

    - Ensures timestamp is first column
    - Optionally drops 'class' label
    - Keeps only numeric features + timestamp
    """
    df = df.copy()

    if drop_label and "class" in df.columns:
        df = df.drop(columns=["class"])

    # Ensure timestamp is first column
    cols = list(df.columns)
    if "timestamp" in cols:
        cols.insert(0, cols.pop(cols.index("timestamp")))
        df = df[cols]

    return df

