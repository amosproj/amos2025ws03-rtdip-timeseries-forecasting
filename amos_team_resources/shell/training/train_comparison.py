"""


REQUIREMENTS:
- pip install autogluon.timeseries
- pip install xgboost
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import gc

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sdk_path = os.path.join(repo_root, "src", "sdk", "python")
sys.path.insert(0, sdk_path)

preprocessing_dir = os.path.abspath(os.path.join(script_dir, "..", "preprocessing"))
sys.path.insert(0, preprocessing_dir)

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from rtdip_sdk.pipelines.forecasting.spark.xgboost_timeseries import XGBoostTimeSeries
from rtdip_sdk.pipelines.forecasting.spark.catboost_timeseries_refactored import CatBoostTimeSeries
from rtdip_sdk.pipelines.visualization.matplotlib.comparison import (
    ModelComparisonPlot,
    ModelMetricsTable,
    ModelsOverlayPlot,
    ComparisonDashboard,
)
from rtdip_sdk.pipelines.visualization.matplotlib.forecasting import ForecastPlot
from pyspark.sql import SparkSession

SHELL_DIR = os.path.abspath(os.path.join(script_dir, ".."))
RAW_DATA_PATHS = [
    os.path.join(SHELL_DIR, "data", "ShellData.parquet"),
    os.path.join(SHELL_DIR, "data", "ShellData.csv"),
]
PREPROCESSED_DATA_PATH = os.path.join(SHELL_DIR, "preprocessing", "ShellData_preprocessed.parquet")
FILTERED_DATA_PATH = os.path.join(SHELL_DIR, "preprocessing", "ShellData_preprocessed_filtered.parquet")
OUTPUT_DIR = "comparison_results"
AUTOGLUON_DIR = os.path.join(OUTPUT_DIR, "autogluon")
XGBOOST_DIR = os.path.join(OUTPUT_DIR, "xgboost")
CATBOOST_DIR = os.path.join(OUTPUT_DIR, "catboost")

N_SIGMA = 10.0  
TOP_N_SENSORS = 10
PREDICTION_LENGTH = 24
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TIME_LIMIT = 600
EVAL_METRIC = "MAE"
PRESET = "medium_quality"
FREQ = "h"


def preprocess_raw_data(raw_data_path, output_path, n_sigma=10.0, sample_ratio=None):
    """
    Preprocess raw Shell data using RTDIP pipeline components.

    Args:
        raw_data_path: Path to raw data file (.parquet or .csv)
        output_path: Path for output parquet file
        n_sigma: Number of MAD-based standard deviations for outlier detection
        sample_ratio: Fraction of data to sample (e.g., 0.1 for 10%). None uses all data.

    Returns:
        Preprocessed DataFrame
    """
    print("=" * 80)
    print("PREPROCESSING RAW SHELL DATA")
    print("=" * 80)

    from rtdip_sdk.pipelines.data_quality.data_manipulation.pandas import (
        MixedTypeSeparation,
        DatetimeStringConversion,
        OneHotEncoding,
        DatetimeFeatures,
        ChronologicalSort,
        MADOutlierDetection,
    )

    print(f"\nLoading raw data from: {raw_data_path}")
    if raw_data_path.endswith('.parquet'):
        df = pd.read_parquet(raw_data_path)
    else:
        df = pd.read_csv(raw_data_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    if sample_ratio is not None and sample_ratio < 1.0:
        original_len = len(df)
        df = df.sample(frac=sample_ratio, random_state=42)
        print(f"Sampled {sample_ratio*100:.1f}% of data: {len(df):,} rows (from {original_len:,})")
        gc.collect()

    print("\nStep 1: Separating text values from numeric Value column")
    separator = MixedTypeSeparation(
        df, column="Value", placeholder=-1, string_fill="NaN", suffix="_str"
    )
    df = separator.apply()
    gc.collect()

    print("Step 2: Converting EventTime to datetime")
    converter = DatetimeStringConversion(
        df,
        column="EventTime",
        output_column="EventTime_DT",
        strip_trailing_zeros=True,
        keep_original=True,
    )
    df = converter.apply()
    gc.collect()

    print("Step 3: One-hot encoding Status column")
    encoder = OneHotEncoding(df, column="Status", sparse=False)
    df = encoder.apply()
    gc.collect()

    print("Step 4: Extracting datetime features")
    extractor = DatetimeFeatures(
        df,
        datetime_column="EventTime_DT",
        features=["day", "week", "weekday", "day_name"],
        prefix="EventTime",
    )
    df = extractor.apply()
    df["EventTime_month"] = df["EventTime_DT"].dt.month_name()
    dt = df["EventTime_DT"].dt
    df["EventTime_seconds"] = (dt.hour * 3600 + dt.minute * 60 + dt.second).astype("Int32")
    gc.collect()

    print("Step 5: Handling missing values")
    initial_rows = len(df)
    df = df.dropna(subset=["EventTime_DT"]).copy()
    print(f"  Dropped {initial_rows - len(df):,} rows with NaT timestamps")

    value_missing = df["Value"].isna().sum()
    if value_missing > 0:
        df = df.dropna(subset=["Value"])
        print(f"  Dropped {value_missing:,} rows with missing Value")
    gc.collect()

    print("Step 6: Sorting chronologically")
    sorter = ChronologicalSort(
        df,
        datetime_column="EventTime_DT",
        ascending=True,
        na_position="last",
        reset_index=True,
    )
    df = sorter.apply()
    gc.collect()

    print(f"Step 7: Detecting outliers using MAD (n_sigma={n_sigma})")
    detector = MADOutlierDetection(
        df,
        column="Value",
        n_sigma=n_sigma,
        action="replace",
        replacement_value=-1,
        exclude_values=[-1],
    )
    df = detector.apply()
    gc.collect()

    if sample_ratio is None or sample_ratio >= 1.0:
        print(f"\nSaving preprocessed data to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False, compression="snappy")
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"Saved ({file_size:.2f} MB)")
    else:
        print(f"\nSkipping save (sampled data, not caching)")

    print(f"\nPreprocessing complete: {len(df):,} rows")
    print("=" * 80)

    return df


def find_and_load_data(data_path=None, sample_ratio=None):
    """
    Find and load Shell data, preprocessing if necessary.

    Priority:
    0. provided data path (if specified)
    1. Filtered preprocessed data (flat sensors removed) - only if no sampling
    2. Preprocessed data - only if no sampling
    3. Raw data (will be preprocessed with optional sampling)

    Args:
        data_path: Direct path to preprocessed data. If provided, uses this file directly.
        sample_ratio: Fraction of data to sample. If provided, skips cached preprocessed files.

    Returns:
        DataFrame with preprocessed data
    """
    if data_path is not None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Specified data file not found: {data_path}")
        print(f"Loading user-specified data: {data_path}")
        df = pd.read_parquet(data_path)
        if sample_ratio is not None and sample_ratio < 1.0:
            original_len = len(df)
            df = df.sample(frac=sample_ratio, random_state=42)
            print(f"Sampled {sample_ratio*100:.1f}% of data: {len(df):,} rows (from {original_len:,})")
        return df

    if sample_ratio is not None and sample_ratio < 1.0:
        print(f"Sampling requested ({sample_ratio*100:.1f}%), will preprocess from raw data")
        for raw_path in RAW_DATA_PATHS:
            if os.path.exists(raw_path):
                print(f"Found raw data: {raw_path}")
                return preprocess_raw_data(raw_path, PREPROCESSED_DATA_PATH, N_SIGMA, sample_ratio)
        raise FileNotFoundError(f"Raw data not found for sampling. Looked in: {RAW_DATA_PATHS}")

    if os.path.exists(FILTERED_DATA_PATH):
        print(f"Found filtered preprocessed data: {FILTERED_DATA_PATH}")
        return pd.read_parquet(FILTERED_DATA_PATH)

    if os.path.exists(PREPROCESSED_DATA_PATH):
        print(f"Found preprocessed data: {PREPROCESSED_DATA_PATH}")
        return pd.read_parquet(PREPROCESSED_DATA_PATH)

    for raw_path in RAW_DATA_PATHS:
        if os.path.exists(raw_path):
            print(f"Found raw data: {raw_path}")
            return preprocess_raw_data(raw_path, PREPROCESSED_DATA_PATH, N_SIGMA)

    raise FileNotFoundError(
        f"No Shell data found. Please place raw data at one of:\n"
        f"  - {RAW_DATA_PATHS[0]}\n"
        f"  - {RAW_DATA_PATHS[1]}\n"
        f"Or preprocessed data at:\n"
        f"  - {PREPROCESSED_DATA_PATH}"
    )


def load_shell_data(data_path=None, top_n_sensors=10, sample_ratio=None):
    """
    Load and prepare Shell data for time series forecasting.
    Will preprocess raw data if preprocessed data is not available.

    Args:
        data_path: Direct path to preprocessed data file. If provided, skips auto-detection.
        top_n_sensors: Number of top sensors by data volume to use
        sample_ratio: Fraction of data to sample during preprocessing (e.g., 0.1 for 10%)
    """
    print("LOADING SHELL DATA")

    df = find_and_load_data(data_path=data_path, sample_ratio=sample_ratio)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    print(f"\nSelecting top {top_n_sensors} sensors by data volume")
    sensor_counts = df["TagName"].value_counts()
    top_sensors = sensor_counts.head(top_n_sensors).index.tolist()

    df_filtered = df[df["TagName"].isin(top_sensors)].copy()
    print(f"Selected {len(df_filtered):,} rows from {top_n_sensors} sensors")

    print("\nTop sensors:")
    for i, (sensor, count) in enumerate(sensor_counts.head(top_n_sensors).items(), 1):
        print(f"  {i}. {sensor}: {count:,} data points")

    timestamp_col = "EventTime_DT" if "EventTime_DT" in df_filtered.columns else "EventTime"
    ts_data = pd.DataFrame(
        {
            "item_id": df_filtered["TagName"],
            "timestamp": df_filtered[timestamp_col],
            "target": df_filtered["Value"],
        }
    )

    # Remove any null values not cleared in preprocessing for some reason (shouldnt really happen)
    original_len = len(ts_data)
    ts_data = ts_data.dropna(subset=["target"])
    if len(ts_data) < original_len:
        print(f"  Removed {original_len - len(ts_data):,} rows with null target values")

    original_len = len(ts_data)
    ts_data = ts_data[ts_data["target"] != -1].copy()
    if len(ts_data) < original_len:
        print(
            f"  Removed {original_len - len(ts_data):,} rows with error markers (Value = -1)"
        )

    ts_data = ts_data.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    print(f"Final dataset: {len(ts_data):,} rows")
    print(f"Time range: {ts_data['timestamp'].min()} to {ts_data['timestamp'].max()}")
    print(
        f"Target range: [{ts_data['target'].min():.2f}, {ts_data['target'].max():.2f}]"
    )

    return ts_data


def split_timeseries_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split time series data into train/val/test sets (time-aware, per sensor).
    Each sensor's timeline is split individually to ensure all sensors appear in all splits.
    """
    print("SPLITTING DATA")

    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
    ), "Split ratios must sum to 1.0"

    train_dfs = []
    val_dfs = []
    test_dfs = []

    item_ids = df["item_id"].unique()
    print(f"Splitting {len(item_ids)} sensors individually...")

    for item_id in item_ids:
        item_data = df[df["item_id"] == item_id].copy()
        item_data = item_data.sort_values("timestamp")

        n = len(item_data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dfs.append(item_data.iloc[:train_end])
        val_dfs.append(item_data.iloc[train_end:val_end])
        test_dfs.append(item_data.iloc[val_end:])

    # Combine splits again
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    n = len(df)
    print(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
    print(f"Train set: {len(train_df):,} rows ({len(train_df)/n:.1%})")
    print(
        f"  Time range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}"
    )
    print(f"  Unique sensors: {train_df['item_id'].nunique()}")
    print(f"Val set:   {len(val_df):,} rows ({len(val_df)/n:.1%})")
    print(f"  Time range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print(f"  Unique sensors: {val_df['item_id'].nunique()}")
    print(f"Test set:  {len(test_df):,} rows ({len(test_df)/n:.1%})")
    print(f"  Time range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"  Unique sensors: {test_df['item_id'].nunique()}")

    return train_df, val_df, test_df


def create_timeseries_dataframe(df, freq="h"):
    """
    Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="timestamp"
    )
    ts_df = ts_df.convert_frequency(freq=freq)

    return ts_df


def train_autogluon(
    train_df, prediction_length, eval_metric, time_limit, preset, freq="h", verbosity=2
):
    """Train AutoGluon time series predictor."""
    print("TRAINING AUTOGLUON MODELS")
    print(f"Prediction length: {prediction_length} time steps")
    print(f"Time frequency: {freq}")
    print(f"Evaluation metric: {eval_metric}")
    print(f"Time limit: {time_limit} seconds ({time_limit/60:.1f} minutes)")
    print(f"Quality preset: {preset}")

    train_data = create_timeseries_dataframe(train_df, freq=freq)
    print(f"\nCreated TimeSeriesDataFrame with {len(train_data)} rows")

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        eval_metric=eval_metric,
        freq=freq,
        verbosity=verbosity,
    )

    # Train models
    print("\nStarting training")
    start_time = datetime.now()

    predictor.fit(train_data=train_data, time_limit=time_limit, presets=preset)

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    print(
        f"Training completed in {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)"
    )

    return predictor


def evaluate_autogluon(predictor, test_df, freq="h"):
    """Evaluate AutoGluon model and return metrics + future forecasts.

    Note: AutoGluon's predict() generates future forecasts, not predictions at test timestamps.
    The returned predictions_df contains forecasts beyond the test set. Not really suited for comparison with LSTM
    /XGBoost predictions
    (afaik, but thats what we have to work with for now).
    """
    print("EVALUATING AUTOGLUON")

    min_points = PREDICTION_LENGTH + 1
    sensor_counts = test_df.groupby("item_id").size()
    valid_sensors = sensor_counts[sensor_counts >= min_points].index.tolist()
    
    if len(valid_sensors) < len(sensor_counts):
        excluded = len(sensor_counts) - len(valid_sensors)
        print(f"  Excluding {excluded} sensors with fewer than {min_points} test points")
    
    test_df_filtered = test_df[test_df["item_id"].isin(valid_sensors)].copy()
    test_data = create_timeseries_dataframe(test_df_filtered, freq=freq)

    print("Computing metrics")
    metrics = predictor.evaluate(
        test_data, metrics=["MAE", "RMSE", "MAPE", "MASE", "SMAPE"]
    )

    print("\nAutoGluon Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:20s}: {metric_value:.4f}")

    leaderboard = predictor.leaderboard()
    best_model = leaderboard.iloc[0]["model"]
    print(f"\nBest Model: {best_model}")

    print("\nGenerating future forecasts")
    predictions = predictor.predict(test_data)
    predictions_df = predictions.reset_index()
    # predictions_df has columns: ['item_id', 'timestamp', 'mean', quantiles...]

    print(f"Generated {len(predictions_df)} future forecast rows")
    print(
        f"Note: These are forecasts beyond the test set, not aligned with test timestamps"
    )

    return metrics, leaderboard, predictions_df


def train_xgboost(train_df, prediction_length=24):
    """Train XGBoost model."""
    print("TRAINING XGBOOST MODEL")

    xgboost_model = XGBoostTimeSeries(
        target_col="target",
        timestamp_col="timestamp",
        item_id_col="item_id",
        prediction_length=prediction_length,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=150,
        n_jobs=-1,
    )

    xgboost_model.train(train_df)

    return xgboost_model


def evaluate_xgboost(xgboost_model, test_df):
    """Evaluate XGBoost model and return metrics + predictions."""
    print("EVALUATING XGBOOST")

    print("Computing metrics")
    metrics = xgboost_model.evaluate(test_df)

    if metrics:
        print("\nXGBoost Metrics:")
        print("-" * 80)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:20s}: {abs(metric_value):.4f}")

    print("\nGenerating future forecasts")
    predictions_df = xgboost_model.predict(test_df)

    print(f"Generated {len(predictions_df)} prediction rows")

    return metrics, predictions_df


def train_catboost(train_df, prediction_length=24):
    """Train CatBoost model."""
    print("TRAINING CATBOOST MODEL")

    catboost_model = CatBoostTimeSeries(
        target_col="target",
        timestamp_col="timestamp",
        item_id_col="item_id",
        prediction_length=prediction_length,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=150,
        n_jobs=-1,
    )

    catboost_model.train(train_df)

    return catboost_model


def evaluate_catboost(catboost_model, test_df):
    """Evaluate CatBoost model and return metrics + predictions."""
    print("EVALUATING CATBOOST")

    print("Computing metrics")
    metrics = catboost_model.evaluate(test_df)

    if metrics:
        print("\nCatBoost Metrics:")
        print("-" * 80)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:20s}: {abs(metric_value):.4f}")

    print("\nGenerating future forecasts")
    predictions_df = catboost_model.predict(test_df)

    print(f"Generated {len(predictions_df)} prediction rows")

    return metrics, predictions_df


def compute_residuals(predictions_df, test_df, model_name):
    """Compute residuals (predicted - actual) for a model."""
    predictions_df = predictions_df.copy()
    test_df = test_df.copy()
    predictions_df["timestamp"] = pd.to_datetime(predictions_df["timestamp"])
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    # Rename 'mean' to 'predicted' if needed (AutoGluon formatting, match others)
    if "mean" in predictions_df.columns:
        predictions_df = predictions_df.rename(columns={"mean": "predicted"})

    # Try exact timestamp match first
    merged = pd.merge(
        test_df[["item_id", "timestamp", "target"]],
        predictions_df[["item_id", "timestamp", "predicted"]],
        on=["item_id", "timestamp"],
        how="inner",
    )

    # If no exact matches found, use hourly normalization
    if len(merged) == 0:
        print(
            f"   No exact timestamp matches for {model_name}, using hourly normalization"
        )
        predictions_df["timestamp_normalized"] = predictions_df["timestamp"].dt.floor(
            "h"
        )
        test_df["timestamp_normalized"] = test_df["timestamp"].dt.floor("h")

        merged = pd.merge(
            test_df[["item_id", "timestamp", "timestamp_normalized", "target"]],
            predictions_df[["item_id", "timestamp_normalized", "predicted"]],
            on=["item_id", "timestamp_normalized"],
            how="inner",
        )

    merged["residual"] = merged["predicted"] - merged["target"]
    merged["model_name"] = model_name

    return merged[
        ["item_id", "timestamp", "target", "predicted", "residual", "model_name"]
    ]


def compute_per_sensor_metrics(predictions_df, test_df, model_name):
    """Compute evaluation metrics for each sensor individually."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        mean_absolute_percentage_error,
    )

    residuals_df = compute_residuals(predictions_df, test_df, model_name)

    per_sensor_results = []

    for item_id in residuals_df["item_id"].unique():
        sensor_data = residuals_df[residuals_df["item_id"] == item_id]

        y_true = sensor_data["target"].values
        y_pred = sensor_data["predicted"].values

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # MAPE (filter near-zero values)
        non_zero_mask = np.abs(y_true) >= 0.1
        if np.sum(non_zero_mask) > 0:
            mape = mean_absolute_percentage_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]
            )
        else:
            mape = np.nan

        # MASE
        if len(y_true) > 1:
            naive_forecast = y_true[:-1]
            mae_naive = mean_absolute_error(y_true[1:], naive_forecast)
            mase = mae / mae_naive if mae_naive != 0 else mae
        else:
            mase = np.nan

        # SMAPE
        smape = (
            100
            * (
                2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
            ).mean()
        )

        per_sensor_results.append(
            {
                "item_id": item_id,
                "model_name": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "MASE": mase,
                "SMAPE": smape,
                "num_predictions": len(y_true),
            }
        )

    return pd.DataFrame(per_sensor_results)


def generate_comparison_report(
    ag_metrics, xgboost_metrics, catboost_metrics, ag_leaderboard, output_dir
):
    """Generate comparison report for AutoGluon vs XGBoost vs CatBoost."""
    print("COMPARISON REPORT")

    os.makedirs(output_dir, exist_ok=True)

    comparison_data = []
    for metric_name in ["MAE", "RMSE", "MAPE", "MASE", "SMAPE"]:
        ag_value = ag_metrics[metric_name]
        xgboost_value = xgboost_metrics[metric_name]
        catboost_value = catboost_metrics[metric_name]

        # Convert back to positive for display (higher negative is better → lower positive is better)
        ag_abs = abs(ag_value)
        xgboost_abs = abs(xgboost_value)
        catboost_abs = abs(catboost_value)

        # Determine winner (higher negative value is better -- i.e., lower absolute error)
        values = {
            "AutoGluon": ag_value,
            "XGBoost": xgboost_value,
            "CatBoost": catboost_value,
        }
        winner = max(values.items(), key=lambda x: x[1])[0]

        comparison_data.append(
            {
                "Metric": metric_name,
                "AutoGluon": ag_abs,
                "XGBoost": xgboost_abs,
                "CatBoost": catboost_abs,
                "Winner": winner,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    print("\nSide-by-Side Comparison:")
    print(comparison_df.to_string(index=False))

    ag_wins = sum(1 for row in comparison_data if row["Winner"] == "AutoGluon")
    xgboost_wins = sum(1 for row in comparison_data if row["Winner"] == "XGBoost")
    catboost_wins = sum(1 for row in comparison_data if row["Winner"] == "CatBoost")

    print(f"\nOverall Performance:")
    print(f"  AutoGluon wins: {ag_wins}/5 metrics")
    print(f"  XGBoost wins: {xgboost_wins}/5 metrics")
    print(f"  CatBoost wins: {catboost_wins}/5 metrics")

    wins = {
        "AutoGluon": ag_wins,
        "XGBoost": xgboost_wins,
        "CatBoost": catboost_wins,
    }
    max_wins = max(wins.values())
    winners = [model for model, count in wins.items() if count == max_wins]

    if len(winners) == 1:
        print(f"\nOverall Winner: {winners[0]}")
    else:
        print(f"\nOverall Result: TIE between {', '.join(winners)}")

    comparison_path = os.path.join(output_dir, "comparison_report.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison report saved to: {comparison_path}")

    leaderboard_path = os.path.join(output_dir, "autogluon_leaderboard.csv")
    ag_leaderboard.to_csv(leaderboard_path, index=False)
    print(f"AutoGluon leaderboard saved to: {leaderboard_path}")

    combined_metrics = {
        "AutoGluon": {k: abs(v) for k, v in ag_metrics.items()},
        "XGBoost": {k: abs(v) for k, v in xgboost_metrics.items()},
        "CatBoost": {k: abs(v) for k, v in catboost_metrics.items()},
    }
    metrics_df = pd.DataFrame(combined_metrics).T
    metrics_path = os.path.join(output_dir, "combined_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"Combined metrics saved to: {metrics_path}")

    return comparison_df


def generate_visualizations(
    ag_metrics,
    xgboost_metrics,
    catboost_metrics,
    ag_predictions_df,
    xgboost_predictions_df,
    catboost_predictions_df,
    test_df,
    output_dir,
):
    """Generate visualizations comparing all models using SDK visualization components."""
    print("\nGENERATING VISUALIZATIONS")

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    metrics_dict = {
        "AutoGluon": {k.lower(): abs(v) for k, v in ag_metrics.items()},
        "XGBoost": {k.lower(): abs(v) for k, v in xgboost_metrics.items()},
        "CatBoost": {k.lower(): abs(v) for k, v in catboost_metrics.items()},
    }

    ag_preds = ag_predictions_df.copy()
    if "mean" in ag_preds.columns and "prediction" not in ag_preds.columns:
        ag_preds = ag_preds.rename(columns={"mean": "prediction"})

    xgb_preds = xgboost_predictions_df.copy()
    if "predicted" in xgb_preds.columns and "prediction" not in xgb_preds.columns:
        xgb_preds = xgb_preds.rename(columns={"predicted": "prediction"})

    cb_preds = catboost_predictions_df.copy()
    if "predicted" in cb_preds.columns and "prediction" not in cb_preds.columns:
        cb_preds = cb_preds.rename(columns={"predicted": "prediction"})

    predictions_dict = {
        "AutoGluon": ag_preds,
        "XGBoost": xgb_preds,
        "CatBoost": cb_preds,
    }

    sensors = test_df["item_id"].unique()
    first_sensor = sensors[0] if len(sensors) > 0 else None

    print("Creating model comparison bar chart")
    try:
        comparison_plot = ModelComparisonPlot(
            metrics_dict=metrics_dict,
            metrics_to_plot=["mae", "rmse", "mape", "smape"],
        )
        fig = comparison_plot.plot()
        comparison_path = os.path.join(viz_dir, "model_comparison.png")
        fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {comparison_path}")
    except Exception as e:
        print(f"    Warning: Could not create comparison plot: {e}")
    print("Creating metrics table")
    try:
        metrics_table = ModelMetricsTable(
            metrics_dict=metrics_dict,
            highlight_best=True,
        )
        fig = metrics_table.plot()
        table_path = os.path.join(viz_dir, "metrics_table.png")
        fig.savefig(table_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {table_path}")
    except Exception as e:
        print(f"    Warning: Could not create metrics table: {e}")

    print("Creating per-sensor forecast overlays")
    for sensor in sensors[:3]:
        try:
            actual_sensor = test_df[test_df["item_id"] == sensor].copy()

            overlay_plot = ModelsOverlayPlot(
                predictions_dict=predictions_dict,
                sensor_id=sensor,
                actual_data=actual_sensor,
            )
            fig = overlay_plot.plot()
            safe_sensor_name = sensor.replace("/", "_").replace("\\", "_")
            overlay_path = os.path.join(viz_dir, f"forecast_overlay_{safe_sensor_name}.png")
            fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
            print(f"    Saved: {overlay_path}")
        except Exception as e:
            print(f"    Warning: Could not create overlay for {sensor}: {e}")

    print("Creating individual forecast plots")
    if first_sensor:
        test_sensor = test_df[test_df["item_id"] == first_sensor].copy()
        test_sensor = test_sensor.sort_values("timestamp")

        historical_data = test_sensor.rename(columns={"target": "value"})

        for model_name, pred_df in [
            ("XGBoost", xgb_preds),
            ("CatBoost", cb_preds),
        ]:
            try:
                pred_sensor = pred_df[pred_df["item_id"] == first_sensor].copy()
                if len(pred_sensor) > 0:
                    forecast_data = pred_sensor.rename(columns={"prediction": "mean"})
                    forecast_start = forecast_data["timestamp"].min()

                    forecast_plot = ForecastPlot(
                        historical_data=historical_data.tail(100),
                        forecast_data=forecast_data,
                        forecast_start=forecast_start,
                        sensor_id=first_sensor,
                    )
                    fig = forecast_plot.plot()
                    safe_sensor_name = first_sensor.replace("/", "_").replace("\\", "_")
                    forecast_path = os.path.join(
                        viz_dir, f"forecast_{model_name.lower()}_{safe_sensor_name}.png"
                    )
                    fig.savefig(forecast_path, dpi=150, bbox_inches="tight")
                    print(f"    Saved: {forecast_path}")
            except Exception as e:
                print(f"    Warning: Could not create {model_name} forecast plot: {e}")

    print("Creating comparison dashboard")
    if first_sensor:
        try:
            actual_sensor = test_df[test_df["item_id"] == first_sensor].copy()

            dashboard = ComparisonDashboard(
                predictions_dict=predictions_dict,
                metrics_dict=metrics_dict,
                sensor_id=first_sensor,
                actual_data=actual_sensor,
            )
            fig = dashboard.plot()
            dashboard_path = os.path.join(viz_dir, "comparison_dashboard.png")
            fig.savefig(dashboard_path, dpi=300, bbox_inches="tight")
            print(f"    Saved: {dashboard_path}")
        except Exception as e:
            print(f"    Warning: Could not create dashboard: {e}")

    print(f"\nVisualizations saved to: {viz_dir}")
    return viz_dir


def main(data_path=None, sample_ratio=None, top_n_sensors=None):
    """
    Main function to run the model comparison.

    Args:
        data_path: Path to preprocessed data file. If provided, skips auto-detection.
        sample_ratio: Fraction of data to sample (e.g., 0.1 for 10%). None uses all data.
        top_n_sensors: Number of top sensors to use. None uses TOP_N_SENSORS config.
    """
    if top_n_sensors is None:
        top_n_sensors = TOP_N_SENSORS

    print("=" * 80)
    print("AUTOGLUON vs XGBOOST vs CATBOOST COMPARISON")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if data_path:
        print(f"Using provided data: {data_path}")
    if sample_ratio:
        print(f"Using {sample_ratio*100:.1f}% sample of data")

    try:
        # 1. Load data (will preprocess if needed)
        df = load_shell_data(data_path=data_path, top_n_sensors=top_n_sensors, sample_ratio=sample_ratio)

        # 2. Split data
        train_df, val_df, test_df = split_timeseries_data(
            df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
        )

        # 3. Train AutoGluon
        ag_start_time = datetime.now()
        ag_predictor = train_autogluon(
            train_df,
            prediction_length=PREDICTION_LENGTH,
            eval_metric=EVAL_METRIC,
            time_limit=TIME_LIMIT,
            preset=PRESET,
            freq=FREQ,
            verbosity=2,
        )
        ag_training_duration = (datetime.now() - ag_start_time).total_seconds()

        # 4. Train XGBoost
        xgboost_start_time = datetime.now()
        xgboost_model = train_xgboost(train_df, prediction_length=PREDICTION_LENGTH)
        xgboost_training_duration = (
            datetime.now() - xgboost_start_time
        ).total_seconds()

        # 5. Train CatBoost
        catboost_start_time = datetime.now()
        catboost_model = train_catboost(train_df, prediction_length=PREDICTION_LENGTH)
        catboost_training_duration = (
            datetime.now() - catboost_start_time
        ).total_seconds()

        # 6. Evaluate AutoGluon
        ag_metrics, ag_leaderboard, ag_predictions_df = evaluate_autogluon(
            ag_predictor, test_df, freq=FREQ
        )

        # 7. Evaluate XGBoost
        xgboost_metrics, xgboost_predictions_df = evaluate_xgboost(
            xgboost_model, test_df
        )

        if xgboost_metrics is None:
            print(
                "\n ERROR: XGBoost evaluation failed. Cannot generate comparison report."
            )
            return 1

        # 8. Evaluate CatBoost
        catboost_metrics, catboost_predictions_df = evaluate_catboost(
            catboost_model, test_df
        )

        if catboost_metrics is None:
            print(
                "\n ERROR: CatBoost evaluation failed. Cannot generate comparison report."
            )
            return 1

        # 9. Save test set actual values for visualization
        print("SAVING DATA FOR VISUALIZATION")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        test_actuals_path = os.path.join(OUTPUT_DIR, "test_actuals.parquet")
        test_df.to_parquet(test_actuals_path, index=False)
        print(f"Test actuals saved to: {test_actuals_path}")

        # 10. Save AutoGluon predictions (future forecasts)
        ag_pred_path = os.path.join(OUTPUT_DIR, "autogluon_predictions.parquet")
        ag_predictions_df.to_parquet(ag_pred_path, index=False)
        print(f"AutoGluon predictions saved to: {ag_pred_path}")
        print(
            f"  Note: AutoGluon predictions are future forecasts, not aligned with test timestamps"
        )

        # 11. Save XGBoost predictions (future forecasts)
        xgboost_pred_path = os.path.join(OUTPUT_DIR, "xgboost_predictions.parquet")
        xgboost_predictions_df.to_parquet(xgboost_pred_path, index=False)
        print(f"XGBoost predictions saved to: {xgboost_pred_path}")
        print(
            f"  Note: XGBoost predictions are future forecasts, not aligned with test timestamps"
        )

        # 12. Save CatBoost predictions (future forecasts)
        catboost_pred_path = os.path.join(OUTPUT_DIR, "catboost_predictions.parquet")
        catboost_predictions_df.to_parquet(catboost_pred_path, index=False)
        print(f"CatBoost predictions saved to: {catboost_pred_path}")
        print(
            f"  Note: CatBoost predictions are future forecasts, not aligned with test timestamps"
        )

        # 13. Save training metadata
        metadata = {
            "run_timestamp": datetime.now().isoformat(),
            "autogluon_training_time_seconds": ag_training_duration,
            "xgboost_training_time_seconds": xgboost_training_duration,
            "catboost_training_time_seconds": catboost_training_duration,
            "num_sensors": train_df["item_id"].nunique(),
            "prediction_length": PREDICTION_LENGTH,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "freq": FREQ,
            "eval_metric": EVAL_METRIC,
            "data_split": {
                "train_start": train_df["timestamp"].min().isoformat(),
                "train_end": train_df["timestamp"].max().isoformat(),
                "val_start": val_df["timestamp"].min().isoformat(),
                "val_end": val_df["timestamp"].max().isoformat(),
                "test_start": test_df["timestamp"].min().isoformat(),
                "test_end": test_df["timestamp"].max().isoformat(),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
            },
            "model_configs": {
                "xgboost": {
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "n_estimators": 150,
                    "lag_features": [1, 6, 12, 24, 48],
                    "rolling_windows": [12, 24],
                },
                "catboost": {
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 150,
                    "lag_features": [1, 6, 12, 24, 48],
                    "rolling_windows": [12, 24],
                },
            },
            "notes": {
                "models": "Comparison includes AutoGluon (ensemble), XGBoost (gradient boosting with feature engineering), and CatBoost (gradient boosting with feature engineering). LSTM is disabled due to TensorFlow compatibility issues.",
                "predictions": "All models generate future forecasts (prediction_length steps beyond test set).",
                "autogluon_predictions": "AutoGluon predictions are future forecasts from ensemble.",
                "xgboost_predictions": "XGBoost predictions are recursive forecasts with engineered lag features.",
                "catboost_predictions": "CatBoost predictions are recursive forecasts with engineered lag features.",
            },
        }
        metadata_path = os.path.join(OUTPUT_DIR, "training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nTraining metadata saved to: {metadata_path}")

        # 14. Save sensor list
        sensor_list = {
            "sensors": sorted(train_df["item_id"].unique().tolist()),
            "num_sensors": train_df["item_id"].nunique(),
        }
        sensor_path = os.path.join(OUTPUT_DIR, "sensor_list.json")
        with open(sensor_path, "w") as f:
            json.dump(sensor_list, f, indent=2)
        print(f"Sensor list saved to: {sensor_path}")

        # 15. Generate comparison report
        comparison_df = generate_comparison_report(
            ag_metrics, xgboost_metrics, catboost_metrics, ag_leaderboard, OUTPUT_DIR
        )

        # 16. Generate visualizations
        generate_visualizations(
            ag_metrics=ag_metrics,
            xgboost_metrics=xgboost_metrics,
            catboost_metrics=catboost_metrics,
            ag_predictions_df=ag_predictions_df,
            xgboost_predictions_df=xgboost_predictions_df,
            catboost_predictions_df=catboost_predictions_df,
            test_df=test_df,
            output_dir=OUTPUT_DIR,
        )

        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nAll results saved to: {OUTPUT_DIR}")
        print("\nComparison completed")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and compare AutoGluon, XGBoost, and CatBoost models on Shell sensor data."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to preprocessed data file (.parquet). Skips preprocessing and uses this file directly.",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Sample a fraction of data (e.g., 0.1 for 10%%).",
    )
    parser.add_argument(
        "--top-n-sensors",
        type=int,
        default=None,
        help=f"Number of top sensors to use (default: {TOP_N_SENSORS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit(main(data_path=args.data, sample_ratio=args.sample, top_n_sensors=args.top_n_sensors))
