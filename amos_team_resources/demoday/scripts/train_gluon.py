import sys

from pathlib import Path
from pyspark.sql import SparkSession

DATA = Path(__file__).parent.parent / "data"

print(f"Data directory is located at: {DATA}")


def setup_python_env():
    project_root = Path(__file__).parent.parent.parent.parent
    print(f"Project root directory is located at: {project_root}")

    sdk_path = project_root / "src" / "sdk" / "python"
    sdk_path = sdk_path.resolve()

    sys.path.insert(0, str(sdk_path))


def main():

    print("Setting up Python environment...")
    setup_python_env()
    from rtdip_sdk.pipelines.forecasting.spark.autogluon_timeseries import (
        AutoGluonTimeSeries,
    )

    print("Starting Spark session...")
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("SCADA-Forecasting")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    print("Reading preprocessed SCADA data...")

    data_path = DATA / "scada_prepro.parquet"
    assert data_path.exists(), f"Data file not found at {data_path}"
    df = spark.read.parquet(str(data_path))

    print("Starting AutoGluon Training...")
    ag_model = AutoGluonTimeSeries()
    train_df, test_df = ag_model.split_data(df)
    res_dict = ag_model.train(train_df)

    print("Saving test dataset...")
    test_df.write.mode("overwrite").parquet(str(DATA / "scada_test.parquet"))


if __name__ == "__main__":
    exit(main())
