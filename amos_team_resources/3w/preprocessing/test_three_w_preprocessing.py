from rtdip_sdk.pipelines.preprocessing.three_w import ThreeWPreprocessing

def test_three_w_preprocessing():
    component = ThreeWPreprocessing(
        data_path="/path/to/data/data folder",   # local 3W path
        scenarios=["0"],            # small subset
        missing_value_strategy="ffill",
    )

    df = component.run()

    # Basic assertions
    assert df is not None
    assert len(df) > 0

    # Required columns
    expected_columns = {
        "timestamp",
        "P-PDG", "P-TPT", "T-TPT",
        "P-MON-CKP", "T-JUS-CKP",
        "P-JUS-CKGL", "QGL",
        "class",
        "scenario_id",
        "well_file",
    }
    assert expected_columns.issubset(df.columns)

    # No NaNs in sensor columns
    sensor_columns = [
        "P-PDG", "P-TPT", "T-TPT",
        "P-MON-CKP", "T-JUS-CKP",
        "P-JUS-CKGL", "QGL",
    ]
    assert df[sensor_columns].isna().sum().sum() == 0

    # Correct dtypes
    assert df["timestamp"].dtype.name.startswith("datetime")
    assert df["class"].dtype == "int64"

    df = component.run()

    print(df.head())
    print(df.shape)
    print(df.dtypes)

    df.to_parquet(
        "amos_team_resources/3w/preprocessing/three_w_preprocessed_sample.parquet",
        index=False
    )

    print("✅ ThreeWPreprocessing component test passed")


if __name__ == "__main__":
    test_three_w_preprocessing()

