from pathlib import Path
from typing import Iterable, Literal
import pandas as pd

from .interfaces import ThreeWPreprocessingInterface
from .preprocess_utils import (
    load_3w_dataset,
    standardize_timestamps,
    validate_schema,
    handle_missing_values,
    prepare_for_anomaly_detection,
)


class ThreeWPreprocessing(ThreeWPreprocessingInterface):
    """
    RTDIP preprocessing component for the 3W dataset.
    """

    def __init__(
        self,
        data_path: str | Path,
        scenarios: Iterable[str] | None = None,
        missing_value_strategy: Literal["ffill", "drop"] = "ffill",
    ):
        self.data_path = data_path
        self.scenarios = scenarios
        self.missing_value_strategy = missing_value_strategy

    def run(self) -> pd.DataFrame:
        df = load_3w_dataset(
            root_path=self.data_path,
            scenarios=self.scenarios,
        )

        validate_schema(df)
        df = standardize_timestamps(df)
        df = handle_missing_values(df, strategy=self.missing_value_strategy)
        df = prepare_for_anomaly_detection(df)

        return df

