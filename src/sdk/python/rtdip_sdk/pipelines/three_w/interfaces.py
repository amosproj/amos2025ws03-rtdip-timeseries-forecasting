from abc import ABC, abstractmethod
import pandas as pd


class ThreeWPreprocessingInterface(ABC):

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Run preprocessing and return a cleaned DataFrame."""
        pass

