""" Abstract base class for ingestion
"""
from abc import ABC
import logging

import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_categorical_dtype,
    is_string_dtype,
)
from .data_structure import Field, TypeEnum

logger = logging.getLogger(__name__)


class IngestionBase(ABC):
    """Abstract base class for ingestion"""

    def get_data(self) -> pd.DataFrame:
        """Returns the data as a pandas DataFrame"""
        raise NotImplementedError

    def get_metadata(self) -> list[Field]:
        """Returns the metadata as a list of Field"""
        raise NotImplementedError


class CSVIngestion(IngestionBase):
    """Ingestion for CSV files"""

    def __init__(self, path: str):
        self.path = path

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def get_metadata(self) -> list[Field]:
        helper = IngestionHelper()
        df = self.get_data()
        return [
            Field(name=name, type=type, summary=summary, samples=samples)
            for name, type, summary, samples in zip(
                df.columns,
                [helper.get_type_from_dtype(t) for t in df.dtypes.values],
                helper.get_summary(df),
                # Get a sample of the data
                [list(col.values()) for col in df.sample(5).to_dict().values()],
            )
        ]


class IngestionHelper:
    """Helper class for summary"""

    def get_summary(self, df: pd.DataFrame) -> list:
        """Returns the summary"""
        return df.describe(include="all").to_dict().values()

    def get_type_from_dtype(self, dtype) -> TypeEnum:
        """Returns the type from the dtype"""
        if is_numeric_dtype(dtype):
            return TypeEnum.NUMERICAL
        if (
            is_object_dtype(dtype)
            or is_categorical_dtype(dtype)
            or is_string_dtype(dtype)
        ):
            return TypeEnum.CATEGORICAL
        return TypeEnum.NONE
