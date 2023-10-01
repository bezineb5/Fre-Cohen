from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel

# Structure of the input data columns, capturing:
# - the basic data types
# - statistics for each column
# - the column names
# - the composite data types (ex.: coordinates, date time)
# - the link between columns (causality, correlation)


class LinkEnum(str, Enum):
    """Link between columns, capturing causality, correlation or none."""

    CAUSALITY = "causality"
    CORRELATION = "correlation"
    NONE = "none"


class CompositeEnum(str, Enum):
    """Composite data types, capturing coordinates, date time, etc."""

    LOCATION = "location"
    VECTOR = "vector"
    DATETIME = "datetime"
    MATRIX = "matrix"
    QUATERNION = "quaternion"
    NONE = "none"


class TypeEnum(str, Enum):
    """Basic data types, capturing categorical, numerical and none."""

    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    NONE = "none"


class Summary(BaseModel):
    mean: float
    median: float
    min: float
    max: float
    std: float
    variance: float
    unique_values: int


class Field(BaseModel):
    """Basic field type, capturing the name, type and summary."""

    name: str
    type: TypeEnum
    summary: dict[str, Any]
    samples: Sequence[Any]


class RichField(BaseModel):
    """Enriched field type, capturing the unit and description."""

    field: Field
    unit: str
    description: str


class CompositeField(BaseModel):
    """Composite data type, capturing the link between columns."""

    columns: Sequence[RichField]
    name: str
    description: str
    composite: CompositeEnum


class IntentType(str, Enum):
    """Intent type, capturing the purpose of the analysis."""

    IDENTITY_TRENDS = "identity_patterns_or_trends"
    COMPARISON = "comparison"
    CAUSALITY = "understand_impact_of_causes"
    RELATIONSHIP = "relationship_between_variables"
    QUALITY = "analyze_quality_of_data"
    MONITORING = "monitoring"
    NONE = "none"


class FieldsGraph(BaseModel):
    """Graph of fields, capturing the links between columns."""

    nodes: Sequence[CompositeField]
    edges: Sequence[tuple[CompositeField, CompositeField, LinkEnum]]

    intents: Sequence[IntentType]
