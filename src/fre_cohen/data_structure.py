from enum import Enum
from typing import Any, Optional, Sequence, Union

from pydantic import BaseModel

# Structure of the input data columns, capturing:
# - the basic data types
# - statistics for each column
# - the column names
# - the composite data types (ex.: coordinates, date time)
# - the link between columns (causality, correlation)


class LinkEnum(str, Enum):
    """Link between columns, capturing causality, correlation or none."""

    DEPENDENCY = "dependency"
    CORRELATION = "correlation"
    CONSTRAINT = "constraint"
    NONE = "none"


class Location(BaseModel):
    """Location data type, capturing the latitude and longitude."""

    latitude: float
    longitude: float
    altitude: Optional[float]


class Vector(BaseModel):
    """Vector data type, capturing the x, y and z coordinates."""

    x: float
    y: float
    z: float


class DateTime(BaseModel):
    """Date time data type, capturing the date and time."""

    date: str
    time: str


class Matrix(BaseModel):
    """Matrix data type, capturing the matrix."""

    matrix: Sequence[Sequence[float]]


class Quaternion(BaseModel):
    """Quaternion data type, capturing the quaternion."""

    w: float
    x: float
    y: float
    z: float


class CompositeUnion(BaseModel):
    """Composite data type, capturing the union of all composite types."""

    composite: Union[Location, Vector, DateTime, Matrix, Quaternion]


class CompositeEnum(str, Enum):
    """Composite data types, capturing coordinates, date time, etc."""

    LOCATION = "location (latitude, longitude, altitude)"
    VECTOR = "vector (x, y, z)"
    DATETIME = "datetime (date, time)"
    QUATERNION = "quaternion (w, x, y, z)"
    MATRIX = "matrix (list of float)"
    NONE = "none (basic data type)"


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


class Edge(BaseModel):
    """Edge, capturing the source and target nodes and the link."""

    source: int
    target: int
    link: LinkEnum


class FieldsGraph(BaseModel):
    """Graph of fields, capturing the links between columns."""

    nodes: Sequence[CompositeField]
    edges: Sequence[Edge]

    intents: Sequence[IntentType]

    description: str


class IndividualGraph(BaseModel):
    """Individual graph, capturing the links between columns."""

    title: str
    chart_description: str
    independent_variables: Sequence[int]
    dependent_variables: Sequence[int]


class GraphsLayout(BaseModel):
    """Layout of the subgraphs composing the visualization."""

    fields_graph: FieldsGraph
    graphs: Sequence[IndividualGraph]


class GraphSpecifications(BaseModel):
    """Specifications of the graph."""

    format_type: str
    specifications: Any
    visualization_type: str
    graph: IndividualGraph


class LayoutSpecifications(BaseModel):
    """Specifications of the layout."""

    fields_graph: FieldsGraph
    specifications: Sequence[GraphSpecifications]
