from enum import Enum
from typing import Any, Optional, Sequence

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


class InstructionOrigin(str, Enum):
    """Instruction origin, capturing the source of the instruction."""

    HUMAN = "human"
    CRITIC = "critic"


class Instruction(BaseModel):
    """Instruction, given by a human or an algorithm."""

    instruction: str
    origin: InstructionOrigin


class IndividualGraph(BaseModel):
    """Individual graph, capturing the links between columns."""

    title: str
    chart_description: str
    independent_variables: Sequence[int]
    dependent_variables: Sequence[int]
    instructions: Sequence[Instruction] = []


class GraphsLayout(BaseModel):
    """Layout of the subgraphs composing the visualization."""

    fields_graph: FieldsGraph
    graphs: Sequence[IndividualGraph]


class GraphSpecifications(BaseModel):
    """Specifications of the graph."""

    format_type: str
    specifications: Any
    graph: IndividualGraph
    map_style: Optional[str] = None


class LayoutSpecifications(BaseModel):
    """Specifications of the layout."""

    fields_graph: FieldsGraph
    specifications: Sequence[GraphSpecifications]
