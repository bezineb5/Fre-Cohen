"""
The visualization layer is responsible for generating the visualization of the fields graph.
"""
from abc import ABC, abstractmethod
from typing import Optional

from fre_cohen.data_structure import FieldsGraph, GraphSpecifications, IndividualGraph


class IndividualVisualizationLayer(ABC):
    """Abstract class for the visualization layer"""

    def __init__(
        self,
        fields_graph: FieldsGraph,
        graph: IndividualGraph,
        previous_specifications: Optional[GraphSpecifications],
    ):
        self._fields_graph = fields_graph
        self._graph = graph
        self._previous_specifications = previous_specifications

    @abstractmethod
    def get_specifications(self) -> GraphSpecifications:
        """Returns the specifications of the graph. The format depends on the implementation."""
