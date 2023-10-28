"""
The visualization layer is responsible for generating the visualization of the fields graph.
The output is a valid Vega-lite specification.
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Sequence

from langchain.chains import LLMChain
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field as PyField

from fre_cohen import configuration
from fre_cohen.data_structure import (
    FieldsGraph,
    GraphSpecifications,
    GraphsLayout,
    IndividualGraph,
    RichField,
)
from fre_cohen.llms import LLMQualityEnum, build_llm_chain

logger = logging.getLogger(__name__)


class IndividualVisualizationLayer(ABC):
    """Abstract class for the visualization layer"""

    def __init__(self, fields_graph: FieldsGraph, graph: IndividualGraph):
        self._fields_graph = fields_graph
        self._graph = graph

    @abstractmethod
    def get_specifications(self) -> GraphSpecifications:
        """Returns the specifications of the graph. The format depends on the implementation."""


class VegaSpecification(BaseModel):
    """Vega-lite specification"""

    format_type: str = PyField(
        "vega-lite",
        description="Format type of the specification, e.g. vega-lite, vega, plotly, etc.",
    )
    specifications: Any = PyField(
        None,
        description="Specifications of the graph, e.g. the vega-lite specification.",
    )


class LLMIndividualVegaVisualizationLayer(IndividualVisualizationLayer):
    def __init__(
        self,
        config: configuration.Config,
        data_source: str,
        fields_graph: FieldsGraph,
        graph: IndividualGraph,
    ):
        super().__init__(fields_graph, graph)

        self._data_source = data_source
        self._llm_layout = self._build_llm_chain_for_vega(config)

    def get_specifications(self) -> GraphSpecifications:
        """Returns the specifications of the graph"""

        # Run the LLM chain
        output = self._get_vega_lite_specifications()

        return GraphSpecifications(
            format_type="vega-lite",
            specifications=output.specifications,
            graph=self._graph,
        )

    def _summarize_fields(self, fields: Sequence[RichField]) -> str:
        """Summarizes the fields"""
        return "\n".join(
            [
                f'{field.field.name} [{field.unit}]: "{field.description}" with summary: "{field.field.summary}"'
                for field in fields
            ]
        )

    def _summarize_composite_fields(self, variable_indexes: Sequence[int]) -> str:
        """Summarizes the variables"""
        variables = [self._fields_graph.nodes[index] for index in variable_indexes]
        return "\n".join(
            [
                f'Group: {composite_field.name}, described as "{composite_field.description}" with fields:\n{self._summarize_fields(composite_field.columns)}\n'
                for composite_field in variables
            ]
        )

    def _get_vega_lite_specifications(self) -> VegaSpecification:
        """Returns the Vega-lite specifications"""

        # Build the input data
        input_data = {
            "data_source": self._data_source,
            "title": self._graph.title,
            "graph_description": self._graph.chart_description,
            "independent_variables_summary": self._summarize_composite_fields(
                self._graph.independent_variables
            ),
            "dependent_variables_summary": self._summarize_composite_fields(
                self._graph.dependent_variables
            ),
        }
        logger.debug("Vega LLM input: %s", input_data)
        output: VegaSpecification = self._llm_layout.run(input_data)
        logger.debug("Vega LLM output: %s", output)
        return output

    def _build_llm_chain_for_vega(self, config: configuration.Config) -> LLMChain:
        """Builds the LLM chain for vega-lite specification"""
        llm_chain = build_llm_chain(
            config,
            VegaSpecification,
            [
                SystemMessagePromptTemplate.from_template(
                    "This is the title for the graph: {title}"
                ),
                SystemMessagePromptTemplate.from_template(
                    'This is the datasource path for the graph: "{data_source}"'
                ),
                SystemMessagePromptTemplate.from_template(
                    "These are the independent variables for the graph:\n{independent_variables_summary}"
                ),
                SystemMessagePromptTemplate.from_template(
                    "These are the dependent variables for the graph:\n{dependent_variables_summary}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "Given a set of data and a description of the graph you want to create, can you generate a vega-lite specification in JSON format that will produce the desired graph? Please include the data source, the encoding of the data, and any necessary transformations or scales."
                ),
            ],
            LLMQualityEnum.ACCURACY,
        )

        return llm_chain


def build_individual_visualization_layers_for_layout(
    config: configuration.Config, data_source: str, layout: GraphsLayout
) -> Sequence[IndividualVisualizationLayer]:
    """Builds the individual visualization layer"""

    return [
        LLMIndividualVegaVisualizationLayer(
            config, data_source, layout.fields_graph, graph
        )
        for graph in layout.graphs
    ]
