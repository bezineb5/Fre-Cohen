from abc import ABC, abstractmethod
import logging

from langchain.chains import LLMChain
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field as PyField

from fre_cohen import configuration
from fre_cohen.data_structure import (
    CompositeField,
    FieldsGraph,
    GraphsLayout,
    IndividualGraph,
)
from fre_cohen.llms import build_llm_chain


logger = logging.getLogger(__name__)


class MultipleVisualizationLayer(ABC):
    """Abstract class for the multiple visualization layer"""

    def __init__(self, fields_graph: FieldsGraph):
        self._fields_graph = fields_graph

    @abstractmethod
    def get_layout(self) -> GraphsLayout:
        """Returns the data structure"""


class LayoutItem(BaseModel):
    """Item of the layout"""

    title: str = PyField("", description="Name of the layout")
    fields: list[str] = PyField([], description="Fields composing the layout")


class LayoutInfo(BaseModel):
    """Layout information"""

    titles: list[str] = PyField([], description="List of titles for the layouts")
    fields: list[list[str]] = PyField(
        [], description="List of the lists of fields for the layouts"
    )


class LLMMultipleVisualizationLayer(MultipleVisualizationLayer):
    """LLM-based multiple visualization layer"""

    def __init__(self, config: configuration.Config, fields_graph: FieldsGraph):
        super().__init__(fields_graph)

        self._llm_layout = self._build_llm_chain_for_layout(config)

    def get_layout(self) -> GraphsLayout:
        """Computes a layout for the graph"""
        layout_info = self._generate_layout()

        pairs = zip(layout_info.titles, layout_info.fields)
        return GraphsLayout(
            graphs=[
                IndividualGraph(
                    title=title,
                    fields=[
                        self._get_node_by_name(field_name) for field_name in fields
                    ],
                )
                for title, fields in pairs
            ]
        )

    def _get_node_by_name(self, name: str) -> CompositeField:
        """Returns the node by name"""
        for node in self._fields_graph.nodes:
            if node.name == name:
                return node
        raise ValueError(f"Node with name {name} not found")

    def _generate_layout(self) -> LayoutInfo:
        """Generates the layout"""
        fields_summary = "\n".join(
            [
                f'{field.name}: "{field.description}"'
                for field in self._fields_graph.nodes
            ]
        )
        input_data = {
            "all_fields_details": fields_summary,
        }
        logger.debug("Enrich LLM input: %s", input_data)
        output: LayoutInfo = self._llm_layout.run(input_data)
        logger.debug("Enrich LLM output: %s", output)
        if len(output.fields) != len(output.titles):
            raise ValueError(
                f"Number of titles ({len(output.titles)}) and number of fields ({len(output.fields)}) do not match"
            )

        return output

    def _build_llm_chain_for_layout(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to enrich a field"""
        return build_llm_chain(
            config,
            LayoutInfo,
            [
                HumanMessagePromptTemplate.from_template(
                    "Here are the fields composing our data set: {all_fields_details}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "Can you group the fields into smaller subsets based on their relationships or similarities, and suggest a way to visualize each subset as a separate graph?"
                ),
            ],
        )
