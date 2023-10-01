""" Methods to add semantic information to the data fields.
"""

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from langchain.chains import LLMChain
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field as PyField

from fre_cohen import configuration
from fre_cohen.data_structure import (
    CompositeEnum,
    CompositeField,
    Field,
    FieldsGraph,
    IntentType,
    LinkEnum,
    RichField,
)
from fre_cohen.llms import build_llm_chain

logger = logging.getLogger(__name__)


class SemanticInterpretation(ABC):
    """Abstract base class for semantic interpretation"""

    def __init__(self, fields: list[Field]):
        self._fields = fields

    @abstractmethod
    def get_data_structure(self) -> FieldsGraph:
        """Returns the data structure"""


class SemanticInfo(BaseModel):
    """Semantic information"""

    unit: str = PyField(description="The unit of the field")
    description: str = PyField(description="The description of the field")


class AggregationInfo(BaseModel):
    """Aggregation information"""

    field_names: list[str] = PyField([], description="The fields to aggregate")
    composite: CompositeEnum = PyField(
        CompositeEnum.NONE, description="The composite type of the fields"
    )
    name: str = PyField("", description="The name of the composite field")


class GroupingInfo(BaseModel):
    """Grouping information"""

    composite_fields: list[AggregationInfo] = PyField(
        [], description="The composite fields"
    )


class LinkCompositeFields(BaseModel):
    """Link between data structures"""

    from_field: str = PyField(description="The data structure from")
    to_field: str = PyField(description="The data structure to")
    link: LinkEnum = PyField(LinkEnum.NONE, description="The link between the fields")


class CausalInfo(BaseModel):
    """Causal information"""

    links: list[LinkCompositeFields] = PyField(
        [], description="The links between the data structures"
    )


class IntentInfo(BaseModel):
    """Intent information"""

    intents: list[IntentType] = PyField(
        [], description="The intent of the visualization"
    )


class OpenAISemanticInterpretation(SemanticInterpretation):
    """Semantic interpretation using OpenAI API"""

    def __init__(self, config: configuration.Config, fields: list[Field]):
        super().__init__(fields)

        self._config = config

        self._llm_enrich_field = self._build_llm_chain_for_rich_field(config)
        self._llm_grouping = self._build_llm_chain_for_grouping(config)
        self._llm_links = self._build_llm_chain_for_links(config)
        self._llm_intent = self._build_llm_chain_for_intent(config)

    def get_data_structure(self) -> FieldsGraph:
        """Returns the data structure"""

        # Enrich the field data
        rich_fields = [self._enrich_field(field) for field in self._fields]

        # Group related fields
        grouped_structure = self._group_fields(rich_fields)

        def _find_composite_field(name: str) -> CompositeField:
            return next(cf for cf in grouped_structure if cf.name == name)

        # Determine links between fields
        links = self._find_links(grouped_structure)
        edges = [
            (
                _find_composite_field(l.from_field),
                _find_composite_field(l.to_field),
                l.link,
            )
            for l in links
            if l.link != LinkEnum.NONE
        ]

        # Determine the intent
        intents = self._determine_intents(grouped_structure)

        # Build the graph
        return FieldsGraph(
            nodes=grouped_structure,
            edges=edges,
            intents=intents,
        )

    def _determine_intents(
        self, composite_fields: Sequence[CompositeField]
    ) -> Sequence[IntentType]:
        """Determines the intent of the visualization"""
        input_data = {
            "all_composite_field_names": [
                composite_field.name for composite_field in composite_fields
            ],
        }

        logger.debug("Intent LLM input: %s", input_data)
        try:
            intent_info: IntentInfo = self._llm_intent.run(input_data)
            logger.debug("Intent LLM output: %s", intent_info)
        except Exception as ex:
            logger.error("Failed to run intent LLM: %s", ex)
            intent_info = IntentInfo(intents=[])

        return intent_info.intents

    def _find_links(
        self, composite_fields: Sequence[CompositeField]
    ) -> Sequence[LinkCompositeFields]:
        """Finds the links between the data structures"""
        input_data = {
            "all_composite_field_names": [
                composite_field.name for composite_field in composite_fields
            ],
        }

        logger.debug("Links LLM input: %s", input_data)
        links: CausalInfo = self._llm_links.run(input_data)
        logger.debug("Links LLM output: %s", links)

        return links.links

    def _group_fields(self, fields: Sequence[RichField]) -> Sequence[CompositeField]:
        """Group fields together"""
        input_data = {
            "all_field_names": [field.field.name for field in fields],
        }

        logger.debug("Group LLM input: %s", input_data)
        grouping_info: GroupingInfo = self._llm_grouping.run(input_data)
        logger.debug("Group LLM output: %s", grouping_info)

        grouped_ds = [
            CompositeField(
                columns=self._string_to_richfields(group.field_names, fields),
                name=group.name,
                description="",  # group.description,
                composite=group.composite,
            )
            for group in grouping_info.composite_fields
        ]

        # Add missing fields not belonging to any group
        included_in_groups = [
            field.field.name for group in grouped_ds for field in group.columns
        ]

        for field in fields:
            if field.field.name not in included_in_groups:
                grouped_ds.append(
                    CompositeField(
                        columns=[field],
                        name=field.field.name,
                        description=field.description,
                        composite=CompositeEnum.NONE,
                    )
                )

        return grouped_ds

    def _string_to_richfields(
        self, fields_to_lookup: Sequence[str], all_fields: Sequence[RichField]
    ) -> Sequence[RichField]:
        """Converts a list of field names to a list of RichField in the order of the field names"""
        return [
            next(field for field in all_fields if field.field.name == field_name)
            for field_name in fields_to_lookup
        ]

    def _enrich_field(self, field: Field) -> RichField:
        """Enriches the field with semantic information"""
        input_data = {
            "field_name": field.name,
            "field_type": field.type,
            "field_summary": field.summary,
            "field_samples": field.samples,
        }
        logger.debug("Enrich LLM input: %s", input_data)
        output: SemanticInfo = self._llm_enrich_field.run(input_data)
        logger.debug("Enrich LLM output: %s", output)

        return RichField(
            field=field,
            unit=output.unit,
            description=output.description,
        )

    def _build_llm_chain_for_rich_field(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to enrich a field"""
        return build_llm_chain(
            config,
            SemanticInfo,
            [
                SystemMessagePromptTemplate.from_template(
                    "Units are important to understand the data. Please write them as symbols."
                ),
                HumanMessagePromptTemplate.from_template(
                    'Given a data set named "{field_name}", with some samples like this: "{field_samples}", what do you think is the unit of the data? How would you describe the data?'
                ),
            ],
        )

    def _build_llm_chain_for_aggregation(
        self, config: configuration.Config
    ) -> LLMChain:
        """Builds a LLMChain to aggregate fields"""
        return build_llm_chain(
            config,
            AggregationInfo,
            [
                HumanMessagePromptTemplate.from_template(
                    'There is a group of fields named "{other_field_names}" with units "{other_field_units}" and described as "{other_field_descriptions}". They all relate together as they form a multi-dimensional field.'
                ),
                HumanMessagePromptTemplate.from_template(
                    'Given a field named "{field_name}", with unit "{field_unit}" and described as "{field_description}", do you think it is another dimension to the same group of fields? What is the logical relationship between the fields?'
                ),
            ],
        )

    def _build_llm_chain_for_grouping(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to aggregate fields"""
        return build_llm_chain(
            config,
            GroupingInfo,
            [
                HumanMessagePromptTemplate.from_template(
                    "Considering the fields: {all_field_names} - which fields would you group together to form meaningful composite fields? Please give a name to each group of fields with a composition reason."
                ),
            ],
        )

    def _build_llm_chain_for_links(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to determine the links between the fields"""
        return build_llm_chain(
            config,
            CausalInfo,
            [
                HumanMessagePromptTemplate.from_template(
                    "Considering the fields: {all_composite_field_names} - what do you think the causality link are between those fields?"
                ),
            ],
        )

    def _build_llm_chain_for_intent(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to determine the intent of the visualization"""
        return build_llm_chain(
            config,
            IntentInfo,
            [
                HumanMessagePromptTemplate.from_template(
                    "Considering the fields: {all_composite_field_names} - What is the main purpose of the visualization? What insights do you hope to gain from it?"
                ),
            ],
        )
