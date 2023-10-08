"""Minimal CLI for fre_cohen package."""

import argparse
import json
import logging
from enum import Enum

from graphviz import Digraph

from fre_cohen import configuration
from fre_cohen.data_structure import CompositeField, Field, FieldsGraph, RichField
from fre_cohen.ingestion import CSVIngestion
from fre_cohen.multi_visualization_layer import LLMMultipleVisualizationLayer
from fre_cohen.semantic_layer import OpenAISemanticInterpretation

logger = logging.getLogger(__name__)


class LayerEnum(Enum):
    """Enumeration of the layers"""

    SEMANTIC = "semantic"
    MULTI_VISUALIZATION = "multi_visualization"


def _parse_arguments():
    """Parses the arguments"""

    parser = argparse.ArgumentParser(description="CLI for fre_cohen package")
    parser.add_argument(
        "--input",
        type=str,
        help="The input CSV file",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The output base filename",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        help="The input JSON file",
    )
    parser.add_argument(
        "--layer",
        type=str,
        help="The layer type",
        choices=[v.value for v in LayerEnum.__members__.values()],
        nargs="+",  # make the argument repeatable
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    # OpenAI API key
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="The OpenAI API key",
    )
    return parser.parse_args()


def _pretty_rich_field_name(rich_field: RichField) -> str:
    """Returns a pretty name for the rich field"""

    return f"{rich_field.field.name} [{rich_field.unit}] ({rich_field.field.type.name}): {rich_field.description}"


def _pretty_composite_field_name(node: CompositeField) -> str:
    """Returns a pretty name for the node"""

    list_fields = "\n* ".join(
        [_pretty_rich_field_name(field) for field in node.columns]
    )

    return f"{node.name} ({node.composite.name}): {node.description}\n{list_fields}"


def _generate_dot_file(fields_graph: FieldsGraph) -> str:
    dot = Digraph(comment="Fields Graph")
    dot.format = "png"

    # Add nodes
    for node in fields_graph.nodes:
        dot.node(node.name, label=_pretty_composite_field_name(node))

    # Add edges
    for edge in fields_graph.edges:
        source = fields_graph.nodes[edge.source]
        target = fields_graph.nodes[edge.target]
        dot.edge(source.name, target.name, label=edge.link.name)

    return dot.source


def _list_layers_to_apply(args: argparse.Namespace) -> list[str]:
    """Returns the list of layers to apply"""

    raw_layers = args.layer or [
        LayerEnum.SEMANTIC.value,
        LayerEnum.MULTI_VISUALIZATION.value,
    ]
    return raw_layers


def _apply_semantic_layer(
    config: configuration.Config, metadata: list[Field], output_base: str, **kwargs
) -> str:
    """Applies the semantic layer"""

    logger.info("Semantic layer")

    # Add semantic information
    sem_interpretation = OpenAISemanticInterpretation(config=config, fields=metadata)
    fields_graph = sem_interpretation.get_data_structure()

    logger.info("Semantic information:")
    logger.info(fields_graph)

    # Serialize fields_graph as JSON
    fields_graph_json = json.dumps(
        fields_graph, default=lambda o: o.__dict__, sort_keys=True, indent=4
    )

    # Generate dot file
    if output_base:
        dot_file = output_base + "_semantic.dot"
        dot = _generate_dot_file(fields_graph)
        with open(dot_file, "w", encoding="utf-8") as f:
            f.write(dot)
        logger.info("Dot file saved to: %s", dot_file)

    return fields_graph_json


def _apply_multi_visualization_layer(
    config: configuration.Config, input_json: str, **kwargs
) -> str:
    """Applies the multi visualization layer"""

    logger.info("Multi visualization layer")

    # Deserialize fields_graph from JSON
    fields_graph = FieldsGraph(**json.loads(input_json))

    # Apply multi visualization layer
    multi_visualization_layer = LLMMultipleVisualizationLayer(
        config=config, fields_graph=fields_graph
    )
    graphs_layout = multi_visualization_layer.get_layout()

    # Serialize graphs_layout as JSON
    graphs_layout_json = json.dumps(
        graphs_layout, default=lambda o: o.__dict__, sort_keys=True, indent=4
    )

    return graphs_layout_json


LAYER_METHODS = {
    LayerEnum.SEMANTIC: _apply_semantic_layer,
    LayerEnum.MULTI_VISUALIZATION: _apply_multi_visualization_layer,
}


def main():
    """Main function"""

    args = _parse_arguments()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    open_ai_key = args.openai_api_key
    config = configuration.Config(openai_api_key=open_ai_key)

    # Ingest data from the CSV
    ingestion = CSVIngestion(path=args.input)
    # data = ingestion.get_data()
    metadata = ingestion.get_metadata()

    # Apply relevant layers
    layers = _list_layers_to_apply(args)
    logger.info("Layers to apply: %s", layers)
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            current_json = f.read()
    else:
        current_json = "{}"

    for layer in layers:
        layer_method = LAYER_METHODS.get(LayerEnum(layer))
        if layer_method is None:
            raise ValueError(f"Unknown layer {layer}")

        current_json = layer_method(
            config=config,
            metadata=metadata,
            input_json=current_json,
            output_base=args.output,
        )

        json_filename = args.output + "_" + layer + ".json"
        with open(json_filename, "w", encoding="utf-8") as f:
            f.write(current_json)
        logger.info("Current JSON [%s]: %s", json_filename, current_json)

    # # Add semantic information
    # sem_interpretation = OpenAISemanticInterpretation(config=config, fields=metadata)
    # fields_graph = sem_interpretation.get_data_structure()

    # logger.info("Semantic information:")
    # logger.info(fields_graph)

    # # Generate dot file
    # if args.output:
    #     dot = _generate_dot_file(fields_graph)
    #     with open(args.output, "w", encoding="utf-8") as f:
    #         f.write(dot)


if __name__ == "__main__":
    main()
