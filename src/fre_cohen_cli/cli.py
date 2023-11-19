"""Minimal CLI for fre_cohen package."""

import argparse
import json
import logging
import pathlib
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple
from xml.etree import ElementTree

from graphviz import Digraph
from pydantic import ValidationError

from fre_cohen import configuration
from fre_cohen.critic_layer import LLMVisualizationCriticLayer
from fre_cohen.data_structure import (
    CompositeField,
    Field,
    FieldsGraph,
    GraphsLayout,
    GraphSpecifications,
    IndividualGraph,
    LayoutSpecifications,
    RichField,
)
from fre_cohen.ingestion import CSVIngestion
from fre_cohen.multi_visualization_layer import LLMMultipleVisualizationLayer
from fre_cohen.rendering.visualization_rendering import render_graph
from fre_cohen.semantic_layer import OpenAISemanticInterpretation
from fre_cohen.vega_visualization_layer import LLMIndividualVegaVisualizationLayer

logger = logging.getLogger(__name__)


class LayerEnum(Enum):
    """Enumeration of the layers"""

    SEMANTIC = "semantic"
    MULTI_VISUALIZATION = "multi_visualization"
    VISUALIZATION = "visualization"
    CRITIC = "critic"
    RENDERING = "rendering"


DEFAULT_LAYERS_EXECUTION = [
    LayerEnum.SEMANTIC.value,
    LayerEnum.MULTI_VISUALIZATION.value,
    LayerEnum.VISUALIZATION.value,
    LayerEnum.CRITIC.value,
    LayerEnum.VISUALIZATION.value,
    LayerEnum.RENDERING.value,
]


def _parse_arguments(arguments: list[str]) -> argparse.Namespace:
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
        "--mbtiles",
        type=str,
        help="The MBTiles file",
    )
    parser.add_argument(
        "--fonts",
        type=str,
        help="The fonts directory",
    )
    parser.add_argument(
        "--layer",
        type=str,
        help="The list of layers to apply",
        choices=[v.value for v in LayerEnum.__members__.values()],
        nargs="+",  # make the argument repeatable
        default=DEFAULT_LAYERS_EXECUTION,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    # OpenAI API key
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="The OpenAI API key",
    )
    return parser.parse_args(arguments)


def _pretty_rich_field_name(rich_field: RichField) -> str:
    """Returns a pretty name for the rich field"""

    return f"{rich_field.field.name} [{rich_field.unit}] ({rich_field.field.type.name}): {rich_field.description}"


def _pretty_composite_field_name(node: CompositeField) -> str:
    """Returns a pretty name for the node"""

    list_fields = "\n* ".join(
        [_pretty_rich_field_name(field) for field in node.columns]
    )

    return f"{node.name}: {node.description}\n{list_fields}"


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

    if not args.layer:
        raise ValueError("No layer to apply")
    return args.layer


def _apply_semantic_layer(
    *, config: configuration.Config, metadata: list[Field], output_base: str, **kwargs
) -> str:
    """Applies the semantic layer"""

    logger.info("Semantic layer")

    # Add semantic information
    sem_interpretation = OpenAISemanticInterpretation(config=config, fields=metadata)
    fields_graph = sem_interpretation.get_data_structure()

    logger.info("Semantic information:")
    logger.info(fields_graph)

    # Generate dot file
    if output_base:
        dot_file = output_base + "_semantic.dot"
        dot = _generate_dot_file(fields_graph)
        with open(dot_file, "w", encoding="utf-8") as f:
            f.write(dot)
        logger.info("Dot file saved to: %s", dot_file)

    # Serialize fields_graph as JSON
    return json.dumps(
        fields_graph, default=lambda o: o.__dict__, sort_keys=True, indent=4
    )


def _apply_multi_visualization_layer(
    *, config: configuration.Config, input_json: str, **kwargs
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
    return json.dumps(
        graphs_layout, default=lambda o: o.__dict__, sort_keys=True, indent=4
    )


def _apply_visualization_layer(
    *,
    config: configuration.Config,
    input_json: str,
    data_source: str,
    **kwargs,
) -> str:
    """Applies the visualization layer on each graph of the multi"""

    logger.info("Visualization layer")

    # Deserialize fields_graph from JSON
    # It can be the output of the multi visualization layer or of the critic layer
    fields_graph: Optional[FieldsGraph] = None
    graph_spec_pairs: list[Tuple[IndividualGraph, Optional[GraphSpecifications]]] = []

    try:
        graphs_layout = GraphsLayout(**json.loads(input_json))
        fields_graph = graphs_layout.fields_graph
        graph_spec_pairs = [(graph, None) for graph in graphs_layout.graphs]
    except ValidationError as e:
        logger.info("Not a GraphsLayout: %s", e)
        layout_specs = LayoutSpecifications(**json.loads(input_json))
        fields_graph = layout_specs.fields_graph
        graph_spec_pairs = [(spec.graph, spec) for spec in layout_specs.specifications]

    # Apply visualization layer
    specs = [
        LLMIndividualVegaVisualizationLayer(
            config, data_source, fields_graph, graph, spec
        ).get_specifications()
        for (graph, spec) in graph_spec_pairs
    ]

    new_layout_specs = LayoutSpecifications(
        fields_graph=fields_graph,
        specifications=specs,
    )

    # Serialize layout_specs as JSON
    return json.dumps(
        new_layout_specs, default=lambda o: o.__dict__, sort_keys=True, indent=4
    )


def _apply_critic_layer(
    *,
    config: configuration.Config,
    input_json: str,
    data_source: str,
    mbtiles_path: Optional[pathlib.Path],
    fonts_path: Optional[pathlib.Path],
    **kwargs,
) -> str:
    """Applies the critic layer"""

    logger.info("Critic layer")

    # Deserialize layout specifications from JSON
    layout_specs = LayoutSpecifications(**json.loads(input_json))

    # Apply critic layer
    critic_layers = []
    for i, spec in enumerate(layout_specs.specifications):
        critic_layer = LLMVisualizationCriticLayer(
            config=config,
            graph=spec,
            data_source=pathlib.Path(data_source),
            mbtiles_path=mbtiles_path,
            fonts_path=fonts_path,
        )
        critic_layers.append(critic_layer)

    # Enrich the layout specifications with critic advices
    enriched_specs = []
    for i, spec in enumerate(layout_specs.specifications):
        enriched_specs.append(critic_layers[i].enrich_with_critic_advices())

    enriched_layout_specs = LayoutSpecifications(
        fields_graph=layout_specs.fields_graph,
        specifications=enriched_specs,
    )

    # Serialize enriched_layout_specs as JSON
    return json.dumps(
        enriched_layout_specs,
        default=lambda o: o.__dict__,
        sort_keys=True,
        indent=4,
    )


def _apply_render_visualization(
    *,
    input_json: str,
    output_base: str,
    data_source: str,
    mbtiles_path: Optional[pathlib.Path],
    fonts_path: Optional[pathlib.Path],
    **kwargs,
):
    """Renders the visualization"""
    if not output_base:
        return None

    logger.info("Rendering visualization")

    # Deserialize layout specifications from JSON
    layout_specs = LayoutSpecifications(**json.loads(input_json))

    # Render each visualization
    svgs = []
    for i, layout_spec in enumerate(layout_specs.specifications):
        try:
            output_file = pathlib.Path(f"{output_base}_visualization_{i}.svg")
            svgs.append(
                render_graph(
                    layout_spec,
                    pathlib.Path(data_source),
                    output_file,
                    mbtiles_path=mbtiles_path,
                    fonts_path=fonts_path,
                )
            )
        except Exception as e:
            logger.error("Error while rendering visualization %d", i, exc_info=e)

    # Now, merge the SVG files into a single one
    svg_layout = _render_svg_layout(svgs)
    output_svg = f"{output_base}_visualization.svg"
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write(svg_layout)
    logger.info("Visualization layout saved to: %s", output_svg)

    return None


@dataclass
class SizedSvg:
    width: int
    height: int
    svg: ElementTree.Element


def _render_svg_layout(svg_files: list[pathlib.Path]) -> str:
    """Renders the SVG layout"""
    # Create the root SVG element
    root = ElementTree.Element("svg", xmlns="http://www.w3.org/2000/svg")

    sized_svgs: list[SizedSvg] = []
    for svg_file in svg_files:
        # Parse the SVG file
        tree = ElementTree.parse(svg_file)
        sized_svg = tree.getroot()

        # Remove the namespace attributes
        for key in sized_svg.attrib.keys():
            if key.startswith("xmlns"):
                del sized_svg.attrib[key]

        # Get the width and height
        width = int(sized_svg.attrib["width"].replace("px", ""))
        height = int(sized_svg.attrib["height"].replace("px", ""))

        # Append the SVG element to the list
        sized_svgs.append(SizedSvg(width=width, height=height, svg=sized_svg))

    # Order the svg files by decreasing height
    sized_svgs.sort(key=lambda svg: svg.height, reverse=True)

    # We want an approximativelly 4/3 aspect ratio
    # Initialize the current position
    x, y = 0, 0

    # Initialize the maximum height of the current row
    max_width = 0
    max_row_height = 0

    # Calculate the total area of the SVGs
    total_area = sum(svg.width * svg.height for svg in sized_svgs)

    # Calculate the width and height of the grid
    grid_width = (total_area * 4 / 3) ** 0.5

    for sized_svg in sized_svgs:
        # If the SVG doesn't fit on the current row, move to the next row
        if x + sized_svg.width > grid_width:
            x = 0
            y += max_row_height
            max_row_height = 0

        # Position the SVG at the current position
        sized_svg.svg.attrib["x"] = str(x)
        sized_svg.svg.attrib["y"] = str(y)

        # Update the current position and the maximum row height
        x += sized_svg.width
        max_width = max(max_width, x)
        max_row_height = max(max_row_height, sized_svg.height)

        # Append the SVG element to the root element
        root.append(sized_svg.svg)

    # Set the width and height of the root SVG
    root.attrib["width"] = str(max_width)
    root.attrib["height"] = str(y + max_row_height)

    # Convert the root element to a string
    return ElementTree.tostring(root, encoding="unicode")


LAYER_METHODS = {
    LayerEnum.SEMANTIC: _apply_semantic_layer,
    LayerEnum.MULTI_VISUALIZATION: _apply_multi_visualization_layer,
    LayerEnum.VISUALIZATION: _apply_visualization_layer,
    LayerEnum.CRITIC: _apply_critic_layer,
    LayerEnum.RENDERING: _apply_render_visualization,
}


def main(arguments: list[str]) -> None:
    """Main function"""

    args = _parse_arguments(arguments)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    open_ai_key = args.openai_api_key
    config = configuration.Config(openai_api_key=open_ai_key)

    # Ingest data from the CSV
    csv_path = args.input
    ingestion = CSVIngestion(path=csv_path)
    data = ingestion.get_data()
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
        layer_method: Optional[Callable] = LAYER_METHODS.get(LayerEnum(layer))
        if layer_method is None:
            raise ValueError(f"Unknown layer {layer}")

        current_json = layer_method(
            config=config,
            metadata=metadata,
            input_json=current_json,
            output_base=args.output,
            data_source=csv_path,
            data=data,
            mbtiles_path=pathlib.Path(args.mbtiles) if args.mbtiles else None,
            fonts_path=pathlib.Path(args.fonts) if args.fonts else None,
        )

        if current_json is None:
            logger.info("No output for layer %s", layer)
            continue
        json_filename = args.output + "_" + layer + ".json"
        with open(json_filename, "w", encoding="utf-8") as f:
            f.write(current_json)
        logger.info("Current JSON [%s]: %s", json_filename, current_json)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
