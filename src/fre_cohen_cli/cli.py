"""Minimal CLI for fre_cohen package."""

import argparse
import http.server
import json
import logging
import pathlib
import multiprocessing
from enum import Enum
import socket
from typing import Optional

import altair as alt
from graphviz import Digraph

from fre_cohen import configuration
from fre_cohen.data_structure import (
    CompositeField,
    Field,
    FieldsGraph,
    GraphsLayout,
    LayoutSpecifications,
    RichField,
)
from fre_cohen.ingestion import CSVIngestion
from fre_cohen.multi_visualization_layer import LLMMultipleVisualizationLayer
from fre_cohen.semantic_layer import OpenAISemanticInterpretation
from fre_cohen.visualization_layer import (
    build_individual_visualization_layers_for_layout,
)

logger = logging.getLogger(__name__)


class LayerEnum(Enum):
    """Enumeration of the layers"""

    SEMANTIC = "semantic"
    MULTI_VISUALIZATION = "multi_visualization"
    VISUALIZATION = "visualization"
    RENDERING = "rendering"


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
    graphs_layout = GraphsLayout(**json.loads(input_json))

    # Apply visualization layer
    viz_layers = build_individual_visualization_layers_for_layout(
        config=config, data_source=data_source, layout=graphs_layout
    )
    specs = [viz_layer.get_specifications() for viz_layer in viz_layers]

    layout_specs = LayoutSpecifications(
        fields_graph=graphs_layout.fields_graph,
        specifications=specs,
    )

    # Serialize layout_specs as JSON
    return json.dumps(
        layout_specs, default=lambda o: o.__dict__, sort_keys=True, indent=4
    )


class FileServer:
    """Simple HTTP server for serving a file"""

    def __init__(self, port: Optional[int], file_to_serve: pathlib.Path) -> None:
        self.process: Optional[multiprocessing.Process] = None
        self.file_to_serve = file_to_serve.absolute()
        self.port = port if port else self._find_available_port()

    def _find_available_port(self) -> int:
        """Finds an available port"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def get_file_url(self) -> str:
        """Returns the URL of the file which is served"""
        return f"http://localhost:{self.port}/{self.file_to_serve.name}"

    def _start(self) -> None:
        server = self._init_server()
        server.serve_forever()

    def start_in_new_process(self) -> None:
        """Starts the server in a new process"""
        self.process = multiprocessing.Process(target=self._start)
        self.process.start()

    def _init_server(self) -> http.server.HTTPServer:
        file_to_serve = self.file_to_serve.name

        class Handler(http.server.SimpleHTTPRequestHandler):
            """Handler for serving the file"""

            def __init__(self, *args, **kwargs):
                self.file_to_serve = file_to_serve
                super().__init__(*args, **kwargs)

            def do_GET(self):
                self.path = self.file_to_serve
                return super().do_GET()

            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                super().end_headers()

        # Create the server
        logger.info("Serving file %s on port %d", self.file_to_serve, self.port)
        return http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)

    def stop(self) -> None:
        """Stops the server"""
        if self.process:
            # Wait for the server process to finish
            self.process.kill()
            self.process.join()
            self.process = None


def _render(vl_spec: str, output_file: str) -> None:
    """Renders the content, made in a separate process to avoid a bug in vl-convert"""
    chart = alt.Chart.from_json(vl_spec)
    chart.save(output_file)


def _apply_render_visualization(
    *,
    input_json: str,
    output_base: str,
    data_source: str,
    **kwargs,
):
    """Renders the visualization"""
    if not output_base:
        return None

    # Deserialize layout specifications from JSON
    layout_specs = LayoutSpecifications(**json.loads(input_json))

    # Bug in vl-convert: it cannot fetch data from a file:// URL
    # So we need to start a webserver to serve the data
    server = FileServer(port=53789, file_to_serve=pathlib.Path(data_source))
    server.start_in_new_process()
    logger.info("Server started")

    # Render each visualization
    for i, layout_spec in enumerate(layout_specs.specifications):
        try:
            output_file = f"{output_base}_visualization_{i}.svg"
            logger.info("Rendering visualization %d to: %s", i, output_file)

            # Hack to fix the bug in vl-convert:
            # update the data source path to the locally served file
            layout_spec.specifications.setdefault("data", {})[
                "url"
            ] = server.get_file_url()
            logger.info(
                "Updated data source: %s", layout_spec.specifications["data"]["url"]
            )

            # Render the visualization directly using vl_convert
            vl_spec = json.dumps(layout_spec.specifications)
            # For no known reason, it must be done in a separate process
            process = multiprocessing.Process(
                target=_render, args=(vl_spec, output_file)
            )
            process.start()
            process.join()
            logger.info("Visualization %s saved to: %s", vl_spec, output_file)
        except Exception as e:
            logger.error("Error while rendering visualization %d", i, exc_info=e)

    server.stop()
    return None


LAYER_METHODS = {
    LayerEnum.SEMANTIC: _apply_semantic_layer,
    LayerEnum.MULTI_VISUALIZATION: _apply_multi_visualization_layer,
    LayerEnum.VISUALIZATION: _apply_visualization_layer,
    LayerEnum.RENDERING: _apply_render_visualization,
}


def main():
    """Main function"""

    args = _parse_arguments()
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
        layer_method = LAYER_METHODS.get(LayerEnum(layer))
        if layer_method is None:
            raise ValueError(f"Unknown layer {layer}")

        current_json = layer_method(
            config=config,
            metadata=metadata,
            input_json=current_json,
            output_base=args.output,
            data_source=csv_path,
            data=data,
        )

        if current_json is None:
            logger.info("No output for layer %s", layer)
            continue
        json_filename = args.output + "_" + layer + ".json"
        with open(json_filename, "w", encoding="utf-8") as f:
            f.write(current_json)
        logger.info("Current JSON [%s]: %s", json_filename, current_json)


if __name__ == "__main__":
    main()
