"""Renders the visualization specifications to a file"""

import json
import logging
import multiprocessing
import pathlib
from typing import Any, Optional

import altair as alt

from fre_cohen.data_structure import GraphSpecifications
from fre_cohen.rendering.file_server import FileServer
from fre_cohen.rendering.map_server import MapConfiguration, MapServer

logger = logging.getLogger(__name__)


def render_graph(
    graph_specs: GraphSpecifications,
    data_source: pathlib.Path,
    output_file: pathlib.Path,
    mbtiles_path: Optional[pathlib.Path],
    fonts_path: Optional[pathlib.Path],
) -> pathlib.Path:
    """Renders the graph specifications"""

    # Bug in vl-convert: it cannot fetch data from a file:// URL
    # So we need to start a webserver to serve the data
    with FileServer(file_to_serve=data_source) as server:
        logger.info("Server started. Rendering visualization to: %s", output_file)

        # Hack to fix the bug in vl-convert:
        # update the data source path to the locally served file
        graph_specs.specifications.setdefault("data", {})["url"] = server.get_file_url()
        logger.info(
            "Updated data source: %s", graph_specs.specifications["data"]["url"]
        )

        if graph_specs.map_style and mbtiles_path and fonts_path:
            # There is a map style, so we need to render using the map server
            map_configuration = MapConfiguration(
                mbtiles_path=mbtiles_path,
                fonts_path=fonts_path,
                style_json=graph_specs.map_style,
            )
            with MapServer(map_configuration) as m:
                graph_specs.specifications = _add_map_layer(
                    graph_specs.specifications, m.get_url()
                )
                _start_render(graph_specs.specifications, output_file)
        else:
            _start_render(graph_specs.specifications, output_file)

    return output_file


def _add_map_layer(vl_spec: Any, server_url: str) -> Any:
    """Adds a tile map layer to the vega visualization.
    Inspired by: https://github.com/vega/vega-lite/issues/5758#issuecomment-1462683219
    """
    map_layer = {
        "data": {
            "name": "tiles",
            "url": {
                "property": "{x}/{y}/{z}",
                "format": {"type": "topojson", "feature": "countries"},
            },
        },
        "mark": {
            "type": "image",
            "url": {"expr": f"'{server_url}/tile/' + datum.property"},
        },
        "encoding": {
            "longitude": {"field": "x", "type": "quantitative"},
            "latitude": {"field": "y", "type": "quantitative"},
            "url": {"field": "url", "type": "nominal"},
        },
    }

    vl_spec["layer"].append(map_layer)
    return vl_spec


def _start_render(vl_spec: Any, output_file: pathlib.Path) -> None:
    vl_spec_json = json.dumps(vl_spec)
    # For no known reason, it must be done in a separate process
    process = multiprocessing.Process(
        target=_render, args=(vl_spec_json, str(output_file))
    )
    process.start()
    process.join()
    logger.info("Visualization %s saved to: %s", vl_spec_json, output_file)


def _render(vl_spec: str, output_file: str) -> None:
    """Renders the content, made in a separate process to avoid a bug in vl-convert"""
    chart = alt.Chart.from_json(vl_spec)
    chart.save(output_file)
