"""A context manager for running a map server in a docker container"""

from dataclasses import dataclass
import json
import logging
import pathlib
import tempfile

import docker

from fre_cohen.rendering import utils

logger = logging.getLogger(__name__)

MAPTILER_IMAGE = "maptiler/tileserver-gl"


@dataclass
class MapConfiguration:
    """A configuration for the map"""

    mbtiles_path: pathlib.Path
    fonts_path: pathlib.Path
    style_json: str


class MapServer:
    """A context manager for running a map server in a docker container"""

    def __init__(
        self,
        map_configuration: MapConfiguration,
        image_name: str = MAPTILER_IMAGE,
        container_port: int = 8080,
    ):
        self.map_configuration = map_configuration
        self.image_name = image_name
        self.container = None
        self.external_port = utils.find_available_port()
        self.port_mapping = {f"{container_port}/tcp": self.external_port}

        # Create a temporary configuration file for the map server
        self.config_path = pathlib.Path(tempfile.mkstemp()[1])
        config_content = self._configuration_file()
        self.config_path.write_text(config_content, encoding="utf-8")
        logger.debug("Map server configuration: %s", config_content)

        # Create a temporary style file for the map server
        self.style_path = pathlib.Path(tempfile.mkstemp()[1])
        self._interpolate_style(self.get_url())

        self.volumes = {
            str(map_configuration.mbtiles_path.absolute()): {
                "bind": "/data/tiles.mbtiles",
                "mode": "ro",
            },
            str(self.style_path.absolute()): {
                "bind": "/data/styles/style.json",
                "mode": "ro",
            },
            str(self.config_path.absolute()): {
                "bind": "/data/config.json",
                "mode": "ro",
            },
            str(map_configuration.fonts_path.absolute()): {
                "bind": "/data/fonts",
                "mode": "ro",
            },
        }

    def _configuration_file(self):
        config = {
            "options": {"paths": {"fonts": "fonts", "styles": "styles"}},
            "styles": {
                "custom-style": {
                    "style": "style.json",
                    "tilejson": {"type": "overlay"},
                },
            },
            "data": {"openmaptiles": {"mbtiles": "tiles.mbtiles"}},
        }

        return json.dumps(config, indent=2)

    def __enter__(self):
        client = docker.from_env()
        self.container = client.containers.run(
            self.image_name,
            ports=self.port_mapping,
            volumes=self.volumes,
            detach=True,
        )

        logger.info("Started map server: %s on %s", self.container.name, self.get_url())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.stop()
        self.container.remove()
        self.container = None
        self.config_path.unlink()
        self.config_path = None
        self.style_path.unlink()
        self.style_path = None
        logger.info("Stopped map server")

    def get_url(self):
        """Get the url of the map server"""
        # self.container.reload()
        # logger.debug(
        #     "Map server port mapping: %s",
        #     self.container.attrs["NetworkSettings"]["Ports"],
        # )
        # host_port = self.container.attrs["NetworkSettings"]["Ports"][
        #     f"{list(self.port_mapping.keys())[0]}"
        # ][0]["HostPort"]
        # host_port_int = int(host_port)
        return f"http://localhost:{self.external_port}"

    def _interpolate_style(self, url: str) -> None:
        """Interpolate the style with the actual url of the map server"""
        interpolated_style = self.map_configuration.style_json.replace("{{host}}", url)
        self.style_path.write_text(interpolated_style, encoding="utf-8")
        logger.debug("Interpolated map server style: %s", interpolated_style)
