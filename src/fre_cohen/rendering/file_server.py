import http.server
import logging
import multiprocessing
import pathlib
from typing import Optional

from fre_cohen.rendering import utils

logger = logging.getLogger(__name__)


class FileServer:
    """Simple HTTP server for serving a file"""

    def __init__(self, file_to_serve: pathlib.Path, port: Optional[int] = None) -> None:
        self.process: Optional[multiprocessing.Process] = None
        self.file_to_serve = file_to_serve.absolute()
        self.port = port if port else self._find_available_port()

    def _find_available_port(self) -> int:
        """Finds an available port"""
        return utils.find_available_port()

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

    def __enter__(self):
        self.start_in_new_process()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
