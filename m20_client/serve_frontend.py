#!/usr/bin/env python3
"""Serve the test dashboard locally; does not connect to or start a robot."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="M20 浏览器验证台（在控制电脑上启动）")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    handler = partial(SimpleHTTPRequestHandler, directory=str(Path(__file__).parent / "web"))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"打开 Chrome / Edge: http://localhost:{args.port} （不会自动启动巡检）", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
