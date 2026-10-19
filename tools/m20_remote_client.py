#!/usr/bin/env python3
"""Minimal client: control routes and show the combined front/back H.264 stream."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess

import websockets

from m20_h264_websocket import unpack_message


async def run(args) -> None:
    async with websockets.connect(f"ws://{args.host}:{args.port}/control", max_size=32 * 1024 * 1024) as control:
        async with websockets.connect(f"ws://{args.host}:{args.port}/all", max_size=32 * 1024 * 1024) as stream:
            players = {
                "front": subprocess.Popen([args.player, "-", "-autoexit", "-loglevel", "error"], stdin=subprocess.PIPE),
                "back": subprocess.Popen([args.player, "-", "-autoexit", "-loglevel", "error"], stdin=subprocess.PIPE),
            }
            await control.send(json.dumps({"action": "start", "route": args.route, "loops": args.loops}))

            async def statuses():
                async for message in control:
                    print(message, flush=True)

            async def video():
                async for message in stream:
                    if isinstance(message, str):
                        continue
                    decoded = unpack_message(message)
                    pipe = players[decoded.camera].stdin
                    if pipe:
                        pipe.write(decoded.access_unit)
                        pipe.flush()

            try:
                await asyncio.gather(statuses(), video())
            finally:
                for process in players.values():
                    if process.stdin:
                        process.stdin.close()
                    process.terminate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--route", choices=("route1", "route2", "route3", "route4"), default="route1")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--player", default="ffplay")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
