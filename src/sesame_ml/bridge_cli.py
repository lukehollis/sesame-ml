"""Lightweight vendor-environment bridge for OpenPI, GR00T, and LeRobot export.

Unlike :mod:`sesame_ml.cli`, this module deliberately imports no Gymnasium or MuJoCo
code. It can therefore be exposed through ``PYTHONPATH=.../sesame-ml/src`` inside the
exact vendor environment without installing Sesame ML's simulator dependency set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


async def _serve(args: argparse.Namespace) -> None:
    from sesame_ml.integrations import GrootRemotePolicy, OpenPIRemotePolicy, RemotePolicyBridge
    from sesame_ml.transport import PolicyWebSocketServer

    if args.policy == "openpi":
        remote = OpenPIRemotePolicy(
            host=args.backend_host,
            port=args.backend_port or 8000,
            api_key=args.api_token,
            max_chunk_steps=args.max_chunk_steps,
            connect_timeout_s=args.backend_timeout_ms / 1000.0,
        )
    else:
        remote = GrootRemotePolicy(
            host=args.backend_host,
            port=args.backend_port or 5555,
            timeout_ms=args.backend_timeout_ms,
            api_token=args.api_token,
            max_chunk_steps=args.max_chunk_steps,
        )
    bridge = RemotePolicyBridge(
        remote,
        valid_for_s=args.valid_for,
        default_instruction=args.default_instruction,
    )
    server = PolicyWebSocketServer(bridge, host=args.host, port=args.port)
    print(
        f"Sesame {args.policy} bridge listening on ws://{args.host}:{args.port}",
        flush=True,
    )
    try:
        await server.serve_forever()
    finally:
        bridge.close()


def command_serve(args: argparse.Namespace) -> int:
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass
    return 0


def command_export(args: argparse.Namespace) -> int:
    from sesame_ml.data import export_to_lerobot

    result = export_to_lerobot(
        args.episodes,
        repo_id=args.repo_id,
        output_dir=args.output,
        overwrite=args.overwrite,
        require_groot_v2=args.groot_v2,
        add_groot_metadata=args.groot_v2,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sesame-ml-bridge",
        description=(
            "MuJoCo-free bridge and dataset exporter for exact OpenPI/GR00T environments"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="bridge a vendor policy server to Sesame")
    serve.add_argument("--policy", choices=("openpi", "groot"), required=True)
    serve.add_argument("--backend-host", default="127.0.0.1")
    serve.add_argument("--backend-port", type=int)
    serve.add_argument("--backend-timeout-ms", type=int, default=15_000)
    serve.add_argument("--api-token")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--valid-for", type=float, default=1.0)
    serve.add_argument("--max-chunk-steps", type=int)
    serve.add_argument("--default-instruction")
    serve.set_defaults(handler=command_serve)

    export = subparsers.add_parser(
        "export-dataset", help="export canonical episodes with the vendor's LeRobot"
    )
    export.add_argument("episodes", nargs="+")
    export.add_argument("--repo-id", required=True)
    export.add_argument("--output")
    export.add_argument("--groot-v2", action="store_true")
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(handler=command_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        if not 0 <= args.port <= 65535:
            parser.error("--port must be between 0 and 65535")
        if args.backend_port is not None and not 1 <= args.backend_port <= 65535:
            parser.error("--backend-port must be between 1 and 65535")
        if args.backend_timeout_ms < 1:
            parser.error("--backend-timeout-ms must be positive")
        if args.max_chunk_steps is not None and args.max_chunk_steps < 1:
            parser.error("--max-chunk-steps must be positive")
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
