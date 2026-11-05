"""Standalone workbench command."""

import argparse
import sys

EXIT_OK = 0
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    ui = argparse.ArgumentParser(prog="xwalk-ui")
    ui.add_argument(
        "--root", default=".", help="the workspace directory the app may read and write"
    )
    ui.add_argument("--host", default="127.0.0.1", help="address to listen on")
    ui.add_argument("--port", type=int, default=8765, help="port to listen on (0: any free)")
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ui.add_argument(
        "--max-calls-cap",
        type=int,
        default=None,
        help="the largest max_calls a run with the job's endpoint may request (default 500)",
    )
    ui.add_argument(
        "--offline-only",
        action="store_true",
        help="refuse runs against the job's endpoint; only the offline stand-in model",
    )
    ui.add_argument(
        "--max-upload-mb", type=int, default=512, help="the largest file the page may upload"
    )
    ui.add_argument(
        "--public",
        action="store_true",
        help="run as a website: a private workspace per visitor, visitors' own model keys, "
        "no server-side URL downloads (see docs/guide/hosting.md)",
    )
    ui.add_argument(
        "--library",
        action="append",
        default=[],
        help="a directory of pre-parsed ontologies to offer read-only (repeatable; made by "
        "scripts/build_ontology_library.py)",
    )
    ui.add_argument("--lookup-cache", default=None, help="public mode: shared search-index cache")
    ui.add_argument(
        "--session-hours", type=float, default=24.0, help="public mode: delete idle sessions after"
    )
    ui.add_argument(
        "--max-tasks", type=int, default=4, help="public mode: runs at once across visitors"
    )
    ui.add_argument(
        "--frame-ancestors", default=None, help="origins allowed to embed the page in a frame"
    )
    ui.add_argument(
        "--trust-proxy",
        action="store_true",
        help="public mode: trust X-Forwarded-Proto/-For from the HTTPS proxy in front "
        "(only when every request reaches the server through it)",
    )
    ui.add_argument(
        "--sessions-per-ip",
        type=int,
        default=20,
        help="public mode: new visitor sessions one address may open per 10 minutes (0: no limit)",
    )
    ui.add_argument(
        "--allow-local-downloads",
        action="store_true",
        help="let library downloads use http and private addresses (an intranet mirror); "
        "not with --public",
    )
    ui.add_argument("--verbose", action="store_true", help="log every request to stderr")
    ui.add_argument(
        "--chatgpt-login",
        action="store_true",
        help="enable local Sign in with ChatGPT (requires xwalk[chatgpt] and --host 127.0.0.1)",
    )
    ui.add_argument(
        "--dense",
        nargs="?",
        const="local",
        default="off",
        choices=("off", "local", "api"),
        help="dense retrieval alongside BM25: 'local' (the default when given alone) adds "
        "general and biomedical sentence-transformers models (requires xwalk[dense]); "
        "'api' adds Qwen3-Embedding 8B through OpenRouter (needs OPENROUTER_API_KEY)",
    )
    ui.add_argument(
        "--dense-device",
        default=None,
        help="local dense encoder device, e.g. cpu or mps (default: automatic)",
    )

    return ui


def main(argv: list[str] | None = None) -> int:
    return _serve_ui(_build_parser().parse_args(argv))


def _serve_ui(args: argparse.Namespace) -> int:
    """Serve the web app until Ctrl-C. Status lines go to stderr."""
    try:
        from xwalk_ui import serve
        from xwalk_ui.api import DEFAULT_MAX_CALLS_CAP

        serve(
            args.root,
            host=args.host,
            port=args.port,
            max_calls_cap=DEFAULT_MAX_CALLS_CAP
            if args.max_calls_cap is None
            else args.max_calls_cap,
            offline_only=args.offline_only,
            max_upload_mb=args.max_upload_mb,
            open_browser=not args.no_browser,
            verbose=args.verbose,
            public=args.public,
            library=args.library,
            lookup_cache=args.lookup_cache,
            session_ttl=args.session_hours * 3600,
            max_tasks=args.max_tasks,
            frame_ancestors=args.frame_ancestors,
            chatgpt_login=args.chatgpt_login,
            dense=args.dense,
            dense_device=args.dense_device,
            trust_proxy=args.trust_proxy,
            sessions_per_ip=args.sessions_per_ip,
            allow_local_downloads=args.allow_local_downloads,
        )
    except KeyboardInterrupt:
        return EXIT_OK
    except (OSError, ValueError) as exc:
        print(f"error: ui: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
