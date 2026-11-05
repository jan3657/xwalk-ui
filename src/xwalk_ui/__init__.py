"""A local web app for exploring xwalk: `xwalk-ui`. See `xwalk_ui.server`."""

from xwalk_ui.server import DEFAULT_HOST, DEFAULT_PORT, make_server, serve

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "make_server", "serve"]
