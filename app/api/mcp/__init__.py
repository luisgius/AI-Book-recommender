"""
MCP driving adapter: the book catalog as read-only tools for AI assistants.

Served at /mcp on the FastAPI app (see server.py); the tools' behaviour is in
tools.py and their output models in schemas.py.
"""

from app.api.mcp.server import MCP_PATH, MCPEndpoint, create_mcp_server, load_mcp_settings, mount_mcp_server

__all__ = ["MCP_PATH", "MCPEndpoint", "create_mcp_server", "load_mcp_settings", "mount_mcp_server"]
