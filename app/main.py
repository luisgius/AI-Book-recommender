"""
Main application entry point.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.mcp import MCP_PATH, load_mcp_settings, mount_mcp_server
from app.api.v1.search_endpoints import router as search_router
from app.api.v1.evaluation_endpoints import router as evaluation_router
from app.api.v1.agent_endpoints import router as agent_router
from app.api.v1.assistant_endpoints import router as assistant_router
from app.api.v1.usage_endpoints import router as usage_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run the MCP transport (when enabled) for as long as the API runs."""
    if mcp_endpoint is None:
        yield
        return
    async with mcp_endpoint.run():
        yield


app = FastAPI(
    title="AI Book Recommender API",
    description="A hybrid search and recommendation API for books.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Include API routers
app.include_router(search_router, prefix="/api/v1", tags=["search"])
app.include_router(evaluation_router, prefix="/api/v1", tags=["evaluation", "recommendation"])
app.include_router(agent_router, prefix="/api/v1", tags=["agent"])
app.include_router(assistant_router, prefix="/api/v1", tags=["assistant"])
app.include_router(usage_router, prefix="/api/v1", tags=["usage"])

# Minimal web UI for manual testing and demos (static files, no build step)
UI_DIR = Path(__file__).parent / "ui" / "static"
app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

# Read-only MCP server for AI assistants (search_books, get_book, similar_books)
# at /mcp; MCP_ENABLED=false leaves it out. None when disabled.
mcp_endpoint = mount_mcp_server(app, load_mcp_settings())

@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Welcome to the AI Book Recommender API",
        "docs": "/docs",
        "ui": "/ui/",
        "health": "/api/v1/health",
        "mcp": MCP_PATH if mcp_endpoint is not None else None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
