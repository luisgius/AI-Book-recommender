"""
Main application entry point.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.v1.search_endpoints import router as search_router
from app.api.v1.evaluation_endpoints import router as evaluation_router
from app.api.v1.agent_endpoints import router as agent_router

app = FastAPI(
    title="AI Book Recommender API",
    description="A hybrid search and recommendation API for books.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include API routers
app.include_router(search_router, prefix="/api/v1", tags=["search"])
app.include_router(evaluation_router, prefix="/api/v1", tags=["evaluation", "recommendation"])
app.include_router(agent_router, prefix="/api/v1", tags=["agent"])

# Minimal web UI for manual testing and demos (static files, no build step)
UI_DIR = Path(__file__).parent / "ui" / "static"
app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Welcome to the AI Book Recommender API",
        "docs": "/docs",
        "ui": "/ui/",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
