"""
Main application entry point.
"""

from fastapi import FastAPI
from app.api.v1.search_endpoints import router as search_router

app = FastAPI(
    title="AI Book Recommender API",
    description="A hybrid search and recommendation API for books.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include API routers
app.include_router(search_router, prefix="/api/v1", tags=["search"])

@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Welcome to the AI Book Recommender API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
