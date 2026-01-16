"""ASGI application entry point for FastAPI."""
from backend.flask import app

# Export for uvicorn: uvicorn shubble:app
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
