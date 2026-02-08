"""Main FastAPI application for Shubble Admin."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqladmin import Admin
from backend.config import Settings
from backend.database import create_async_db_engine
from backend.admin.admin_models import (
    VehicleAdmin,
    GeofenceEventAdmin,
    VehicleLocationAdmin,
    DriverAdmin,
    DriverVehicleAssignmentAdmin,
    ETAAdmin,
    PredictedLocationAdmin,
)

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    app.state.engine = create_async_db_engine(settings.DATABASE_URL, echo=settings.DEBUG)
    
    admin = Admin(
        app,
        app.state.engine,
        title="Shubble Admin",
        base_url="/admin"
    )
    
    # Add model views
    admin.add_view(VehicleAdmin)
    admin.add_view(GeofenceEventAdmin)
    admin.add_view(VehicleLocationAdmin)
    admin.add_view(DriverAdmin)
    admin.add_view(DriverVehicleAssignmentAdmin)
    admin.add_view(ETAAdmin)
    admin.add_view(PredictedLocationAdmin)
    
    yield
    
    await app.state.engine.dispose()


app = FastAPI(
    title="Shubble Admin API",
    description="Admin interface for Shubble shuttle tracking system",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Shubble Admin API", "admin_panel": "/admin"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
