"""
API routes package for InsureSense Backend
"""

from app.api.auth import router as auth_router
from app.api.predictions import router as predictions_router
from app.api.models import router as models_router

__all__ = ["auth_router", "predictions_router", "models_router"]