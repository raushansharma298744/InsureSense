"""
Model Management and Dashboard API routes for InsureSense Backend
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_active_user, require_employee_or_admin, require_admin
from app.db.session import get_db
from app.services.prediction_service import PredictionService
from app.schemas.prediction import (
    ModelVersionResponse, DashboardSummaryResponse,
    BatchJobStatus
)
from app.models.prediction import ModelVersion

router = APIRouter(prefix=f"{settings.api_prefix}", tags=["Models & Dashboard"])


# Dashboard routes
@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Get dashboard analytics summary"""
    pred_service = PredictionService(db)
    user_id = None if (current_user.role == "admin" or getattr(current_user.role, 'value', None) == "admin") else current_user.id
    return pred_service.get_dashboard_summary(user_id=user_id)


# Model Management routes
@router.get("/models", response_model=List[ModelVersionResponse])
def list_models(
    skip: int = 0,
    limit: int = 50,
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """List all model versions"""
    pred_service = PredictionService(db)
    models = pred_service.get_model_versions(skip=skip, limit=limit)
    return [ModelVersionResponse.model_validate(m) for m in models]


@router.get("/models/active", response_model=ModelVersionResponse)
def get_active_model(
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Get currently active model"""
    pred_service = PredictionService(db)
    model = pred_service.get_active_model()
    if not model:
        raise HTTPException(status_code=404, detail="No active model found")
    return ModelVersionResponse.model_validate(model)


@router.get("/models/{model_id}", response_model=ModelVersionResponse)
def get_model(
    model_id: int,
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Get model version by ID"""
    pred_service = PredictionService(db)
    model = pred_service.get_model_version(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelVersionResponse.model_validate(model)


@router.post("/models/{model_id}/activate", response_model=ModelVersionResponse)
def activate_model(
    model_id: int,
    current_user = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Activate a model version (admin only)"""
    pred_service = PredictionService(db)
    model = pred_service.activate_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelVersionResponse.model_validate(model)


# Health check
@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "InsureSense API",
        "version": "1.0.0"
    }