"""
Schemas package for InsureSense Backend
"""

from app.schemas.auth import (
    UserBase, UserCreate, UserUpdate, UserResponse,
    TokenData, Token, LoginRequest, RefreshTokenRequest, MessageResponse
)
from app.schemas.prediction import (
    PropensityCategory, VehicleAge, VehicleDamage, Gender,
    PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse,
    PredictionHistoryItem, PredictionHistoryResponse, PredictionFilter,
    BatchJobStatus, BatchJobCreateResponse, BatchJobResponse, BatchJobListResponse,
    ModelVersionResponse, DashboardSummaryResponse
)

__all__ = [
    # Auth
    "UserBase", "UserCreate", "UserUpdate", "UserResponse",
    "TokenData", "Token", "LoginRequest", "RefreshTokenRequest", "MessageResponse",
    # Prediction
    "PropensityCategory", "VehicleAge", "VehicleDamage", "Gender",
    "PredictionRequest", "PredictionResponse", "BatchPredictionRequest", "BatchPredictionResponse",
    "PredictionHistoryItem", "PredictionHistoryResponse", "PredictionFilter",
    "BatchJobStatus", "BatchJobCreateResponse", "BatchJobResponse", "BatchJobListResponse",
    "ModelVersionResponse", "DashboardSummaryResponse",
]