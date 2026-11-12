"""
Prediction schemas for InsureSense Backend
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class PropensityCategory(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VehicleAge(str, Enum):
    LESS_THAN_1_YEAR = "< 1 Year"
    ONE_TO_2_YEAR = "1-2 Year"
    MORE_THAN_2_YEARS = "> 2 Years"


class VehicleDamage(str, Enum):
    NO = "No"
    YES = "Yes"


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class PredictionRequest(BaseModel):
    """Single customer prediction request"""
    gender: Gender
    age: int = Field(..., ge=18, le=100)
    driving_license: int = Field(..., ge=0, le=1)
    region_code: float = Field(..., ge=0, le=100)
    previously_insured: int = Field(..., ge=0, le=1)
    vehicle_age: VehicleAge
    vehicle_damage: VehicleDamage
    annual_premium: float = Field(..., ge=0, le=1000000)
    policy_sales_channel: float = Field(..., ge=0, le=200)
    vintage: int = Field(..., ge=0, le=1000)

    @field_validator('driving_license', 'previously_insured')
    @classmethod
    def validate_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError('Must be 0 or 1')
        return v


class PredictionResponse(BaseModel):
    """Single prediction response"""
    model_config = {"protected_namespaces": ()}
    prediction: int
    probability: float
    propensity_category: PropensityCategory
    threshold_used: float
    model_version: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (for direct API calls with JSON)"""
    customers: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total: int
    successful: int
    failed: int


class PredictionHistoryItem(BaseModel):
    """Prediction history item for listing"""
    model_config = {"protected_namespaces": (), "from_attributes": True}
    id: int
    gender: str
    age: int
    vehicle_age: str
    vehicle_damage: str
    annual_premium: float
    predicted_response: int
    purchase_probability: float
    propensity_category: PropensityCategory
    model_version: Optional[str] = None
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    """Paginated prediction history response"""
    items: List[PredictionHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class PredictionFilter(BaseModel):
    """Filters for prediction history"""
    propensity_category: Optional[PropensityCategory] = None
    predicted_response: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_probability: Optional[float] = None
    max_probability: Optional[float] = None


class BatchJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchJobCreateResponse(BaseModel):
    """Response when creating a batch job"""
    job_id: int
    status: BatchJobStatus
    message: str


class BatchJobResponse(BaseModel):
    """Batch job response"""
    model_config = {"protected_namespaces": (), "from_attributes": True}
    id: int
    filename: str
    total_rows: int
    processed_rows: int
    failed_rows: int
    status: BatchJobStatus
    error_message: Optional[str] = None
    result_file_path: Optional[str] = None
    model_version: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BatchJobListResponse(BaseModel):
    """Paginated batch job list response"""
    items: List[BatchJobResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ModelVersionResponse(BaseModel):
    """Model version response"""
    model_config = {"protected_namespaces": (), "from_attributes": True}
    id: int
    version: str
    model_name: str
    model_type: str
    training_date: datetime
    training_dataset_size: Optional[int] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    is_active: bool
    status: str
    created_at: datetime
    activated_at: Optional[datetime] = None
    notes: Optional[str] = None


class DashboardSummaryResponse(BaseModel):
    """Dashboard analytics summary"""
    model_config = {"protected_namespaces": ()}
    total_customers_analyzed: int
    total_predictions: int
    total_batch_jobs: int
    predicted_purchase_distribution: Dict[str, int]
    propensity_distribution: Dict[str, int]
    recent_predictions: int
    recent_batch_jobs: int
    model_info: Optional[ModelVersionResponse] = None