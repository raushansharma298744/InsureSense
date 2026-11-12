"""
Prediction and batch job models for InsureSense Backend
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Float, Text,
    ForeignKey, Enum as SQLEnum, Index, JSON
)
from sqlalchemy.orm import relationship

from app.models.base import Base
import enum


class PropensityCategory(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Input features (stored for audit/history)
    gender = Column(String(20), nullable=False)
    age = Column(Integer, nullable=False)
    driving_license = Column(Integer, nullable=False)
    region_code = Column(Float, nullable=False)
    previously_insured = Column(Integer, nullable=False)
    vehicle_age = Column(String(20), nullable=False)
    vehicle_damage = Column(String(10), nullable=False)
    annual_premium = Column(Float, nullable=False)
    policy_sales_channel = Column(Float, nullable=False)
    vintage = Column(Integer, nullable=False)

    # Prediction results
    predicted_response = Column(Integer, nullable=False)
    purchase_probability = Column(Float, nullable=False)
    propensity_category = Column(SQLEnum(PropensityCategory), nullable=False)
    threshold_used = Column(Float, default=0.5, nullable=False)

    # Metadata
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="predictions")

    # Indexes
    __table_args__ = (
        Index("ix_predictions_user_created", "user_id", "created_at"),
        Index("ix_predictions_category", "propensity_category"),
    )

    def __repr__(self):
        return f"<Prediction(id={self.id}, prob={self.purchase_probability:.4f}, cat={self.propensity_category.value})>"


class BatchJobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # File info
    filename = Column(String(255), nullable=False)
    total_rows = Column(Integer, default=0)
    processed_rows = Column(Integer, default=0)
    failed_rows = Column(Integer, default=0)

    # Status
    status = Column(SQLEnum(BatchJobStatus), default=BatchJobStatus.PENDING, nullable=False)
    error_message = Column(Text, nullable=True)

    # Results file path
    result_file_path = Column(String(500), nullable=True)

    # Metadata
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="batch_jobs")
    predictions = relationship("BatchPrediction", back_populates="batch_job", lazy="dynamic")

    # Indexes
    __table_args__ = (
        Index("ix_batch_jobs_user_created", "user_id", "created_at"),
        Index("ix_batch_jobs_status", "status"),
    )

    def __repr__(self):
        return f"<BatchJob(id={self.id}, status={self.status.value}, rows={self.total_rows})>"


class BatchPrediction(Base):
    __tablename__ = "batch_predictions"

    id = Column(Integer, primary_key=True, index=True)
    batch_job_id = Column(Integer, ForeignKey("batch_jobs.id"), nullable=False)

    # Row identifier (from input CSV)
    row_id = Column(Integer, nullable=True)
    customer_id = Column(String(100), nullable=True)

    # Input features (JSON for flexibility)
    input_features = Column(JSON, nullable=True)

    # Prediction results
    predicted_response = Column(Integer, nullable=True)
    purchase_probability = Column(Float, nullable=True)
    propensity_category = Column(SQLEnum(PropensityCategory), nullable=True)

    # Error info (if processing failed for this row)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    batch_job = relationship("BatchJob", back_populates="predictions")

    # Indexes
    __table_args__ = (
        Index("ix_batch_predictions_job", "batch_job_id"),
    )

    def __repr__(self):
        return f"<BatchPrediction(id={self.id}, job_id={self.batch_job_id}, prob={self.purchase_probability})>"


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(100), nullable=False)

    # Training info
    training_date = Column(DateTime, nullable=False)
    training_dataset_size = Column(Integer, nullable=True)

    # Metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    pr_auc = Column(Float, nullable=True)

    # Status
    is_active = Column(Boolean, default=False, nullable=False)
    status = Column(String(20), default="staged", nullable=False)  # staged, active, archived

    # Artifacts
    model_path = Column(String(500), nullable=True)
    pipeline_path = Column(String(500), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    activated_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_model_versions_active", "is_active"),
    )

    def __repr__(self):
        return f"<ModelVersion(version={self.version}, active={self.is_active})>"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Action info
    action = Column(String(100), nullable=False)  # login, predict, batch_upload, etc.
    resource_type = Column(String(50), nullable=True)  # prediction, batch_job, model
    resource_id = Column(String(100), nullable=True)

    # Request info
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

    # Details
    details = Column(JSON, nullable=True)
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index("ix_audit_logs_user_created", "user_id", "created_at"),
        Index("ix_audit_logs_action", "action"),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, user={self.user_id})>"