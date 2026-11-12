"""
Prediction Service for InsureSense Backend
Handles prediction CRUD, history, and analytics
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from datetime import datetime, timedelta
import pandas as pd

from app.models.prediction import (
    Prediction, BatchJob, BatchPrediction, ModelVersion,
    PropensityCategory, BatchJobStatus
)
from app.schemas.prediction import (
    PredictionRequest, PredictionFilter,
    BatchJobCreateResponse, BatchJobResponse
)
from app.services.ml_inference import get_ml_service


class PredictionService:
    """Service for prediction management"""

    def __init__(self, db: Session):
        self.db = db
        self.ml_service = get_ml_service()

    def create_prediction(
        self,
        user_id: int,
        request: PredictionRequest,
        threshold: float = 0.5
    ) -> Prediction:
        """Create and save a single prediction"""
        # Get ML prediction
        input_dict = request.model_dump()
        result = self.ml_service.predict(input_dict, threshold)

        # Get active model version
        active_model = self.get_active_model()
        model_version = active_model.version if active_model else "unknown"

        # Create prediction record
        prediction = Prediction(
            user_id=user_id,
            gender=request.gender.value,
            age=request.age,
            driving_license=request.driving_license,
            region_code=request.region_code,
            previously_insured=request.previously_insured,
            vehicle_age=request.vehicle_age.value,
            vehicle_damage=request.vehicle_damage.value,
            annual_premium=request.annual_premium,
            policy_sales_channel=request.policy_sales_channel,
            vintage=request.vintage,
            predicted_response=result['prediction'],
            purchase_probability=result['probability'],
            propensity_category=PropensityCategory(result['propensity_category']) if isinstance(result['propensity_category'], str) else result['propensity_category'],
            threshold_used=result['threshold_used'],
            model_version=model_version
        )

        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)

        return prediction

    def get_prediction(self, prediction_id: int, user_id: Optional[int] = None) -> Optional[Prediction]:
        """Get a single prediction by ID"""
        query = self.db.query(Prediction).filter(Prediction.id == prediction_id)
        if user_id:
            query = query.filter(Prediction.user_id == user_id)
        return query.first()

    def get_predictions(
        self,
        user_id: Optional[int] = None,
        filters: Optional[PredictionFilter] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[Prediction]:
        """Get predictions with filters and pagination"""
        query = self.db.query(Prediction)

        if user_id:
            query = query.filter(Prediction.user_id == user_id)

        if filters:
            if filters.propensity_category:
                query = query.filter(Prediction.propensity_category == filters.propensity_category)
            if filters.predicted_response is not None:
                query = query.filter(Prediction.predicted_response == filters.predicted_response)
            if filters.start_date:
                query = query.filter(Prediction.created_at >= filters.start_date)
            if filters.end_date:
                query = query.filter(Prediction.created_at <= filters.end_date)
            if filters.min_probability is not None:
                query = query.filter(Prediction.purchase_probability >= filters.min_probability)
            if filters.max_probability is not None:
                query = query.filter(Prediction.purchase_probability <= filters.max_probability)

        query = query.order_by(desc(Prediction.created_at))
        return query.offset(skip).limit(limit).all()

    def count_predictions(
        self,
        user_id: Optional[int] = None,
        filters: Optional[PredictionFilter] = None
    ) -> int:
        """Count predictions with filters"""
        query = self.db.query(func.count(Prediction.id))

        if user_id:
            query = query.filter(Prediction.user_id == user_id)

        if filters:
            if filters.propensity_category:
                query = query.filter(Prediction.propensity_category == filters.propensity_category)
            if filters.predicted_response is not None:
                query = query.filter(Prediction.predicted_response == filters.predicted_response)
            if filters.start_date:
                query = query.filter(Prediction.created_at >= filters.start_date)
            if filters.end_date:
                query = query.filter(Prediction.created_at <= filters.end_date)
            if filters.min_probability is not None:
                query = query.filter(Prediction.purchase_probability >= filters.min_probability)
            if filters.max_probability is not None:
                query = query.filter(Prediction.purchase_probability <= filters.max_probability)

        return query.scalar()

    def get_dashboard_summary(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get dashboard analytics summary"""
        query = self.db.query(Prediction)
        if user_id:
            query = query.filter(Prediction.user_id == user_id)

        total_predictions = query.count()

        # Predicted purchase distribution
        purchase_dist = query.with_entities(
            Prediction.predicted_response,
            func.count(Prediction.id)
        ).group_by(Prediction.predicted_response).all()
        purchase_distribution = {str(k): v for k, v in purchase_dist}

        # Propensity distribution
        propensity_dist = query.with_entities(
            Prediction.propensity_category,
            func.count(Prediction.id)
        ).group_by(Prediction.propensity_category).all()
        propensity_distribution = {(k.value if hasattr(k, 'value') else str(k)): v for k, v in propensity_dist if k is not None}

        # Recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_predictions = query.filter(Prediction.created_at >= week_ago).count()

        # Batch jobs
        batch_query = self.db.query(BatchJob)
        if user_id:
            batch_query = batch_query.filter(BatchJob.user_id == user_id)
        total_batch_jobs = batch_query.count()
        recent_batch_jobs = batch_query.filter(BatchJob.created_at >= week_ago).count()

        # Active model
        active_model = self.get_active_model()

        return {
            "total_customers_analyzed": total_predictions,
            "total_predictions": total_predictions,
            "total_batch_jobs": total_batch_jobs,
            "predicted_purchase_distribution": purchase_distribution,
            "propensity_distribution": propensity_distribution,
            "recent_predictions": recent_predictions,
            "recent_batch_jobs": recent_batch_jobs,
            "model_info": active_model
        }

    def get_active_model(self) -> Optional[ModelVersion]:
        """Get the currently active model version"""
        return self.db.query(ModelVersion).filter(ModelVersion.is_active == True).first()

    # Batch Job methods
    def create_batch_job(
        self,
        user_id: int,
        filename: str,
        total_rows: int
    ) -> BatchJob:
        """Create a new batch job record"""
        active_model = self.get_active_model()
        model_version = active_model.version if active_model else "unknown"

        batch_job = BatchJob(
            user_id=user_id,
            filename=filename,
            total_rows=total_rows,
            status=BatchJobStatus.PENDING,
            model_version=model_version
        )
        self.db.add(batch_job)
        self.db.commit()
        self.db.refresh(batch_job)
        return batch_job

    def update_batch_job_status(
        self,
        job_id: int,
        status: BatchJobStatus,
        processed_rows: Optional[int] = None,
        failed_rows: Optional[int] = None,
        error_message: Optional[str] = None,
        result_file_path: Optional[str] = None
    ) -> Optional[BatchJob]:
        """Update batch job status"""
        job = self.db.query(BatchJob).filter(BatchJob.id == job_id).first()
        if not job:
            return None

        job.status = status
        if processed_rows is not None:
            job.processed_rows = processed_rows
        if failed_rows is not None:
            job.failed_rows = failed_rows
        if error_message is not None:
            job.error_message = error_message
        if result_file_path is not None:
            job.result_file_path = result_file_path

        if status == BatchJobStatus.PROCESSING and job.started_at is None:
            job.started_at = datetime.utcnow()
        elif status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
            job.completed_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(job)
        return job

    def save_batch_predictions(
        self,
        job_id: int,
        predictions: List[Dict[str, Any]]
    ) -> List[BatchPrediction]:
        """Save batch prediction results"""
        batch_predictions = []
        for pred in predictions:
            bp = BatchPrediction(
                batch_job_id=job_id,
                row_id=pred.get('row_id'),
                customer_id=pred.get('customer_id'),
                input_features=pred.get('input_features'),
                predicted_response=pred.get('predicted_response'),
                purchase_probability=pred.get('purchase_probability'),
                propensity_category=PropensityCategory(pred.get('propensity_category')) if pred.get('propensity_category') else None,
                error_message=pred.get('error_message')
            )
            batch_predictions.append(bp)

        self.db.bulk_save_objects(batch_predictions)
        self.db.commit()
        return batch_predictions

    def get_batch_job(self, job_id: int, user_id: Optional[int] = None) -> Optional[BatchJob]:
        """Get batch job by ID"""
        query = self.db.query(BatchJob).filter(BatchJob.id == job_id)
        if user_id:
            query = query.filter(BatchJob.user_id == user_id)
        return query.first()

    def get_batch_jobs(
        self,
        user_id: Optional[int] = None,
        status: Optional[BatchJobStatus] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[BatchJob]:
        """Get batch jobs with filters"""
        query = self.db.query(BatchJob)
        if user_id:
            query = query.filter(BatchJob.user_id == user_id)
        if status:
            query = query.filter(BatchJob.status == status)
        query = query.order_by(desc(BatchJob.created_at))
        return query.offset(skip).limit(limit).all()

    def count_batch_jobs(
        self,
        user_id: Optional[int] = None,
        status: Optional[BatchJobStatus] = None
    ) -> int:
        """Count batch jobs"""
        query = self.db.query(func.count(BatchJob.id))
        if user_id:
            query = query.filter(BatchJob.user_id == user_id)
        if status:
            query = query.filter(BatchJob.status == status)
        return query.scalar()

    def get_batch_predictions(
        self,
        job_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[BatchPrediction]:
        """Get predictions for a batch job"""
        return self.db.query(BatchPrediction).filter(
            BatchPrediction.batch_job_id == job_id
        ).offset(skip).limit(limit).all()

    # Model Version methods
    def get_model_versions(self, skip: int = 0, limit: int = 50) -> List[ModelVersion]:
        return self.db.query(ModelVersion).order_by(
            desc(ModelVersion.created_at)
        ).offset(skip).limit(limit).all()

    def get_model_version(self, version_id: int) -> Optional[ModelVersion]:
        return self.db.query(ModelVersion).filter(ModelVersion.id == version_id).first()

    def activate_model(self, version_id: int) -> Optional[ModelVersion]:
        """Activate a model version (deactivates others)"""
        # Deactivate all
        self.db.query(ModelVersion).update({ModelVersion.is_active: False})

        # Activate selected
        model = self.db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if model:
            model.is_active = True
            model.status = "active"
            model.activated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(model)

            # Reload ML service with new model
            self.ml_service._load_artifacts()

        return model