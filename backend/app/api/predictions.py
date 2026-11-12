"""
Prediction API routes for InsureSense Backend
"""

import os
import tempfile
from typing import Optional, List
from datetime import datetime
import pandas as pd
from fastapi import (
    APIRouter, Depends, HTTPException, status, UploadFile, File,
    Form, Query, Request, BackgroundTasks
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_active_user, require_employee_or_admin
from app.db.session import get_db
from app.services.prediction_service import PredictionService
from app.services.audit_service import AuditService
from app.services.ml_inference import get_ml_service
from app.schemas.prediction import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    PredictionHistoryItem, PredictionHistoryResponse, PredictionFilter,
    BatchJobStatus, BatchJobCreateResponse, BatchJobResponse, BatchJobListResponse,
    ModelVersionResponse, DashboardSummaryResponse,
    PropensityCategory
)
from app.models.prediction import BatchJob, BatchPrediction

router = APIRouter(prefix=f"{settings.api_prefix}/predictions", tags=["Predictions"])


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def create_prediction(
    request: PredictionRequest,
    req: Request,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Create a single customer prediction"""
    pred_service = PredictionService(db)
    audit_service = AuditService(db)

    try:
        prediction = pred_service.create_prediction(current_user.id, request, threshold)
        audit_service.log_action(
            action="predict",
            user_id=current_user.id,
            resource_type="prediction",
            resource_id=str(prediction.id),
            ip_address=req.client.host if req.client else None,
            user_agent=req.headers.get("user-agent"),
            details={
                "probability": prediction.purchase_probability,
                "category": prediction.propensity_category.value
            },
            success=True
        )
        return PredictionResponse(
            prediction=prediction.predicted_response,
            probability=prediction.purchase_probability,
            propensity_category=prediction.propensity_category,
            threshold_used=prediction.threshold_used,
            model_version=prediction.model_version
        )
    except Exception as e:
        audit_service.log_action(
            action="predict",
            user_id=current_user.id,
            ip_address=req.client.host if req.client else None,
            user_agent=req.headers.get("user-agent"),
            success=False,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch", response_model=BatchPredictionResponse)
def create_batch_predictions(
    request: BatchPredictionRequest,
    req: Request,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Create multiple predictions in one request"""
    pred_service = PredictionService(db)
    audit_service = AuditService(db)

    results = []
    successful = 0
    failed = 0

    for i, cust_request in enumerate(request.customers):
        try:
            prediction = pred_service.create_prediction(current_user.id, cust_request, threshold)
            results.append(PredictionResponse(
                prediction=prediction.predicted_response,
                probability=prediction.purchase_probability,
                propensity_category=prediction.propensity_category,
                threshold_used=prediction.threshold_used,
                model_version=prediction.model_version
            ))
            successful += 1
        except Exception as e:
            results.append(PredictionResponse(
                prediction=0,
                probability=0.0,
                propensity_category=PropensityCategory.LOW,
                threshold_used=threshold
            ))
            failed += 1

    audit_service.log_action(
        action="batch_predict",
        user_id=current_user.id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("user-agent"),
        details={"total": len(request.customers), "successful": successful, "failed": failed},
        success=True
    )

    return BatchPredictionResponse(
        predictions=results,
        total=len(request.customers),
        successful=successful,
        failed=failed
    )


def normalize_column_name(col: str) -> str:
    cleaned = str(col).strip().lower().replace(" ", "_").replace("-", "_")
    alias_map = {
        "drivinglicense": "driving_license",
        "regioncode": "region_code",
        "previouslyinsured": "previously_insured",
        "vehicleage": "vehicle_age",
        "vehicledamage": "vehicle_damage",
        "annualpremium": "annual_premium",
        "premium": "annual_premium",
        "policysaleschannel": "policy_sales_channel",
        "sales_channel": "policy_sales_channel",
        "saleschannel": "policy_sales_channel",
        "channel": "policy_sales_channel",
    }
    return alias_map.get(cleaned, cleaned)


@router.post("/batch/upload", response_model=BatchJobCreateResponse)
async def upload_batch_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Upload CSV file for batch prediction"""
    pred_service = PredictionService(db)
    audit_service = AuditService(db)

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Validate file size
    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File size exceeds {settings.max_file_size_mb}MB limit")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    try:
        # Read CSV to count rows and validate columns
        df = pd.read_csv(temp_path)
        # Normalize column names for robust matching
        col_map = {c: normalize_column_name(c) for c in df.columns}
        df_norm = df.rename(columns=col_map)
        required_cols = [
            'gender', 'age', 'driving_license', 'region_code',
            'previously_insured', 'vehicle_age', 'vehicle_damage',
            'annual_premium', 'policy_sales_channel', 'vintage'
        ]

        missing_cols = set(required_cols) - set(df_norm.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

        total_rows = len(df_norm)
        # Save normalized CSV back to temp_path so background task receives normalized columns
        df_norm.to_csv(temp_path, index=False)

        # Create batch job record
        batch_job = pred_service.create_batch_job(current_user.id, file.filename, total_rows)

        # Start background processing
        background_tasks.add_task(
            process_batch_job,
            job_id=batch_job.id,
            file_path=temp_path,
            threshold=threshold,
            user_id=current_user.id
        )

        audit_service.log_action(
            action="batch_upload",
            user_id=current_user.id,
            resource_type="batch_job",
            resource_id=str(batch_job.id),
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"filename": file.filename, "rows": total_rows},
            success=True
        )

        return BatchJobCreateResponse(
            job_id=batch_job.id,
            status=batch_job.status,
            message=f"Batch job created with {total_rows} rows. Processing started."
        )

    except HTTPException:
        os.unlink(temp_path)
        raise
    except Exception as e:
        os.unlink(temp_path)
        audit_service.log_action(
            action="batch_upload",
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            success=False,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def process_batch_job(job_id: int, file_path: str, threshold: float, user_id: int):
    """Background task to process batch predictions"""
    from app.db.session import get_db_context
    import pandas as pd

    required_cols = [
        'gender', 'age', 'driving_license', 'region_code',
        'previously_insured', 'vehicle_age', 'vehicle_damage',
        'annual_premium', 'policy_sales_channel', 'vintage'
    ]

    with get_db_context() as db:
        pred_service = PredictionService(db)
        ml_service = get_ml_service()

        try:
            pred_service.update_batch_job_status(job_id, BatchJobStatus.PROCESSING)

            df = pd.read_csv(file_path)
            col_map = {c: normalize_column_name(c) for c in df.columns}
            df = df.rename(columns=col_map)
            total_rows = len(df)

            results = []
            successful = 0
            failed = 0

            for idx, row in df.iterrows():
                try:
                    input_dict = row[required_cols].to_dict()
                    result = ml_service.predict(input_dict, threshold)

                    results.append({
                        'row_id': idx,
                        'customer_id': row.get('id', str(idx)),
                        'input_features': input_dict,
                        'predicted_response': result['prediction'],
                        'purchase_probability': result['probability'],
                        'propensity_category': result['propensity_category']
                    })
                    successful += 1
                except Exception as e:
                    results.append({
                        'row_id': idx,
                        'customer_id': row.get('id', str(idx)),
                        'input_features': row[required_cols].to_dict() if idx < len(df) else {},
                        'error_message': str(e)
                    })
                    failed += 1

                # Update progress every 100 rows
                if (idx + 1) % 100 == 0:
                    pred_service.update_batch_job_status(job_id, BatchJobStatus.PROCESSING, processed_rows=idx + 1)

            # Save results
            pred_service.save_batch_predictions(job_id, results)

            # Create result CSV
            result_df = df.copy()
            result_df['predicted_response'] = [r.get('predicted_response') for r in results]
            result_df['purchase_probability'] = [r.get('purchase_probability') for r in results]
            result_df['propensity_category'] = [r.get('propensity_category') for r in results]

            result_filename = f"batch_result_{job_id}.csv"
            result_path = os.path.join(tempfile.gettempdir(), result_filename)
            result_df.to_csv(result_path, index=False)

            pred_service.update_batch_job_status(
                job_id,
                BatchJobStatus.COMPLETED if failed == 0 else BatchJobStatus.COMPLETED,
                processed_rows=successful,
                failed_rows=failed,
                result_file_path=result_path
            )

        except Exception as e:
            pred_service.update_batch_job_status(
                job_id,
                BatchJobStatus.FAILED,
                error_message=str(e)
            )
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)


@router.get("/batch/jobs", response_model=BatchJobListResponse)
def list_batch_jobs(
    status: Optional[BatchJobStatus] = None,
    skip: int = 0,
    limit: int = 50,
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """List batch jobs with pagination"""
    pred_service = PredictionService(db)

    # Non-admin users only see their own jobs
    user_id = None if current_user.role == "admin" else current_user.id

    jobs = pred_service.get_batch_jobs(user_id=user_id, status=status, skip=skip, limit=limit)
    total = pred_service.count_batch_jobs(user_id=user_id, status=status)

    return BatchJobListResponse(
        items=[BatchJobResponse.model_validate(job) for job in jobs],
        total=total,
        page=skip // limit + 1,
        page_size=limit,
        total_pages=(total + limit - 1) // limit
    )


@router.get("/batch/jobs/{job_id}", response_model=BatchJobResponse)
def get_batch_job(
    job_id: int,
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Get batch job details"""
    pred_service = PredictionService(db)
    user_id = None if current_user.role == "admin" else current_user.id

    job = pred_service.get_batch_job(job_id, user_id=user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")

    return BatchJobResponse.model_validate(job)


@router.get("/batch/jobs/{job_id}/download")
def download_batch_results(
    job_id: int,
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Download batch prediction results as CSV"""
    pred_service = PredictionService(db)
    user_id = None if current_user.role == "admin" else current_user.id

    job = pred_service.get_batch_job(job_id, user_id=user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found")

    if job.status != BatchJobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Batch job not completed")

    if not job.result_file_path or not os.path.exists(job.result_file_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        job.result_file_path,
        media_type='text/csv',
        filename=f"batch_predictions_{job_id}.csv"
    )


@router.get("/history", response_model=PredictionHistoryResponse)
def get_prediction_history(
    propensity_category: Optional[PropensityCategory] = None,
    predicted_response: Optional[int] = Query(None, ge=0, le=1),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_probability: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_probability: Optional[float] = Query(None, ge=0.0, le=1.0),
    skip: int = 0,
    limit: int = 50,
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Get prediction history with filters and pagination"""
    pred_service = PredictionService(db)

    # Parse dates
    from datetime import datetime
    start_dt = datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.fromisoformat(end_date) if end_date else None

    filters = PredictionFilter(
        propensity_category=propensity_category,
        predicted_response=predicted_response,
        start_date=start_dt,
        end_date=end_dt,
        min_probability=min_probability,
        max_probability=max_probability
    )

    user_id = None if current_user.role == "admin" else current_user.id
    predictions = pred_service.get_predictions(user_id=user_id, filters=filters, skip=skip, limit=limit)
    total = pred_service.count_predictions(user_id=user_id, filters=filters)

    return PredictionHistoryResponse(
        items=[PredictionHistoryItem.model_validate(p) for p in predictions],
        total=total,
        page=skip // limit + 1,
        page_size=limit,
        total_pages=(total + limit - 1) // limit
    )


@router.get("/history/{prediction_id}", response_model=PredictionHistoryItem)
def get_prediction_detail(
    prediction_id: int,
    current_user = Depends(require_employee_or_admin),
    db: Session = Depends(get_db)
):
    """Get single prediction detail"""
    pred_service = PredictionService(db)
    user_id = None if current_user.role == "admin" else current_user.id

    prediction = pred_service.get_prediction(prediction_id, user_id=user_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return PredictionHistoryItem.model_validate(prediction)