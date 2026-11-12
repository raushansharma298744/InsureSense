"""
Database session management for InsureSense Backend
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

from app.core.config import settings

# Create engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.debug
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for DB session outside of FastAPI"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables and seed default users if not present"""
    from app.models.base import Base
    from app.models.user import User, UserRole
    from app.models.prediction import ModelVersion
    from app.core.security import get_password_hash
    import logging

    log = logging.getLogger(__name__)
    Base.metadata.create_all(bind=engine)

    # Auto-seed initial users and model version
    db = SessionLocal()
    try:
        from datetime import datetime
        admin = db.query(User).filter(User.email == "admin@insuresense.com").first()
        if not admin:
            admin = User(
                email="admin@insuresense.com",
                hashed_password=get_password_hash("admin123"),
                full_name="System Administrator",
                role=UserRole.ADMIN,
                is_active=True,
                is_superuser=True
            )
            db.add(admin)
            log.info("Created admin user: admin@insuresense.com")
        else:
            # Ensure admin password hash is synced with current bcrypt
            admin.hashed_password = get_password_hash("admin123")
            admin.is_active = True
            log.info("Updated existing admin user: admin@insuresense.com")

        employee = db.query(User).filter(User.email == "employee@insuresense.com").first()
        if not employee:
            employee = User(
                email="employee@insuresense.com",
                hashed_password=get_password_hash("employee123"),
                full_name="Demo Employee",
                role=UserRole.EMPLOYEE,
                is_active=True
            )
            db.add(employee)
            log.info("Created employee user: employee@insuresense.com")
        else:
            employee.hashed_password = get_password_hash("employee123")
            employee.is_active = True

        db.commit()

        model_version = db.query(ModelVersion).filter(ModelVersion.version == "1.0.0").first()
        if not model_version:
            model_version = ModelVersion(
                version="1.0.0",
                model_name="InsureSense XGBoost",
                model_type="XGBClassifier",
                training_date=datetime(2026, 11, 7),
                training_dataset_size=50000,
                accuracy=0.7335,
                precision=0.2995,
                recall=0.8738,
                f1_score=0.4461,
                roc_auc=0.8520,
                pr_auc=0.3574,
                is_active=True,
                status="active",
                model_path="ml/artifacts/best_model.pkl",
                pipeline_path="ml/artifacts/preprocessing_pipeline.pkl",
                notes="Initial XGBoost model trained on Vehicle Insurance Cross-Sell dataset"
            )
            db.add(model_version)
            db.commit()
            log.info("Seeded model version 1.0.0")

    except Exception as e:
        db.rollback()
        log.warning(f"Failed to auto-seed initial data: {e}")
    finally:
        db.close()


def drop_db() -> None:
    """Drop all database tables (use with caution!)"""
    from app.models.base import Base
    Base.metadata.drop_all(bind=engine)