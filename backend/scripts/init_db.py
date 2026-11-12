"""
Database initialization script for InsureSense
Creates tables via Alembic migration and seeds initial data
"""
import subprocess
import sys
import os

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.db.session import engine, SessionLocal
from app.models.base import Base
from app.models.user import User, UserRole
from app.models.prediction import ModelVersion
from app.core.security import get_password_hash
from app.core.config import settings


def run_migrations():
    """Run Alembic migrations"""
    print("Running database migrations...")
    try:
        result = subprocess.run(
            ['alembic', 'upgrade', 'head'],
            cwd=backend_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Migration failed: {result.stderr}")
            return False
        print("Migrations completed successfully")
        return True
    except FileNotFoundError:
        print("Alembic not found. Creating tables directly via SQLAlchemy...")
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully")
        return True


def seed_data():
    """Seed initial data"""
    print("Seeding initial data...")
    db = SessionLocal()
    try:
        from datetime import datetime
        # Check if admin user exists
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
            print("Created admin user: admin@insuresense.com / admin123")
        else:
            admin.hashed_password = get_password_hash("admin123")
            admin.is_active = True
            print("Updated existing admin user: admin@insuresense.com / admin123")

        # Check if demo employee exists
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
            print("Created employee user: employee@insuresense.com / employee123")
        else:
            employee.hashed_password = get_password_hash("employee123")
            employee.is_active = True
            print("Updated existing employee user: employee@insuresense.com / employee123")

        # Seed model version
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
            print("Created model version 1.0.0")
        else:
            print("Model version 1.0.0 already exists")

        db.commit()
        print("Initial data seeded successfully")
        return True
    except Exception as e:
        db.rollback()
        print(f"Seeding failed: {e}")
        return False
    finally:
        db.close()


def main():
    print("=" * 60)
    print("InsureSense Database Initialization")
    print("=" * 60)

    if not run_migrations():
        sys.exit(1)

    if not seed_data():
        sys.exit(1)

    print("=" * 60)
    print("Database initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()