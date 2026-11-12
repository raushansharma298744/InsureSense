"""
Core configuration for InsureSense Backend
"""

import os
from functools import lru_cache
from typing import List, Optional, Any, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # App
    app_name: str = "InsureSense"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = Field(
        default="postgresql://postgres:changeme@localhost:5432/insuresense",
        description="PostgreSQL connection string"
    )

    # JWT Authentication
    jwt_secret_key: str = Field(
        default="your-super-secret-jwt-key-change-in-production",
        description="Secret key for JWT token signing"
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "insuresense-training"

    # Model
    model_path: str = "ml/artifacts/best_model.pkl"
    pipeline_path: str = "ml/artifacts/preprocessing_pipeline.pkl"

    # File Upload
    max_file_size_mb: int = 10
    allowed_extensions: Union[str, List[str]] = ["csv"]

    @field_validator("allowed_extensions", mode="after")
    @classmethod
    def parse_allowed_extensions(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [ext.strip().lstrip(".") for ext in v.split(",") if ext.strip()]
        return v

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Frontend URL (for CORS)
    frontend_url: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


settings = get_settings()