"""
User Service for InsureSense Backend
"""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime

logger = logging.getLogger(__name__)

from app.models.user import User, UserRole
from app.schemas.auth import UserCreate, UserUpdate
from app.core.security import get_password_hash, verify_password


class UserService:
    """Service for user management"""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None
    ) -> List[User]:
        query = self.db.query(User)
        if role:
            query = query.filter(User.role == role)
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        return query.offset(skip).limit(limit).all()

    def create(self, user_data: UserCreate) -> User:
        logger.info(f"UserService.create called for email: {user_data.email}")
        existing_user = self.get_by_email(user_data.email)
        if existing_user:
            logger.warning(f"Registration rejected: Email '{user_data.email}' already exists with ID {existing_user.id}")
            raise ValueError(f"Email '{user_data.email}' is already registered")

        try:
            hashed_password = get_password_hash(user_data.password)
            role_val = UserRole(user_data.role) if isinstance(user_data.role, str) else user_data.role
            user = User(
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                role=role_val
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            logger.info(f"User created successfully: ID {user.id}, email {user.email}")
            return user
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user in DB: {e}", exc_info=True)
            raise

    def update(self, user_id: int, user_data: UserUpdate) -> Optional[User]:
        user = self.get_by_id(user_id)
        if not user:
            return None

        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        user.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user_id: int) -> bool:
        user = self.get_by_id(user_id)
        if not user:
            return False
        self.db.delete(user)
        self.db.commit()
        return True

    def authenticate(self, email: str, password: str) -> Optional[User]:
        user = self.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def update_last_login(self, user_id: int) -> None:
        user = self.get_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.db.commit()

    def get_stats(self) -> dict:
        total = self.db.query(func.count(User.id)).scalar()
        active = self.db.query(func.count(User.id)).filter(User.is_active == True).scalar()
        admins = self.db.query(func.count(User.id)).filter(User.role == UserRole.ADMIN).scalar()
        employees = self.db.query(func.count(User.id)).filter(User.role == UserRole.EMPLOYEE).scalar()
        return {
            "total": total,
            "active": active,
            "admins": admins,
            "employees": employees
        }