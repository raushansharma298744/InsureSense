"""
Authentication API routes for InsureSense Backend
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import EmailStr

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    authenticate_user,
    get_current_user,
    get_current_active_user
)
from app.db.session import get_db
from app.models.user import User
from app.services.user_service import UserService
from app.services.audit_service import AuditService
from app.schemas.auth import (
    UserCreate, UserUpdate, UserResponse,
    Token, LoginRequest, RefreshTokenRequest, MessageResponse
)

router = APIRouter(prefix=f"{settings.api_prefix}/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    user_service = UserService(db)
    audit_service = AuditService(db)

    try:
        user = user_service.create(user_data)
        audit_service.log_action(
            action="register",
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"email": user.email, "role": str(user.role.value if hasattr(user.role, 'value') else user.role)},
            success=True
        )
        return user
    except ValueError as e:
        logger.warning(f"Registration failed for email {user_data.email}: {e}")
        try:
            audit_service.log_action(
                action="register",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"email": user_data.email},
                success=False,
                error_message=str(e)
            )
        except Exception as audit_err:
            logger.error(f"Audit log failed: {audit_err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected registration error for {user_data.email}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Registration error: {str(e)}")


@router.post("/login", response_model=Token)
def login(
    login_data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """User login - returns access and refresh tokens"""
    user_service = UserService(db)
    audit_service = AuditService(db)

    user = authenticate_user(db, login_data.email, login_data.password)
    if not user:
        audit_service.log_action(
            action="login",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"email": login_data.email},
            success=False,
            error_message="Invalid credentials"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        audit_service.log_action(
            action="login",
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"email": login_data.email},
            success=False,
            error_message="Inactive user"
        )
        raise HTTPException(status_code=400, detail="Inactive user")

    # Create tokens
    role_str = user.role.value if hasattr(user.role, "value") else str(user.role)
    access_token = create_access_token(
        data={"sub": str(user.id), "role": role_str}
    )
    refresh_token = create_refresh_token(
        data={"sub": str(user.id), "role": role_str}
    )

    # Update last login
    user_service.update_last_login(user.id)

    audit_service.log_action(
        action="login",
        user_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"email": user.email},
        success=True
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60
    )


@router.post("/refresh", response_model=Token)
def refresh_token(
    refresh_data: RefreshTokenRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token"""
    user_service = UserService(db)
    audit_service = AuditService(db)

    token_data = decode_token(refresh_data.refresh_token)
    if not token_data or token_data.token_type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_service.get_by_id(token_data.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=400, detail="User not found or inactive")

    role_str = user.role.value if hasattr(user.role, "value") else str(user.role)
    access_token = create_access_token(data={"sub": str(user.id), "role": role_str})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id), "role": role_str})

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return current_user


@router.put("/me", response_model=UserResponse)
def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    user_service = UserService(db)
    audit_service = AuditService(db)

    # Only allow updating own profile unless admin
    updated_user = user_service.update(current_user.id, user_data)
    audit_service.log_action(
        action="update_profile",
        user_id=current_user.id,
        details=user_data.model_dump(exclude_unset=True),
        success=True
    )
    return updated_user


@router.get("/users", response_model=list[UserResponse])
def list_users(
    skip: int = 0,
    limit: int = 100,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List users (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    user_service = UserService(db)
    role_enum = None
    if role:
        from app.models.user import UserRole
        try:
            role_enum = UserRole(role)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid role")

    return user_service.get_all(skip=skip, limit=limit, role=role_enum, is_active=is_active)


@router.get("/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user by ID (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    user_service = UserService(db)
    user = user_service.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    user_service = UserService(db)
    audit_service = AuditService(db)

    updated_user = user_service.update(user_id, user_data)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    audit_service.log_action(
        action="update_user",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(user_id),
        details=user_data.model_dump(exclude_unset=True),
        success=True
    )
    return updated_user


@router.delete("/users/{user_id}", response_model=MessageResponse)
def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    user_service = UserService(db)
    audit_service = AuditService(db)

    success = user_service.delete(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    audit_service.log_action(
        action="delete_user",
        user_id=current_user.id,
        resource_type="user",
        resource_id=str(user_id),
        success=True
    )
    return MessageResponse(message="User deleted successfully")