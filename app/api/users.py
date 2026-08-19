import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, or_
from pydantic import BaseModel, EmailStr, ConfigDict

from app.database.database import get_db
from app.database.models import User, UserRole, LoginHistory
from app.schemas.auth import UserResponse
from app.dependencies.auth import get_current_user, require_admin

router = APIRouter(prefix="/api/users", tags=["Users"])

# Pydantic schemas for User & Login History management
class UpdateUserRoleRequest(BaseModel):
    role: UserRole

class CreateInternalUserRequest(BaseModel):
    full_name: str
    email: EmailStr
    role: UserRole = UserRole.RECRUITER
    password: Optional[str] = None
    phone: Optional[str] = None

class LoginHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    login_time: datetime
    ip_address: Optional[str] = None
    device: Optional[str] = None

class InternalUserItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    full_name: str
    name: str
    email: str
    role: UserRole
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    login_count: int = 0
    microsoft_id: Optional[str] = None


@router.get(
    "/me", 
    response_model=UserResponse,
    summary="Get current user details"
)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Returns the details of the currently authenticated user (via JWT in the Authorization header).
    """
    return current_user


@router.get(
    "/internal",
    response_model=List[InternalUserItem],
    summary="Get all internal staff and admins"
)
async def get_internal_users(
    role: Optional[str] = Query(None, description="Filter by role: admin, recruiter, or all"),
    search: Optional[str] = Query(None, description="Search by name or email"),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin)
):
    """
    Retrieve all internal platform users (admins, recruiters, staff) along with their login statistics.
    Only accessible by administrators.
    """
    stmt = select(User)
    
    if role and role.lower() in [r.value for r in UserRole]:
        stmt = stmt.where(User.role == UserRole(role.lower()))
    else:
        # By default return recruiters and admins (internal users) or all if specified
        if role != 'all':
            stmt = stmt.where(or_(User.role == UserRole.ADMIN, User.role == UserRole.RECRUITER))

    if search:
        search_pattern = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                User.full_name.ilike(search_pattern),
                User.email.ilike(search_pattern)
            )
        )

    stmt = stmt.order_by(desc(User.created_at))
    result = await db.execute(stmt)
    users = result.scalars().all()

    response_items = []
    for u in users:
        # Calculate last login and login count from user.login_history
        histories = u.login_history or []
        last_login_dt = max([h.login_time for h in histories], default=None) if histories else None
        
        response_items.append(InternalUserItem(
            id=u.id,
            full_name=u.full_name,
            name=u.full_name,
            email=u.email,
            role=u.role,
            created_at=u.created_at,
            updated_at=u.updated_at,
            last_login=last_login_dt,
            login_count=len(histories),
            microsoft_id=u.microsoft_id
        ))

    return response_items


@router.post(
    "/internal",
    response_model=InternalUserItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new internal team user"
)
async def create_internal_user(
    payload: CreateInternalUserRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin)
):
    """
    Manually add an internal admin or recruiter.
    """
    from app.core import security
    
    # Check if email is already registered
    existing_result = await db.execute(select(User).where(User.email == payload.email))
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists."
        )

    hashed_pw = security.get_password_hash(payload.password) if payload.password else None

    new_user = User(
        full_name=payload.full_name,
        email=payload.email,
        password=hashed_pw,
        phone=payload.phone,
        role=payload.role
    )
    
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error creating internal user: {str(e)}"
        )

    return InternalUserItem(
        id=new_user.id,
        full_name=new_user.full_name,
        name=new_user.full_name,
        email=new_user.email,
        role=new_user.role,
        created_at=new_user.created_at,
        updated_at=new_user.updated_at,
        last_login=None,
        login_count=0,
        microsoft_id=None
    )


@router.patch(
    "/{user_id}/role",
    response_model=InternalUserItem,
    summary="Update a user's role"
)
async def update_user_role(
    user_id: uuid.UUID,
    payload: UpdateUserRoleRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin)
):
    """
    Change a user's system role (e.g. promote recruiter to admin, or switch roles).
    Cannot demote self if that would leave zero admins.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )

    # Protect current logged in admin from accidentally stripping their own admin role
    if user.id == admin_user.id and payload.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot revoke your own administrator permissions."
        )

    user.role = payload.role
    try:
        await db.commit()
        await db.refresh(user)
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error updating user role: {str(e)}"
        )

    histories = user.login_history or []
    last_login_dt = max([h.login_time for h in histories], default=None) if histories else None

    return InternalUserItem(
        id=user.id,
        full_name=user.full_name,
        name=user.full_name,
        email=user.email,
        role=user.role,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=last_login_dt,
        login_count=len(histories),
        microsoft_id=user.microsoft_id
    )


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an internal user"
)
async def delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin)
):
    """
    Remove an internal user or revoke their access.
    """
    if user_id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own administrator account."
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )

    await db.delete(user)
    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error deleting user: {str(e)}"
        )

    return {"message": "User deleted successfully", "deleted_user_id": str(user_id)}


@router.get(
    "/login-history",
    response_model=List[LoginHistoryResponse],
    summary="Get recent authentication and login audit history"
)
async def get_login_history(
    limit: int = Query(100, ge=1, le=500),
    user_id: Optional[uuid.UUID] = Query(None),
    role: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(require_admin)
):
    """
    Retrieve login history logs with timestamp, user role, client IP, and device user-agent.
    Only accessible by administrators.
    """
    stmt = (
        select(LoginHistory, User)
        .join(User, LoginHistory.user_id == User.id)
        .order_by(desc(LoginHistory.login_time))
        .limit(limit)
    )

    if user_id:
        stmt = stmt.where(LoginHistory.user_id == user_id)

    if role and role.lower() in [r.value for r in UserRole]:
        stmt = stmt.where(User.role == UserRole(role.lower()))

    result = await db.execute(stmt)
    rows = result.all()

    logs = []
    for history, user in rows:
        logs.append(LoginHistoryResponse(
            id=history.id,
            user_id=history.user_id,
            user_name=user.full_name,
            user_email=user.email,
            user_role=user.role.value if user.role else "candidate",
            login_time=history.login_time,
            ip_address=history.ip_address or "127.0.0.1",
            device=history.device or "Browser / Web Session"
        ))

    return logs
