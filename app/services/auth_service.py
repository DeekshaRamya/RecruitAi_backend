import httpx
from datetime import datetime, timezone
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
from app.core.config import settings
from app.core import security
from app.database.models import User, UserRole, LoginHistory
from app.schemas.candidate import CandidateRegisterRequest, CandidateLoginRequest
from app.schemas.auth import TokenResponse, UserResponse

class AuthService:
    
    @staticmethod
    def get_navigation_data(role: UserRole) -> dict:
        """Helper to get navigation redirect URL based on role."""
        if role == UserRole.CANDIDATE:
            return {"role": UserRole.CANDIDATE, "redirect": "/candidate/dashboard"}
        return {"role": UserRole.RECRUITER, "redirect": "/recruiter/dashboard"}

    @classmethod
    async def register_candidate(cls, request: CandidateRegisterRequest, db: AsyncSession) -> TokenResponse:
        """Register a new candidate and return tokens and profile."""
        # 1. Check if email is already registered
        result = await db.execute(select(User).where(User.email == request.email))
        existing_user = result.scalar_one_or_none()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already registered"
            )

        # 2. Hash the password
        hashed_password = security.get_password_hash(request.password)

        # 3. Create candidate user
        new_candidate = User(
            full_name=request.name,
            email=request.email,
            password=hashed_password,
            phone=request.phone,
            role=UserRole.CANDIDATE
        )
        
        try:
            db.add(new_candidate)
            await db.commit()
            await db.refresh(new_candidate)
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error during candidate registration: {str(e)}"
            )

        # 4. Generate local JWTs
        access_token = security.create_access_token(new_candidate.id, new_candidate.email, new_candidate.role.value)
        refresh_token = security.create_refresh_token(new_candidate.id, new_candidate.email, new_candidate.role.value)

        nav_data = cls.get_navigation_data(new_candidate.role)

        return TokenResponse(
            user=UserResponse.model_validate(new_candidate),
            access_token=access_token,
            refresh_token=refresh_token,
            role=nav_data["role"],
            redirect=nav_data["redirect"]
        )

    @classmethod
    async def login_candidate(cls, request: CandidateLoginRequest, db: AsyncSession, ip_address: str | None = None, device: str | None = None) -> TokenResponse:
        """Authenticate a candidate or recruiter using email and password."""
        # 1. Retrieve user
        result = await db.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()
        if not user or not user.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # 2. Verify password
        if not security.verify_password(request.password, user.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # 3. Store login history details
        login_history = LoginHistory(
            user_id=user.id,
            ip_address=ip_address,
            device=device
        )
        try:
            db.add(login_history)
            await db.commit()
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error storing login history: {str(e)}"
            )

        # 4. Generate tokens
        access_token = security.create_access_token(user.id, user.email, user.role.value)
        refresh_token = security.create_refresh_token(user.id, user.email, user.role.value)

        nav_data = cls.get_navigation_data(user.role)

        return TokenResponse(
            user=UserResponse.model_validate(user),
            access_token=access_token,
            refresh_token=refresh_token,
            role=nav_data["role"],
            redirect=nav_data["redirect"]
        )

    @classmethod
    def get_microsoft_login_url(cls) -> str:
        """Generate the Microsoft Entra ID login redirect URL."""
        tenant = settings.MICROSOFT_TENANT_ID
        client_id = settings.MICROSOFT_CLIENT_ID
        redirect_uri = settings.MICROSOFT_REDIRECT_URI
        
        # Define scopes: openid profile email and User.Read (graph)
        scope = "openid profile email User.Read"
        
        auth_url = (
            f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?"
            f"client_id={client_id}&"
            f"response_type=code&"
            f"redirect_uri={redirect_uri}&"
            f"response_mode=query&"
            f"scope={scope}&"
            f"state=recruitai_oauth"
        )
        return auth_url

    @classmethod
    async def authenticate_microsoft_user(cls, code: str, db: AsyncSession, ip_address: str | None = None, device: str | None = None) -> TokenResponse:
        """Exchange auth code for Microsoft access token, get user profile, and register/login recruiter."""
        tenant = settings.MICROSOFT_TENANT_ID
        token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
        
        # 1. Exchange authorization code for Microsoft access token
        data = {
            "client_id": settings.MICROSOFT_CLIENT_ID,
            "client_secret": settings.MICROSOFT_CLIENT_SECRET,
            "code": code,
            "redirect_uri": settings.MICROSOFT_REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        
        async with httpx.AsyncClient() as client:
            try:
                token_resp = await client.post(token_url, data=data)
                token_data = token_resp.json()
                if token_resp.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Microsoft token exchange failed: {token_data.get('error_description', 'Unknown error')}"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Network error during Microsoft auth exchange: {str(e)}"
                )
            
            microsoft_access_token = token_data.get("access_token")
            if not microsoft_access_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Access token not found in Microsoft response"
                )

            # 2. Retrieve user profile info from Microsoft Graph API
            headers = {"Authorization": f"Bearer {microsoft_access_token}"}
            try:
                profile_resp = await client.get("https://graph.microsoft.com/v1.0/me", headers=headers)
                profile = profile_resp.json()
                if profile_resp.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to fetch Microsoft user profile: {profile.get('error', {}).get('message', 'Unknown error')}"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Network error during Microsoft profile retrieval: {str(e)}"
                )

        microsoft_id = profile.get("id")
        name = profile.get("displayName") or profile.get("givenName", "Microsoft User")
        email = profile.get("mail") or profile.get("userPrincipalName")
        
        if not microsoft_id or not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incomplete user profile data returned by Microsoft (missing id or email)"
            )

        # 3. Lookup recruiter in the local DB. Check by microsoft_id first, then email.
        result = await db.execute(select(User).where(User.microsoft_id == microsoft_id))
        recruiter = result.scalar_one_or_none()
        
        try:
            if not recruiter:
                # If not found by microsoft_id, check if email exists
                result = await db.execute(select(User).where(User.email == email))
                recruiter = result.scalar_one_or_none()
                if recruiter:
                    # User exists but hasn't linked Microsoft accounts
                    # Link Microsoft ID and update role to Recruiter
                    recruiter.microsoft_id = microsoft_id
                    recruiter.role = UserRole.RECRUITER
                    await db.commit()
                    await db.refresh(recruiter)
                else:
                    # Create a new Recruiter record automatically
                    recruiter = User(
                        full_name=name,
                        email=email,
                        microsoft_id=microsoft_id,
                        role=UserRole.RECRUITER
                    )
                    db.add(recruiter)
                    await db.commit()
                    await db.refresh(recruiter)
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error linking or creating Microsoft user: {str(e)}"
            )

        # 4. Store login history details
        login_history = LoginHistory(
            user_id=recruiter.id,
            ip_address=ip_address,
            device=device
        )
        try:
            db.add(login_history)
            await db.commit()
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error storing Microsoft login history: {str(e)}"
            )

        # 5. Generate local JWT access & refresh tokens
        access_token = security.create_access_token(recruiter.id, recruiter.email, recruiter.role.value)
        refresh_token = security.create_refresh_token(recruiter.id, recruiter.email, recruiter.role.value)

        nav_data = cls.get_navigation_data(recruiter.role)

        return TokenResponse(
            user=UserResponse.model_validate(recruiter),
            access_token=access_token,
            refresh_token=refresh_token,
            role=nav_data["role"],
            redirect=nav_data["redirect"]
        )

    @classmethod
    async def refresh_user_tokens(cls, refresh_token: str, db: AsyncSession) -> TokenResponse:
        """Validate refresh token and issue new access & refresh tokens."""
        # 1. Decode token
        payload = security.decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 2. Get user
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        try:
            user_int_id = int(user_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID format in token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        result = await db.execute(select(User).where(User.id == user_int_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 3. Generate new access and refresh tokens (token rotation)
        new_access_token = security.create_access_token(user.id, user.email, user.role.value)
        new_refresh_token = security.create_refresh_token(user.id, user.email, user.role.value)

        nav_data = cls.get_navigation_data(user.role)

        return TokenResponse(
            user=UserResponse.model_validate(user),
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            role=nav_data["role"],
            redirect=nav_data["redirect"]
        )
