from fastapi import APIRouter, Depends, Query, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.database.models import User
from app.schemas.auth import TokenResponse, UserResponse, RefreshTokenRequest, MessageResponse
from app.schemas.candidate import CandidateRegisterRequest, CandidateLoginRequest
from app.services.auth_service import AuthService
from app.dependencies.auth import get_current_user
from app.core.config import settings

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post(
    "/register", 
    response_model=TokenResponse, 
    status_code=status.HTTP_201_CREATED,
    summary="Register a new Candidate"
)
async def register_candidate(request: CandidateRegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Registers a new candidate using their Name, Email, Phone number, and Password.
    Returns the user details, access token, and refresh token on successful registration.
    """
    return await AuthService.register_candidate(request, db)


@router.post(
    "/login", 
    response_model=TokenResponse,
    summary="Login as User"
)
async def login(
    request: CandidateLoginRequest, 
    fastapi_request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticates a candidate or recruiter with Email and Password.
    Returns the user details, access token, refresh token, role, and navigation path on success.
    """
    ip_address = fastapi_request.client.host if fastapi_request.client else None
    device = fastapi_request.headers.get("user-agent")
    
    return await AuthService.login_candidate(request, db, ip_address=ip_address, device=device)


@router.get(
    "/me", 
    response_model=UserResponse,
    summary="Get current user details (Legacy/Fallback)"
)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Returns the details of the currently authenticated user.
    """
    return current_user


@router.get(
    "/microsoft/login",
    summary="Initiate Microsoft Entra ID Login"
)
def microsoft_login():
    """
    Redirects the user's browser to the Microsoft Entra ID (Azure AD) OAuth2 authorization page.
    """
    auth_url = AuthService.get_microsoft_login_url()
    return RedirectResponse(url=auth_url)


@router.get(
    "/microsoft/callback",
    response_model=TokenResponse,
    summary="Microsoft Entra ID OAuth Callback"
)
async def microsoft_callback(
    fastapi_request: Request,
    code: str = Query(None, description="Authorization code from Microsoft"),
    error: str = Query(None, description="OAuth error code if any"),
    error_description: str = Query(None, description="OAuth error description if any"),
    db: AsyncSession = Depends(get_db),
    redirect: bool = Query(False, description="If True, redirects to frontend dashboard with tokens in query params")
):
    """
    Handles the Microsoft OAuth callback, exchanges the code for tokens, retrieves profile,
    and automatically registers or logs in the recruiter user.
    """
    import urllib.parse
    from fastapi import HTTPException

    if error or not code:
        err_msg = urllib.parse.quote(error_description or error or "Authentication cancelled or failed")
        if redirect:
            return RedirectResponse(url=f"{settings.FRONTEND_URL}/login?error={err_msg}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_description or error or "Missing auth code")

    ip_address = fastapi_request.client.host if fastapi_request.client else None
    device = fastapi_request.headers.get("user-agent")
    
    try:
        token_response = await AuthService.authenticate_microsoft_user(code, db, ip_address=ip_address, device=device)
    except Exception as e:
        if redirect:
            detail = getattr(e, "detail", str(e))
            err_msg = urllib.parse.quote(str(detail))
            return RedirectResponse(url=f"{settings.FRONTEND_URL}/login?error={err_msg}")
        raise
    
    if redirect:
        encoded_name = urllib.parse.quote(token_response.user.full_name or "")
        encoded_email = urllib.parse.quote(token_response.user.email or "")
        user_json = urllib.parse.quote(token_response.user.model_dump_json())
        redirect_url = (
            f"{settings.FRONTEND_URL}/oauth/callback?"
            f"token={token_response.access_token}&"
            f"access_token={token_response.access_token}&"
            f"refresh_token={token_response.refresh_token}&"
            f"role={token_response.role.value}&"
            f"redirect={token_response.redirect}&"
            f"name={encoded_name}&"
            f"email={encoded_email}&"
            f"user_id={token_response.user.id}&"
            f"user={user_json}"
        )
        return RedirectResponse(url=redirect_url)
        
    return token_response


@router.post(
    "/refresh", 
    response_model=TokenResponse,
    summary="Refresh Access & Refresh Tokens"
)
async def refresh_token(request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    """
    Exchanges a valid Refresh Token for a brand new Access Token and Refresh Token (token rotation).
    """
    return await AuthService.refresh_user_tokens(request.refresh_token, db)


@router.post(
    "/logout", 
    response_model=MessageResponse,
    summary="Logout user"
)
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logs out the current user. Client should delete local tokens.
    """
    return MessageResponse(message="Successfully logged out")

