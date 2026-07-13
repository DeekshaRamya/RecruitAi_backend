from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database.models import User
from app.schemas.auth import TokenResponse, UserResponse, RefreshTokenRequest, MessageResponse
from app.schemas.candidate import CandidateRegisterRequest, CandidateLoginRequest
from app.services.auth_service import AuthService
from app.dependencies.auth import get_current_user
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post(
    "/candidate/register", 
    response_model=TokenResponse, 
    status_code=status.HTTP_201_CREATED,
    summary="Register a new Candidate"
)
def register_candidate(request: CandidateRegisterRequest, db: Session = Depends(get_db)):
    """
    Registers a new candidate using their Name, Email, Phone number, and Password.
    Returns the user details, access token, and refresh token on successful registration.
    """
    return AuthService.register_candidate(request, db)


@router.post(
    "/candidate/login", 
    response_model=TokenResponse,
    summary="Login as Candidate"
)
def login_candidate(request: CandidateLoginRequest, db: Session = Depends(get_db)):
    """
    Authenticates a candidate with Email and Password.
    Returns the user details, access token, refresh token, role, and navigation path on success.
    """
    return AuthService.login_candidate(request, db)


@router.get(
    "/me", 
    response_model=UserResponse,
    summary="Get current user details"
)
def get_me(current_user: User = Depends(get_current_user)):
    """
    Returns the details of the currently authenticated user (via JWT in the Authorization header).
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
    code: str = Query(..., description="Authorization code from Microsoft"),
    db: Session = Depends(get_db),
    redirect: bool = Query(False, description="If True, redirects to frontend dashboard with tokens in query params")
):
    """
    Handles the Microsoft OAuth callback, exchanges the code for tokens, retrieves profile,
    and automatically registers or logs in the recruiter user.
    """
    token_response = await AuthService.authenticate_microsoft_user(code, db)
    
    if redirect:
        # Construct redirect URI to React frontend dashboard
        # This allows a seamless SPA login flow where Microsoft redirects to backend and backend redirects back to frontend
        redirect_url = (
            f"{settings.FRONTEND_URL}/oauth/callback?"
            f"access_token={token_response.access_token}&"
            f"refresh_token={token_response.refresh_token}&"
            f"role={token_response.role.value}&"
            f"redirect={token_response.redirect}"
        )
        return RedirectResponse(url=redirect_url)
        
    return token_response


@router.post(
    "/refresh", 
    response_model=TokenResponse,
    summary="Refresh Access & Refresh Tokens"
)
def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """
    Exchanges a valid Refresh Token for a brand new Access Token and Refresh Token (token rotation).
    """
    return AuthService.refresh_user_tokens(request.refresh_token, db)


@router.post(
    "/logout", 
    response_model=MessageResponse,
    summary="Logout user"
)
def logout(current_user: User = Depends(get_current_user)):
    """
    Logs out the current user. Since JWTs are stateless, this endpoint invalidates the session
    from the application context. The client should delete the local tokens.
    """
    return MessageResponse(message="Successfully logged out")
