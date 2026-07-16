from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
import bcrypt
from app.core.config import settings

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its bcrypt hashed counterpart."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:72],
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8")[:72], salt)
    return hashed.decode("utf-8")

def create_jwt_token(user_id: str, email: str, role: str, expires_delta: timedelta, token_type: str = "access") -> str:
    """Generate a JWT token containing user identity, email, and role."""
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "type": token_type,
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp())
    }
    
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def create_access_token(user_id: str, email: str, role: str) -> str:
    """Create a short-lived access token."""
    expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return create_jwt_token(user_id, email, role, expires, "access")

def create_refresh_token(user_id: str, email: str, role: str) -> str:
    """Create a long-lived refresh token."""
    expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    return create_jwt_token(user_id, email, role, expires, "refresh")

def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token, returning the payload if valid."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None
