"""
Authentication Utilities Module (Alternative Implementation)

Purpose: JWT token management, password hashing, and user authentication utilities
Features:
- JWT token creation with configurable expiration
- Password hashing using pbkdf2_sha256 (supports passwords of any length)
- Password verification supporting both bcrypt and pbkdf2_sha256
- Current user extraction from JWT tokens
- Manager role verification

Security:
- Uses pbkdf2_sha256 as primary hashing scheme (no 72-byte limit)
- Falls back to bcrypt for backward compatibility
- JWT tokens with configurable expiration
- Secret key should be changed in production (use environment variable)

Note: This is an alternative implementation to utils.py with enhanced password handling

@author Orbit Skill Development Team
@date 2025
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database import get_db_async
from app.models import User

# Configuration for JWT
# TODO: Move SECRET_KEY to environment variable for production
SECRET_KEY = "your-super-secret-key"  # CHANGE THIS!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing configuration
# Use pbkdf2_sha256 as primary (no 72-byte limit like bcrypt), bcrypt as fallback
# pbkdf2_sha256 is listed first to be the default, and bcrypt is kept for backward compatibility
pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")

# OAuth2 password bearer scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token with expiration.
    
    Args:
        data: Dictionary containing token payload (typically username and role)
        expires_delta: Optional custom expiration time (defaults to ACCESS_TOKEN_EXPIRE_MINUTES)
        
    Returns:
        str: Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_password_hash(password: str):
    """
    Hash password using pbkdf2_sha256 (no 72-byte limit like bcrypt).
    Supports passwords of any length.
    """
    # Use pbkdf2_sha256 explicitly to avoid bcrypt's 72-byte limit
    # This ensures we can handle passwords of any length
    try:
        # Create a context specifically for pbkdf2_sha256 hashing
        pbkdf2_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
        return pbkdf2_context.hash(password)
    except Exception as e:
        # Fallback to the main context if there's an issue
        import logging
        logging.warning(f"pbkdf2_sha256 hashing failed, using fallback: {e}")
        return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    """
    Verify password against hash. Supports both bcrypt and pbkdf2_sha256 hashes.
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # If hash format is invalid or unrecognized, try to handle it gracefully
        # This can happen if the hash was stored incorrectly
        import logging
        logging.warning(f"Password verification failed: {e}")
        return False

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Extract current user information from JWT token.
    
    This is a dependency function used in protected routes to get the authenticated user.
    Validates the JWT token and extracts username and role.
    
    Args:
        token: JWT token from Authorization header (extracted by oauth2_scheme)
        
    Returns:
        dict: Dictionary containing 'username' and 'role'
        
    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        print("❌ No token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication token provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        print(f"🔍 Validating token: {token[:20]}..." if len(token) > 20 else f"🔍 Validating token: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        print(f"✅ Token decoded - Username: {username}, Role: {role}")

        if username is None or role is None:
            print("❌ Username or role is None in token payload")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token payload - Username: {username}, Role: {role}",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except jwt.ExpiredSignatureError:
        print("❌ Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        print(f"❌ JWT Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return username and role as a dictionary for easy access
    return {"username": username, "role": role}

async def get_current_active_user(user_data: dict = Depends(get_current_user)):
    """
    Get current active user (alias for get_current_user).
    
    Args:
        user_data: User data from get_current_user dependency
        
    Returns:
        dict: User data dictionary
    """
    return user_data

async def get_current_active_manager(user_data: dict = Depends(get_current_user)):
    """
    Get current active manager user with role verification.
    
    Verifies that the current user has manager role, raises 403 if not.
    Use this dependency in routes that require manager privileges.
    
    Args:
        user_data: User data from get_current_user dependency
        
    Returns:
        dict: User data dictionary (only if role is 'manager')
        
    Raises:
        HTTPException: 403 if user is not a manager
    """
    if user_data["role"] != "manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource"
        )
    return user_data
