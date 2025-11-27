"""
Authentication Utilities Module

Purpose: JWT token management, password hashing, and user authentication utilities
Features:
- JWT token creation and validation
- Password hashing and verification using bcrypt
- Current user extraction from JWT tokens
- OAuth2 password bearer scheme integration

Security:
- Uses bcrypt for password hashing
- JWT tokens with configurable expiration
- Secret key should be changed in production (use environment variable)

@author Orbit Skill Development Team
@date 2025
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from app.models import User
from app.database import get_db_async
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# JWT Configuration
# TODO: Move SECRET_KEY to environment variable for production
SECRET_KEY = "your-super-secret-key"  # CHANGE THIS in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")

# OAuth2 password bearer scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary containing token payload (typically username and role)
        expires_delta: Optional custom expiration time (defaults to 15 minutes)
        
    Returns:
        str: Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    """
    Verify a plain text password against a hashed password.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Bcrypt hashed password from database
        
    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)

async def get_current_user_and_role(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db_async)):
    """
    Extract current user and role from JWT token.
    
    This is a dependency function used in protected routes to get the authenticated user.
    Validates the JWT token and retrieves user from database.
    
    Args:
        token: JWT token from Authorization header (extracted by oauth2_scheme)
        db: Database session dependency
        
    Returns:
        dict: Dictionary containing 'user' (User object) and 'role' (string)
        
    Raises:
        HTTPException: 401 if token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode and validate JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        token_data = {"username": username, "role": role}
    except JWTError:
        raise credentials_exception
    
    # Verify user exists in database
    user_statement = select(User).where(User.username == token_data["username"])
    user_result = await db.execute(user_statement)
    user = user_result.scalars().first()
    
    if user is None:
        raise credentials_exception
    return {"user": user, "role": role}

def get_db_session(db: AsyncSession = Depends(get_db_async)):
    """
    Database session dependency (alias for get_db_async).
    
    Args:
        db: Database session from dependency injection
        
    Yields:
        AsyncSession: Database session object
    """
    yield db
