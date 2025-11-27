"""
Authentication Routes Module

Purpose: Handle user registration and login functionality
Features:
- User registration with password hashing
- User login with JWT token generation
- Role detection (manager/employee) based on manager_employee table
- Password verification using bcrypt

Endpoints:
- POST /register: Register a new user
- POST /login: Authenticate user and return JWT token

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, ManagerEmployee
from app.schemas import UserCreate, UserLogin
from app.utils import hash_password, verify_password, create_access_token

router = APIRouter()

@router.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user account.
    
    Creates a new user with hashed password. Username must be unique.
    
    Args:
        user: UserCreate schema containing username and password
        db: Database session dependency
        
    Returns:
        dict: Success message
        
    Raises:
        HTTPException: 400 if username already exists
    """
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Hash password before storing
    hashed = hash_password(user.password)
    new_user = User(username=user.username, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user and return JWT access token.
    
    Verifies user credentials and determines role (manager/employee) based on
    manager_employee table relationships.
    
    Args:
        user: UserLogin schema containing username and password
        db: Database session dependency
        
    Returns:
        dict: Access token, token type, and user role
        
    Raises:
        HTTPException: 401 if credentials are invalid
    """
    # Verify user exists and password is correct
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Determine user role based on manager_employee relationships
    emp_id = user.username
    is_manager = db.query(ManagerEmployee).filter(ManagerEmployee.manager_empid == emp_id).first()
    is_employee = db.query(ManagerEmployee).filter(ManagerEmployee.employee_empid == emp_id).first()

    if is_manager:
        role = "manager"
    elif is_employee:
        role = "employee"
    else:
        role = "unknown"

    # Generate JWT token with username and role
    token = create_access_token({"sub": user.username, "role": role})
    return {"access_token": token, "token_type": "bearer", "role": role}
