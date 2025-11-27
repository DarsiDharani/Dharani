"""
Register Routes Module

Purpose: User registration endpoint
Features:
- Create new user account
- Password hashing using pbkdf2_sha256
- Username uniqueness validation

Endpoint:
- POST /register: Register a new user account

Note: This is an alternative registration route implementation.
The main registration route is in app/auth.py

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from sqlalchemy import select

from app.schemas import UserRegister
from app.database import get_db_async
from app.models import User
from app.auth_utils import get_password_hash

router = APIRouter()

@router.post("/register")
async def register_user(user: UserRegister, db: AsyncSession = Depends(get_db_async)):
    # Check if user already exists
    stmt = select(User).where(User.username == user.emp_id)
    result = await db.execute(stmt)
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Hash password and create user
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.emp_id,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Check if there's existing employee_competency data for this employee
    from app.models import EmployeeCompetency
    competency_check = select(EmployeeCompetency).where(EmployeeCompetency.employee_empid == user.emp_id)
    competency_result = await db.execute(competency_check)
    competency_count = len(competency_result.scalars().all())
    
    message = "User registered successfully"
    if competency_count > 0:
        message += f". Found {competency_count} competency record(s) linked to this employee."

    return {"message": message}