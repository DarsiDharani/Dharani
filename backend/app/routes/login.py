"""
Login Routes Module

Purpose: User authentication endpoint
Features:
- User login with username and password
- JWT token generation
- Role detection (manager/employee)

Endpoint:
- POST /login: Authenticate user and return JWT token

Note: This is an alternative login route implementation.
The main login route is in app/auth.py

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database import get_db_async
from app.models import User, ManagerEmployee
from app.auth_utils import verify_password, create_access_token
from app.schemas import UserLogin

router = APIRouter()

@router.post("/login")
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db_async)):
    # Check user
    result = await db.execute(select(User).where(User.username == user_data.username))
    user = result.scalars().first()

    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Determine role from manager_employee table
    # Check if user is a manager (has employees reporting to them)
    manager_check = await db.execute(
        select(ManagerEmployee).where(ManagerEmployee.manager_empid == user_data.username)
    )
    is_manager = manager_check.scalars().first() is not None
    
    # Check if user is an employee (reports to a manager)
    employee_check = await db.execute(
        select(ManagerEmployee).where(ManagerEmployee.employee_empid == user_data.username)
    )
    is_employee = employee_check.scalars().first() is not None
    
    # Determine role: manager takes precedence if user is both
    if is_manager:
        role = "manager"
    elif is_employee:
        role = "employee"
    else:
        # User exists but not in manager_employee table - default to employee
        role = "employee"

    # Fetch employee name to add to the token
    name_result = await db.execute(
        select(ManagerEmployee.employee_name).where(ManagerEmployee.employee_empid == user_data.username)
    )
    employee_name = name_result.scalars().first()
    
    # If employee_name is None, use username as fallback
    if employee_name is None:
        # Try to get manager name if user is a manager
        # Note: A manager can have multiple rows (one per employee), so we use first() instead of scalar_one_or_none()
        manager_name_result = await db.execute(
            select(ManagerEmployee.manager_name).where(ManagerEmployee.manager_empid == user_data.username)
        )
        manager_name = manager_name_result.scalars().first()
        employee_name = manager_name or user_data.username

    # Create token with username, role, and employee_name
    token = create_access_token({"sub": user_data.username, "role": role, "employee_name": employee_name})
    return {"access_token": token, "token_type": "bearer", "role": role}