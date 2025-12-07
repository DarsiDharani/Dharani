"""
Admin Routes Module

Purpose: API routes for admin dashboard and administrative operations
Features:
- Admin dashboard data endpoint
- User management (CRUD, role assignment)
- Training management (admin override)
- Skills management (system-wide)
- Analytics endpoints
- Data management

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, or_, and_
from sqlalchemy.orm import selectinload
from typing import Optional, List
from datetime import datetime, date

from app.database import get_db_async
from app.auth_utils import get_current_active_admin
from app.models import (
    User, Admin, ManagerEmployee, EmployeeCompetency, AdditionalSkill,
    TrainingDetail, TrainingAssignment, TrainingRequest, TrainingAttendance,
    Trainer, AssignmentSubmission, FeedbackSubmission, Notification
)
from app.schemas import TrainingCreate, TrainingResponse
from app.auth_utils import get_password_hash
from pydantic import BaseModel

router = APIRouter(prefix="/admin", tags=["Admin"])

# ==================== SCHEMAS ====================

class UserCreateAdmin(BaseModel):
    username: str
    password: str
    name: str
    role: str  # employee, manager, admin
    manager_empid: Optional[str] = None
    is_trainer: bool = False

class UserUpdateAdmin(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    manager_empid: Optional[str] = None
    is_trainer: Optional[bool] = None

class UserResponseAdmin(BaseModel):
    username: str
    name: str
    role: str
    is_trainer: bool
    manager_name: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None

class AdminDashboardResponse(BaseModel):
    admin_name: str
    admin_id: str
    metrics: dict
    recent_activities: List[dict]

class TrainingAssignRequest(BaseModel):
    employee_empids: List[str]
    manager_empid: str

class SkillUpdateAdmin(BaseModel):
    current_expertise: str
    target_expertise: str

class AdditionalSkillReview(BaseModel):
    action: str  # approve, reject
    admin_notes: Optional[str] = None

# ==================== DASHBOARD ====================

@router.get("/dashboard")
async def get_admin_dashboard(
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Get admin dashboard data with metrics and recent activities"""
    admin_username = current_user.get("username")
    
    # Get admin name
    admin_name_result = await db.execute(
        select(ManagerEmployee.manager_name).where(ManagerEmployee.manager_empid == admin_username)
    )
    admin_name_row = admin_name_result.first()
    admin_name = admin_name_row[0] if admin_name_row else admin_username
    
    # Get employee name if manager name not found
    if admin_name == admin_username:
        emp_name_result = await db.execute(
            select(ManagerEmployee.employee_name).where(ManagerEmployee.employee_empid == admin_username)
        )
        emp_name_row = emp_name_result.first()
        admin_name = emp_name_row[0] if emp_name_row else admin_username
    
    # Calculate metrics
    total_users = await db.execute(select(func.count(User.id)))
    total_users_count = total_users.scalar() or 0
    
    total_trainings = await db.execute(select(func.count(TrainingDetail.id)))
    total_trainings_count = total_trainings.scalar() or 0
    
    total_assignments = await db.execute(select(func.count(TrainingAssignment.id)))
    total_assignments_count = total_assignments.scalar() or 0
    
    total_skills = await db.execute(select(func.count(EmployeeCompetency.id)))
    total_skills_count = total_skills.scalar() or 0
    
    pending_requests = await db.execute(
        select(func.count(TrainingRequest.id)).where(TrainingRequest.status == 'pending')
    )
    pending_requests_count = pending_requests.scalar() or 0
    
    # Count users by role - count actual users, not just relationships
    # Count admins (users who exist in Admin table)
    admins_result = await db.execute(select(func.count(Admin.id)))
    admins_count = admins_result.scalar() or 0
    
    # Count managers - distinct users who exist in User table AND appear as manager_empid
    # Join with User table to ensure we only count actual users
    managers_query = select(func.count(func.distinct(ManagerEmployee.manager_empid))).join(
        User, ManagerEmployee.manager_empid == User.username
    )
    managers_result = await db.execute(managers_query)
    managers_count = managers_result.scalar() or 0
    
    # Count employees - distinct users who exist in User table AND appear as employee_empid
    # Join with User table to ensure we only count actual users
    employees_query = select(func.count(func.distinct(ManagerEmployee.employee_empid))).join(
        User, ManagerEmployee.employee_empid == User.username
    )
    employees_result = await db.execute(employees_query)
    employees_total = employees_result.scalar() or 0
    
    # Count active trainers
    trainers_result = await db.execute(
        select(func.count(func.distinct(ManagerEmployee.manager_empid))).where(
            ManagerEmployee.manager_is_trainer == True
        )
    )
    trainers_count = trainers_result.scalar() or 0
    
    emp_trainers_result = await db.execute(
        select(func.count(func.distinct(ManagerEmployee.employee_empid))).where(
            ManagerEmployee.employee_is_trainer == True
        )
    )
    trainers_count += emp_trainers_result.scalar() or 0
    
    # Recent activities (last 10)
    recent_trainings = await db.execute(
        select(TrainingDetail).order_by(TrainingDetail.id.desc()).limit(5)
    )
    recent_trainings_list = recent_trainings.scalars().all()
    
    activities = []
    for training in recent_trainings_list:
        activities.append({
            "type": "training_created",
            "description": f"Training '{training.training_name}' created",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return {
        "admin_name": admin_name,
        "admin_id": admin_username,
        "metrics": {
            "total_users": total_users_count,
            "total_managers": managers_count,
            "total_employees": employees_total,
            "total_trainings": total_trainings_count,
            "total_assignments": total_assignments_count,
            "total_skills": total_skills_count,
            "pending_requests": pending_requests_count,
            "active_trainers": trainers_count
        },
        "recent_activities": activities[:10]
    }

# ==================== USER MANAGEMENT ====================

@router.get("/users")
async def get_all_users(
    role: Optional[str] = Query(None),
    is_trainer: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Get all users with optional filtering"""
    offset = (page - 1) * limit
    
    # Base query
    query = select(User)
    
    # Apply filters
    if search:
        query = query.where(User.username.ilike(f"%{search}%"))
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    users_result = await db.execute(query)
    users = users_result.scalars().all()
    
    # Build response with role information
    users_list = []
    for user in users:
        # Check if admin
        admin_check = await db.execute(
            select(Admin).where(Admin.username == user.username)
        )
        is_admin = admin_check.scalars().first() is not None
        
        # Get manager/employee info
        manager_info = await db.execute(
            select(ManagerEmployee).where(
                or_(
                    ManagerEmployee.manager_empid == user.username,
                    ManagerEmployee.employee_empid == user.username
                )
            ).limit(1)
        )
        manager_row = manager_info.scalars().first()
        
        user_role = "admin" if is_admin else "unknown"
        user_name = user.username
        is_trainer_user = False
        
        if manager_row:
            if manager_row.manager_empid == user.username:
                user_role = "manager"
                user_name = manager_row.manager_name or user.username
                is_trainer_user = manager_row.manager_is_trainer
            elif manager_row.employee_empid == user.username:
                user_role = "employee"
                user_name = manager_row.employee_name or user.username
                is_trainer_user = manager_row.employee_is_trainer
        
        # Apply role filter
        if role and user_role != role:
            continue
        
        # Apply trainer filter
        if is_trainer is not None and is_trainer_user != is_trainer:
            continue
        
        users_list.append({
            "username": user.username,
            "name": user_name,
            "role": user_role,
            "is_trainer": is_trainer_user,
            "created_at": user.created_at.isoformat() if user.created_at else None
        })
    
    return {
        "users": users_list,
        "total": total,
        "page": page,
        "limit": limit
    }

@router.post("/users")
async def create_user(
    user_data: UserCreateAdmin,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Create a new user"""
    # Check if user already exists
    existing_user = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if existing_user.scalars().first():
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        hashed_password=hashed_password
    )
    db.add(new_user)
    await db.flush()
    
    # Create manager-employee relationship if needed
    if user_data.role in ["employee", "manager"]:
        if user_data.role == "manager":
            # Manager entry
            manager_emp = ManagerEmployee(
                manager_empid=user_data.username,
                manager_name=user_data.name,
                employee_empid=user_data.username,  # Self-reference for manager
                employee_name=user_data.name,
                manager_is_trainer=user_data.is_trainer,
                employee_is_trainer=False
            )
        else:
            # Employee entry
            if not user_data.manager_empid:
                raise HTTPException(status_code=400, detail="Manager ID required for employees")
            
            manager_emp = ManagerEmployee(
                manager_empid=user_data.manager_empid,
                manager_name="",  # Will be updated if manager exists
                employee_empid=user_data.username,
                employee_name=user_data.name,
                manager_is_trainer=False,
                employee_is_trainer=user_data.is_trainer
            )
        db.add(manager_emp)
    
    # Make admin if role is admin
    if user_data.role == "admin":
        admin_entry = Admin(
            username=user_data.username,
            created_by=current_user.get("username")
        )
        db.add(admin_entry)
    
    await db.commit()
    
    return {"message": "User created successfully", "username": user_data.username}

@router.put("/users/{username}")
async def update_user(
    username: str,
    user_data: UserUpdateAdmin,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Update user information"""
    user = await db.execute(select(User).where(User.username == username))
    user_obj = user.scalars().first()
    
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update manager-employee relationship
    manager_emp = await db.execute(
        select(ManagerEmployee).where(
            or_(
                ManagerEmployee.manager_empid == username,
                ManagerEmployee.employee_empid == username
            )
        ).limit(1)
    )
    manager_emp_obj = manager_emp.first()
    
    if manager_emp_obj:
        if user_data.name:
            if manager_emp_obj.manager_empid == username:
                manager_emp_obj.manager_name = user_data.name
            else:
                manager_emp_obj.employee_name = user_data.name
        
        if user_data.is_trainer is not None:
            if manager_emp_obj.manager_empid == username:
                manager_emp_obj.manager_is_trainer = user_data.is_trainer
            else:
                manager_emp_obj.employee_is_trainer = user_data.is_trainer
    
    await db.commit()
    
    return {"message": "User updated successfully"}

@router.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Delete a user"""
    if username == current_user.get("username"):
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    user = await db.execute(select(User).where(User.username == username))
    user_obj = user.scalars().first()
    
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Delete will cascade due to foreign keys
    await db.delete(user_obj)
    await db.commit()
    
    return {"message": "User deleted successfully"}

@router.post("/users/{username}/reset-password")
async def reset_password(
    username: str,
    password_data: dict,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Reset user password"""
    user = await db.execute(select(User).where(User.username == username))
    user_obj = user.scalars().first()
    
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_password = password_data.get("new_password")
    if not new_password:
        raise HTTPException(status_code=400, detail="New password required")
    
    user_obj.hashed_password = get_password_hash(new_password)
    await db.commit()
    
    return {"message": "Password reset successfully"}

# ==================== TRAINING MANAGEMENT ====================

@router.get("/trainings")
async def get_all_trainings(
    skill: Optional[str] = Query(None),
    trainer: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Get all trainings (admin override)"""
    query = select(TrainingDetail)
    
    if skill:
        query = query.where(TrainingDetail.skill.ilike(f"%{skill}%"))
    if trainer:
        query = query.where(TrainingDetail.trainer_name.ilike(f"%{trainer}%"))
    
    trainings_result = await db.execute(query.order_by(TrainingDetail.id.desc()))
    trainings = trainings_result.scalars().all()
    
    trainings_list = []
    for training in trainings:
        # Get assignment count
        assign_count = await db.execute(
            select(func.count(TrainingAssignment.id)).where(
                TrainingAssignment.training_id == training.id
            )
        )
        assigned_count = assign_count.scalar() or 0
        
        # Get attendance count
        attend_count = await db.execute(
            select(func.count(TrainingAttendance.id)).where(
                and_(
                    TrainingAttendance.training_id == training.id,
                    TrainingAttendance.attended == True
                )
            )
        )
        attended_count = attend_count.scalar() or 0
        
        completion_rate = (attended_count / assigned_count * 100) if assigned_count > 0 else 0
        
        trainings_list.append({
            "id": training.id,
            "training_name": training.training_name,
            "trainer_name": training.trainer_name,
            "training_date": training.training_date.isoformat() if training.training_date else None,
            "skill": training.skill,
            "assigned_count": assigned_count,
            "attended_count": attended_count,
            "completion_rate": round(completion_rate, 2)
        })
    
    return {"trainings": trainings_list, "total": len(trainings_list)}

@router.post("/trainings")
async def create_training(
    training_data: TrainingCreate,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Create training (admin override - no trainer restriction)"""
    new_training = TrainingDetail(**training_data.dict())
    db.add(new_training)
    await db.commit()
    await db.refresh(new_training)
    
    return {"message": "Training created successfully", "training_id": new_training.id}

@router.put("/trainings/{training_id}")
async def update_training(
    training_id: int,
    training_data: TrainingCreate,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Update training"""
    training = await db.execute(
        select(TrainingDetail).where(TrainingDetail.id == training_id)
    )
    training_obj = training.scalars().first()
    
    if not training_obj:
        raise HTTPException(status_code=404, detail="Training not found")
    
    for key, value in training_data.dict().items():
        setattr(training_obj, key, value)
    
    await db.commit()
    
    return {"message": "Training updated successfully"}

@router.delete("/trainings/{training_id}")
async def delete_training(
    training_id: int,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Delete training"""
    training = await db.execute(
        select(TrainingDetail).where(TrainingDetail.id == training_id)
    )
    training_obj = training.scalars().first()
    
    if not training_obj:
        raise HTTPException(status_code=404, detail="Training not found")
    
    await db.delete(training_obj)
    await db.commit()
    
    return {"message": "Training deleted successfully"}

@router.post("/trainings/{training_id}/assign")
async def assign_training(
    training_id: int,
    assign_data: TrainingAssignRequest,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Assign training to employees (admin override)"""
    training = await db.execute(
        select(TrainingDetail).where(TrainingDetail.id == training_id)
    )
    if not training.scalars().first():
        raise HTTPException(status_code=404, detail="Training not found")
    
    assignments = []
    for emp_id in assign_data.employee_empids:
        assignment = TrainingAssignment(
            training_id=training_id,
            employee_empid=emp_id,
            manager_empid=assign_data.manager_empid
        )
        assignments.append(assignment)
    
    db.add_all(assignments)
    await db.commit()
    
    return {"message": f"Training assigned to {len(assignments)} employees"}

# ==================== SKILLS MANAGEMENT ====================

@router.get("/skills/competencies")
async def get_all_competencies(
    employee_empid: Optional[str] = Query(None),
    skill: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Get all competencies (system-wide)"""
    query = select(EmployeeCompetency)
    
    if employee_empid:
        query = query.where(EmployeeCompetency.employee_empid == employee_empid)
    if skill:
        query = query.where(EmployeeCompetency.skill.ilike(f"%{skill}%"))
    
    competencies_result = await db.execute(query)
    competencies = competencies_result.scalars().all()
    
    competencies_list = []
    for comp in competencies:
        # Determine status
        current = comp.current_expertise or ""
        target = comp.target_expertise or ""
        status_val = "Error"
        if current and target:
            try:
                current_num = int(current.replace("L", "")) if current.startswith("L") else 0
                target_num = int(target.replace("L", "")) if target.startswith("L") else 0
                status_val = "Met" if current_num >= target_num else "Gap"
            except:
                status_val = "Error"
        
        competencies_list.append({
            "id": comp.id,
            "employee_empid": comp.employee_empid,
            "employee_name": comp.employee_name,
            "skill": comp.skill,
            "competency": comp.competency,
            "current_expertise": comp.current_expertise,
            "target_expertise": comp.target_expertise,
            "status": status_val,
            "department": comp.department
        })
    
    return {"competencies": competencies_list, "total": len(competencies_list)}

@router.put("/skills/competencies/{competency_id}")
async def update_competency(
    competency_id: int,
    skill_data: SkillUpdateAdmin,
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Update any employee's skill (admin override)"""
    competency = await db.execute(
        select(EmployeeCompetency).where(EmployeeCompetency.id == competency_id)
    )
    comp_obj = competency.scalars().first()
    
    if not comp_obj:
        raise HTTPException(status_code=404, detail="Competency not found")
    
    comp_obj.current_expertise = skill_data.current_expertise
    comp_obj.target_expertise = skill_data.target_expertise
    
    await db.commit()
    
    return {"message": "Skill updated successfully"}

@router.get("/skills/gap-analysis")
async def get_skill_gap_analysis(
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Get system-wide skill gap analysis"""
    all_competencies = await db.execute(select(EmployeeCompetency))
    competencies = all_competencies.scalars().all()
    
    total = len(competencies)
    met = 0
    gap = 0
    error = 0
    
    for comp in competencies:
        current = comp.current_expertise or ""
        target = comp.target_expertise or ""
        if not current or not target:
            error += 1
            continue
        
        try:
            current_num = int(current.replace("L", "")) if current.startswith("L") else 0
            target_num = int(target.replace("L", "")) if target.startswith("L") else 0
            if current_num >= target_num:
                met += 1
            else:
                gap += 1
        except:
            error += 1
    
    gap_percentage = (gap / total * 100) if total > 0 else 0
    
    return {
        "total_skills": total,
        "skills_met": met,
        "skills_gap": gap,
        "gap_percentage": round(gap_percentage, 2)
    }

# ==================== ANALYTICS ====================

@router.get("/analytics/overview")
async def get_system_analytics(
    current_user: dict = Depends(get_current_active_admin),
    db: AsyncSession = Depends(get_db_async)
):
    """Get system-wide analytics - matches dashboard data exactly"""
    # User statistics - using same logic as dashboard endpoint
    total_users = await db.execute(select(func.count(User.id)))
    total_users_count = total_users.scalar() or 0
    
    # Count admins (users who exist in Admin table)
    admins_result = await db.execute(select(func.count(Admin.id)))
    admins_count = admins_result.scalar() or 0
    
    # Count managers - distinct users who exist in User table AND appear as manager_empid
    # Join with User table to ensure we only count actual users
    managers_query = select(func.count(func.distinct(ManagerEmployee.manager_empid))).join(
        User, ManagerEmployee.manager_empid == User.username
    )
    managers_result = await db.execute(managers_query)
    managers_count = managers_result.scalar() or 0
    
    # Count employees - distinct users who exist in User table AND appear as employee_empid
    # Join with User table to ensure we only count actual users
    employees_query = select(func.count(func.distinct(ManagerEmployee.employee_empid))).join(
        User, ManagerEmployee.employee_empid == User.username
    )
    employees_result = await db.execute(employees_query)
    employees_count = employees_result.scalar() or 0
    
    # Training statistics
    total_trainings = await db.execute(select(func.count(TrainingDetail.id)))
    trainings_count = total_trainings.scalar() or 0
    
    total_assignments = await db.execute(select(func.count(TrainingAssignment.id)))
    assignments_count = total_assignments.scalar() or 0
    
    # Completion rate
    completed_assignments = await db.execute(
        select(func.count(TrainingAttendance.id)).where(TrainingAttendance.attended == True)
    )
    completed = completed_assignments.scalar() or 0
    completion_rate = (completed / assignments_count * 100) if assignments_count > 0 else 0
    
    # Skill statistics
    total_skills = await db.execute(select(func.count(EmployeeCompetency.id)))
    skills_count = total_skills.scalar() or 0
    
    # Pending requests
    pending_requests = await db.execute(
        select(func.count(TrainingRequest.id)).where(TrainingRequest.status == 'pending')
    )
    pending_requests_count = pending_requests.scalar() or 0
    
    # Active trainers
    trainers_mgr_result = await db.execute(
        select(func.count(func.distinct(ManagerEmployee.manager_empid))).where(
            ManagerEmployee.manager_is_trainer == True
        )
    )
    trainers_emp_result = await db.execute(
        select(func.count(func.distinct(ManagerEmployee.employee_empid))).where(
            ManagerEmployee.employee_is_trainer == True
        )
    )
    active_trainers_count = (trainers_mgr_result.scalar() or 0) + (trainers_emp_result.scalar() or 0)
    
    return {
        "user_statistics": {
            "total_users": total_users_count,
            "managers": managers_count,
            "employees": employees_count
        },
        "training_statistics": {
            "total_trainings": trainings_count,
            "total_assignments": assignments_count,
            "completion_rate": round(completion_rate, 2)
        },
        "skill_statistics": {
            "total_competencies": skills_count
        },
        "additional_metrics": {
            "pending_requests": pending_requests_count,
            "active_trainers": active_trainers_count
        }
    }

