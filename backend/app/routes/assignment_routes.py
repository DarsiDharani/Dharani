"""
Assignment Routes Module

Purpose: API routes for training assignment management
Features:
- Assign trainings to employees (managers only)
- Get assignments for current user
- Get team assignments (managers only)
- Delete assignments

Endpoints:
- POST /assignments/: Assign training to employee
- GET /assignments/my: Get current user's assignments
- GET /assignments/manager/team: Get team assignments (manager only)
- DELETE /assignments/{id}: Delete assignment

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import date, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from sqlalchemy.future import select
from sqlalchemy import delete

from app.database import get_db_async
from app import models
from app.auth_utils import get_current_active_user # Using your auth dependency

router = APIRouter(
    prefix="/assignments",
    tags=["Assignments"]
)

class AssignmentCreate(BaseModel):
    """Request schema for creating a training assignment"""
    training_id: int
    employee_username: str

@router.post("/", status_code=201)
async def assign_training_to_employee(
    assignment: AssignmentCreate,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user) # Get the logged-in manager
):
    """
    Creates an assignment record linking a training to an employee.
    Called by the manager's dashboard.
    """
    manager_username = current_user.get("username")

    # Verify training exists
    training_stmt = select(models.TrainingDetail).where(
        models.TrainingDetail.id == assignment.training_id
    )
    training_result = await db.execute(training_stmt)
    training = training_result.scalar_one_or_none()
    
    if not training:
        raise HTTPException(
            status_code=404,
            detail="Training not found"
        )
    
    # Verify employee exists in manager_employee relationship
    employee_check_stmt = select(models.ManagerEmployee).where(
        models.ManagerEmployee.employee_empid == assignment.employee_username,
        models.ManagerEmployee.manager_empid == manager_username
    )
    employee_check_result = await db.execute(employee_check_stmt)
    if not employee_check_result.scalar_one_or_none():
        raise HTTPException(
            status_code=403,
            detail="You can only assign trainings to employees in your team"
        )
    
    # Check if assignment already exists
    existing_assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == assignment.training_id,
        models.TrainingAssignment.employee_empid == assignment.employee_username
    )
    existing_assignment_result = await db.execute(existing_assignment_stmt)
    if existing_assignment_result.scalar_one_or_none():
        raise HTTPException(
            status_code=400, 
            detail="This training is already assigned to this employee"
        )

    try:
        # Create the new assignment record
        db_assignment = models.TrainingAssignment(
            training_id=assignment.training_id,
            employee_empid=assignment.employee_username,
            manager_empid=manager_username
        )
        db.add(db_assignment)
        await db.commit()
        await db.refresh(db_assignment)
        
        return {"message": "Training assigned successfully"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign training: {str(e)}"
        )

@router.get("/my")
async def get_my_assigned_trainings(
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Returns training details for trainings assigned to the current logged-in user (employee).
    Returns all assigned trainings, regardless of attendance status.
    Note: Access to assignments and feedback is still restricted to attended trainings only.
    """
    employee_username = current_user.get("username")

    # Get all assigned trainings (show all, not just those with attendance marked)
    stmt = select(models.TrainingDetail).join(
        models.TrainingAssignment,
        models.TrainingAssignment.training_id == models.TrainingDetail.id
    ).where(
        models.TrainingAssignment.employee_empid == employee_username
    )

    result = await db.execute(stmt)
    trainings = result.scalars().all()

    # Serialize minimal fields
    def to_iso(val):
        if isinstance(val, (date, datetime)):
            return val.isoformat()
        if isinstance(val, str):
            try:
                # Try parse ISO-like strings
                return datetime.fromisoformat(val).date().isoformat()
            except Exception:
                return val
        return None

    def serialize(td: models.TrainingDetail):
        return {
            "id": td.id,
            "division": td.division,
            "department": td.department,
            "competency": td.competency,
            "skill": td.skill,
            "training_name": td.training_name,
            "training_topics": td.training_topics,
            "prerequisites": td.prerequisites,
            "skill_category": td.skill_category,
            "trainer_name": td.trainer_name,
            "email": td.email,
            "training_date": to_iso(td.training_date),
            "duration": td.duration,
            "time": td.time,
            "training_type": td.training_type,
            "seats": td.seats,
            "assessment_details": td.assessment_details,
        }

    return [serialize(t) for t in trainings]

@router.get("/manager/team")
async def get_team_assigned_trainings(
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Returns all training assignments for the manager's team members.
    Returns a list of assignments with training_id and employee_empid for duplicate checking.
    """
    manager_username = current_user.get("username")
    
    # Get all team member IDs for this manager
    team_members_stmt = select(models.ManagerEmployee.employee_empid).where(
        models.ManagerEmployee.manager_empid == manager_username
    )
    team_result = await db.execute(team_members_stmt)
    team_member_ids = [row[0] for row in team_result.all()]
    
    if not team_member_ids:
        return []
    
    # Get all assignments for team members managed by this manager
    assignments_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.employee_empid.in_(team_member_ids),
        models.TrainingAssignment.manager_empid == manager_username
    )
    assignments_result = await db.execute(assignments_stmt)
    assignments = assignments_result.scalars().all()
    
    # Return simple structure for duplicate checking
    return [
        {
            "training_id": assignment.training_id,
            "employee_empid": assignment.employee_empid
        }
        for assignment in assignments
    ]

@router.delete("/{training_id}/{employee_empid}", status_code=200)
async def delete_assignment(
    training_id: int,
    employee_empid: str,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Deletes an assignment record. Used when manager wants to reassign a training.
    Only the manager who assigned the training can delete it.
    """
    manager_username = current_user.get("username")
    if not manager_username:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )
    
    # Find the assignment
    assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == training_id,
        models.TrainingAssignment.employee_empid == employee_empid,
        models.TrainingAssignment.manager_empid == manager_username
    )
    assignment_result = await db.execute(assignment_stmt)
    assignment = assignment_result.scalar_one_or_none()
    
    if not assignment:
        raise HTTPException(
            status_code=404,
            detail="Assignment not found or you are not authorized to delete it"
        )
    
    # Delete the assignment
    delete_stmt = delete(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == training_id,
        models.TrainingAssignment.employee_empid == employee_empid,
        models.TrainingAssignment.manager_empid == manager_username
    )
    await db.execute(delete_stmt)
    await db.commit()
    
    return {"message": "Assignment deleted successfully"}

@router.get("/training/{training_id}/candidates")
async def get_training_candidates(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Returns list of all candidates assigned to a specific training.
    Includes their attendance status.
    Only accessible by the trainer of that training.
    """
    trainer_username = current_user.get("username")
    if not trainer_username:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )
    
    # Verify the training exists and get trainer name
    training_stmt = select(models.TrainingDetail).where(
        models.TrainingDetail.id == training_id
    )
    training_result = await db.execute(training_stmt)
    training = training_result.scalar_one_or_none()
    
    if not training:
        raise HTTPException(
            status_code=404,
            detail="Training not found"
        )
    
    # Verify the current user is the trainer for this training
    trainer_name = str(training.trainer_name or "").strip()
    if not trainer_name:
        raise HTTPException(
            status_code=403,
            detail="Training has no trainer assigned"
        )
    
    # Get employee/manager name for matching
    employee_result = await db.execute(
        select(models.ManagerEmployee.employee_name).where(
            models.ManagerEmployee.employee_empid == trainer_username
        ).distinct()
    )
    employee_name = employee_result.scalar_one_or_none()
    
    manager_result = await db.execute(
        select(models.ManagerEmployee.manager_name).where(
            models.ManagerEmployee.manager_empid == trainer_username
        ).distinct()
    )
    manager_name = manager_result.scalar_one_or_none()
    
    display_name = employee_name or manager_name
    trainer_username_lower = str(trainer_username).lower().strip()
    display_name_lower = (display_name or "").lower().strip() if display_name else ""
    trainer_name_lower = trainer_name.lower().strip()
    
    # Use improved matching logic (same as my-trainings endpoint and shared_content_routes)
    # Split trainer_name by comma (Excel format) or newline, then check each
    trainer_names = []
    if ',' in trainer_name:
        trainer_names = [t.strip() for t in trainer_name.split(',') if t.strip()]
    elif '\n' in trainer_name:
        trainer_names = [t.strip() for t in trainer_name.split('\n') if t.strip()]
    else:
        trainer_names = [trainer_name.strip()]
    
    # Check multiple matching strategies for each trainer name:
    is_trainer = False
    for single_trainer_name in trainer_names:
        single_trainer_name_lower = single_trainer_name.lower().strip()
        
        # 1. Exact match with username
        if single_trainer_name_lower == trainer_username_lower:
            is_trainer = True
            break
        
        # 2. Exact match with display name
        if display_name_lower and single_trainer_name_lower == display_name_lower:
            is_trainer = True
            break
        
        # 3. Contains username (for partial matches)
        if trainer_username_lower in single_trainer_name_lower or single_trainer_name_lower in trainer_username_lower:
            is_trainer = True
            break
        
        # 4. Contains display name (for partial matches)
        if display_name_lower:
            if display_name_lower in single_trainer_name_lower or single_trainer_name_lower in display_name_lower:
                is_trainer = True
                break
        
        # 5. Check if any part of display name matches (e.g., "Sharib Jawed" matches "Sharib" or "Jawed")
        if display_name_lower:
            name_parts = [part.strip() for part in display_name_lower.split() if len(part.strip()) > 2]
            for part in name_parts:
                if part in single_trainer_name_lower or single_trainer_name_lower in part:
                    is_trainer = True
                    break
            if is_trainer:
                break
    
    if not is_trainer:
        raise HTTPException(
            status_code=403,
            detail="Only the trainer of this training can view candidates"
        )
    
    # Get all assignments for this training
    assignments_stmt = select(
        models.TrainingAssignment.employee_empid,
        models.ManagerEmployee.employee_name
    ).join(
        models.ManagerEmployee,
        models.TrainingAssignment.employee_empid == models.ManagerEmployee.employee_empid
    ).where(
        models.TrainingAssignment.training_id == training_id
    ).distinct()
    
    assignments_result = await db.execute(assignments_stmt)
    assignments = assignments_result.all()
    
    # Get attendance records for this training
    attendance_stmt = select(models.TrainingAttendance).where(
        models.TrainingAttendance.training_id == training_id
    )
    attendance_result = await db.execute(attendance_stmt)
    attendance_records = {rec.employee_empid: rec.attended for rec in attendance_result.scalars().all()}
    
    # Build response with candidate info and attendance status
    candidates = []
    for empid, empname in assignments:
        candidates.append({
            "employee_empid": empid,
            "employee_name": empname or empid,
            "attended": attendance_records.get(empid, False)
        })
    
    return candidates

class AttendanceMarkRequest(BaseModel):
    """Request schema for marking attendance"""
    candidate_empids: list[str]  # List of employee IDs who attended

@router.post("/training/{training_id}/attendance")
async def mark_training_attendance(
    training_id: int,
    attendance_data: AttendanceMarkRequest,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Marks attendance for candidates who attended the training.
    Only accessible by the trainer of that training.
    """
    trainer_username = current_user.get("username")
    if not trainer_username:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )
    
    # Verify the training exists and get trainer name
    training_stmt = select(models.TrainingDetail).where(
        models.TrainingDetail.id == training_id
    )
    training_result = await db.execute(training_stmt)
    training = training_result.scalar_one_or_none()
    
    if not training:
        raise HTTPException(
            status_code=404,
            detail="Training not found"
        )
    
    # Verify the current user is the trainer for this training
    trainer_name = str(training.trainer_name or "").strip()
    if not trainer_name:
        raise HTTPException(
            status_code=403,
            detail="Training has no trainer assigned"
        )
    
    # Get employee/manager name for matching
    employee_result = await db.execute(
        select(models.ManagerEmployee.employee_name).where(
            models.ManagerEmployee.employee_empid == trainer_username
        ).distinct()
    )
    employee_name = employee_result.scalar_one_or_none()
    
    manager_result = await db.execute(
        select(models.ManagerEmployee.manager_name).where(
            models.ManagerEmployee.manager_empid == trainer_username
        ).distinct()
    )
    manager_name = manager_result.scalar_one_or_none()
    
    display_name = employee_name or manager_name
    trainer_username_lower = str(trainer_username).lower().strip()
    display_name_lower = (display_name or "").lower().strip() if display_name else ""
    
    # Use improved matching logic (same as my-trainings endpoint and shared_content_routes)
    # Split trainer_name by comma (Excel format) or newline, then check each
    trainer_names = []
    if ',' in trainer_name:
        trainer_names = [t.strip() for t in trainer_name.split(',') if t.strip()]
    elif '\n' in trainer_name:
        trainer_names = [t.strip() for t in trainer_name.split('\n') if t.strip()]
    else:
        trainer_names = [trainer_name.strip()]
    
    # Check multiple matching strategies for each trainer name:
    is_trainer = False
    for single_trainer_name in trainer_names:
        single_trainer_name_lower = single_trainer_name.lower().strip()
        
        # 1. Exact match with username
        if single_trainer_name_lower == trainer_username_lower:
            is_trainer = True
            break
        
        # 2. Exact match with display name
        if display_name_lower and single_trainer_name_lower == display_name_lower:
            is_trainer = True
            break
        
        # 3. Contains username (for partial matches)
        if trainer_username_lower in single_trainer_name_lower or single_trainer_name_lower in trainer_username_lower:
            is_trainer = True
            break
        
        # 4. Contains display name (for partial matches)
        if display_name_lower:
            if display_name_lower in single_trainer_name_lower or single_trainer_name_lower in display_name_lower:
                is_trainer = True
                break
        
        # 5. Check if any part of display name matches (e.g., "Sharib Jawed" matches "Sharib" or "Jawed")
        if display_name_lower:
            name_parts = [part.strip() for part in display_name_lower.split() if len(part.strip()) > 2]
            for part in name_parts:
                if part in single_trainer_name_lower or single_trainer_name_lower in part:
                    is_trainer = True
                    break
            if is_trainer:
                break
    
    if not is_trainer:
        raise HTTPException(
            status_code=403,
            detail="Only the trainer of this training can mark attendance"
        )
    
    # Get all assignments for this training to validate candidate IDs
    assignments_stmt = select(models.TrainingAssignment.employee_empid).where(
        models.TrainingAssignment.training_id == training_id
    )
    assignments_result = await db.execute(assignments_stmt)
    valid_empids = {row[0] for row in assignments_result.all()}
    
    # Validate that all provided employee IDs are assigned to this training
    invalid_empids = set(attendance_data.candidate_empids) - valid_empids
    if invalid_empids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid employee IDs: {', '.join(invalid_empids)}. These employees are not assigned to this training."
        )
    
    # Delete existing attendance records for this training
    delete_stmt = delete(models.TrainingAttendance).where(
        models.TrainingAttendance.training_id == training_id
    )
    await db.execute(delete_stmt)
    
    # Create new attendance records
    for empid in attendance_data.candidate_empids:
        attendance = models.TrainingAttendance(
            training_id=training_id,
            employee_empid=empid,
            attended=True
        )
        db.add(attendance)
    
    # Also mark non-attended candidates as False (optional, but helps with tracking)
    for empid in valid_empids:
        if empid not in attendance_data.candidate_empids:
            attendance = models.TrainingAttendance(
                training_id=training_id,
                employee_empid=empid,
                attended=False
            )
            db.add(attendance)
    
    await db.commit()
    
    return {
        "message": "Attendance marked successfully",
        "attended_count": len(attendance_data.candidate_empids),
        "total_assigned": len(valid_empids)
    }