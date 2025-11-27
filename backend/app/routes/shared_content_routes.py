"""
Shared Content Routes Module

Purpose: API routes for shared assignments, feedback, and performance tracking
Features:
- Trainers can share assignments and feedback forms
- Employees can take shared assignments and submit feedback
- Managers can view team submissions and provide performance feedback
- Assignment scoring and result tracking

Endpoints:
- POST /shared-content/assignments: Share an assignment (trainer)
- GET /shared-content/assignments/{trainingId}: Get shared assignment
- POST /shared-content/assignments/submit: Submit assignment answers
- GET /shared-content/assignments/{trainingId}/result: Get assignment results
- POST /shared-content/feedback: Share a feedback form (trainer)
- POST /shared-content/feedback/submit: Submit feedback
- GET /shared-content/manager/team/assignments: Get team assignment submissions
- GET /shared-content/manager/team/feedback: Get team feedback submissions
- POST /shared-content/manager/performance-feedback: Provide performance feedback

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from app.database import get_db_async
from app import models
from app.auth_utils import get_current_active_user, get_current_active_manager

router = APIRouter(
    prefix="/shared-content",
    tags=["Shared Content"]
)

# --- Pydantic Schemas ---

class AssignmentQuestionOption(BaseModel):
    text: str
    isCorrect: bool

class AssignmentQuestion(BaseModel):
    text: str
    helperText: Optional[str] = ""
    type: str  # single-choice, multiple-choice, text, etc.
    options: List[AssignmentQuestionOption] = []

class SharedAssignmentCreate(BaseModel):
    training_id: int
    title: str
    description: Optional[str] = ""
    questions: List[AssignmentQuestion]

class SharedAssignmentResponse(BaseModel):
    id: int
    training_id: int
    trainer_username: str
    title: str
    description: Optional[str]
    questions: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class FeedbackQuestion(BaseModel):
    text: str
    options: List[str]
    isDefault: bool = False

class SharedFeedbackCreate(BaseModel):
    training_id: int
    defaultQuestions: Optional[List[Dict[str, Any]]] = []
    customQuestions: List[FeedbackQuestion]

class SharedFeedbackResponse(BaseModel):
    id: int
    training_id: int
    trainer_username: str
    defaultQuestions: List[Dict[str, Any]]
    customQuestions: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# --- Routes ---

@router.post("/assignments", response_model=SharedAssignmentResponse, status_code=status.HTTP_201_CREATED)
async def share_assignment(
    assignment_data: SharedAssignmentCreate,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows a trainer to share an assignment for a training they have scheduled.
    """
    trainer_username = current_user.get("username")
    if not trainer_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training exists
    training_stmt = select(models.TrainingDetail).where(
        models.TrainingDetail.id == assignment_data.training_id
    )
    training_result = await db.execute(training_stmt)
    training = training_result.scalar_one_or_none()
    
    if not training:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training not found"
        )

    # Verify the current user is the trainer for this training
    # trainer_name can contain multiple trainers separated by commas (from Excel) or newlines
    trainer_name = str(training.trainer_name or "").strip()
    if not trainer_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Training has no trainer assigned"
        )
    
    # Get employee/manager name from ManagerEmployee table for additional matching
    # Check both employee_empid and manager_empid to support both employees and managers
    employee_result = await db.execute(
        select(models.ManagerEmployee.employee_name).where(
            models.ManagerEmployee.employee_empid == trainer_username
        )
    )
    employee_name = employee_result.scalars().first()
    
    manager_result = await db.execute(
        select(models.ManagerEmployee.manager_name).where(
            models.ManagerEmployee.manager_empid == trainer_username
        )
    )
    manager_name = manager_result.scalars().first()
    
    display_name = employee_name or manager_name
    
    trainer_username_lower = str(trainer_username).lower().strip()
    display_name_lower = (display_name or "").lower().strip() if display_name else ""
    
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
        trainer_name_lower = single_trainer_name.lower().strip()
        
        # 1. Exact match with username
        if trainer_name_lower == trainer_username_lower:
            is_trainer = True
            break
        
        # 2. Exact match with display name
        if display_name_lower and trainer_name_lower == display_name_lower:
            is_trainer = True
            break
        
        # 3. Contains username (for partial matches)
        if trainer_username_lower in trainer_name_lower or trainer_name_lower in trainer_username_lower:
            is_trainer = True
            break
        
        # 4. Contains display name (for partial matches)
        if display_name_lower:
            if display_name_lower in trainer_name_lower or trainer_name_lower in display_name_lower:
                is_trainer = True
                break
        
        # 5. Check if any part of display name matches (e.g., "Sharib Jawed" matches "Sharib" or "Jawed")
        if display_name_lower:
            name_parts = [part.strip() for part in display_name_lower.split() if len(part.strip()) > 2]
            for part in name_parts:
                if part in trainer_name_lower or trainer_name_lower in part:
                    is_trainer = True
                    break
            if is_trainer:
                break
    
    if not is_trainer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only share assignments for trainings you have scheduled"
        )

    # Convert questions to JSON string
    questions_json = json.dumps([q.dict() for q in assignment_data.questions])

    # Check if assignment already exists for this training (update existing)
    # Get most recent if multiple exist
    existing_stmt = select(models.SharedAssignment).where(
        models.SharedAssignment.training_id == assignment_data.training_id
    ).order_by(models.SharedAssignment.updated_at.desc())
    existing_result = await db.execute(existing_stmt)
    existing_assignment = existing_result.scalars().first()

    if existing_assignment:
        # Update existing assignment
        existing_assignment.title = assignment_data.title
        existing_assignment.description = assignment_data.description
        existing_assignment.assignment_data = questions_json
        existing_assignment.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing_assignment)
        
        # Parse and return
        try:
            questions_data = json.loads(existing_assignment.assignment_data)
        except (json.JSONDecodeError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse assignment data: {str(e)}"
            )
        return SharedAssignmentResponse(
            id=existing_assignment.id,
            training_id=existing_assignment.training_id,
            trainer_username=existing_assignment.trainer_username,
            title=existing_assignment.title,
            description=existing_assignment.description,
            questions=questions_data,
            created_at=existing_assignment.created_at,
            updated_at=existing_assignment.updated_at
        )
    else:
        # Create new assignment
        new_assignment = models.SharedAssignment(
            training_id=assignment_data.training_id,
            trainer_username=trainer_username,
            title=assignment_data.title,
            description=assignment_data.description,
            assignment_data=questions_json
        )
        db.add(new_assignment)
        await db.commit()
        await db.refresh(new_assignment)

        # Parse and return
        try:
            questions_data = json.loads(new_assignment.assignment_data)
        except (json.JSONDecodeError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse assignment data: {str(e)}"
            )
        return SharedAssignmentResponse(
            id=new_assignment.id,
            training_id=new_assignment.training_id,
            trainer_username=new_assignment.trainer_username,
            title=new_assignment.title,
            description=new_assignment.description,
            questions=questions_data,
            created_at=new_assignment.created_at,
            updated_at=new_assignment.updated_at
        )

@router.post("/feedback", response_model=SharedFeedbackResponse, status_code=status.HTTP_201_CREATED)
async def share_feedback(
    feedback_data: SharedFeedbackCreate,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows a trainer to share feedback form for a training they have scheduled.
    """
    trainer_username = current_user.get("username")
    if not trainer_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training exists
    training_stmt = select(models.TrainingDetail).where(
        models.TrainingDetail.id == feedback_data.training_id
    )
    training_result = await db.execute(training_stmt)
    training = training_result.scalar_one_or_none()
    
    if not training:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training not found"
        )

    # Verify the current user is the trainer for this training
    # trainer_name can contain multiple trainers separated by commas (from Excel) or newlines
    trainer_name = str(training.trainer_name or "").strip()
    if not trainer_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Training has no trainer assigned"
        )
    
    # Get employee/manager name from ManagerEmployee table for additional matching
    # Check both employee_empid and manager_empid to support both employees and managers
    employee_result = await db.execute(
        select(models.ManagerEmployee.employee_name).where(
            models.ManagerEmployee.employee_empid == trainer_username
        )
    )
    employee_name = employee_result.scalars().first()
    
    manager_result = await db.execute(
        select(models.ManagerEmployee.manager_name).where(
            models.ManagerEmployee.manager_empid == trainer_username
        )
    )
    manager_name = manager_result.scalars().first()
    
    display_name = employee_name or manager_name
    
    trainer_username_lower = str(trainer_username).lower().strip()
    display_name_lower = (display_name or "").lower().strip() if display_name else ""
    
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
        trainer_name_lower = single_trainer_name.lower().strip()
        
        # 1. Exact match with username
        if trainer_name_lower == trainer_username_lower:
            is_trainer = True
            break
        
        # 2. Exact match with display name
        if display_name_lower and trainer_name_lower == display_name_lower:
            is_trainer = True
            break
        
        # 3. Contains username (for partial matches)
        if trainer_username_lower in trainer_name_lower or trainer_name_lower in trainer_username_lower:
            is_trainer = True
            break
        
        # 4. Contains display name (for partial matches)
        if display_name_lower:
            if display_name_lower in trainer_name_lower or trainer_name_lower in display_name_lower:
                is_trainer = True
                break
        
        # 5. Check if any part of display name matches (e.g., "Sharib Jawed" matches "Sharib" or "Jawed")
        if display_name_lower:
            name_parts = [part.strip() for part in display_name_lower.split() if len(part.strip()) > 2]
            for part in name_parts:
                if part in trainer_name_lower or trainer_name_lower in part:
                    is_trainer = True
                    break
            if is_trainer:
                break
    
    if not is_trainer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only share feedback for trainings you have scheduled"
        )

    # Convert feedback data to JSON string
    feedback_json = json.dumps({
        "defaultQuestions": feedback_data.defaultQuestions or [],
        "customQuestions": [q.dict() for q in feedback_data.customQuestions]
    })

    # Check if feedback already exists for this training (update existing)
    # Get most recent if multiple exist
    existing_stmt = select(models.SharedFeedback).where(
        models.SharedFeedback.training_id == feedback_data.training_id
    ).order_by(models.SharedFeedback.updated_at.desc())
    existing_result = await db.execute(existing_stmt)
    existing_feedback = existing_result.scalars().first()

    if existing_feedback:
        # Update existing feedback
        existing_feedback.feedback_data = feedback_json
        existing_feedback.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing_feedback)
        
        # Parse and return
        feedback_data_parsed = json.loads(existing_feedback.feedback_data)
        return SharedFeedbackResponse(
            id=existing_feedback.id,
            training_id=existing_feedback.training_id,
            trainer_username=existing_feedback.trainer_username,
            defaultQuestions=feedback_data_parsed.get("defaultQuestions", []),
            customQuestions=feedback_data_parsed.get("customQuestions", []),
            created_at=existing_feedback.created_at,
            updated_at=existing_feedback.updated_at
        )
    else:
        # Create new feedback
        new_feedback = models.SharedFeedback(
            training_id=feedback_data.training_id,
            trainer_username=trainer_username,
            feedback_data=feedback_json
        )
        db.add(new_feedback)
        await db.commit()
        await db.refresh(new_feedback)

        # Parse and return
        feedback_data_parsed = json.loads(new_feedback.feedback_data)
        return SharedFeedbackResponse(
            id=new_feedback.id,
            training_id=new_feedback.training_id,
            trainer_username=new_feedback.trainer_username,
            defaultQuestions=feedback_data_parsed.get("defaultQuestions", []),
            customQuestions=feedback_data_parsed.get("customQuestions", []),
            created_at=new_feedback.created_at,
            updated_at=new_feedback.updated_at
        )

@router.get("/assignments/{training_id}", response_model=Optional[SharedAssignmentResponse])
async def get_shared_assignment(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows engineers to retrieve shared assignment for a training assigned to them.
    Only accessible if the employee attended the training (attendance was marked).
    """
    employee_username = current_user.get("username")
    if not employee_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training is assigned to this employee
    # Use first() instead of scalar_one_or_none() to handle cases where multiple assignments exist
    assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == training_id,
        models.TrainingAssignment.employee_empid == employee_username
    ).order_by(models.TrainingAssignment.id.desc())
    assignment_result = await db.execute(assignment_stmt)
    assignment = assignment_result.scalars().first()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access assignments for trainings assigned to you"
        )

    # Verify the employee attended the training
    attendance_stmt = select(models.TrainingAttendance).where(
        models.TrainingAttendance.training_id == training_id,
        models.TrainingAttendance.employee_empid == employee_username,
        models.TrainingAttendance.attended == True
    )
    attendance_result = await db.execute(attendance_stmt)
    attendance = attendance_result.scalar_one_or_none()

    if not attendance:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access assignments for trainings you attended. Please contact your trainer if you believe this is an error."
        )

    # Get the shared assignment (get most recent if multiple exist)
    assignment_stmt = select(models.SharedAssignment).where(
        models.SharedAssignment.training_id == training_id
    ).order_by(models.SharedAssignment.updated_at.desc())
    result = await db.execute(assignment_stmt)
    shared_assignment = result.scalars().first()

    if not shared_assignment:
        return None

    # Parse and return
    questions_data = json.loads(shared_assignment.assignment_data)
    return SharedAssignmentResponse(
        id=shared_assignment.id,
        training_id=shared_assignment.training_id,
        trainer_username=shared_assignment.trainer_username,
        title=shared_assignment.title,
        description=shared_assignment.description,
        questions=questions_data,
        created_at=shared_assignment.created_at,
        updated_at=shared_assignment.updated_at
    )

@router.get("/feedback/{training_id}", response_model=Optional[SharedFeedbackResponse])
async def get_shared_feedback(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows engineers to retrieve shared feedback form for a training assigned to them.
    Only accessible if the employee attended the training (attendance was marked).
    """
    employee_username = current_user.get("username")
    if not employee_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training is assigned to this employee
    assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == training_id,
        models.TrainingAssignment.employee_empid == employee_username
    )
    assignment_result = await db.execute(assignment_stmt)
    assignment = assignment_result.scalar_one_or_none()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access feedback for trainings assigned to you"
        )

    # Verify the employee attended the training
    attendance_stmt = select(models.TrainingAttendance).where(
        models.TrainingAttendance.training_id == training_id,
        models.TrainingAttendance.employee_empid == employee_username,
        models.TrainingAttendance.attended == True
    )
    attendance_result = await db.execute(attendance_stmt)
    attendance = attendance_result.scalar_one_or_none()

    if not attendance:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access feedback for trainings you attended. Please contact your trainer if you believe this is an error."
        )

    # Get the shared feedback (get most recent if multiple exist)
    feedback_stmt = select(models.SharedFeedback).where(
        models.SharedFeedback.training_id == training_id
    ).order_by(models.SharedFeedback.updated_at.desc())
    result = await db.execute(feedback_stmt)
    shared_feedback = result.scalars().first()

    if not shared_feedback:
        return None

    # Parse and return
    feedback_data_parsed = json.loads(shared_feedback.feedback_data)
    return SharedFeedbackResponse(
        id=shared_feedback.id,
        training_id=shared_feedback.training_id,
        trainer_username=shared_feedback.trainer_username,
        defaultQuestions=feedback_data_parsed.get("defaultQuestions", []),
        customQuestions=feedback_data_parsed.get("customQuestions", []),
        created_at=shared_feedback.created_at,
        updated_at=shared_feedback.updated_at
    )

@router.get("/trainer/assignments/{training_id}", response_model=Optional[SharedAssignmentResponse])
async def get_shared_assignment_for_trainer(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows trainers to check if assignment is already shared for their training.
    """
    trainer_username = current_user.get("username")
    if not trainer_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training exists
    training_stmt = select(models.TrainingDetail).where(
        models.TrainingDetail.id == training_id
    )
    training_result = await db.execute(training_stmt)
    training = training_result.scalar_one_or_none()
    
    if not training:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training not found"
        )

    # Verify the current user is a trainer for this training
    trainer_name = str(training.trainer_name or "").strip()
    if not trainer_name:
        return None
    
    # Get employee/manager name for matching
    # Check both employee_empid and manager_empid to support both employees and managers
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
        trainer_name_lower = single_trainer_name.lower().strip()
        
        # 1. Exact match with username
        if trainer_name_lower == trainer_username_lower:
            is_trainer = True
            break
        
        # 2. Exact match with display name
        if display_name_lower and trainer_name_lower == display_name_lower:
            is_trainer = True
            break
        
        # 3. Contains username (for partial matches)
        if trainer_username_lower in trainer_name_lower or trainer_name_lower in trainer_username_lower:
            is_trainer = True
            break
        
        # 4. Contains display name (for partial matches)
        if display_name_lower:
            if display_name_lower in trainer_name_lower or trainer_name_lower in display_name_lower:
                is_trainer = True
                break
        
        # 5. Check if any part of display name matches (e.g., "Sharib Jawed" matches "Sharib" or "Jawed")
        if display_name_lower:
            name_parts = [part.strip() for part in display_name_lower.split() if len(part.strip()) > 2]
            for part in name_parts:
                if part in trainer_name_lower or trainer_name_lower in part:
                    is_trainer = True
                    break
            if is_trainer:
                break
    
    if not is_trainer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only check assignments for trainings you have scheduled"
        )

    # Get the shared assignment (get most recent if multiple exist)
    assignment_stmt = select(models.SharedAssignment).where(
        models.SharedAssignment.training_id == training_id
    ).order_by(models.SharedAssignment.updated_at.desc())
    result = await db.execute(assignment_stmt)
    shared_assignment = result.scalars().first()

    if not shared_assignment:
        return None

    # Parse and return
    try:
        questions_data = json.loads(shared_assignment.assignment_data)
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse assignment data: {str(e)}"
        )
    return SharedAssignmentResponse(
        id=shared_assignment.id,
        training_id=shared_assignment.training_id,
        trainer_username=shared_assignment.trainer_username,
        title=shared_assignment.title,
        description=shared_assignment.description,
        questions=questions_data,
        created_at=shared_assignment.created_at,
        updated_at=shared_assignment.updated_at
    )

@router.get("/trainer/feedback/{training_id}", response_model=Optional[SharedFeedbackResponse])
async def get_shared_feedback_for_trainer(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows trainers to check if feedback is already shared for their training.
    """
    trainer_username = current_user.get("username")
    if not trainer_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training exists
    training_stmt = select(models.TrainingDetail).where(
        models.TrainingDetail.id == training_id
    )
    training_result = await db.execute(training_stmt)
    training = training_result.scalar_one_or_none()
    
    if not training:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training not found"
        )

    # Verify the current user is a trainer for this training
    trainer_name = str(training.trainer_name or "").strip()
    if not trainer_name:
        return None
    
    # Get employee/manager name for matching
    # Check both employee_empid and manager_empid to support both employees and managers
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
        trainer_name_lower = single_trainer_name.lower().strip()
        
        # 1. Exact match with username
        if trainer_name_lower == trainer_username_lower:
            is_trainer = True
            break
        
        # 2. Exact match with display name
        if display_name_lower and trainer_name_lower == display_name_lower:
            is_trainer = True
            break
        
        # 3. Contains username (for partial matches)
        if trainer_username_lower in trainer_name_lower or trainer_name_lower in trainer_username_lower:
            is_trainer = True
            break
        
        # 4. Contains display name (for partial matches)
        if display_name_lower:
            if display_name_lower in trainer_name_lower or trainer_name_lower in display_name_lower:
                is_trainer = True
                break
        
        # 5. Check if any part of display name matches (e.g., "Sharib Jawed" matches "Sharib" or "Jawed")
        if display_name_lower:
            name_parts = [part.strip() for part in display_name_lower.split() if len(part.strip()) > 2]
            for part in name_parts:
                if part in trainer_name_lower or trainer_name_lower in part:
                    is_trainer = True
                    break
            if is_trainer:
                break
    
    if not is_trainer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only check feedback for trainings you have scheduled"
        )

    # Get the shared feedback (get most recent if multiple exist)
    feedback_stmt = select(models.SharedFeedback).where(
        models.SharedFeedback.training_id == training_id
    ).order_by(models.SharedFeedback.updated_at.desc())
    result = await db.execute(feedback_stmt)
    shared_feedback = result.scalars().first()

    if not shared_feedback:
        return None

    # Parse and return
    feedback_data_parsed = json.loads(shared_feedback.feedback_data)
    return SharedFeedbackResponse(
        id=shared_feedback.id,
        training_id=shared_feedback.training_id,
        trainer_username=shared_feedback.trainer_username,
        defaultQuestions=feedback_data_parsed.get("defaultQuestions", []),
        customQuestions=feedback_data_parsed.get("customQuestions", []),
        created_at=shared_feedback.created_at,
        updated_at=shared_feedback.updated_at
    )

# --- Assignment Submission Schemas ---

class AnswerSubmission(BaseModel):
    questionIndex: int
    type: str
    selectedOptions: List[int] = []  # For single/multiple choice: indices of selected options
    textAnswer: Optional[str] = ""  # For text-input questions

class AssignmentSubmissionCreate(BaseModel):
    training_id: int
    shared_assignment_id: int
    answers: List[AnswerSubmission]

class QuestionResult(BaseModel):
    questionIndex: int
    isCorrect: bool
    correctAnswers: List[int]  # Indices of correct options
    userAnswers: List[int]  # Indices of user's selected options
    userTextAnswer: Optional[str] = ""  # For text-input questions

class AssignmentResultResponse(BaseModel):
    id: int
    training_id: int
    score: int
    total_questions: int
    correct_answers: int
    question_results: List[QuestionResult]
    submitted_at: datetime

@router.post("/assignments/submit", response_model=AssignmentResultResponse, status_code=status.HTTP_201_CREATED)
async def submit_assignment(
    submission_data: AssignmentSubmissionCreate,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows engineers to submit their answers for an assignment and get evaluated results.
    """
    employee_username = current_user.get("username")
    if not employee_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training is assigned to this employee
    assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == submission_data.training_id,
        models.TrainingAssignment.employee_empid == employee_username
    )
    assignment_result = await db.execute(assignment_stmt)
    assignment = assignment_result.scalar_one_or_none()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only submit assignments for trainings assigned to you"
        )

    # Verify the employee attended the training
    attendance_stmt = select(models.TrainingAttendance).where(
        models.TrainingAttendance.training_id == submission_data.training_id,
        models.TrainingAttendance.employee_empid == employee_username,
        models.TrainingAttendance.attended == True
    )
    attendance_result = await db.execute(attendance_stmt)
    attendance = attendance_result.scalar_one_or_none()

    if not attendance:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only submit assignments for trainings you attended. Please contact your trainer if you believe this is an error."
        )

    # Get the shared assignment
    # Note: This query filters by id (primary key) so should be unique, but using first() for consistency
    shared_stmt = select(models.SharedAssignment).where(
        models.SharedAssignment.id == submission_data.shared_assignment_id,
        models.SharedAssignment.training_id == submission_data.training_id
    )
    shared_result = await db.execute(shared_stmt)
    shared_assignment = shared_result.scalars().first()

    if not shared_assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found"
        )

    # Parse assignment questions
    questions_data = json.loads(shared_assignment.assignment_data)
    total_questions = len(questions_data)
    correct_count = 0
    question_results = []

    # Evaluate answers
    for answer in submission_data.answers:
        if answer.questionIndex >= len(questions_data):
            continue
        
        question = questions_data[answer.questionIndex]
        is_correct = False
        correct_indices = []
        user_indices = answer.selectedOptions.copy()
        
        # Get correct answer indices
        if question.get("type") in ["single-choice", "multiple-choice"]:
            options = question.get("options", [])
            for idx, opt in enumerate(options):
                if opt.get("isCorrect", False):
                    correct_indices.append(idx)
            
            # Check if answer is correct
            if question.get("type") == "single-choice":
                # For single-choice, user should select exactly one option that matches the correct one
                if len(answer.selectedOptions) == 1 and answer.selectedOptions[0] in correct_indices:
                    is_correct = True
            elif question.get("type") == "multiple-choice":
                # For multiple-choice, user's selections must exactly match correct answers
                if set(answer.selectedOptions) == set(correct_indices):
                    is_correct = True
        
        elif question.get("type") == "text-input":
            # For text-input, we'll mark as correct if answer is provided (manual evaluation needed)
            # For now, we'll mark it as needs review
            is_correct = False  # Text answers need manual evaluation
            user_text = answer.textAnswer or ""
        
        if is_correct:
            correct_count += 1
        
        question_results.append({
            "questionIndex": answer.questionIndex,
            "isCorrect": is_correct,
            "correctAnswers": correct_indices,
            "userAnswers": user_indices,
            "userTextAnswer": answer.textAnswer if question.get("type") == "text-input" else None
        })

    # Calculate score (percentage)
    score = int((correct_count / total_questions * 100)) if total_questions > 0 else 0

    # Store submission
    answers_json = json.dumps([a.dict() for a in submission_data.answers])
    submission = models.AssignmentSubmission(
        training_id=submission_data.training_id,
        shared_assignment_id=submission_data.shared_assignment_id,
        employee_empid=employee_username,
        answers_data=answers_json,
        score=score,
        total_questions=total_questions,
        correct_answers=correct_count
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    return AssignmentResultResponse(
        id=submission.id,
        training_id=submission.training_id,
        score=submission.score,
        total_questions=submission.total_questions,
        correct_answers=submission.correct_answers,
        question_results=[QuestionResult(**qr) for qr in question_results],
        submitted_at=submission.submitted_at
    )

@router.get("/assignments/{training_id}/result", response_model=Optional[AssignmentResultResponse])
async def get_assignment_result(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows engineers to retrieve their assignment result for a training.
    """
    employee_username = current_user.get("username")
    if not employee_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training is assigned to this employee
    assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == training_id,
        models.TrainingAssignment.employee_empid == employee_username
    )
    assignment_result = await db.execute(assignment_stmt)
    # Use first() instead of scalar_one_or_none() to handle cases where multiple assignments exist
    assignment = assignment_result.scalars().first()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access results for trainings assigned to you"
        )

    # Verify the employee attended the training
    attendance_stmt = select(models.TrainingAttendance).where(
        models.TrainingAttendance.training_id == training_id,
        models.TrainingAttendance.employee_empid == employee_username,
        models.TrainingAttendance.attended == True
    )
    attendance_result = await db.execute(attendance_stmt)
    attendance = attendance_result.scalar_one_or_none()

    if not attendance:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access results for trainings you attended. Please contact your trainer if you believe this is an error."
        )

    # Get the shared assignment (get most recent if multiple exist)
    shared_stmt = select(models.SharedAssignment).where(
        models.SharedAssignment.training_id == training_id
    ).order_by(models.SharedAssignment.updated_at.desc())
    shared_result = await db.execute(shared_stmt)
    shared_assignment = shared_result.scalars().first()

    if not shared_assignment:
        return None

    # Get the submission
    submission_stmt = select(models.AssignmentSubmission).where(
        models.AssignmentSubmission.training_id == training_id,
        models.AssignmentSubmission.employee_empid == employee_username,
        models.AssignmentSubmission.shared_assignment_id == shared_assignment.id
    ).order_by(models.AssignmentSubmission.submitted_at.desc())
    submission_result = await db.execute(submission_stmt)
    # Use first() instead of scalar_one_or_none() to handle cases where multiple submissions exist
    submission = submission_result.scalars().first()

    if not submission:
        return None

    # Parse answers and reconstruct question results
    answers_data = json.loads(submission.answers_data)
    questions_data = json.loads(shared_assignment.assignment_data)
    question_results = []

    for answer in answers_data:
        question_idx = answer.get("questionIndex", 0)
        if question_idx >= len(questions_data):
            continue
        
        question = questions_data[question_idx]
        correct_indices = []
        user_indices = answer.get("selectedOptions", [])
        
        # Get correct answer indices
        if question.get("type") in ["single-choice", "multiple-choice"]:
            options = question.get("options", [])
            for idx, opt in enumerate(options):
                if opt.get("isCorrect", False):
                    correct_indices.append(idx)
        
        # Check if answer is correct
        is_correct = False
        if question.get("type") == "single-choice":
            if len(user_indices) == 1 and user_indices[0] in correct_indices:
                is_correct = True
        elif question.get("type") == "multiple-choice":
            if set(user_indices) == set(correct_indices):
                is_correct = True
        
        question_results.append({
            "questionIndex": question_idx,
            "isCorrect": is_correct,
            "correctAnswers": correct_indices,
            "userAnswers": user_indices,
            "userTextAnswer": answer.get("textAnswer") if question.get("type") == "text-input" else None
        })

    return AssignmentResultResponse(
        id=submission.id,
        training_id=submission.training_id,
        score=submission.score,
        total_questions=submission.total_questions,
        correct_answers=submission.correct_answers,
        question_results=[QuestionResult(**qr) for qr in question_results],
        submitted_at=submission.submitted_at
    )

# --- Feedback Submission Schemas ---

class FeedbackResponseSubmission(BaseModel):
    questionIndex: int
    questionText: str
    selectedOption: str  # The selected option text

class FeedbackSubmissionCreate(BaseModel):
    training_id: int
    shared_feedback_id: int
    responses: List[FeedbackResponseSubmission]

class FeedbackSubmissionResponse(BaseModel):
    id: int
    training_id: int
    employee_empid: str
    responses: List[Dict[str, Any]]
    submitted_at: datetime

    class Config:
        from_attributes = True

@router.post("/feedback/submit", response_model=FeedbackSubmissionResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    submission_data: FeedbackSubmissionCreate,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows engineers to submit their feedback responses for a training.
    """
    employee_username = current_user.get("username")
    if not employee_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training is assigned to this employee
    assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == submission_data.training_id,
        models.TrainingAssignment.employee_empid == employee_username
    )
    assignment_result = await db.execute(assignment_stmt)
    assignment = assignment_result.scalar_one_or_none()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only submit feedback for trainings assigned to you"
        )

    # Verify the employee attended the training
    attendance_stmt = select(models.TrainingAttendance).where(
        models.TrainingAttendance.training_id == submission_data.training_id,
        models.TrainingAttendance.employee_empid == employee_username,
        models.TrainingAttendance.attended == True
    )
    attendance_result = await db.execute(attendance_stmt)
    attendance = attendance_result.scalar_one_or_none()

    if not attendance:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only submit feedback for trainings you attended. Please contact your trainer if you believe this is an error."
        )

    # Get the shared feedback
    # Note: This query filters by id (primary key) so should be unique, but using first() for consistency
    shared_stmt = select(models.SharedFeedback).where(
        models.SharedFeedback.id == submission_data.shared_feedback_id,
        models.SharedFeedback.training_id == submission_data.training_id
    )
    shared_result = await db.execute(shared_stmt)
    shared_feedback = shared_result.scalars().first()

    if not shared_feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback form not found"
        )

    # Check if feedback already submitted (prevent duplicates)
    existing_stmt = select(models.FeedbackSubmission).where(
        models.FeedbackSubmission.training_id == submission_data.training_id,
        models.FeedbackSubmission.employee_empid == employee_username,
        models.FeedbackSubmission.shared_feedback_id == submission_data.shared_feedback_id
    )
    existing_result = await db.execute(existing_stmt)
    existing_submission = existing_result.scalar_one_or_none()

    if existing_submission:
        # Update existing submission
        responses_json = json.dumps([r.dict() for r in submission_data.responses])
        existing_submission.responses_data = responses_json
        await db.commit()
        await db.refresh(existing_submission)
        
        responses_data = json.loads(existing_submission.responses_data)
        return FeedbackSubmissionResponse(
            id=existing_submission.id,
            training_id=existing_submission.training_id,
            employee_empid=existing_submission.employee_empid,
            responses=responses_data,
            submitted_at=existing_submission.submitted_at
        )

    # Store submission
    responses_json = json.dumps([r.dict() for r in submission_data.responses])
    submission = models.FeedbackSubmission(
        training_id=submission_data.training_id,
        shared_feedback_id=submission_data.shared_feedback_id,
        employee_empid=employee_username,
        responses_data=responses_json
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    responses_data = json.loads(submission.responses_data)
    return FeedbackSubmissionResponse(
        id=submission.id,
        training_id=submission.training_id,
        employee_empid=submission.employee_empid,
        responses=responses_data,
        submitted_at=submission.submitted_at
    )

@router.get("/feedback/{training_id}/result", response_model=Optional[FeedbackSubmissionResponse])
async def get_feedback_submission_result(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows engineers to check if they have submitted feedback for a training.
    Returns the feedback submission if it exists, None otherwise.
    """
    employee_username = current_user.get("username")
    if not employee_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the training is assigned to this employee
    assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == training_id,
        models.TrainingAssignment.employee_empid == employee_username
    )
    assignment_result = await db.execute(assignment_stmt)
    assignment = assignment_result.scalars().first()

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access feedback results for trainings assigned to you"
        )

    # Verify the employee attended the training
    attendance_stmt = select(models.TrainingAttendance).where(
        models.TrainingAttendance.training_id == training_id,
        models.TrainingAttendance.employee_empid == employee_username,
        models.TrainingAttendance.attended == True
    )
    attendance_result = await db.execute(attendance_stmt)
    attendance = attendance_result.scalar_one_or_none()

    if not attendance:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access feedback results for trainings you attended. Please contact your trainer if you believe this is an error."
        )

    # Get the shared feedback (get most recent if multiple exist)
    shared_stmt = select(models.SharedFeedback).where(
        models.SharedFeedback.training_id == training_id
    ).order_by(models.SharedFeedback.updated_at.desc())
    shared_result = await db.execute(shared_stmt)
    shared_feedback = shared_result.scalars().first()

    if not shared_feedback:
        return None

    # Get the feedback submission
    submission_stmt = select(models.FeedbackSubmission).where(
        models.FeedbackSubmission.training_id == training_id,
        models.FeedbackSubmission.employee_empid == employee_username,
        models.FeedbackSubmission.shared_feedback_id == shared_feedback.id
    ).order_by(models.FeedbackSubmission.submitted_at.desc())
    submission_result = await db.execute(submission_stmt)
    submission = submission_result.scalars().first()

    if not submission:
        return None

    responses_data = json.loads(submission.responses_data)
    return FeedbackSubmissionResponse(
        id=submission.id,
        training_id=submission.training_id,
        employee_empid=submission.employee_empid,
        responses=responses_data,
        submitted_at=submission.submitted_at
    )

# --- Manager Endpoints for Team Submissions ---

class TeamAssignmentSubmissionResponse(BaseModel):
    id: int
    training_id: int
    training_name: str
    employee_empid: str
    employee_name: str
    score: int
    total_questions: int
    correct_answers: int
    submitted_at: datetime
    has_feedback: bool = False  # Whether manager has provided feedback
    feedback_count: int = 0  # Number of feedback submissions (0 or 1)

class TeamFeedbackSubmissionResponse(BaseModel):
    id: int
    training_id: int
    training_name: str
    employee_empid: str
    employee_name: str
    responses: List[Dict[str, Any]]
    submitted_at: datetime

@router.get("/manager/team/assignments", response_model=List[TeamAssignmentSubmissionResponse])
async def get_team_assignment_submissions(
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_manager)
):
    """
    Returns all assignment submissions from team members for trainings assigned by this manager.
    """
    manager_username = current_user.get("username")
    
    # Get all team member IDs for this manager
    team_members_stmt = select(models.ManagerEmployee.employee_empid, models.ManagerEmployee.employee_name).where(
        models.ManagerEmployee.manager_empid == manager_username
    )
    team_result = await db.execute(team_members_stmt)
    team_members = {row[0]: row[1] for row in team_result.all()}
    
    if not team_members:
        return []
    
    # Get all assignments for team members managed by this manager
    assignments_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.employee_empid.in_(list(team_members.keys())),
        models.TrainingAssignment.manager_empid == manager_username
    )
    assignments_result = await db.execute(assignments_stmt)
    assignments = assignments_result.scalars().all()
    
    if not assignments:
        return []
    
    # Get training IDs assigned by this manager
    training_ids = [assignment.training_id for assignment in assignments]
    
    # Get all assignment submissions for these trainings
    submissions_stmt = select(models.AssignmentSubmission, models.TrainingDetail).join(
        models.TrainingDetail, models.AssignmentSubmission.training_id == models.TrainingDetail.id
    ).where(
        models.AssignmentSubmission.training_id.in_(training_ids),
        models.AssignmentSubmission.employee_empid.in_(list(team_members.keys()))
    ).order_by(models.AssignmentSubmission.submitted_at.desc())
    
    submissions_result = await db.execute(submissions_stmt)
    submissions = submissions_result.all()
    
    # Get all manager feedback for these submissions to check if feedback exists
    submission_ids = [submission.id for submission, _ in submissions]
    if submission_ids:
        # Get feedback for all training-employee combinations
        feedback_stmt = select(models.ManagerPerformanceFeedback).where(
            models.ManagerPerformanceFeedback.manager_empid == manager_username
        )
        feedback_result = await db.execute(feedback_stmt)
        all_feedback = feedback_result.scalars().all()
        
        # Create a set of (training_id, employee_empid) tuples that have feedback
        feedback_set = {(fb.training_id, fb.employee_empid) for fb in all_feedback}
    else:
        feedback_set = set()
    
    result = []
    for submission, training in submissions:
        employee_name = team_members.get(submission.employee_empid, submission.employee_empid)
        # Check if feedback exists for this training-employee combination
        has_feedback = (submission.training_id, submission.employee_empid) in feedback_set
        # Handle score: use actual score if not None, otherwise default to 0
        # Note: score can be 0 (valid score) or None (not calculated), so we check for None explicitly
        submission_score = submission.score if submission.score is not None else 0
        result.append(TeamAssignmentSubmissionResponse(
            id=submission.id,
            training_id=submission.training_id,
            training_name=training.training_name,
            employee_empid=submission.employee_empid,
            employee_name=employee_name,
            score=submission_score,
            total_questions=submission.total_questions,
            correct_answers=submission.correct_answers,
            submitted_at=submission.submitted_at,
            has_feedback=has_feedback,
            feedback_count=1 if has_feedback else 0
        ))
    
    return result

@router.get("/manager/team/feedback", response_model=List[TeamFeedbackSubmissionResponse])
async def get_team_feedback_submissions(
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_manager)
):
    """
    Returns all feedback submissions from team members for trainings assigned by this manager.
    """
    manager_username = current_user.get("username")
    
    # Get all team member IDs for this manager
    team_members_stmt = select(models.ManagerEmployee.employee_empid, models.ManagerEmployee.employee_name).where(
        models.ManagerEmployee.manager_empid == manager_username
    )
    team_result = await db.execute(team_members_stmt)
    team_members = {row[0]: row[1] for row in team_result.all()}
    
    if not team_members:
        return []
    
    # Get all assignments for team members managed by this manager
    assignments_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.employee_empid.in_(list(team_members.keys())),
        models.TrainingAssignment.manager_empid == manager_username
    )
    assignments_result = await db.execute(assignments_stmt)
    assignments = assignments_result.scalars().all()
    
    if not assignments:
        return []
    
    # Get training IDs assigned by this manager
    training_ids = [assignment.training_id for assignment in assignments]
    
    # Get all feedback submissions for these trainings
    submissions_stmt = select(models.FeedbackSubmission, models.TrainingDetail).join(
        models.TrainingDetail, models.FeedbackSubmission.training_id == models.TrainingDetail.id
    ).where(
        models.FeedbackSubmission.training_id.in_(training_ids),
        models.FeedbackSubmission.employee_empid.in_(list(team_members.keys()))
    ).order_by(models.FeedbackSubmission.submitted_at.desc())
    
    submissions_result = await db.execute(submissions_stmt)
    submissions = submissions_result.all()
    
    result = []
    for submission, training in submissions:
        employee_name = team_members.get(submission.employee_empid, submission.employee_empid)
        responses_data = json.loads(submission.responses_data)
        result.append(TeamFeedbackSubmissionResponse(
            id=submission.id,
            training_id=submission.training_id,
            training_name=training.training_name,
            employee_empid=submission.employee_empid,
            employee_name=employee_name,
            responses=responses_data,
            submitted_at=submission.submitted_at
        ))
    
    return result

# --- Manager Performance Feedback Endpoints ---

class ManagerPerformanceFeedbackCreate(BaseModel):
    training_id: int
    employee_empid: str
    application_of_training: Optional[int] = None  # 1-5
    quality_of_deliverables: Optional[int] = None  # 1-5
    problem_solving_capability: Optional[int] = None  # 1-5
    productivity_independence: Optional[int] = None  # 1-5
    process_compliance_adherence: Optional[int] = None  # 1-5
    improvement_areas: Optional[str] = None
    strengths: Optional[str] = None
    overall_performance: int  # 1-5 (required)
    additional_comments: Optional[str] = None

class ManagerPerformanceFeedbackResponse(BaseModel):
    id: int
    training_id: int
    training_name: str
    employee_empid: str
    employee_name: str
    manager_empid: str
    manager_name: str
    application_of_training: Optional[int]
    quality_of_deliverables: Optional[int]
    problem_solving_capability: Optional[int]
    productivity_independence: Optional[int]
    process_compliance_adherence: Optional[int]
    improvement_areas: Optional[str]
    strengths: Optional[str]
    overall_performance: int
    additional_comments: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

@router.post("/manager/performance-feedback", response_model=ManagerPerformanceFeedbackResponse, status_code=status.HTTP_201_CREATED)
async def create_or_update_performance_feedback(
    feedback_data: ManagerPerformanceFeedbackCreate,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_manager)
):
    """
    Create manager performance feedback for an employee's training.
    Each submission creates a new feedback entry to maintain complete history.
    All previous feedback entries are preserved and visible to the employee.
    """
    manager_username = current_user.get("username")
    if not manager_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Validate ratings are between 1-5
    if feedback_data.overall_performance < 1 or feedback_data.overall_performance > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Overall performance rating must be between 1 and 5"
        )
    
    for field_name, field_value in [
        ("application_of_training", feedback_data.application_of_training),
        ("quality_of_deliverables", feedback_data.quality_of_deliverables),
        ("problem_solving_capability", feedback_data.problem_solving_capability),
        ("productivity_independence", feedback_data.productivity_independence),
        ("process_compliance_adherence", feedback_data.process_compliance_adherence)
    ]:
        if field_value is not None and (field_value < 1 or field_value > 5):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} rating must be between 1 and 5"
            )

    # Verify the employee is managed by this manager
    manager_relation_stmt = select(models.ManagerEmployee).where(
        models.ManagerEmployee.manager_empid == manager_username,
        models.ManagerEmployee.employee_empid == feedback_data.employee_empid
    )
    manager_relation_result = await db.execute(manager_relation_stmt)
    manager_relation = manager_relation_result.scalar_one_or_none()
    
    if not manager_relation:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only provide feedback for employees in your team"
        )

    # Verify the training was assigned to this employee by this manager
    assignment_stmt = select(models.TrainingAssignment).where(
        models.TrainingAssignment.training_id == feedback_data.training_id,
        models.TrainingAssignment.employee_empid == feedback_data.employee_empid,
        models.TrainingAssignment.manager_empid == manager_username
    )
    assignment_result = await db.execute(assignment_stmt)
    assignment = assignment_result.scalar_one_or_none()
    
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only provide feedback for trainings you assigned to this employee"
        )

    # Get training and employee details
    training_stmt = select(models.TrainingDetail).where(
        models.TrainingDetail.id == feedback_data.training_id
    )
    training_result = await db.execute(training_stmt)
    training = training_result.scalar_one_or_none()
    
    if not training:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training not found"
        )

    # Get manager name
    manager_name_stmt = select(models.ManagerEmployee.manager_name).where(
        models.ManagerEmployee.manager_empid == manager_username
    ).limit(1)
    manager_name_result = await db.execute(manager_name_stmt)
    manager_name_row = manager_name_result.first()
    manager_name = manager_name_row[0] if manager_name_row and manager_name_row[0] else manager_username

    employee_name = manager_relation.employee_name
    
    # Store training_name before commit to avoid lazy-loading issues
    training_name = training.training_name

    # Always create a new feedback entry to maintain history
    # This allows employees to see all previous feedback updates
    new_feedback = models.ManagerPerformanceFeedback(
        training_id=feedback_data.training_id,
        employee_empid=feedback_data.employee_empid,
        manager_empid=manager_username,
        application_of_training=feedback_data.application_of_training,
        quality_of_deliverables=feedback_data.quality_of_deliverables,
        problem_solving_capability=feedback_data.problem_solving_capability,
        productivity_independence=feedback_data.productivity_independence,
        process_compliance_adherence=feedback_data.process_compliance_adherence,
        improvement_areas=feedback_data.improvement_areas,
        strengths=feedback_data.strengths,
        overall_performance=feedback_data.overall_performance,
        additional_comments=feedback_data.additional_comments
    )
    db.add(new_feedback)
    await db.commit()
    await db.refresh(new_feedback)
    
    return ManagerPerformanceFeedbackResponse(
        id=new_feedback.id,
        training_id=new_feedback.training_id,
        training_name=training_name,
        employee_empid=new_feedback.employee_empid,
        employee_name=employee_name,
        manager_empid=new_feedback.manager_empid,
        manager_name=manager_name,
        application_of_training=new_feedback.application_of_training,
        quality_of_deliverables=new_feedback.quality_of_deliverables,
        problem_solving_capability=new_feedback.problem_solving_capability,
        productivity_independence=new_feedback.productivity_independence,
        process_compliance_adherence=new_feedback.process_compliance_adherence,
        improvement_areas=new_feedback.improvement_areas,
        strengths=new_feedback.strengths,
        overall_performance=new_feedback.overall_performance,
        additional_comments=new_feedback.additional_comments,
        created_at=new_feedback.created_at,
        updated_at=new_feedback.updated_at
    )

@router.get("/manager/performance-feedback/{training_id}/{employee_empid}", response_model=Optional[ManagerPerformanceFeedbackResponse])
async def get_performance_feedback(
    training_id: int,
    employee_empid: str,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_manager)
):
    """
    Get the latest performance feedback for a specific employee's training.
    """
    manager_username = current_user.get("username")
    if not manager_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the employee is managed by this manager
    manager_relation_stmt = select(models.ManagerEmployee).where(
        models.ManagerEmployee.manager_empid == manager_username,
        models.ManagerEmployee.employee_empid == employee_empid
    )
    manager_relation_result = await db.execute(manager_relation_stmt)
    manager_relation = manager_relation_result.scalar_one_or_none()
    
    if not manager_relation:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view feedback for employees in your team"
        )

    # Get the latest feedback
    feedback_stmt = select(models.ManagerPerformanceFeedback, models.TrainingDetail).join(
        models.TrainingDetail, models.ManagerPerformanceFeedback.training_id == models.TrainingDetail.id
    ).where(
        models.ManagerPerformanceFeedback.training_id == training_id,
        models.ManagerPerformanceFeedback.employee_empid == employee_empid,
        models.ManagerPerformanceFeedback.manager_empid == manager_username
    ).order_by(models.ManagerPerformanceFeedback.updated_at.desc()).limit(1)
    
    feedback_result = await db.execute(feedback_stmt)
    feedback_row = feedback_result.first()
    
    if not feedback_row:
        return None
    
    feedback, training = feedback_row
    
    # Get manager name
    manager_name_stmt = select(models.ManagerEmployee.manager_name).where(
        models.ManagerEmployee.manager_empid == manager_username
    ).limit(1)
    manager_name_result = await db.execute(manager_name_stmt)
    manager_name_row = manager_name_result.first()
    manager_name = manager_name_row[0] if manager_name_row and manager_name_row[0] else manager_username

    return ManagerPerformanceFeedbackResponse(
        id=feedback.id,
        training_id=feedback.training_id,
        training_name=training.training_name,
        employee_empid=feedback.employee_empid,
        employee_name=manager_relation.employee_name,
        manager_empid=feedback.manager_empid,
        manager_name=manager_name,
        application_of_training=feedback.application_of_training,
        quality_of_deliverables=feedback.quality_of_deliverables,
        problem_solving_capability=feedback.problem_solving_capability,
        productivity_independence=feedback.productivity_independence,
        process_compliance_adherence=feedback.process_compliance_adherence,
        improvement_areas=feedback.improvement_areas,
        strengths=feedback.strengths,
        overall_performance=feedback.overall_performance,
        additional_comments=feedback.additional_comments,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at
    )

@router.get("/manager/performance-feedback/{training_id}/{employee_empid}/history", response_model=List[ManagerPerformanceFeedbackResponse])
async def get_performance_feedback_history(
    training_id: int,
    employee_empid: str,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_manager)
):
    """
    Get all performance feedback history for a specific employee's training (all entries, not just latest).
    """
    manager_username = current_user.get("username")
    if not manager_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify the employee is managed by this manager
    manager_relation_stmt = select(models.ManagerEmployee).where(
        models.ManagerEmployee.manager_empid == manager_username,
        models.ManagerEmployee.employee_empid == employee_empid
    )
    manager_relation_result = await db.execute(manager_relation_stmt)
    manager_relation = manager_relation_result.scalar_one_or_none()
    
    if not manager_relation:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view feedback for employees in your team"
        )

    # Get all feedback entries for this training-employee combination
    feedback_stmt = select(
        models.ManagerPerformanceFeedback,
        models.TrainingDetail
    ).join(
        models.TrainingDetail, models.ManagerPerformanceFeedback.training_id == models.TrainingDetail.id
    ).where(
        models.ManagerPerformanceFeedback.training_id == training_id,
        models.ManagerPerformanceFeedback.employee_empid == employee_empid,
        models.ManagerPerformanceFeedback.manager_empid == manager_username
    ).order_by(models.ManagerPerformanceFeedback.updated_at.desc())
    
    feedback_result = await db.execute(feedback_stmt)
    all_feedback = feedback_result.all()
    
    if not all_feedback:
        return []
    
    # Get manager name
    manager_name_stmt = select(models.ManagerEmployee.manager_name).where(
        models.ManagerEmployee.manager_empid == manager_username
    ).limit(1)
    manager_name_result = await db.execute(manager_name_stmt)
    manager_name_row = manager_name_result.first()
    manager_name = manager_name_row[0] if manager_name_row and manager_name_row[0] else manager_username

    result = []
    for feedback, training in all_feedback:
        result.append(ManagerPerformanceFeedbackResponse(
            id=feedback.id,
            training_id=feedback.training_id,
            training_name=training.training_name,
            employee_empid=feedback.employee_empid,
            employee_name=manager_relation.employee_name,
            manager_empid=feedback.manager_empid,
            manager_name=manager_name,
            application_of_training=feedback.application_of_training,
            quality_of_deliverables=feedback.quality_of_deliverables,
            problem_solving_capability=feedback.problem_solving_capability,
            productivity_independence=feedback.productivity_independence,
            process_compliance_adherence=feedback.process_compliance_adherence,
            improvement_areas=feedback.improvement_areas,
            strengths=feedback.strengths,
            overall_performance=feedback.overall_performance,
            additional_comments=feedback.additional_comments,
            created_at=feedback.created_at,
            updated_at=feedback.updated_at
        ))
    
    return result

@router.get("/employee/performance-feedback", response_model=List[ManagerPerformanceFeedbackResponse])
async def get_employee_performance_feedback(
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get all performance feedback history for the current employee (all feedback entries, not just latest).
    """
    employee_username = current_user.get("username")
    if not employee_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Get all feedback for this employee, ordered by updated_at (most recent first)
    # This returns ALL feedback entries, not just the latest per training
    feedback_stmt = select(
        models.ManagerPerformanceFeedback,
        models.TrainingDetail,
        models.ManagerEmployee.manager_name,
        models.ManagerEmployee.employee_name
    ).join(
        models.TrainingDetail, models.ManagerPerformanceFeedback.training_id == models.TrainingDetail.id
    ).join(
        models.ManagerEmployee,
        (models.ManagerPerformanceFeedback.manager_empid == models.ManagerEmployee.manager_empid) &
        (models.ManagerPerformanceFeedback.employee_empid == models.ManagerEmployee.employee_empid)
    ).where(
        models.ManagerPerformanceFeedback.employee_empid == employee_username
    ).order_by(
        models.ManagerPerformanceFeedback.updated_at.desc()
    )
    
    feedback_result = await db.execute(feedback_stmt)
    all_feedback = feedback_result.all()
    
    # Return all feedback entries (not just latest)
    result = []
    for feedback, training, manager_name, employee_name in all_feedback:
        result.append(ManagerPerformanceFeedbackResponse(
            id=feedback.id,
            training_id=feedback.training_id,
            training_name=training.training_name,
            employee_empid=feedback.employee_empid,
            employee_name=employee_name or feedback.employee_empid,
            manager_empid=feedback.manager_empid,
            manager_name=manager_name or feedback.manager_empid,
            application_of_training=feedback.application_of_training,
            quality_of_deliverables=feedback.quality_of_deliverables,
            problem_solving_capability=feedback.problem_solving_capability,
            productivity_independence=feedback.productivity_independence,
            process_compliance_adherence=feedback.process_compliance_adherence,
            improvement_areas=feedback.improvement_areas,
            strengths=feedback.strengths,
            overall_performance=feedback.overall_performance,
            additional_comments=feedback.additional_comments,
            created_at=feedback.created_at,
            updated_at=feedback.updated_at
        ))
    
    return result

