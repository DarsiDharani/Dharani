"""
Training Files Routes Module

Purpose: API routes for managing training question and solution PDF files
Features:
- Trainers can upload question PDF files for trainings
- Engineers can download question PDF files for assigned trainings
- Engineers can upload solution PDF files
- Trainers can view/download solution PDF files submitted by engineers

Endpoints:
- POST /training-files/questions/upload: Upload question PDF (trainer)
- GET /training-files/questions/{training_id}: Download question PDF (engineer)
- POST /training-files/solutions/upload: Upload solution PDF (engineer)
- GET /training-files/solutions/{training_id}/{employee_empid}: Get solution PDF (trainer)
- GET /training-files/trainer/solutions/{training_id}: Get all solutions for a training (trainer)

@author Orbit Skill Development Team
@date 2025
"""

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.database import get_db_async
from app import models
from app.auth_utils import get_current_active_user

router = APIRouter(
    prefix="/training-files",
    tags=["Training Files"]
)

# Directory for storing uploaded files
UPLOAD_DIR = Path("uploads")
QUESTION_FILES_DIR = UPLOAD_DIR / "question_files"
SOLUTION_FILES_DIR = UPLOAD_DIR / "solution_files"

# Ensure directories exist
QUESTION_FILES_DIR.mkdir(parents=True, exist_ok=True)
SOLUTION_FILES_DIR.mkdir(parents=True, exist_ok=True)

# --- Pydantic Schemas ---

class SolutionFileResponse(BaseModel):
    id: int
    training_id: int
    training_name: str
    employee_empid: str
    employee_name: str
    file_name: str
    file_size: Optional[int]
    uploaded_at: datetime

    class Config:
        from_attributes = True

# --- Helper Functions ---

async def verify_trainer_for_training(training: models.TrainingDetail, trainer_username: str, db: AsyncSession):
    """Verify that the current user is the trainer for the given training."""
    trainer_name = str(training.trainer_name or "").strip()
    if not trainer_name:
        return False
    
    # Get employee/manager name from ManagerEmployee table
    employee_result = await db.execute(
        select(models.ManagerEmployee.employee_name).where(
            models.ManagerEmployee.employee_empid == trainer_username
        )
    )
    employee_name = employee_result.scalar_one_or_none()
    
    manager_result = await db.execute(
        select(models.ManagerEmployee.manager_name).where(
            models.ManagerEmployee.manager_empid == trainer_username
        )
    )
    manager_name = manager_result.scalar_one_or_none()
    
    display_name = employee_name or manager_name
    trainer_username_lower = str(trainer_username).lower().strip()
    display_name_lower = (display_name or "").lower().strip() if display_name else ""
    
    # Split trainer_name by comma or newline
    trainer_names = []
    if ',' in trainer_name:
        trainer_names = [t.strip() for t in trainer_name.split(',') if t.strip()]
    elif '\n' in trainer_name:
        trainer_names = [t.strip() for t in trainer_name.split('\n') if t.strip()]
    else:
        trainer_names = [trainer_name.strip()]
    
    # Check matching strategies
    for single_trainer_name in trainer_names:
        trainer_name_lower = single_trainer_name.lower().strip()
        if (trainer_name_lower == trainer_username_lower or
            (display_name_lower and trainer_name_lower == display_name_lower) or
            trainer_username_lower in trainer_name_lower or
            trainer_name_lower in trainer_username_lower or
            (display_name_lower and (display_name_lower in trainer_name_lower or trainer_name_lower in display_name_lower))):
            return True
    
    return False

# --- Routes ---

@router.post("/questions/upload", status_code=status.HTTP_201_CREATED)
async def upload_question_file(
    file: UploadFile = File(...),
    training_id: int = Form(...),
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows a trainer to upload a question PDF file for a training they have scheduled.
    """
    trainer_username = current_user.get("username")
    if not trainer_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify file is PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
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

    # Verify the current user is the trainer for this training
    is_trainer = await verify_trainer_for_training(training, trainer_username, db)
    if not is_trainer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only upload question files for trainings you have scheduled"
        )

    # Read file content
    file_content = await file.read()
    file_size = len(file_content)

    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{training_id}_{uuid.uuid4()}{file_extension}"
    file_path = QUESTION_FILES_DIR / unique_filename

    # Save file
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Check if question file already exists for this training (update existing)
    existing_stmt = select(models.TrainingQuestionFile).where(
        models.TrainingQuestionFile.training_id == training_id
    )
    existing_result = await db.execute(existing_stmt)
    existing_file = existing_result.scalar_one_or_none()

    if existing_file:
        # Delete old file
        old_file_path = Path(existing_file.file_path)
        if old_file_path.exists():
            old_file_path.unlink()
        
        # Update existing record
        existing_file.file_path = str(file_path)
        existing_file.file_name = file.filename
        existing_file.file_size = file_size
        existing_file.uploaded_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing_file)
        
        return {
            "message": "Question file updated successfully",
            "file_id": existing_file.id,
            "file_name": existing_file.file_name,
            "file_size": existing_file.file_size
        }
    else:
        # Create new record
        new_file = models.TrainingQuestionFile(
            training_id=training_id,
            trainer_username=trainer_username,
            file_path=str(file_path),
            file_name=file.filename,
            file_size=file_size
        )
        db.add(new_file)
        await db.commit()
        await db.refresh(new_file)
        
        return {
            "message": "Question file uploaded successfully",
            "file_id": new_file.id,
            "file_name": new_file.file_name,
            "file_size": new_file.file_size
        }

@router.get("/questions/{training_id}")
async def download_question_file(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows engineers to download the question PDF file for a training assigned to them.
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
            detail="You can only download question files for trainings assigned to you"
        )

    # Get the question file
    file_stmt = select(models.TrainingQuestionFile).where(
        models.TrainingQuestionFile.training_id == training_id
    )
    file_result = await db.execute(file_stmt)
    question_file = file_result.scalar_one_or_none()

    if not question_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question file not found for this training"
        )

    file_path = Path(question_file.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on server"
        )

    return FileResponse(
        path=str(file_path),
        filename=question_file.file_name,
        media_type='application/pdf'
    )

@router.post("/solutions/upload", status_code=status.HTTP_201_CREATED)
async def upload_solution_file(
    file: UploadFile = File(...),
    training_id: int = Form(...),
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows engineers to upload a solution PDF file for a training assigned to them.
    """
    employee_username = current_user.get("username")
    if not employee_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

    # Verify file is PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
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
            detail="You can only upload solution files for trainings assigned to you"
        )

    # Read file content
    file_content = await file.read()
    file_size = len(file_content)

    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{training_id}_{employee_username}_{uuid.uuid4()}{file_extension}"
    file_path = SOLUTION_FILES_DIR / unique_filename

    # Save file
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Check if solution file already exists for this training-employee combination (update existing)
    existing_stmt = select(models.TrainingSolutionFile).where(
        models.TrainingSolutionFile.training_id == training_id,
        models.TrainingSolutionFile.employee_empid == employee_username
    )
    existing_result = await db.execute(existing_stmt)
    existing_file = existing_result.scalar_one_or_none()

    if existing_file:
        # Delete old file
        old_file_path = Path(existing_file.file_path)
        if old_file_path.exists():
            old_file_path.unlink()
        
        # Update existing record
        existing_file.file_path = str(file_path)
        existing_file.file_name = file.filename
        existing_file.file_size = file_size
        existing_file.uploaded_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing_file)
        
        return {
            "message": "Solution file updated successfully",
            "file_id": existing_file.id,
            "file_name": existing_file.file_name,
            "file_size": existing_file.file_size
        }
    else:
        # Create new record
        new_file = models.TrainingSolutionFile(
            training_id=training_id,
            employee_empid=employee_username,
            file_path=str(file_path),
            file_name=file.filename,
            file_size=file_size
        )
        db.add(new_file)
        await db.commit()
        await db.refresh(new_file)
        
        return {
            "message": "Solution file uploaded successfully",
            "file_id": new_file.id,
            "file_name": new_file.file_name,
            "file_size": new_file.file_size
        }

@router.get("/solutions/{training_id}/{employee_empid}")
async def download_solution_file(
    training_id: int,
    employee_empid: str,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows trainers to download a solution PDF file submitted by an engineer.
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

    # Verify the current user is the trainer for this training
    is_trainer = await verify_trainer_for_training(training, trainer_username, db)
    if not is_trainer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view solution files for trainings you have scheduled"
        )

    # Get the solution file
    file_stmt = select(models.TrainingSolutionFile).where(
        models.TrainingSolutionFile.training_id == training_id,
        models.TrainingSolutionFile.employee_empid == employee_empid
    )
    file_result = await db.execute(file_stmt)
    solution_file = file_result.scalar_one_or_none()

    if not solution_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Solution file not found"
        )

    file_path = Path(solution_file.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on server"
        )

    return FileResponse(
        path=str(file_path),
        filename=solution_file.file_name,
        media_type='application/pdf'
    )

@router.get("/trainer/solutions/{training_id}", response_model=List[SolutionFileResponse])
async def get_all_solutions_for_training(
    training_id: int,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Allows trainers to view all solution files submitted by engineers for their training.
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

    # Verify the current user is the trainer for this training
    is_trainer = await verify_trainer_for_training(training, trainer_username, db)
    if not is_trainer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view solution files for trainings you have scheduled"
        )

    # Get all solution files for this training
    files_stmt = select(
        models.TrainingSolutionFile,
        models.ManagerEmployee.employee_name
    ).join(
        models.ManagerEmployee,
        models.TrainingSolutionFile.employee_empid == models.ManagerEmployee.employee_empid
    ).where(
        models.TrainingSolutionFile.training_id == training_id
    ).order_by(models.TrainingSolutionFile.uploaded_at.desc())
    
    files_result = await db.execute(files_stmt)
    files = files_result.all()

    result = []
    for solution_file, employee_name in files:
        result.append(SolutionFileResponse(
            id=solution_file.id,
            training_id=solution_file.training_id,
            training_name=training.training_name,
            employee_empid=solution_file.employee_empid,
            employee_name=employee_name or solution_file.employee_empid,
            file_name=solution_file.file_name,
            file_size=solution_file.file_size,
            uploaded_at=solution_file.uploaded_at
        ))

    return result

