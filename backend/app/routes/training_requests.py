"""
Training Requests Routes Module

Purpose: API routes for training request management (approval workflow)
Features:
- Employees can request training enrollment
- Managers can approve/reject training requests
- View pending requests (managers)
- View user's own requests (employees)

Endpoints:
- POST /training-requests/: Create a training request
- GET /training-requests/my: Get current user's training requests
- GET /training-requests/pending: Get pending requests (manager only)
- PATCH /training-requests/{id}/respond: Approve/reject request (manager only)

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import List
from datetime import datetime

from app.database import get_db_async
from app.models import TrainingRequest, TrainingDetail, User, ManagerEmployee, EmployeeCompetency
from app.schemas import TrainingRequestCreate, TrainingRequestResponse, TrainingRequestUpdate, TrainingResponse, UserResponse
from app.auth_utils import get_current_active_user
from app.email_service import get_email_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training-requests", tags=["Training Requests"])

@router.post("/", response_model=TrainingRequestResponse, status_code=status.HTTP_201_CREATED)
async def create_training_request(
    request_data: TrainingRequestCreate,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Endpoint for engineers to request enrollment in a training course.
    This initiates the approval workflow with their manager.
    """
    try:
        current_username = current_user.get("username")
        if not current_username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
            )

        # Verify the training exists and extract all data immediately
        training_stmt = select(TrainingDetail).where(TrainingDetail.id == request_data.training_id)
        training_result = await db.execute(training_stmt)
        training = training_result.scalar_one_or_none()
        
        if not training:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training not found"
            )
        
        # Extract ALL training data as simple Python types IMMEDIATELY
        # This prevents lazy loading issues when accessing training attributes later
        training_id_int = int(training.id)
        training_name_str = str(training.training_name or "")
        training_division = training.division
        training_department = training.department
        training_competency = training.competency
        training_skill = training.skill
        training_topics = training.training_topics
        training_prerequisites = training.prerequisites
        training_skill_category = training.skill_category
        training_trainer_name = training.trainer_name
        training_email = training.email
        training_date = training.training_date
        training_duration = training.duration
        training_time = training.time
        training_type = training.training_type
        training_seats = training.seats
        training_assessment_details = training.assessment_details

        # Find the employee's manager
        manager_stmt = select(
            ManagerEmployee.manager_empid,
            ManagerEmployee.manager_name,
            ManagerEmployee.employee_empid,
            ManagerEmployee.employee_name
        ).where(
            ManagerEmployee.employee_empid == current_username
        )
        manager_result = await db.execute(manager_stmt)
        manager_row = manager_result.first()
        
        if not manager_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No manager found for this employee"
            )
        
        # Extract values directly from row tuple to avoid lazy loading
        manager_empid = manager_row[0]
        manager_name = manager_row[1]
        employee_empid_from_relation = manager_row[2]
        employee_name_from_relation = manager_row[3]

        # Check if request already exists
        existing_request_stmt = select(TrainingRequest).where(
            TrainingRequest.training_id == request_data.training_id,
            TrainingRequest.employee_empid == current_username
        )
        existing_request_result = await db.execute(existing_request_stmt)
        existing_request = existing_request_result.scalar_one_or_none()
        
        if existing_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You have already requested this training"
            )

        # Create the training request
        new_request = TrainingRequest(
            training_id=request_data.training_id,
            employee_empid=current_username,
            manager_empid=manager_empid,
            status='pending'
        )

        db.add(new_request)
        await db.commit()
        await db.refresh(new_request)

        # Fetch the complete request with training details and employee
        complete_request_stmt = select(TrainingRequest).options(
            selectinload(TrainingRequest.training),
            selectinload(TrainingRequest.employee)
        ).where(TrainingRequest.id == new_request.id)
        
        complete_request_result = await db.execute(complete_request_stmt)
        complete_request = complete_request_result.scalar_one()

        # Extract ALL complete_request data as simple Python types IMMEDIATELY
        # This prevents lazy loading issues when accessing request attributes later
        request_id_int = int(complete_request.id)
        request_training_id = int(complete_request.training_id)
        request_employee_empid = str(complete_request.employee_empid)
        request_manager_empid = str(complete_request.manager_empid)
        request_date = complete_request.request_date or new_request.request_date or datetime.utcnow()
        request_status = str(complete_request.status)
        request_manager_notes = complete_request.manager_notes
        request_response_date = complete_request.response_date

        # Extract other values as simple Python types
        # Note: training data was already extracted earlier (lines 67-85)
        manager_empid_str = str(manager_empid)  # Already extracted from query
        employee_name_str = str(employee_name_from_relation or current_username)
        
        # Get manager email from employee_competency table
        try:
            manager_email_stmt = select(EmployeeCompetency.email).where(
                EmployeeCompetency.employee_empid == manager_empid_str
            ).limit(1)
            manager_email_result = await db.execute(manager_email_stmt)
            manager_email = manager_email_result.scalar_one_or_none()
            manager_email_str = str(manager_email) if manager_email else None
            
            logger.info(f"📧 Preparing to send email notification to manager {manager_empid_str}")
            logger.info(f"   Manager email from DB: {manager_email_str}")
            
            # Send email in background thread to avoid async/COM conflicts
            import asyncio
            
            def send_email_sync():
                try:
                    email_service = get_email_service()
                    return email_service.send_training_request_notification(
                        manager_username=manager_empid_str,
                        employee_username=current_username,
                        employee_name=employee_name_str,
                        training_name=training_name_str,
                        training_id=training_id_int,
                        manager_email=manager_email_str
                    )
                except Exception as e:
                    logger.error(f"❌ Error in email thread: {str(e)}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")
                    return False
            
            # Run email sending in thread pool (non-blocking)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, send_email_sync)
            
            logger.info(f"📧 Email notification queued for manager {manager_empid_str}")
            
        except Exception as e:
            # Log error but don't fail the request creation
            logger.error(f"❌ Failed to queue email notification for training request {request_id_int}: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")

        # Create in-app notification for manager
        try:
            from app.notification_service import notify_training_request_received
            await notify_training_request_received(
                db=db,
                manager_empid=manager_empid_str,
                employee_name=employee_name_str,
                training_id=training_id_int,
                training_name=training_name_str
            )
        except Exception as e:
            logger.error(f"Failed to create notification for training request: {str(e)}")

        # Manually construct response using extracted values (no database access)
        training_response = TrainingResponse(
            id=training_id_int,
            division=training_division,
            department=training_department,
            competency=training_competency,
            skill=training_skill,
            training_name=training_name_str,
            training_topics=training_topics,
            prerequisites=training_prerequisites,
            skill_category=training_skill_category,
            trainer_name=training_trainer_name,
            email=training_email,
            training_date=training_date,
            duration=training_duration,
            time=training_time,
            training_type=training_type,
            seats=training_seats,
            assessment_details=training_assessment_details
        )
        
        employee_response = UserResponse(
            username=current_username,
            name=employee_name_from_relation
        )
        
        return TrainingRequestResponse(
            id=request_id_int,
            training_id=request_training_id,
            employee_empid=request_employee_empid,
            manager_empid=request_manager_empid,
            request_date=request_date,
            status=request_status,
            manager_notes=request_manager_notes,
            response_date=request_response_date,
            training=training_response,
            employee=employee_response
        )
    except HTTPException:
        # Re-raise HTTP exceptions (like 400, 404) as-is
        raise
    except Exception as e:
        # Log unexpected errors and return a proper error response
        logger.error(f"Unexpected error in create_training_request: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while creating the training request: {str(e)}"
        )

@router.get("/my", response_model=List[TrainingRequestResponse])
async def get_my_training_requests(
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get all training requests made by the current user (engineer).
    """
    current_username = current_user.get("username")
    if not current_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    stmt = select(TrainingRequest).options(
        selectinload(TrainingRequest.training),
        selectinload(TrainingRequest.employee)
    ).where(TrainingRequest.employee_empid == current_username).order_by(TrainingRequest.request_date.desc())
    
    result = await db.execute(stmt)
    requests = result.scalars().all()
    
    return requests

@router.get("/pending", response_model=List[TrainingRequestResponse])
async def get_pending_requests(
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get all pending training requests for the current manager to review.
    """
    current_username = current_user.get("username")
    if not current_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    # Join with ManagerEmployee to get employee name
    stmt = select(TrainingRequest, ManagerEmployee.employee_name).options(
        selectinload(TrainingRequest.training),
        selectinload(TrainingRequest.employee)
    ).join(
        ManagerEmployee, 
        TrainingRequest.employee_empid == ManagerEmployee.employee_empid
    ).where(
        TrainingRequest.manager_empid == current_username,
        TrainingRequest.status == 'pending'
    ).order_by(TrainingRequest.request_date.desc())
    
    result = await db.execute(stmt)
    rows = result.all()
    
    # Convert to TrainingRequestResponse with employee name
    requests = []
    for row in rows:
        training_request = row[0]
        employee_name = row[1]
        
        # Create a modified training request with employee name
        request_dict = {
            "id": training_request.id,
            "training_id": training_request.training_id,
            "employee_empid": training_request.employee_empid,
            "manager_empid": training_request.manager_empid,
            "request_date": training_request.request_date,
            "status": training_request.status,
            "manager_notes": training_request.manager_notes,
            "response_date": training_request.response_date,
            "training": training_request.training,
            "employee": {
                "username": training_request.employee.username,
                "name": employee_name
            }
        }
        requests.append(TrainingRequestResponse(**request_dict))
    
    return requests

@router.put("/{request_id}/respond", response_model=TrainingRequestResponse)
async def respond_to_request(
    request_id: int,
    response_data: TrainingRequestUpdate,
    db: AsyncSession = Depends(get_db_async),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Endpoint for managers to approve or reject training requests.
    """
    current_username = current_user.get("username")
    if not current_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    # Get the request
    stmt = select(TrainingRequest).where(TrainingRequest.id == request_id)
    result = await db.execute(stmt)
    request = result.scalar_one_or_none()
    
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training request not found"
        )

    # Verify the current user is the manager for this request
    if request.manager_empid != current_username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to respond to this request"
        )

    # Check if request is still pending
    if request.status != 'pending':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This request has already been responded to"
        )

    # Validate status value
    if response_data.status not in ['approved', 'rejected']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Status must be either 'approved' or 'rejected'"
        )

    # Update the request
    request.status = response_data.status
    request.manager_notes = response_data.manager_notes
    request.response_date = datetime.utcnow()

    # If approved, create a training assignment
    if response_data.status == 'approved':
        from app.models import TrainingAssignment
        assignment = TrainingAssignment(
            training_id=request.training_id,
            employee_empid=request.employee_empid,
            manager_empid=request.manager_empid
        )
        db.add(assignment)

    await db.commit()
    await db.refresh(request)

    # Fetch the complete request with training details and employee info
    complete_request_stmt = select(TrainingRequest, ManagerEmployee.employee_name).options(
        selectinload(TrainingRequest.training),
        selectinload(TrainingRequest.employee)
    ).join(
        ManagerEmployee, 
        TrainingRequest.employee_empid == ManagerEmployee.employee_empid
    ).where(TrainingRequest.id == request.id)
    
    complete_request_result = await db.execute(complete_request_stmt)
    row = complete_request_result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training request not found after update"
        )
    
    training_request = row[0]
    employee_name = row[1]
    
    # Extract ALL data from training_request and related objects IMMEDIATELY
    # This prevents lazy loading issues when accessing attributes later
    request_id_int = int(training_request.id)
    request_training_id = int(training_request.training_id)
    request_employee_empid = str(training_request.employee_empid)
    request_manager_empid = str(training_request.manager_empid)
    request_date = training_request.request_date
    request_status = str(training_request.status)
    request_manager_notes = training_request.manager_notes
    request_response_date = training_request.response_date
    
    # Extract ALL training data as simple Python types IMMEDIATELY
    training = training_request.training
    training_id_int = int(training.id)
    training_name_str = str(training.training_name or "")
    training_division = training.division
    training_department = training.department
    training_competency = training.competency
    training_skill = training.skill
    training_topics = training.training_topics
    training_prerequisites = training.prerequisites
    training_skill_category = training.skill_category
    training_trainer_name = training.trainer_name
    training_email = training.email
    training_date = training.training_date
    training_duration = training.duration
    training_time = training.time
    training_type = training.training_type
    training_seats = training.seats
    training_assessment_details = training.assessment_details
    
    # Extract employee data
    employee = training_request.employee
    employee_username_str = str(employee.username)
    employee_name_str = str(employee_name or employee_username_str)
    
    # Send email notification to employee about the decision (run in background)
    status_str = str(response_data.status)
    manager_notes_str = str(response_data.manager_notes) if response_data.manager_notes else None
    
    try:
        # Get employee email from employee_competency table
        employee_email_stmt = select(EmployeeCompetency.email).where(
            EmployeeCompetency.employee_empid == request_employee_empid
        ).limit(1)
        employee_email_result = await db.execute(employee_email_stmt)
        employee_email = employee_email_result.scalar_one_or_none()
        employee_email_str = str(employee_email) if employee_email else None
        
        logger.info(f"📧 Preparing to send email notification to employee {request_employee_empid}")
        logger.info(f"   Employee email from DB: {employee_email_str}")
        logger.info(f"   Status: {status_str}")
        
        # Send email in background thread to avoid async/COM conflicts
        import asyncio
        
        def send_email_sync():
            try:
                email_service = get_email_service()
                return email_service.send_request_decision_notification(
                    employee_username=request_employee_empid,
                    employee_name=employee_name_str,
                    manager_username=current_username,
                    training_name=training_name_str,
                    status=status_str,
                    manager_notes=manager_notes_str,
                    employee_email=employee_email_str
                )
            except Exception as e:
                logger.error(f"❌ Error in email thread: {str(e)}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                return False
        
        # Run email sending in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, send_email_sync)
        
        logger.info(f"📧 Email notification queued for employee {request_employee_empid}")
        
    except Exception as e:
        # Log error but don't fail the request update
        logger.error(f"❌ Failed to queue email notification for training request {request_id_int}: {str(e)}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
    
    # Create in-app notification for employee
    try:
        from app.notification_service import notify_training_request_approved, notify_training_request_rejected
        if status_str == 'approved':
            await notify_training_request_approved(
                db=db,
                employee_empid=request_employee_empid,
                training_id=training_id_int,
                training_name=training_name_str
            )
        elif status_str == 'rejected':
            await notify_training_request_rejected(
                db=db,
                employee_empid=request_employee_empid,
                training_id=training_id_int,
                training_name=training_name_str,
                manager_notes=manager_notes_str
            )
    except Exception as e:
        logger.error(f"Failed to create notification for training request response: {str(e)}")
    
    # Manually construct response using extracted values (no database access)
    training_response = TrainingResponse(
        id=training_id_int,
        division=training_division,
        department=training_department,
        competency=training_competency,
        skill=training_skill,
        training_name=training_name_str,
        training_topics=training_topics,
        prerequisites=training_prerequisites,
        skill_category=training_skill_category,
        trainer_name=training_trainer_name,
        email=training_email,
        training_date=training_date,
        duration=training_duration,
        time=training_time,
        training_type=training_type,
        seats=training_seats,
        assessment_details=training_assessment_details
    )
    
    employee_response = UserResponse(
        username=employee_username_str,
        name=employee_name_str
    )
    
    return TrainingRequestResponse(
        id=request_id_int,
        training_id=request_training_id,
        employee_empid=request_employee_empid,
        manager_empid=request_manager_empid,
        request_date=request_date,
        status=request_status,
        manager_notes=request_manager_notes,
        response_date=request_response_date,
        training=training_response,
        employee=employee_response
    )
