"""
Notification Service Module

Purpose: Utility functions to create notifications for various events in the system
Features:
- Create notifications for training assignments
- Create notifications for training request approvals/rejections
- Create notifications for new assignments available
- Create notifications for feedback received
- Helper function to create custom notifications

@author Orbit Skill Development Team
@date 2025
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime
from app.models import Notification, TrainingDetail, TrainingRequest, TrainingAssignment
from typing import Optional

async def create_notification(
    db: AsyncSession,
    user_empid: str,
    title: str,
    message: str,
    type: str = "info",
    related_id: Optional[int] = None,
    related_type: Optional[str] = None,
    action_url: Optional[str] = None
) -> Notification:
    """
    Create a new notification for a user.
    
    Args:
        db: Database session
        user_empid: Employee ID of the user receiving the notification
        title: Notification title
        message: Notification message
        type: Notification type (info, success, warning, error, assignment, approval, etc.)
        related_id: Optional ID of related entity
        related_type: Optional type of related entity
        action_url: Optional URL to navigate to when clicked
        
    Returns:
        Created notification object
    """
    notification = Notification(
        user_empid=user_empid,
        title=title,
        message=message,
        type=type,
        related_id=related_id,
        related_type=related_type,
        action_url=action_url,
        is_read=False,
        created_at=datetime.utcnow()
    )
    
    db.add(notification)
    await db.commit()
    await db.refresh(notification)
    
    return notification

async def notify_training_assigned(
    db: AsyncSession,
    employee_empid: str,
    training_id: int,
    training_name: str
) -> Notification:
    """
    Create a notification when a training is assigned to an employee.
    
    Args:
        db: Database session
        employee_empid: Employee ID receiving the assignment
        training_id: ID of the assigned training
        training_name: Name of the training
        
    Returns:
        Created notification
    """
    return await create_notification(
        db=db,
        user_empid=employee_empid,
        title="New Training Assigned",
        message=f"You have been assigned to the training: {training_name}",
        type="assignment",
        related_id=training_id,
        related_type="training",
        action_url=f"/engineer-dashboard?tab=assignedTrainings"
    )

async def notify_training_request_approved(
    db: AsyncSession,
    employee_empid: str,
    training_id: int,
    training_name: str
) -> Notification:
    """
    Create a notification when a training request is approved.
    
    Args:
        db: Database session
        employee_empid: Employee ID who requested the training
        training_id: ID of the approved training
        training_name: Name of the training
        
    Returns:
        Created notification
    """
    return await create_notification(
        db=db,
        user_empid=employee_empid,
        title="Training Request Approved",
        message=f"Your request for '{training_name}' has been approved by your manager.",
        type="success",
        related_id=training_id,
        related_type="training_request",
        action_url=f"/engineer-dashboard?tab=assignedTrainings"
    )

async def notify_training_request_rejected(
    db: AsyncSession,
    employee_empid: str,
    training_id: int,
    training_name: str,
    manager_notes: Optional[str] = None
) -> Notification:
    """
    Create a notification when a training request is rejected.
    
    Args:
        db: Database session
        employee_empid: Employee ID who requested the training
        training_id: ID of the rejected training
        training_name: Name of the training
        manager_notes: Optional notes from the manager
        
    Returns:
        Created notification
    """
    message = f"Your request for '{training_name}' has been rejected by your manager."
    if manager_notes:
        message += f" Notes: {manager_notes}"
    
    return await create_notification(
        db=db,
        user_empid=employee_empid,
        title="Training Request Rejected",
        message=message,
        type="warning",
        related_id=training_id,
        related_type="training_request",
        action_url=f"/engineer-dashboard?tab=trainingRequests"
    )

async def notify_new_assignment_available(
    db: AsyncSession,
    employee_empid: str,
    training_id: int,
    training_name: str,
    assignment_title: str
) -> Notification:
    """
    Create a notification when a new assignment is available for a training.
    
    Args:
        db: Database session
        employee_empid: Employee ID who should see the assignment
        training_id: ID of the training
        training_name: Name of the training
        assignment_title: Title of the assignment
        
    Returns:
        Created notification
    """
    return await create_notification(
        db=db,
        user_empid=employee_empid,
        title="New Assignment Available",
        message=f"A new assignment '{assignment_title}' is available for training: {training_name}",
        type="assignment",
        related_id=training_id,
        related_type="assignment",
        action_url=f"/engineer-dashboard?tab=assignedTrainings"
    )

async def notify_new_feedback_available(
    db: AsyncSession,
    employee_empid: str,
    training_id: int,
    training_name: str
) -> Notification:
    """
    Create a notification when a new feedback form is available for a training.
    
    Args:
        db: Database session
        employee_empid: Employee ID who should submit feedback
        training_id: ID of the training
        training_name: Name of the training
        
    Returns:
        Created notification
    """
    return await create_notification(
        db=db,
        user_empid=employee_empid,
        title="Feedback Requested",
        message=f"Please submit feedback for the training: {training_name}",
        type="info",
        related_id=training_id,
        related_type="feedback",
        action_url=f"/engineer-dashboard?tab=assignedTrainings"
    )

async def notify_performance_feedback_received(
    db: AsyncSession,
    employee_empid: str,
    training_id: int,
    training_name: str
) -> Notification:
    """
    Create a notification when a manager provides performance feedback.
    
    Args:
        db: Database session
        employee_empid: Employee ID who received feedback
        training_id: ID of the training
        training_name: Name of the training
        
    Returns:
        Created notification
    """
    return await create_notification(
        db=db,
        user_empid=employee_empid,
        title="Performance Feedback Received",
        message=f"Your manager has provided performance feedback for: {training_name}",
        type="info",
        related_id=training_id,
        related_type="performance_feedback",
        action_url=f"/engineer-dashboard?tab=assignedTrainings"
    )

async def notify_training_request_received(
    db: AsyncSession,
    manager_empid: str,
    employee_name: str,
    training_id: int,
    training_name: str
) -> Notification:
    """
    Create a notification when a manager receives a training request from an employee.
    
    Args:
        db: Database session
        manager_empid: Manager ID receiving the request
        employee_name: Name of the employee who requested
        training_id: ID of the requested training
        training_name: Name of the training
        
    Returns:
        Created notification
    """
    return await create_notification(
        db=db,
        user_empid=manager_empid,
        title="Training Request Received",
        message=f"{employee_name} has requested approval for: {training_name}",
        type="info",
        related_id=training_id,
        related_type="training_request",
        action_url=f"/manager-dashboard?tab=trainingRequests"
    )



