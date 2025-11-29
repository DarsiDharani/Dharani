"""
Notifications Routes Module

Purpose: API routes for managing user notifications
Features:
- Get all notifications for a user
- Get unread notifications count
- Mark notification as read
- Mark all notifications as read
- Delete notification

Endpoints:
- GET /notifications: Get all notifications for current user
- GET /notifications/unread-count: Get count of unread notifications
- PATCH /notifications/{id}/read: Mark a notification as read
- PATCH /notifications/read-all: Mark all notifications as read
- DELETE /notifications/{id}: Delete a notification

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func, delete
from datetime import datetime
from typing import List
from app.database import get_db_async
from app.models import Notification
from app.schemas import NotificationResponse, NotificationUpdate
from app.auth_utils import get_current_active_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications", tags=["Notifications"])

@router.get("/", response_model=List[NotificationResponse])
async def get_notifications(
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async),
    limit: int = 50,
    unread_only: bool = False
):
    """
    Get all notifications for the current user.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        limit: Maximum number of notifications to return (default: 50)
        unread_only: If True, only return unread notifications
        
    Returns:
        List of notifications sorted by creation date (newest first)
    """
    user_empid = current_user.get("username")
    
    query = select(Notification).where(Notification.user_empid == user_empid)
    
    if unread_only:
        query = query.where(Notification.is_read == False)
    
    query = query.order_by(Notification.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    notifications = result.scalars().all()
    
    return notifications

@router.get("/unread-count")
async def get_unread_count(
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """
    Get count of unread notifications for the current user.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dictionary with unread_count
    """
    user_empid = current_user.get("username")
    
    result = await db.execute(
        select(func.count(Notification.id)).where(
            Notification.user_empid == user_empid,
            Notification.is_read == False
        )
    )
    count = result.scalar() or 0
    
    return {"unread_count": count}

@router.patch("/{notification_id}/read", response_model=NotificationResponse)
async def mark_notification_read(
    notification_id: int,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """
    Mark a specific notification as read.
    
    Args:
        notification_id: ID of the notification to mark as read
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated notification
        
    Raises:
        HTTPException: 404 if notification not found or doesn't belong to user
    """
    user_empid = current_user.get("username")
    
    # Get the notification
    result = await db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_empid == user_empid
        )
    )
    notification = result.scalar_one_or_none()
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )
    
    # Mark as read
    notification.is_read = True
    notification.read_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(notification)
    
    return notification

@router.patch("/read-all")
async def mark_all_notifications_read(
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """
    Mark all notifications as read for the current user.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dictionary with count of notifications marked as read
    """
    user_empid = current_user.get("username")
    now = datetime.utcnow()
    
    # Update all unread notifications
    result = await db.execute(
        update(Notification)
        .where(
            Notification.user_empid == user_empid,
            Notification.is_read == False
        )
        .values(is_read=True, read_at=now)
    )
    
    await db.commit()
    
    return {"marked_read": result.rowcount}

@router.delete("/all")
async def delete_all_notifications(
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """
    Delete all notifications for the current user.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dictionary with count of notifications deleted
    """
    user_empid = current_user.get("username")
    
    if not user_empid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    logger.info(f"Deleting all notifications for user: {user_empid}")
    
    # First, get count of notifications to delete
    count_result = await db.execute(
        select(func.count(Notification.id)).where(Notification.user_empid == user_empid)
    )
    count = count_result.scalar() or 0
    
    logger.info(f"Found {count} notifications to delete for user {user_empid}")
    
    if count == 0:
        return {
            "deleted_count": 0,
            "remaining_count": 0,
            "success": True,
            "message": "No notifications to delete"
        }
    
    # Use bulk delete - this is the correct way for async SQLAlchemy
    try:
        delete_stmt = delete(Notification).where(Notification.user_empid == user_empid)
        result = await db.execute(delete_stmt)
        await db.commit()
        
        logger.info(f"Deleted {result.rowcount} notifications for user {user_empid}")
        
        # Verify deletion by checking count
        verify_result = await db.execute(
            select(func.count(Notification.id)).where(Notification.user_empid == user_empid)
        )
        remaining_count = verify_result.scalar() or 0
        
        logger.info(f"Remaining notifications after deletion: {remaining_count}")
        
        return {
            "deleted_count": result.rowcount,
            "remaining_count": remaining_count,
            "success": True,
            "message": f"Deleted {result.rowcount} notification(s)"
        }
    except Exception as e:
        logger.error(f"Error deleting notifications: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete notifications: {str(e)}"
        )

@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: int,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """
    Delete a specific notification.
    
    Args:
        notification_id: ID of the notification to delete
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: 404 if notification not found or doesn't belong to user
    """
    user_empid = current_user.get("username")
    
    # Get the notification
    result = await db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_empid == user_empid
        )
    )
    notification = result.scalar_one_or_none()
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )
    
    await db.delete(notification)
    await db.commit()
    
    return {"message": "Notification deleted successfully"}

