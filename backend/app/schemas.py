"""
Pydantic Schemas Module

Purpose: Define request/response data models using Pydantic for API validation
Features:
- Request validation for API endpoints
- Response serialization
- Type safety and automatic documentation
- Data transformation between API and database models

Schemas:
- User schemas: Registration, login, and response models
- Additional Skills: CRUD schemas for self-reported skills
- Training: Training creation and response schemas
- Training Requests: Request creation, update, and response schemas

@author Orbit Skill Development Team
@date 2025
"""

from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Optional

class UserRegister(BaseModel):
    """Schema for user registration request"""
    emp_id: str
    password: str

class UserLogin(BaseModel):
    """Schema for user login request"""
    username: str
    password: str

class UserResponse(BaseModel):
    """Schema for user response data"""
    username: str
    name: Optional[str] = None
    
    class Config:
        from_attributes = True

# Additional Skills Schemas
class AdditionalSkillBase(BaseModel):
    """Base schema for additional skills with common fields"""
    skill_name: str
    skill_level: str
    skill_category: str
    description: Optional[str] = None

class AdditionalSkillCreate(AdditionalSkillBase):
    pass

class AdditionalSkillUpdate(AdditionalSkillBase):
    skill_name: Optional[str] = None
    skill_level: Optional[str] = None
    skill_category: Optional[str] = None

class AdditionalSkillResponse(AdditionalSkillBase):
    id: int
    employee_empid: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# --- Schemas for Training Feature ---

# CORRECTED: Added all the fields sent by the Angular form to match the request
class TrainingCreate(BaseModel):
    division: Optional[str] = None
    department: Optional[str] = None
    competency: Optional[str] = None
    skill: Optional[str] = None
    training_name: str
    training_topics: Optional[str] = None
    prerequisites: Optional[str] = None
    skill_category: Optional[str] = None
    training_date: Optional[date] = None
    duration: Optional[str] = None
    time: Optional[str] = None
    training_type: Optional[str] = None
    seats: Optional[str] = None
    assessment_details: Optional[str] = None

class TrainingResponse(TrainingCreate):
    id: int
    trainer_name: Optional[str] = None 
    email: Optional[str] = None

    class Config:
        from_attributes = True

# --- Schemas for Training Requests (Exploration Path) ---

class TrainingRequestCreate(BaseModel):
    training_id: int

class TrainingRequestResponse(BaseModel):
    id: int
    training_id: int
    employee_empid: str
    manager_empid: str
    request_date: datetime
    status: str
    manager_notes: Optional[str] = None
    response_date: Optional[datetime] = None
    training: TrainingResponse
    employee: Optional[UserResponse] = None

    class Config:
        from_attributes = True

class TrainingRequestUpdate(BaseModel):
    status: str  # approved, rejected
    manager_notes: Optional[str] = None

# --- Schemas for Notifications ---

class NotificationCreate(BaseModel):
    """Schema for creating a notification"""
    user_empid: str
    title: str
    message: str
    type: str = "info"  # info, success, warning, error, assignment, approval, etc.
    related_id: Optional[int] = None
    related_type: Optional[str] = None
    action_url: Optional[str] = None

class NotificationResponse(BaseModel):
    """Schema for notification response"""
    id: int
    user_empid: str
    title: str
    message: str
    type: str
    is_read: bool
    related_id: Optional[int] = None
    related_type: Optional[str] = None
    action_url: Optional[str] = None
    created_at: datetime
    read_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class NotificationUpdate(BaseModel):
    """Schema for updating notification (mark as read)"""
    is_read: bool = True
