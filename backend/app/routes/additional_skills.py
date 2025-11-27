"""
Additional Skills Routes Module

Purpose: API routes for managing self-reported additional skills
Features:
- Get user's additional skills
- Create new additional skill
- Update existing additional skill
- Delete additional skill

Endpoints:
- GET /additional-skills/: Get all additional skills for current user
- POST /additional-skills/: Create a new additional skill
- PUT /additional-skills/{id}: Update an additional skill
- DELETE /additional-skills/{id}: Delete an additional skill

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
from app.database import get_db_async
from app.models import AdditionalSkill
from app.schemas import AdditionalSkillCreate, AdditionalSkillUpdate, AdditionalSkillResponse
from app.auth_utils import get_current_active_user

router = APIRouter(prefix="/additional-skills", tags=["Additional Skills"])

@router.get("/", response_model=List[AdditionalSkillResponse])
async def get_additional_skills(
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """Get all additional skills for the current user"""
    employee_empid = current_user.get("username")
    
    result = await db.execute(
        select(AdditionalSkill).where(AdditionalSkill.employee_empid == employee_empid)
    )
    skills = result.scalars().all()
    return skills

@router.post("/", response_model=AdditionalSkillResponse)
async def create_additional_skill(
    skill_data: AdditionalSkillCreate,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """Create a new additional skill for the current user"""
    employee_empid = current_user.get("username")
    
    # Check if skill already exists for this user
    existing_skill = await db.execute(
        select(AdditionalSkill).where(
            AdditionalSkill.employee_empid == employee_empid,
            AdditionalSkill.skill_name == skill_data.skill_name
        )
    )
    if existing_skill.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skill already exists for this user"
        )
    
    new_skill = AdditionalSkill(
        employee_empid=employee_empid,
        **skill_data.dict()
    )
    
    db.add(new_skill)
    await db.commit()
    await db.refresh(new_skill)
    
    return new_skill

@router.put("/{skill_id}", response_model=AdditionalSkillResponse)
async def update_additional_skill(
    skill_id: int,
    skill_data: AdditionalSkillUpdate,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """Update an additional skill"""
    employee_empid = current_user.get("username")
    
    # Get the skill and verify ownership
    result = await db.execute(
        select(AdditionalSkill).where(
            AdditionalSkill.id == skill_id,
            AdditionalSkill.employee_empid == employee_empid
        )
    )
    skill = result.scalar_one_or_none()
    
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Skill not found"
        )
    
    # Update fields
    update_data = skill_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(skill, field, value)
    
    await db.commit()
    await db.refresh(skill)
    
    return skill

@router.delete("/{skill_id}")
async def delete_additional_skill(
    skill_id: int,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """Delete an additional skill"""
    employee_empid = current_user.get("username")
    
    # Get the skill and verify ownership
    result = await db.execute(
        select(AdditionalSkill).where(
            AdditionalSkill.id == skill_id,
            AdditionalSkill.employee_empid == employee_empid
        )
    )
    skill = result.scalar_one_or_none()
    
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Skill not found"
        )
    
    await db.delete(skill)
    await db.commit()
    
    return {"message": "Skill deleted successfully"}
