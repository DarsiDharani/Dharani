"""
Dashboard Routes Module

Purpose: API routes for engineer and manager dashboard data
Features:
- Engineer dashboard data endpoint
- Manager dashboard data endpoint (with team information)
- Skill update functionality for managers
- Additional skills management

Endpoints:
- GET /data/engineer: Get engineer dashboard data
- GET /data/manager/dashboard: Get manager dashboard data
- PATCH /data/manager/team-skill: Update team member skill levels

@author Orbit Skill Development Team
@date 2025
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from app.database import get_db_async
# Ensure you import your AdditionalSkill model
from app.models import User, ManagerEmployee, EmployeeCompetency, AdditionalSkill
from app.auth_utils import get_current_active_user, get_current_active_manager
from pydantic import BaseModel

# Create a single router for both endpoints with a common prefix
router = APIRouter(prefix="/data", tags=["Dashboard"])

# Pydantic models for API requests
class SkillUpdateRequest(BaseModel):
    """Request schema for updating team member skill levels"""
    employee_username: str
    skill_name: str
    current_expertise: str
    target_expertise: str

# Helper function to get status based on string levels
def get_status_from_levels(current_level_str: str, target_level_str: str) -> str:
    """
    Compares two level strings (e.g., 'L0', 'L2' or 'Beginner', 'Expert') to determine competency status.
    
    Supports both L-format (L1-L5) and text format (Beginner, Intermediate, Advanced, Expert).
    Returns 'Met' if current >= target, 'Gap' if current < target, 'Error' if invalid.
    
    Args:
        current_level_str: Current expertise level
        target_level_str: Target expertise level
        
    Returns:
        str: 'Met', 'Gap', or 'Error'
    """
    # Check for None values and return 'Error' immediately
    if current_level_str is None or target_level_str is None:
        return "Error"

    def convert_to_numeric(level_str: str) -> int:
        """Convert level string to numeric value for comparison"""
        level_str = level_str.strip()

        # Handle L-format (L0, L1, L2, L3, L4, L5)
        if level_str.upper().startswith('L'):
            return int(level_str.upper().lstrip('L'))

        # Handle text format (Beginner, Intermediate, Advanced, Expert)
        level_mapping = {
            'BEGINNER': 1,
            'INTERMEDIATE': 2,
            'ADVANCED': 3,
            'EXPERT': 4
        }
        return level_mapping.get(level_str.upper(), -1)

    try:
        current_level_num = convert_to_numeric(current_level_str)
        target_level_num = convert_to_numeric(target_level_str)

        # If either conversion failed, return Error
        if current_level_num == -1 or target_level_num == -1:
            return "Error"

        if current_level_num >= target_level_num:
            return "Met"
        else:
            return "Gap"
    except (ValueError, IndexError, TypeError):
        return "Error"

@router.get("/manager/dashboard")
async def get_manager_data(
    current_user: dict = Depends(get_current_active_manager),
    db: AsyncSession = Depends(get_db_async)
):
    """
    Fetches dashboard data for a manager, including their own skills 
    and their team's core AND additional skills.
    """
    manager_username = current_user.get("username")

    # Resolve manager display name from ManagerEmployee table if available
    manager_name_result = await db.execute(
        select(ManagerEmployee.manager_name).where(ManagerEmployee.manager_empid == manager_username)
    )
    manager_name_row = manager_name_result.first()
    manager_display_name = manager_name_row[0] if manager_name_row and manager_name_row[0] else manager_username

    # Fetch manager's own skills
    manager_skills_result = await db.execute(
        select(EmployeeCompetency).where(EmployeeCompetency.employee_empid == manager_username)
    )
    manager_skills_orm = manager_skills_result.scalars().all()
    manager_skills_list = [
        {
            "skill": comp.skill, "competency": comp.competency,
            "current_expertise": comp.current_expertise, "target_expertise": comp.target_expertise,
            "status": get_status_from_levels(comp.current_expertise, comp.target_expertise)
        } for comp in manager_skills_orm
    ]

    # Fetch manager_is_trainer from ManagerEmployee table
    manager_trainer_result = await db.execute(
        select(ManagerEmployee.manager_is_trainer).where(ManagerEmployee.manager_empid == manager_username)
    )
    manager_is_trainer_row = manager_trainer_result.first()
    manager_is_trainer = manager_is_trainer_row[0] if manager_is_trainer_row else False
    
    # Step 1: Get all employee IDs and names reporting to the current manager
    manager_relations_result = await db.execute(
        select(ManagerEmployee.employee_empid, ManagerEmployee.employee_name).where(ManagerEmployee.manager_empid == manager_username)
    )
    team_members = manager_relations_result.all()
    team_member_usernames = [member.employee_empid for member in team_members]
    team_member_names = {member.employee_empid: member.employee_name for member in team_members}

    # Step 2: Prepare the base structure for all team members, including 'additional_skills'
    team_members_data = {
        username: {"id": username, "name": team_member_names.get(username, username), "skills": [], "additional_skills": []}
        for username in team_member_usernames
    }
    
    if not team_member_usernames:
        return {
            "name": manager_display_name,
            "role": "manager",
            "id": manager_username,
            "skills": manager_skills_list,
            "team": [],
            "manager_is_trainer": manager_is_trainer
        }

    # Step 3: Fetch all CORE competency data for the team members in a single query
    competencies_result = await db.execute(
        select(EmployeeCompetency).where(EmployeeCompetency.employee_empid.in_(team_member_usernames))
    )
    competencies_data = competencies_result.scalars().all()
    
    # Step 4: Populate the CORE skills for each team member
    for competency in competencies_data:
        username = competency.employee_empid
        if username in team_members_data:
            status_val = get_status_from_levels(competency.current_expertise, competency.target_expertise)
            
            team_members_data[username]["skills"].append({
                "skill": competency.skill,
                "competency": competency.competency,
                "current_expertise": competency.current_expertise,
                "target_expertise": competency.target_expertise,
                "status": status_val
            })
    
    # Step 5: Fetch all ADDITIONAL skill data for the team in a single query
    additional_skills_result = await db.execute(
        select(AdditionalSkill).where(AdditionalSkill.employee_empid.in_(team_member_usernames))
    )
    additional_skills_data = additional_skills_result.scalars().all()

    # Step 6: Populate the ADDITIONAL skills for each team member
    for add_skill in additional_skills_data:
        username = add_skill.employee_empid
        if username in team_members_data:
            team_members_data[username]["additional_skills"].append({
                "id": add_skill.id,
                "skill_name": add_skill.skill_name,
                "skill_level": add_skill.skill_level,
                "skill_category": add_skill.skill_category,
                "description": add_skill.description,
                "created_at": add_skill.created_at.isoformat() if add_skill.created_at else None
            })

    return {
        "name": manager_display_name,
        "role": "manager",
        "id": manager_username,
        "skills": manager_skills_list,
        "team": list(team_members_data.values()),
        "manager_is_trainer": manager_is_trainer
    }

@router.get("/engineer")
async def get_engineer_data(
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_async)
):
    """
    Fetches skill competency data for a single engineer.
    """
    employee_username = current_user.get("username")

    if current_user.get("role") != "employee":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource"
        )

    # MODIFIED: Fetch employee's name and trainer status in one query
    employee_details_result = await db.execute(
        select(
            ManagerEmployee.employee_name,
            ManagerEmployee.employee_is_trainer
        ).where(ManagerEmployee.employee_empid == employee_username)
    )
    employee_details = employee_details_result.first()

    # Safely unpack details, providing default values if the employee is not found
    employee_name = employee_details.employee_name if employee_details else None
    is_trainer = employee_details.employee_is_trainer if employee_details else False
    
    # Fetch user ID
    user_id_result = await db.execute(
        select(User.id).where(User.username == employee_username)
    )
    user_id = user_id_result.scalar_one_or_none()

    # Fetch employee's competencies
    competencies_result = await db.execute(
        select(EmployeeCompetency).where(EmployeeCompetency.employee_empid == employee_username)
    )
    competencies_orm = competencies_result.scalars().all()

    skills_list = [
        {
            "id": comp.id,
            "skill": comp.skill,
            "current_expertise": comp.current_expertise,
            "target_expertise": comp.target_expertise,
            "status": get_status_from_levels(comp.current_expertise, comp.target_expertise),
        }
        for comp in competencies_orm
    ]

    # MODIFIED: Added 'employee_is_trainer' to the response
    return {
        "username": employee_username,
        "employee_name": employee_name,
        "employee_id": user_id,
        "employee_is_trainer": is_trainer,
        "skills": skills_list
    }

@router.put("/manager/team-skill")
async def update_team_member_skill(
    skill_update: SkillUpdateRequest,
    current_manager: dict = Depends(get_current_active_manager),
    db: AsyncSession = Depends(get_db_async)
):
    """
    Update a team member's skill current and target expertise levels.
    """
    try:
        # Verify the employee is part of the manager's team
        team_check_result = await db.execute(
            select(ManagerEmployee.employee_empid)
            .where(
                ManagerEmployee.manager_empid == current_manager['username'],
                ManagerEmployee.employee_empid == skill_update.employee_username
            )
        )
        if not team_check_result.first():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only update skills for your team members"
            )

        new_status = get_status_from_levels(
            skill_update.current_expertise,
            skill_update.target_expertise
        )

        update_stmt = (
            update(EmployeeCompetency)
            .where(
                EmployeeCompetency.employee_empid == skill_update.employee_username,
                EmployeeCompetency.skill == skill_update.skill_name
            )
            .values(
                current_expertise=skill_update.current_expertise,
                target_expertise=skill_update.target_expertise
            )
        )
        result = await db.execute(update_stmt)

        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Skill not found for this employee"
            )

        await db.commit()

        return {
            "message": "Skill updated successfully",
            "employee_username": skill_update.employee_username,
            "skill_name": skill_update.skill_name,
            "current_expertise": skill_update.current_expertise,
            "target_expertise": skill_update.target_expertise,
            "status": new_status
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update skill: {str(e)}"
        )
    