"""
SQLAlchemy Database Models

Purpose: Define database schema using SQLAlchemy ORM
Contains all table definitions and relationships for the Orbit Skill application

Models:
- User: User accounts with authentication
- ManagerEmployee: Manager-employee relationships
- EmployeeCompetency: Employee skill competencies and targets
- AdditionalSkill: Self-reported additional skills
- Trainer: Trainer information and expertise
- TrainingDetail: Training session details
- TrainingAssignment: Training assignments to employees
- TrainingRequest: Training approval requests
- AssignmentSubmission: Assignment exam submissions
- FeedbackSubmission: Training feedback submissions
- SharedAssignment: Shared assignments from trainers
- SharedFeedback: Shared feedback forms from trainers
- ManagerPerformanceFeedback: Manager feedback on employee performance

@author Orbit Skill Development Team
@date 2025
"""

from datetime import datetime, date
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Date, Boolean, Text
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class User(Base):
    """
    User model - Stores user account information and authentication data.
    
    Attributes:
        id: Primary key
        username: Unique employee ID (used for login)
        hashed_password: Bcrypt hashed password
        created_at: Account creation timestamp
    """
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ManagerEmployee(Base):
    __tablename__ = 'manager_employee'
    manager_empid = Column(String, ForeignKey('users.username'), primary_key=True)
    manager_name = Column(String)
    employee_empid = Column(String, ForeignKey('users.username'), primary_key=True)
    employee_name = Column(String)
    manager_is_trainer = Column(Boolean, default=False, nullable=False)
    employee_is_trainer = Column(Boolean, default=False, nullable=False)

class EmployeeCompetency(Base):
    __tablename__ = 'employee_competency'
    id = Column(Integer, primary_key=True, index=True)
    employee_empid = Column(String, ForeignKey('users.username'))
    employee_name = Column(String)
    department = Column(String)
    division = Column(String)
    project = Column(String)
    role_specific_comp = Column(String)
    destination = Column(String)
    competency = Column(String)
    skill = Column(String)
    current_expertise = Column(String)
    target_expertise = Column(String)
    comments = Column(String)
    target_date = Column(Date)
    email = Column(String, nullable=True)  # Email address from employee_competency table
    employee = relationship("User")

class AdditionalSkill(Base):
    __tablename__ = 'additional_skills'
    id = Column(Integer, primary_key=True, index=True)
    employee_empid = Column(String, ForeignKey('users.username'), nullable=False)
    skill_name = Column(String, nullable=False)
    skill_level = Column(String, nullable=False)
    skill_category = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    employee = relationship("User")

class Trainer(Base):
    __tablename__ = "trainers"
    id = Column(Integer, primary_key=True, index=True)
    skill = Column(String, nullable=False)
    competency = Column(String, nullable=False)
    trainer_name = Column(String, nullable=False)
    expertise_level = Column(String, nullable=False)

class TrainingDetail(Base):
    __tablename__ = "training_details"
    id = Column(Integer, primary_key=True, index=True)
    division = Column(String, nullable=True)
    department = Column(String, nullable=True)
    competency = Column(String, nullable=True)
    skill = Column(String, nullable=True)
    training_name = Column(String, nullable=False)
    training_topics = Column(String, nullable=True)
    prerequisites = Column(String, nullable=True)
    skill_category = Column(String, nullable=True)
    trainer_name = Column(String, nullable=False)
    email = Column(String, nullable=True)
    training_date = Column(Date, nullable=True) # CHANGED: From String to Date for proper sorting/filtering
    duration = Column(String, nullable=True)
    time = Column(String, nullable=True)
    training_type = Column(String, nullable=True)
    seats = Column(String, nullable=True)
    assessment_details = Column(String, nullable=True)

class TrainingAssignment(Base):
    __tablename__ = 'training_assignments'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    employee_empid = Column(String, ForeignKey('users.username'), nullable=False)
    manager_empid = Column(String, ForeignKey('users.username'), nullable=False)
    # Match existing DB column name 'assignment_date' (timestamp)
    assignment_date = Column(DateTime, default=datetime.utcnow)

class TrainingAttendance(Base):
    __tablename__ = 'training_attendance'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    employee_empid = Column(String, ForeignKey('users.username'), nullable=False)
    attended = Column(Boolean, default=False, nullable=False)
    marked_at = Column(DateTime, default=datetime.utcnow)
    # Relationships
    training = relationship("TrainingDetail")
    employee = relationship("User", foreign_keys=[employee_empid])

class TrainingRequest(Base):
    __tablename__ = 'training_requests'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    employee_empid = Column(String, ForeignKey('users.username'), nullable=False)
    manager_empid = Column(String, ForeignKey('users.username'), nullable=False)
    request_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default='pending')  # pending, approved, rejected
    manager_notes = Column(String, nullable=True)
    response_date = Column(DateTime, nullable=True)
    # Relationships
    training = relationship("TrainingDetail")
    employee = relationship("User", foreign_keys=[employee_empid])
    manager = relationship("User", foreign_keys=[manager_empid])

class SharedAssignment(Base):
    __tablename__ = 'shared_assignments'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    trainer_username = Column(String, ForeignKey('users.username'), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    assignment_data = Column(Text, nullable=False)  # JSON string storing questions and options
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    training = relationship("TrainingDetail")
    trainer = relationship("User", foreign_keys=[trainer_username])

class SharedFeedback(Base):
    __tablename__ = 'shared_feedback'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    trainer_username = Column(String, ForeignKey('users.username'), nullable=False)
    feedback_data = Column(Text, nullable=False)  # JSON string storing feedback questions
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    training = relationship("TrainingDetail")
    trainer = relationship("User", foreign_keys=[trainer_username])

class AssignmentSubmission(Base):
    __tablename__ = 'assignment_submissions'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    shared_assignment_id = Column(Integer, ForeignKey('shared_assignments.id'), nullable=False)
    employee_empid = Column(String, ForeignKey('users.username'), nullable=False)
    answers_data = Column(Text, nullable=False)  # JSON string storing user answers
    score = Column(Integer, nullable=True)  # Score out of 100
    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    # Relationships
    training = relationship("TrainingDetail")
    shared_assignment = relationship("SharedAssignment")
    employee = relationship("User", foreign_keys=[employee_empid])

class FeedbackSubmission(Base):
    __tablename__ = 'feedback_submissions'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    shared_feedback_id = Column(Integer, ForeignKey('shared_feedback.id'), nullable=False)
    employee_empid = Column(String, ForeignKey('users.username'), nullable=False)
    responses_data = Column(Text, nullable=False)  # JSON string storing feedback responses
    submitted_at = Column(DateTime, default=datetime.utcnow)
    # Relationships
    training = relationship("TrainingDetail")
    shared_feedback = relationship("SharedFeedback")
    employee = relationship("User", foreign_keys=[employee_empid])

class ManagerPerformanceFeedback(Base):
    __tablename__ = 'manager_performance_feedback'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    employee_empid = Column(String, ForeignKey('users.username'), nullable=False)
    manager_empid = Column(String, ForeignKey('users.username'), nullable=False)
    # Performance factors (ratings 1-5)
    application_of_training = Column(Integer, nullable=True)  # How effectively the employee is using the learned concepts/tools in real tasks
    quality_of_deliverables = Column(Integer, nullable=True)  # Impact of training on code quality, test quality, design accuracy, defect reduction, etc.
    problem_solving_capability = Column(Integer, nullable=True)  # Ability to apply trained methods to analyze issues, debug, and provide correct solutions
    productivity_independence = Column(Integer, nullable=True)  # Whether the employee completes tasks faster, with less support, and shows improved efficiency after training
    process_compliance_adherence = Column(Integer, nullable=True)  # Correct use of processes, tools, templates, and standards learned during training (ASPICE, ISO26262, etc.)
    improvement_areas = Column(Text, nullable=True)  # Areas that need improvement
    strengths = Column(Text, nullable=True)  # Key strengths demonstrated
    overall_performance = Column(Integer, nullable=False)  # Overall performance rating (1-5)
    additional_comments = Column(Text, nullable=True)  # Additional comments from manager
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Relationships
    training = relationship("TrainingDetail")
    employee = relationship("User", foreign_keys=[employee_empid])
    manager = relationship("User", foreign_keys=[manager_empid])

class TrainingQuestionFile(Base):
    __tablename__ = 'training_question_files'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    trainer_username = Column(String, ForeignKey('users.username'), nullable=False)
    file_path = Column(String, nullable=False)  # Path to the uploaded PDF file
    file_name = Column(String, nullable=False)  # Original filename
    file_size = Column(Integer, nullable=True)  # File size in bytes
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    # Relationships
    training = relationship("TrainingDetail")
    trainer = relationship("User", foreign_keys=[trainer_username])

class TrainingSolutionFile(Base):
    __tablename__ = 'training_solution_files'
    id = Column(Integer, primary_key=True, index=True)
    training_id = Column(Integer, ForeignKey('training_details.id'), nullable=False)
    employee_empid = Column(String, ForeignKey('users.username'), nullable=False)
    file_path = Column(String, nullable=False)  # Path to the uploaded PDF file
    file_name = Column(String, nullable=False)  # Original filename
    file_size = Column(Integer, nullable=True)  # File size in bytes
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    # Relationships
    training = relationship("TrainingDetail")
    employee = relationship("User", foreign_keys=[employee_empid])

