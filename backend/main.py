"""
Orbit Skill Backend API - Main Application Entry Point

Purpose: FastAPI application for managing employee skills, trainings, and assignments
Features:
- User authentication and authorization
- Skill and competency management
- Training catalog and assignment management
- Assignment and feedback submission tracking
- Manager-employee relationship management
- Data import from Excel/CSV files

API Endpoints:
- /register: User registration
- /login: User authentication
- /data/engineer: Engineer dashboard data
- /data/manager/dashboard: Manager dashboard data
- /trainings/: Training CRUD operations
- /assignments/: Assignment management
- /training-requests/: Training request management
- /additional-skills/: Additional skills management
- /shared-content/: Shared assignments and feedback
- /upload-and-refresh: Excel data import
- /upload-manager-employee-csv: CSV data import

@author Orbit Skill Development Team
@date 2025
"""

import sys
import os

# --- ADDED: Fix ModuleNotFoundError: No module named 'app' ---
# This ensures the directory containing 'app' (which is 'backend') is on the Python path
# Required for proper module resolution when running the application
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
# -------------------------------------------------------------

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.routes import register, login, dashboard_routes, additional_skills, training_routes, assignment_routes, training_requests, shared_content_routes, training_files_routes
from app.database import AsyncSessionLocal, create_db_and_tables
from app.excel_loader import load_all_from_excel, load_manager_employee_from_csv

# --- Configuration ---
# Set up logging with timestamp and level information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Initialization ---
# Create FastAPI application instance with metadata
app = FastAPI(
    title="SkillOrbit API",
    description="API for managing skills and training data.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Configure CORS to allow requests from Angular frontend
# Update origins list for production deployment
origins = [
    "http://localhost:4200",  # Angular development server
    "http://127.0.0.1:4200",  # Alternative localhost
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --- API Routers ---
app.include_router(register.router)
app.include_router(login.router)
app.include_router(dashboard_routes.router)
app.include_router(additional_skills.router)
app.include_router(training_routes.router)
app.include_router(assignment_routes.router)
app.include_router(training_requests.router)
app.include_router(shared_content_routes.router)
app.include_router(training_files_routes.router)

# <<< NEW: Root Endpoint for Welcome Message >>>
@app.get("/", tags=["Default"])
async def read_root():
    """
    Root endpoint - Welcome message to confirm the API is running.
    
    Returns:
        dict: Welcome message with link to API documentation
    """
    return {"message": "Welcome to the SkillOrbit API. Please go to /docs for the API documentation."}

# <<< PERMANENT SOLUTION: File Upload Endpoint >>>
@app.post("/upload-and-refresh", status_code=200, tags=["Admin"])
async def upload_and_refresh_data(file: UploadFile = File(...)):
    """
    Admin endpoint: Upload Excel file and refresh database with training/competency data.
    
    Accepts an Excel file (.xlsx, .xls) containing:
    - Trainer data
    - Training details
    - Employee competencies
    
    The file is processed and data is loaded into the database, replacing existing records.
    
    Args:
        file: Excel file upload containing training and competency data
        
    Returns:
        dict: Success message with counts of inserted records
        
    Raises:
        HTTPException: 400 if file type is invalid or validation fails
        HTTPException: 500 if processing error occurs
    """
    logging.info(f"API: Received file '{file.filename}' for data refresh.")

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an Excel file.")

    try:
        async with AsyncSessionLocal() as db:
            await load_all_from_excel(file.file, db)
            
            # Verify data was inserted
            from sqlalchemy import select, func
            from app.models import Trainer, TrainingDetail, EmployeeCompetency
            
            trainers_result = await db.execute(select(func.count(Trainer.id)))
            trainers_count = trainers_result.scalar()
            trainings_result = await db.execute(select(func.count(TrainingDetail.id)))
            trainings_count = trainings_result.scalar()
            competencies_result = await db.execute(select(func.count(EmployeeCompetency.id)))
            competencies_count = competencies_result.scalar()
        
        logging.info(f"Successfully processed and loaded data from '{file.filename}'.")
        return {
            "message": f"Data from '{file.filename}' has been successfully uploaded and the database has been refreshed.",
            "trainers_inserted": trainers_count,
            "trainings_inserted": trainings_count,
            "employee_competencies_inserted": competencies_count,
            "status": "success"
        }
    
    except ValueError as ve:
        # This is raised when all rows are skipped
        logging.error(f"Validation error: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"An error occurred during file processing and database refresh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


@app.post("/upload-manager-employee-csv", status_code=200, tags=["Admin"])
async def upload_manager_employee_csv(file: UploadFile = File(...)):
    """
    Admin endpoint: Upload CSV file and load manager-employee relationships.
    
    Accepts a CSV file (.csv) containing manager-employee relationships.
    Expected CSV columns:
    - manager_empid: Manager's employee ID
    - manager_name: Manager's name
    - employee_empid: Employee's ID
    - employee_name: Employee's name
    - manager_is_trainer: Boolean indicating if manager is a trainer
    - employee_is_trainer: Boolean indicating if employee is a trainer
    
    Args:
        file: CSV file upload containing manager-employee relationship data
        
    Returns:
        dict: Success message with count of relationships inserted
        
    Raises:
        HTTPException: 400 if file type is invalid or validation fails
        HTTPException: 500 if processing error occurs
    """
    logging.info(f"API: Received CSV file '{file.filename}' for manager-employee data load.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        async with AsyncSessionLocal() as db:
            await load_manager_employee_from_csv(file.file, db)
            
            # Verify data was inserted
            from sqlalchemy import select, func
            from app.models import ManagerEmployee
            
            # Count all manager-employee relationships
            count_result = await db.execute(
                select(func.count()).select_from(ManagerEmployee)
            )
            total_count = count_result.scalar()
        
        logging.info(f"Successfully processed and loaded manager-employee data from '{file.filename}'.")
        return {
            "message": f"Manager-employee data from '{file.filename}' has been successfully uploaded and the database has been refreshed.",
            "relationships_inserted": total_count,
            "status": "success"
        }
    
    except ValueError as ve:
        logging.error(f"Validation error: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"An error occurred during CSV file processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- Application Lifecycle Events ---
@app.on_event("startup")
async def on_startup():
    """
    Application startup event handler.
    
    This function runs automatically when the FastAPI application starts.
    It initializes the database by creating all required tables if they don't exist.
    
    Actions:
    1. Initialize database connection
    2. Create all database tables (if not exist)
    3. Log startup completion
    """
    logging.info("STARTUP: Initializing database...")
    await create_db_and_tables()
    logging.info("STARTUP: Database initialization complete.")
    logging.info("STARTUP: Server is ready. Please go to /docs for the API documentation and to upload data.")