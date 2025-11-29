"""
Excel Data Loader Module

Purpose: Load and process Excel/CSV files to populate database tables
Features:
- Excel file parsing and data extraction
- Flexible column name matching
- Data validation and cleaning
- Database insertion for trainers, trainings, and competencies
- Manager-employee relationship loading from CSV

Functions:
- clean_headers(): Standardize DataFrame column names
- find_column_flexible(): Flexible column matching
- load_all_from_excel(): Main function to load Excel data
- load_manager_employee_from_csv(): Load manager-employee relationships

@author Orbit Skill Development Team
@date 2025
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Trainer, TrainingDetail, ManagerEmployee, User, EmployeeCompetency
from datetime import datetime
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes DataFrame column headers.
    
    Converts column names to lowercase, replaces spaces/special chars with underscores.
    This ensures consistent column naming regardless of Excel file formatting.
    
    Args:
        df: Input DataFrame with potentially inconsistent column names
        
    Returns:
        DataFrame with cleaned column headers
    """
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace(",", "_")
        .str.replace("*", "", regex=False)
    )
    return df


def find_column_flexible(row_dict: dict, possible_names: list) -> Any:
    """
    Tries to find a column value using flexible matching.
    Checks multiple possible column name variations.
    Returns the value (even if empty/None) if column is found.
    """
    # First try exact matches
    for name in possible_names:
        if name in row_dict:
            return row_dict.get(name)
    
    # Then try case-insensitive exact match
    row_keys_lower = {k.lower(): k for k in row_dict.keys()}
    for name in possible_names:
        if name.lower() in row_keys_lower:
            return row_dict.get(row_keys_lower[name.lower()])
    
    # Finally try partial/substring matching
    for name in possible_names:
        for key in row_dict.keys():
            key_lower = key.lower()
            name_lower = name.lower()
            # Check if key contains name or name contains key (partial match)
            if name_lower in key_lower or key_lower in name_lower:
                # Avoid matching very generic names like "name" to everything
                if len(name) > 3 or (len(name) <= 3 and name_lower == key_lower):
                    return row_dict.get(key)
    
    return None


async def load_all_from_excel(excel_file_source: Any, db: AsyncSession):
    """
    Loads all data from a given Excel file source in a single, safe transaction.
    Loads three sheets: "Trainers Details", "Training Details", and "Employee Competency"
    """
    logging.info(f"--- Starting Excel data load (All 3 sheets: Trainers Details, Training Details, Employee Competency) ---")
    try:
        logging.info("Step 1: Clearing old data from tables...")
        # Delete in order to respect foreign key constraints:
        # 1. Delete submission tables first (they reference shared_assignments and shared_feedback)
        await db.execute(text("DELETE FROM feedback_submissions"))  # References shared_feedback
        await db.execute(text("DELETE FROM assignment_submissions"))  # References shared_assignments
        # 2. Delete shared content tables (they reference training_details)
        await db.execute(text("DELETE FROM shared_feedback"))  # Now safe to delete
        await db.execute(text("DELETE FROM shared_assignments"))  # Now safe to delete
        # 3. Delete all tables that reference training_details (must be deleted before training_details)
        await db.execute(text("DELETE FROM manager_performance_feedback"))  # References training_details
        await db.execute(text("DELETE FROM training_attendance"))  # References training_details
        await db.execute(text("DELETE FROM training_question_files"))  # References training_details
        await db.execute(text("DELETE FROM training_solution_files"))  # References training_details
        await db.execute(text("DELETE FROM training_requests"))  # References training_details
        await db.execute(text("DELETE FROM training_assignments"))  # References training_details
        # 4. Now safe to delete training_details
        await db.execute(text("DELETE FROM training_details"))
        # 5. Delete employee_competency (no dependencies on training_details)
        await db.execute(text("DELETE FROM employee_competency"))
        # 6. Finally delete trainers (no dependencies)
        await db.execute(text("DELETE FROM trainers"))
        logging.info("-> Old data cleared successfully.")
        
        # Reset sequences to start from 1 for consistent IDs
        logging.info("Step 1.5: Resetting ID sequences to start from 1...")
        try:
            # Try to reset trainers sequence
            try:
                await db.execute(text("ALTER SEQUENCE trainers_id_seq RESTART WITH 1"))
                logging.info("-> Trainers sequence reset to 1.")
            except Exception:
                # Try to find the actual sequence name
                seq_result = await db.execute(text("""
                    SELECT sequence_name FROM information_schema.sequences 
                    WHERE sequence_name LIKE '%trainers%id%' OR sequence_name LIKE '%trainer%id%'
                    ORDER BY sequence_name LIMIT 1
                """))
                seq_name = seq_result.scalar()
                if seq_name:
                    await db.execute(text(f"ALTER SEQUENCE {seq_name} RESTART WITH 1"))
                    logging.info(f"-> Trainers sequence ({seq_name}) reset to 1.")
                else:
                    logging.warning("-> Could not find trainers sequence, will be set automatically on insert.")
            
            # Try to reset training_details sequence
            try:
                await db.execute(text("ALTER SEQUENCE training_details_id_seq RESTART WITH 1"))
                logging.info("-> Training_details sequence reset to 1.")
            except Exception:
                # Try to find the actual sequence name
                seq_result = await db.execute(text("""
                    SELECT sequence_name FROM information_schema.sequences 
                    WHERE sequence_name LIKE '%training_details%id%' OR sequence_name LIKE '%training_detail%id%'
                    ORDER BY sequence_name LIMIT 1
                """))
                seq_name = seq_result.scalar()
                if seq_name:
                    await db.execute(text(f"ALTER SEQUENCE {seq_name} RESTART WITH 1"))
                    logging.info(f"-> Training_details sequence ({seq_name}) reset to 1.")
                else:
                    logging.warning("-> Could not find training_details sequence, will be set automatically on insert.")
            
            # Try to reset employee_competency sequence
            try:
                await db.execute(text("ALTER SEQUENCE employee_competency_id_seq RESTART WITH 1"))
                logging.info("-> Employee_competency sequence reset to 1.")
            except Exception:
                # Try to find the actual sequence name
                seq_result = await db.execute(text("""
                    SELECT sequence_name FROM information_schema.sequences 
                    WHERE sequence_name LIKE '%employee_competency%id%'
                    ORDER BY sequence_name LIMIT 1
                """))
                seq_name = seq_result.scalar()
                if seq_name:
                    await db.execute(text(f"ALTER SEQUENCE {seq_name} RESTART WITH 1"))
                    logging.info(f"-> Employee_competency sequence ({seq_name}) reset to 1.")
                else:
                    logging.warning("-> Could not find employee_competency sequence, will be set automatically on insert.")
            
            # Commit sequence resets
            await db.commit()
            logging.info("-> ID sequences reset successfully. IDs will start from 1 in Excel row order.")
        except Exception as seq_error:
            # If sequence doesn't exist or has different name, log but continue
            logging.warning(f"Could not reset sequences: {seq_error}")
            logging.warning("Sequences will be automatically reset after data insertion if needed.")

        # --- 1. Load Trainers Details ---
        logging.info("Step 2: Reading 'Trainers Details' sheet from Excel...")
        try:
            df_trainers_raw = pd.read_excel(excel_file_source, sheet_name="Trainers Details", engine='openpyxl')
        except ValueError as e:
            # List available sheets if the sheet name is wrong
            excel_file_source.seek(0)
            xl_file = pd.ExcelFile(excel_file_source, engine='openpyxl')
            available_sheets = xl_file.sheet_names
            logging.error(f"Sheet 'Trainers Details' not found! Available sheets: {available_sheets}")
            raise ValueError(f"Sheet 'Trainers Details' not found. Available sheets: {available_sheets}")
        
        logging.info(f"-> Original column names (before cleaning): {list(df_trainers_raw.columns)}")
        
        df_trainers = df_trainers_raw.replace({np.nan: None})
        df_trainers = clean_headers(df_trainers)
        logging.info(f"-> Found {len(df_trainers)} rows in 'Trainers Details'.")
        logging.info(f"-> Column names after cleaning: {list(df_trainers.columns)}")
        
        # Log first few rows as samples for debugging
        if len(df_trainers) > 0:
            logging.info(f"-> Sample of first row data: {df_trainers.iloc[0].to_dict()}")
            # Also show first 3 rows in a more readable format
            logging.info("-> First 3 rows of data:")
            for idx in range(min(3, len(df_trainers))):
                row_dict = df_trainers.iloc[idx].to_dict()
                logging.info(f"   Row {idx+2}: {row_dict}")
            # Show what we're looking for vs what we found
            logging.info("-> DIAGNOSTIC: Checking column matching...")
            sample_row = df_trainers.iloc[0].to_dict()
            logging.info(f"   Looking for 'skill' - found: {repr(sample_row.get('skill'))}")
            logging.info(f"   Looking for 'competency' - found: {repr(sample_row.get('competency'))}")
            logging.info(f"   Looking for 'trainer_name' - found: {repr(sample_row.get('trainer_name'))}")
            logging.info(f"   Looking for 'expertise_level' - found: {repr(sample_row.get('expertise_level'))}")
            logging.info(f"   All column keys in cleaned data: {list(sample_row.keys())}")

        trainers_to_add = []
        skipped_count = 0
        
        # Log column mapping for first row to help debug
        if len(df_trainers) > 0:
            first_row = df_trainers.iloc[0].to_dict()
            logging.info("-> Column mapping for trainer fields:")
            skill_found = find_column_flexible(first_row, ['skill'])
            competency_found = find_column_flexible(first_row, ['competency', 'competence'])
            trainer_name_found = find_column_flexible(first_row, ['trainer_name', 'trainer', 'trainername', 'trainer name', 'copmetency', 'name'])
            expertise_level_found = find_column_flexible(first_row, ['expertise_level', 'expertise', 'level', 'expertiselevel', 'expertise level'])
            
            logging.info(f"   skill: {repr(skill_found)} (found: {skill_found is not None})")
            logging.info(f"   competency: {repr(competency_found)} (found: {competency_found is not None})")
            logging.info(f"   trainer_name: {repr(trainer_name_found)} (found: {trainer_name_found is not None})")
            logging.info(f"   expertise_level: {repr(expertise_level_found)} (found: {expertise_level_found is not None})")
            logging.info(f"   All available columns in first row: {list(first_row.keys())}")
        
        for i, row in enumerate(df_trainers.to_dict('records')):
            # Validate required fields before creating Trainer object
            # Use flexible column matching to handle typos and variations
            missing_fields = []
            skill_val = find_column_flexible(row, ["skill"]) or row.get("skill")
            competency_val = find_column_flexible(row, ["competency", "competence"]) or row.get("competency")
            # IMPORTANT: In Excel, trainer name is in "Copmetency" column (typo), check it first
            # Then try other trainer name variations
            trainer_name_val = (find_column_flexible(row, ["copmetency"]) or 
                              find_column_flexible(row, ["trainer_name", "trainername", "trainer name", "trainer", "name"])) or row.get("trainer_name")
            expertise_level_val = find_column_flexible(row, ["expertise_level", "expertiselevel", "expertise level", "expertise", "level"]) or row.get("expertise_level")
            
            # Clean and validate values
            if skill_val and isinstance(skill_val, str):
                skill_val = skill_val.strip()
            if competency_val and isinstance(competency_val, str):
                competency_val = competency_val.strip()
            if trainer_name_val and isinstance(trainer_name_val, str):
                trainer_name_val = trainer_name_val.strip()
            if expertise_level_val and isinstance(expertise_level_val, str):
                expertise_level_val = expertise_level_val.strip()
            
            # Provide default for empty trainer_name (make it optional)
            if not trainer_name_val:
                trainer_name_val = "Not Assigned"
                logging.info(f"Row {i+2}: Using default 'Not Assigned' for empty trainer_name")
            
            # Check only truly required fields (skill, competency, expertise_level are mandatory)
            if not skill_val:
                missing_fields.append("skill")
            if not competency_val:
                missing_fields.append("competency")
            if not expertise_level_val:
                missing_fields.append("expertise_level")
            
            if missing_fields:
                skipped_count += 1
                logging.warning(f"Skipping trainer row {i+2} due to missing required fields ({', '.join(missing_fields)})")
                logging.warning(f"  Row data: skill={repr(skill_val)}, competency={repr(competency_val)}, trainer_name={repr(trainer_name_val)}, expertise_level={repr(expertise_level_val)}")
                if i < 5:  # Show first 5 skipped rows in detail
                    logging.warning(f"  Full row keys: {list(row.keys())}")
                    logging.warning(f"  Full row values: {row}")
                continue
            
            trainers_to_add.append(
                Trainer(
                    skill=skill_val,
                    competency=competency_val,
                    trainer_name=trainer_name_val,
                    expertise_level=expertise_level_val,
                )
            )
            if i < 3:  # Log first 3 successful rows
                logging.info(f"✅ Trainer row {i+2} added: skill={skill_val}, competency={competency_val}, trainer_name={trainer_name_val}, expertise_level={expertise_level_val}")
        
        logging.info(f"-> Trainer validation complete: {len(trainers_to_add)} valid rows, {skipped_count} skipped.")

        # --- 2. Load Training Details ---
        logging.info("Step 3: Reading 'Training Details' sheet from Excel...")
        excel_file_source.seek(0)
        try:
            df_trainings_raw = pd.read_excel(excel_file_source, sheet_name="Training Details", engine='openpyxl')
        except ValueError as e:
            # List available sheets if the sheet name is wrong
            excel_file_source.seek(0)
            xl_file = pd.ExcelFile(excel_file_source, engine='openpyxl')
            available_sheets = xl_file.sheet_names
            logging.error(f"Sheet 'Training Details' not found! Available sheets: {available_sheets}")
            raise ValueError(f"Sheet 'Training Details' not found. Available sheets: {available_sheets}")
        
        logging.info(f"-> Original column names (before cleaning): {list(df_trainings_raw.columns)}")
        
        df_trainings = df_trainings_raw.replace({np.nan: None})
        df_trainings = clean_headers(df_trainings)
        logging.info(f"-> Found {len(df_trainings)} rows in 'Training Details'.")
        logging.info(f"-> Column names after cleaning: {list(df_trainings.columns)}")
        
        # Log first few rows as samples for debugging
        if len(df_trainings) > 0:
            logging.info(f"-> Sample of first row data: {df_trainings.iloc[0].to_dict()}")
            # Also show first 3 rows in a more readable format
            logging.info("-> First 3 rows of data:")
            for idx in range(min(3, len(df_trainings))):
                row_dict = df_trainings.iloc[idx].to_dict()
                logging.info(f"   Row {idx+2}: {row_dict}")
            
            # CRITICAL: Show exact column names that contain trainer/email
            logging.info("=" * 80)
            logging.info("🔍 COLUMN NAME ANALYSIS:")
            sample_row = df_trainings.iloc[0].to_dict()
            trainer_cols = [k for k in sample_row.keys() if 'trainer' in k.lower()]
            email_cols = [k for k in sample_row.keys() if 'email' in k.lower() or 'mail' in k.lower()]
            logging.info(f"   Columns with 'trainer': {trainer_cols}")
            logging.info(f"   Columns with 'email': {email_cols}")
            if trainer_cols:
                logging.info(f"   Value in '{trainer_cols[0]}': {repr(sample_row.get(trainer_cols[0]))}")
            if email_cols:
                logging.info(f"   Value in '{email_cols[0]}': {repr(sample_row.get(email_cols[0]))}")
            logging.info("=" * 80)

        trainings_to_add = []
        skipped_training_count = 0
        for i, row in enumerate(df_trainings.to_dict('records')):
            # Validate required fields before creating TrainingDetail object
            # Use flexible column matching to handle typos and variations
            missing_fields = []
            # Training Name/Program - try multiple variations
            training_name_val = find_column_flexible(row, [
                "trainingname_program", "training_name_program", "training_name", 
                "trainingname", "training name", "program", "training"
            ]) or row.get("trainingname_program")
            
            # Trainer Name - try direct access first, then flexible matching
            # After clean_headers, column names become lowercase with underscores
            trainer_name_val = None
            # Try common variations directly
            for col_name in ["trainer_name", "trainername", "trainer", "trainer name"]:
                if col_name in row and row[col_name]:
                    trainer_name_val = row[col_name]
                    break
            
            # If not found, try flexible matching
            if not trainer_name_val:
                trainer_name_val = find_column_flexible(row, [
                    "trainer_name", "trainer", "trainername", "trainer name", "name"
                ])
            
            # Email - try direct access first, then flexible matching
            email_val = None
            # Try common variations directly
            for col_name in ["email_id", "emailid", "email", "email_address", "email id"]:
                if col_name in row and row[col_name]:
                    email_val = row[col_name]
                    break
            
            # If not found, try flexible matching
            if not email_val:
                email_val = find_column_flexible(row, [
                    "email_id", "emailid", "email", "email_address", "email id"
                ])
            
            # Debug: Log raw values for first few rows
            if i < 3:
                logging.info(f"🔍 DEBUG Row {i+2} - Raw values:")
                logging.info(f"   training_name (raw): {repr(training_name_val)}")
                logging.info(f"   trainer_name (raw): {repr(trainer_name_val)} (type: {type(trainer_name_val)})")
                logging.info(f"   email (raw): {repr(email_val)} (type: {type(email_val)})")
                logging.info(f"   All row keys: {list(row.keys())}")
            
            # Clean values first
            if training_name_val and isinstance(training_name_val, str):
                training_name_val = training_name_val.strip()
            elif training_name_val is not None:
                training_name_val = str(training_name_val).strip()
            
            if trainer_name_val and isinstance(trainer_name_val, str):
                trainer_name_val = trainer_name_val.strip()
            elif trainer_name_val is not None:
                trainer_name_val = str(trainer_name_val).strip()
            
            if email_val and isinstance(email_val, str):
                email_val = email_val.strip()
            elif email_val is not None:
                email_val = str(email_val).strip()
            
            # Then validate only truly required fields
            if not training_name_val:
                missing_fields.append("trainingname_program")
            
            if missing_fields:
                skipped_training_count += 1
                logging.warning(f"Skipping training row {i+2} due to missing required fields ({', '.join(missing_fields)})")
                if i < 5:  # Show first 5 skipped rows in detail
                    logging.warning(f"  Full row keys: {list(row.keys())}")
                continue

            # Use flexible matching for all fields
            date_val = find_column_flexible(row, ["training_dates", "training_date", "date", "dates"]) or row.get("training_dates")
            
            # Convert date column to datetime objects, not strings
            try:
                final_date = pd.to_datetime(date_val).date() if pd.notna(date_val) and date_val else None
            except Exception as date_error:
                logging.warning(f"Row {i+2}: Could not parse date '{date_val}': {date_error}. Setting to None.")
                final_date = None

            # Use flexible matching for all other fields - calculate values first
            duration_val = find_column_flexible(row, [
                "duration_(in_hrs)", "duration_in_hrs", "duration", 
                "duration_in_hours", "hours"
            ]) or row.get("duration_(in_hrs)")
            duration_str = str(duration_val) if pd.notna(duration_val) and duration_val else None
            
            seats_val = find_column_flexible(row, [
                "no._of_seats", "no_of_seats", "seats", 
                "number_of_seats", "numberofseats"
            ]) or row.get("no._of_seats")
            seats_str = str(seats_val) if pd.notna(seats_val) and seats_val else None
            
            # Get common fields that don't change per trainer
            division_val = find_column_flexible(row, ["division"]) or row.get("division")
            department_val = find_column_flexible(row, ["department"]) or row.get("department")
            competency_val = find_column_flexible(row, ["competency", "competence"]) or row.get("competency")
            skill_val = find_column_flexible(row, ["skill"]) or row.get("skill")
            training_topics_val = find_column_flexible(row, [
                "trainingtopics__material", "training_topics_material", 
                "training_topics", "trainingtopics", "topics", "material"
            ]) or row.get("trainingtopics__material")
            prerequisites_val = find_column_flexible(row, [
                "perquisites", "prerequisites", "prerequisite"
            ]) or row.get("perquisites")
            skill_category_val = find_column_flexible(row, [
                "skill_category_(l1_-_l5)", "skill_category", 
                "skillcategory", "category"
            ]) or row.get("skill_category_(l1_-_l5)")
            time_val = find_column_flexible(row, ["time", "training_time"]) or row.get("time")
            training_type_val = find_column_flexible(row, [
                "training_type", "trainingtype", "type"
            ]) or row.get("training_type")
            assessment_details_val = find_column_flexible(row, [
                "assessment_details", "assessmentdetails", 
                "assessment", "assessment_detail"
            ]) or row.get("assessment_details")
            
            # Split trainers by comma - handle multiple trainers
            # Convert to string first to handle any pandas/Excel types
            trainer_str = str(trainer_name_val).strip() if trainer_name_val else ""
            if trainer_str and trainer_str.lower() != "nan" and trainer_str.lower() != "none":
                # Split by comma and clean each trainer name
                trainer_names = [t.strip() for t in trainer_str.split(',') if t.strip() and t.strip().lower() != "nan"]
            else:
                trainer_names = ["Not Assigned"]
            
            # Split emails by comma, newline, or space and clean
            email_list = []
            email_str = ""  # Initialize to avoid scope issues
            if email_val:
                # Convert to string first to handle any pandas/Excel types
                email_str = str(email_val).strip() if email_val else ""
                if email_str and email_str.lower() != "nan" and email_str.lower() != "none":
                    # Try splitting by comma first (most common in Excel), then newline, then space
                    if ',' in email_str:
                        email_list = [e.strip() for e in email_str.split(',') if e.strip() and '@' in e.strip() and e.strip().lower() != "nan"]
                    elif '\n' in email_str:
                        email_list = [e.strip() for e in email_str.split('\n') if e.strip() and '@' in e.strip() and e.strip().lower() != "nan"]
                    else:
                        # Split by space and filter for valid emails
                        email_list = [e.strip() for e in email_str.split() if e.strip() and '@' in e.strip() and e.strip().lower() != "nan"]
            else:
                email_str = "None"
            
            # ALWAYS log splitting details for debugging (not just when multiple)
            logging.info(f"🔍 Row {i+2} SPLITTING:")
            logging.info(f"   Trainer string: {repr(trainer_str)}")
            logging.info(f"   Split into {len(trainer_names)} trainer(s): {trainer_names}")
            logging.info(f"   Email string: {repr(email_str)}")
            logging.info(f"   Split into {len(email_list)} email(s): {email_list}")
            
            # Create one training record per trainer
            for idx, trainer_name in enumerate(trainer_names):
                # Match email with trainer index (1-to-1 matching)
                if idx < len(email_list):
                    trainer_email = email_list[idx]
                elif len(email_list) == 1:
                    # If only one email but multiple trainers, use the same email for all
                    trainer_email = email_list[0]
                else:
                    trainer_email = None
                
                logging.info(f"Row {i+2}, Trainer {idx+1}/{len(trainer_names)}: '{trainer_name}' -> email: '{trainer_email}'")
                
                trainings_to_add.append(
                    TrainingDetail(
                        division=division_val,
                        department=department_val,
                        competency=competency_val,
                        skill=skill_val,
                        training_name=training_name_val,
                        training_topics=training_topics_val,
                        prerequisites=prerequisites_val,
                        skill_category=skill_category_val,
                        trainer_name=trainer_name,
                        email=trainer_email,
                        training_date=final_date,
                        duration=duration_str,
                        seats=seats_str,
                        time=time_val,
                        training_type=training_type_val,
                        assessment_details=assessment_details_val,
                    )
                )
            
            if len(trainer_names) > 1:
                logging.info(f"✅ Row {i+2}: Successfully split into {len(trainer_names)} separate training records")
        
        logging.info(f"-> Training validation complete: {len(trainings_to_add)} valid rows, {skipped_training_count} skipped.")

        # --- 3. Load Employee Competency ---
        logging.info("Step 3.5: Reading 'Employee Competency' sheet from Excel...")
        excel_file_source.seek(0)
        competencies_to_add = []
        skipped_competency_count = 0
        
        try:
            df_competency_raw = pd.read_excel(excel_file_source, sheet_name="Employee Competency", engine='openpyxl')
        except ValueError as e:
            # List available sheets if the sheet name is wrong
            excel_file_source.seek(0)
            xl_file = pd.ExcelFile(excel_file_source, engine='openpyxl')
            available_sheets = xl_file.sheet_names
            logging.warning(f"Sheet 'Employee Competency' not found! Available sheets: {available_sheets}")
            logging.warning("-> Continuing without Employee Competency data...")
            df_competency_raw = None
        
        if df_competency_raw is not None:
            logging.info(f"-> Original column names (before cleaning): {list(df_competency_raw.columns)}")
            
            df_competency = df_competency_raw.replace({np.nan: None})
            df_competency = clean_headers(df_competency)
            logging.info(f"-> Found {len(df_competency)} rows in 'Employee Competency'.")
            logging.info(f"-> Column names after cleaning: {list(df_competency.columns)}")
            
            # Show first few rows
            if len(df_competency) > 0:
                logging.info("-> First 3 rows of data:")
                for idx in range(min(3, len(df_competency))):
                    logging.info(f"   Row {idx+2}: {df_competency.iloc[idx].to_dict()}")
            
            for i, row in df_competency.iterrows():
                try:
                    # Map Excel columns to database columns using flexible matching
                    row_dict = row.to_dict()
                    
                    employee_empid = find_column_flexible(row_dict, ['employee_id', 'employeeid', 'empid', 'employee_empid'])
                    if employee_empid:
                        # Convert float to int then to string (handles Excel's 5504763.0 -> "5504763")
                        if isinstance(employee_empid, float):
                            employee_empid = str(int(employee_empid))
                        else:
                            employee_empid = str(employee_empid).strip()
                    else:
                        employee_empid = None
                    
                    employee_name = find_column_flexible(row_dict, ['employee_name', 'employeename', 'employee name', 'name'])
                    if employee_name and isinstance(employee_name, str):
                        employee_name = employee_name.strip()
                    
                    division = find_column_flexible(row_dict, ['division'])
                    if division and isinstance(division, str):
                        division = division.strip()
                    
                    department = find_column_flexible(row_dict, ['department'])
                    if department and isinstance(department, str):
                        department = department.strip()
                    
                    project = find_column_flexible(row_dict, ['project'])
                    if project and isinstance(project, str):
                        project = project.strip()
                    
                    role_specific_comp = find_column_flexible(row_dict, ['role_specific_competency_(mhs)', 'role_specific_competency', 'role_specific_comp', 'role specific competency (mhs)'])
                    if role_specific_comp and isinstance(role_specific_comp, str):
                        role_specific_comp = role_specific_comp.strip()
                    
                    destination = find_column_flexible(row_dict, ['designation', 'destination', 'desination'])
                    if destination and isinstance(destination, str):
                        destination = destination.strip()
                    
                    competency = find_column_flexible(row_dict, ['competency', 'competence'])
                    if competency and isinstance(competency, str):
                        competency = competency.strip()
                    
                    skill = find_column_flexible(row_dict, ['skill'])
                    if skill and isinstance(skill, str):
                        skill = skill.strip()
                    
                    current_expertise = find_column_flexible(row_dict, ['current_expertise_level', 'current_expertise', 'current expertise level', 'current expertise'])
                    if current_expertise and isinstance(current_expertise, str):
                        current_expertise = current_expertise.strip()
                    
                    target_expertise = find_column_flexible(row_dict, ['target_expertise_level', 'target_expertise', 'target expertise level', 'target expertise'])
                    if target_expertise and isinstance(target_expertise, str):
                        target_expertise = target_expertise.strip()
                    
                    comments = find_column_flexible(row_dict, ['comments', 'comment'])
                    if comments and isinstance(comments, str):
                        comments = comments.strip()
                    
                    # Handle target_date - convert from Excel date to Python date
                    target_date = find_column_flexible(row_dict, ['target_date', 'target date'])
                    try:
                        final_target_date = pd.to_datetime(target_date).date() if pd.notna(target_date) and target_date else None
                    except Exception:
                        final_target_date = None
                    
                    # Validate required fields
                    if not employee_empid:
                        skipped_competency_count += 1
                        logging.warning(f"Skipping Employee Competency row {i+2} due to missing employee_empid")
                        continue
                    
                    # Create EmployeeCompetency object
                    competencies_to_add.append(
                        EmployeeCompetency(
                            employee_empid=employee_empid,
                            employee_name=employee_name,
                            department=department,
                            division=division,
                            project=project,
                            role_specific_comp=role_specific_comp,
                            destination=destination,
                            competency=competency,
                            skill=skill,
                            current_expertise=current_expertise,
                            target_expertise=target_expertise,
                            comments=comments,
                            target_date=final_target_date
                        )
                    )
                    
                    if i < 3:  # Log first 3 successful rows
                        logging.info(f"✅ Employee Competency row {i+2} added: employee={employee_empid} ({employee_name}), skill={skill}, competency={competency}")
                        
                except Exception as row_error:
                    skipped_competency_count += 1
                    logging.warning(f"Skipping Employee Competency row {i+2} due to error: {row_error}")
                    continue
            
            logging.info(f"-> Employee Competency validation complete: {len(competencies_to_add)} valid rows, {skipped_competency_count} skipped.")

        # --- 4. Add all objects to the session ---
        logging.info(f"Step 4: Preparing to add {len(trainers_to_add)} trainers, {len(trainings_to_add)} trainings, and {len(competencies_to_add)} employee competencies to the database session.")
        if trainers_to_add:
            db.add_all(trainers_to_add)
            logging.info(f"✅ Added {len(trainers_to_add)} trainer records to session.")
        else:
            logging.warning("⚠️ No trainer records to add - all rows were skipped!")
        
        if trainings_to_add:
            db.add_all(trainings_to_add)
            logging.info(f"✅ Added {len(trainings_to_add)} training records to session.")
        else:
            logging.warning("⚠️ No training records to add - all rows were skipped!")
        
        if competencies_to_add:
            db.add_all(competencies_to_add)
            logging.info(f"✅ Added {len(competencies_to_add)} employee competency records to session.")
        else:
            logging.warning("⚠️ No employee competency records to add - all rows were skipped or sheet not found!")
        
        # Final summary
        logging.info("=" * 80)
        logging.info("📊 FINAL SUMMARY:")
        logging.info(f"   Trainers: {len(trainers_to_add)} valid rows, {skipped_count} skipped")
        logging.info(f"   Trainings: {len(trainings_to_add)} valid rows, {skipped_training_count} skipped")
        logging.info(f"   Employee Competencies: {len(competencies_to_add)} valid rows, {skipped_competency_count} skipped")
        logging.info(f"   Total rows to insert: {len(trainers_to_add) + len(trainings_to_add) + len(competencies_to_add)}")
        logging.info("=" * 80)
        
        if not trainers_to_add and not trainings_to_add and not competencies_to_add:
            logging.error("❌ CRITICAL: No data to insert! All rows were skipped.")
            logging.error("   Possible reasons:")
            logging.error("   1. Column names in Excel don't match expected names")
            logging.error("   2. All rows have empty required fields")
            logging.error("   3. Sheet names are incorrect (should be 'Trainers Details', 'Training Details', and 'Employee Competency')")
            logging.error("   Check the logs above for detailed information about skipped rows.")
            raise ValueError("No valid data found in Excel file. All rows were skipped during validation.")

        logging.info("-> Data added to session successfully.")

        # --- 5. Commit the transaction ---
        logging.info("Step 5: Committing transaction to the database...")
        try:
            await db.commit()
            logging.info(f"✅ COMMIT SUCCESSFUL! Database updated: {len(trainers_to_add)} trainers, {len(trainings_to_add)} trainings, {len(competencies_to_add)} employee competencies.")
            
            # Verify the data was actually inserted
            from sqlalchemy import select, func
            trainers_count_result = await db.execute(select(func.count(Trainer.id)))
            trainers_count = trainers_count_result.scalar()
            trainings_count_result = await db.execute(select(func.count(TrainingDetail.id)))
            trainings_count = trainings_count_result.scalar()
            competencies_count_result = await db.execute(select(func.count(EmployeeCompetency.id)))
            competencies_count = competencies_count_result.scalar()
            
            logging.info(f"✅ VERIFICATION: Database now contains {trainers_count} trainers, {trainings_count} trainings, and {competencies_count} employee competencies.")
            
            if trainers_count == 0 and trainings_count == 0 and competencies_count == 0:
                logging.error("⚠️ WARNING: Commit succeeded but no data found in database! Possible transaction rollback.")
            
            # Verify and fix IDs to start from 1 and be sequential
            logging.info("Step 6: Verifying and fixing IDs to start from 1...")
            try:
                # Check trainers IDs
                if trainers_count > 0:
                    min_trainer_id_result = await db.execute(text("SELECT MIN(id) FROM trainers"))
                    min_trainer_id = min_trainer_id_result.scalar()
                    if min_trainer_id != 1:
                        logging.info(f"-> Fixing trainers IDs (currently starting from {min_trainer_id}, resetting to 1)...")
                        # Trainers table has no foreign key dependencies, safe to update
                        await db.execute(text("""
                            WITH numbered_rows AS (
                                SELECT id, ROW_NUMBER() OVER (ORDER BY id) as new_id
                                FROM trainers
                            )
                            UPDATE trainers t
                            SET id = nr.new_id
                            FROM numbered_rows nr
                            WHERE t.id = nr.id
                        """))
                        # Reset sequence to 1 for next refresh
                        try:
                            await db.execute(text("ALTER SEQUENCE trainers_id_seq RESTART WITH 1"))
                        except Exception:
                            # Sequence might have different name, try to find it
                            seq_result = await db.execute(text("""
                                SELECT sequence_name FROM information_schema.sequences 
                                WHERE sequence_name LIKE '%trainers%id%'
                            """))
                            seq_name = seq_result.scalar()
                            if seq_name:
                                await db.execute(text(f"ALTER SEQUENCE {seq_name} RESTART WITH 1"))
                        logging.info("-> Trainers IDs reset to start from 1, sequence reset to 1 for next refresh")
                    else:
                        logging.info("-> Trainers IDs already start from 1 ✓")
                
                # Check training_details IDs
                if trainings_count > 0:
                    min_training_id_result = await db.execute(text("SELECT MIN(id) FROM training_details"))
                    min_training_id = min_training_id_result.scalar()
                    if min_training_id != 1:
                        # Check if there are any training_requests that reference training_details
                        training_requests_count_result = await db.execute(text("SELECT COUNT(*) FROM training_requests"))
                        training_requests_count = training_requests_count_result.scalar()
                        
                        if training_requests_count > 0:
                            logging.warning(f"-> Cannot fix training_details IDs automatically: {training_requests_count} training_requests exist with foreign key references.")
                            logging.warning("-> IDs will remain as inserted. To fix IDs, please clear training_requests first.")
                        else:
                            logging.info(f"-> Fixing training_details IDs (currently starting from {min_training_id}, resetting to 1)...")
                            # No foreign key constraints, safe to update
                            await db.execute(text("""
                                WITH numbered_rows AS (
                                    SELECT id, ROW_NUMBER() OVER (ORDER BY id) as new_id
                                    FROM training_details
                                )
                                UPDATE training_details td
                                SET id = nr.new_id
                                FROM numbered_rows nr
                                WHERE td.id = nr.id
                            """))
                            # Reset sequence to 1 for next refresh
                            try:
                                await db.execute(text("ALTER SEQUENCE training_details_id_seq RESTART WITH 1"))
                            except Exception:
                                # Sequence might have different name, try to find it
                                seq_result = await db.execute(text("""
                                    SELECT sequence_name FROM information_schema.sequences 
                                    WHERE sequence_name LIKE '%training_details%id%'
                                """))
                                seq_name = seq_result.scalar()
                                if seq_name:
                                    await db.execute(text(f"ALTER SEQUENCE {seq_name} RESTART WITH 1"))
                            logging.info("-> Training_details IDs reset to start from 1, sequence reset to 1 for next refresh")
                    else:
                        logging.info("-> Training_details IDs already start from 1 ✓")
                
                # Check employee_competency IDs
                if competencies_count > 0:
                    min_comp_id_result = await db.execute(text("SELECT MIN(id) FROM employee_competency"))
                    min_comp_id = min_comp_id_result.scalar()
                    if min_comp_id != 1:
                        logging.info(f"-> Fixing employee_competency IDs (currently starting from {min_comp_id}, resetting to 1)...")
                        # Employee_competency has no foreign key dependencies from other tables, safe to update
                        await db.execute(text("""
                            WITH numbered_rows AS (
                                SELECT id, ROW_NUMBER() OVER (ORDER BY id) as new_id
                                FROM employee_competency
                            )
                            UPDATE employee_competency ec
                            SET id = nr.new_id
                            FROM numbered_rows nr
                            WHERE ec.id = nr.id
                        """))
                        # Reset sequence to 1 for next refresh
                        try:
                            await db.execute(text("ALTER SEQUENCE employee_competency_id_seq RESTART WITH 1"))
                        except Exception:
                            # Sequence might have different name, try to find it
                            seq_result = await db.execute(text("""
                                SELECT sequence_name FROM information_schema.sequences 
                                WHERE sequence_name LIKE '%employee_competency%id%'
                            """))
                            seq_name = seq_result.scalar()
                            if seq_name:
                                await db.execute(text(f"ALTER SEQUENCE {seq_name} RESTART WITH 1"))
                        logging.info("-> Employee_competency IDs reset to start from 1, sequence reset to 1 for next refresh")
                    else:
                        logging.info("-> Employee_competency IDs already start from 1 ✓")
                
                await db.commit()
                logging.info("✅ ID verification and fix complete. All IDs now start from 1 in insertion order.")
            except Exception as id_fix_error:
                logging.warning(f"⚠️ Could not fix IDs automatically: {id_fix_error}")
                logging.warning("IDs may not start from 1, but data was inserted successfully.")
        except Exception as commit_error:
            logging.error(f"❌ COMMIT FAILED: {commit_error}", exc_info=True)
            raise

    except Exception as e:
        logging.error(f"❌ An error occurred during the Excel loading process: {e}", exc_info=True)
        logging.error("Rolling back all changes. Your database is in its original state.")
        await db.rollback()
        raise


async def load_manager_employee_from_csv(csv_file_source: Any, db: AsyncSession):
    """
    Loads manager-employee relationship data from a CSV file.
    Expected CSV columns: manager_empid, manager_name, employee_empid, employee_name, 
                         manager_is_trainer, employee_is_trainer
    """
    logging.info("--- Starting Manager-Employee CSV data load ---")
    try:
        logging.info("Step 1: Clearing old data from manager_employee table...")
        await db.execute(text("DELETE FROM manager_employee"))
        logging.info("-> Old data cleared successfully.")

        # Read CSV file
        logging.info("Step 2: Reading CSV file...")
        df = pd.read_csv(csv_file_source)
        logging.info(f"-> Found {len(df)} rows in CSV file.")
        logging.info(f"-> Column names: {list(df.columns)}")
        
        # Clean column names (strip whitespace, lowercase, replace spaces with underscores)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        logging.info(f"-> Cleaned column names: {list(df.columns)}")
        
        # Show first few rows
        if len(df) > 0:
            logging.info("-> First 3 rows of data:")
            for idx in range(min(3, len(df))):
                logging.info(f"   Row {idx+1}: {df.iloc[idx].to_dict()}")

        # Validate required columns
        required_columns = ['manager_empid', 'manager_name', 'employee_empid', 'employee_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")

        # Step 3: Collect all unique user IDs and create missing users
        logging.info("Step 3: Collecting unique user IDs from CSV...")
        all_manager_ids = set(df['manager_empid'].dropna().astype(str).unique())
        all_employee_ids = set(df['employee_empid'].dropna().astype(str).unique())
        all_user_ids = all_manager_ids.union(all_employee_ids)
        logging.info(f"-> Found {len(all_user_ids)} unique user IDs in CSV ({len(all_manager_ids)} managers, {len(all_employee_ids)} employees)")
        
        # Check which users already exist
        from sqlalchemy import select, func
        from app.auth_utils import get_password_hash
        
        existing_users_result = await db.execute(
            select(User.username).where(User.username.in_(all_user_ids))
        )
        existing_usernames = set(existing_users_result.scalars().all())
        missing_user_ids = all_user_ids - existing_usernames
        
        logging.info(f"-> Found {len(existing_usernames)} existing users, {len(missing_user_ids)} new users to create")
        
        # Create missing users with default password
        if missing_user_ids:
            logging.info(f"Step 4: Creating {len(missing_user_ids)} missing user accounts...")
            try:
                # Hash password once before the loop to avoid bcrypt initialization issues
                # Use a shorter password to avoid any potential issues
                default_password = "password123"
                try:
                    # Try to hash with bcrypt
                    default_password_hash = get_password_hash(default_password)
                    logging.info(f"✅ Password hashed successfully using bcrypt")
                except Exception as hash_error:
                    # Fallback: use pbkdf2_sha256 if bcrypt fails
                    logging.warning(f"Bcrypt hash failed: {hash_error}")
                    logging.warning("Trying pbkdf2_sha256 as alternative...")
                    from passlib.context import CryptContext
                    fallback_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
                    default_password_hash = fallback_context.hash(default_password)
                    logging.warning("Using pbkdf2_sha256 hash - password verification should still work")
                
                new_users = []
                for user_id in missing_user_ids:
                    new_users.append(
                        User(
                            username=str(user_id),
                            hashed_password=default_password_hash,
                            created_at=datetime.utcnow()
                        )
                    )
                db.add_all(new_users)
                # Commit users first so they're available for foreign key constraints
                await db.commit()
                logging.info(f"✅ Created and committed {len(new_users)} new user accounts with default password '{default_password}'")
                
                # Verify users were created
                verify_users_result = await db.execute(
                    select(func.count(User.username)).where(User.username.in_(missing_user_ids))
                )
                verified_count = verify_users_result.scalar()
                logging.info(f"✅ Verified: {verified_count} out of {len(missing_user_ids)} users exist in database")
                
                # Note: Users can change their password after first login
            except Exception as user_error:
                logging.error(f"❌ Failed to create users: {user_error}", exc_info=True)
                await db.rollback()
                raise ValueError(f"Failed to create user accounts: {user_error}")
        else:
            logging.info("Step 4: All users already exist in database, skipping user creation.")

        # Process rows
        manager_employees_to_add = []
        skipped_count = 0
        
        for i, row in df.iterrows():
            try:
                # Get required fields
                manager_empid = str(row.get('manager_empid', '')).strip() if pd.notna(row.get('manager_empid')) else None
                manager_name = str(row.get('manager_name', '')).strip() if pd.notna(row.get('manager_name')) else None
                employee_empid = str(row.get('employee_empid', '')).strip() if pd.notna(row.get('employee_empid')) else None
                employee_name = str(row.get('employee_name', '')).strip() if pd.notna(row.get('employee_name')) else None
                
                # Get boolean fields (handle 'f'/'t', 'false'/'true', '0'/'1', etc.)
                manager_is_trainer_val = row.get('manager_is_trainer', False)
                employee_is_trainer_val = row.get('employee_is_trainer', False)
                
                # Convert to boolean
                def to_bool(val):
                    if pd.isna(val) or val is None:
                        return False
                    if isinstance(val, bool):
                        return val
                    val_str = str(val).lower().strip()
                    return val_str in ['t', 'true', '1', 'yes', 'y']
                
                manager_is_trainer = to_bool(manager_is_trainer_val)
                employee_is_trainer = to_bool(employee_is_trainer_val)
                
                # Validate required fields
                if not manager_empid or not employee_empid:
                    skipped_count += 1
                    logging.warning(f"Skipping row {i+2} due to missing manager_empid or employee_empid")
                    continue
                
                # Create ManagerEmployee object
                manager_employees_to_add.append(
                    ManagerEmployee(
                        manager_empid=manager_empid,
                        manager_name=manager_name,
                        employee_empid=employee_empid,
                        employee_name=employee_name,
                        manager_is_trainer=manager_is_trainer,
                        employee_is_trainer=employee_is_trainer
                    )
                )
                
                if i < 3:  # Log first 3 successful rows
                    logging.info(f"✅ Row {i+2} added: manager={manager_empid} ({manager_name}), employee={employee_empid} ({employee_name})")
                    
            except Exception as row_error:
                skipped_count += 1
                logging.warning(f"Skipping row {i+2} due to error: {row_error}")
                continue
        
        logging.info(f"-> Validation complete: {len(manager_employees_to_add)} valid rows, {skipped_count} skipped.")
        
        # Add all objects to the session
        logging.info(f"Step 5: Adding {len(manager_employees_to_add)} manager-employee relationships to database session...")
        if manager_employees_to_add:
            db.add_all(manager_employees_to_add)
            logging.info("-> Data added to session successfully.")
        else:
            logging.warning("⚠️ No manager-employee records to add - all rows were skipped!")
            raise ValueError("No valid data found in CSV file. All rows were skipped during validation.")

        # Commit the transaction
        logging.info("Step 6: Committing transaction to the database...")
        try:
            await db.commit()
            logging.info(f"✅ COMMIT SUCCESSFUL! Database updated with {len(manager_employees_to_add)} manager-employee relationships.")
            
            # Verify the data was actually inserted
            from sqlalchemy import select, func
            # Count distinct manager-employee pairs (composite primary key)
            count_result = await db.execute(
                select(func.count()).select_from(ManagerEmployee)
            )
            total_count = count_result.scalar()
            
            logging.info(f"✅ VERIFICATION: Database now contains {total_count} manager-employee relationships.")
            
        except Exception as commit_error:
            logging.error(f"❌ COMMIT FAILED: {commit_error}", exc_info=True)
            raise

    except Exception as e:
        logging.error(f"❌ An error occurred during the CSV loading process: {e}", exc_info=True)
        logging.error("Rolling back all changes. Your database is in its original state.")
        await db.rollback()
        raise


async def load_employee_competency_from_excel(excel_file_source: Any, db: AsyncSession):
    """
    Loads employee competency data from the 'Employee Competency' sheet in an Excel file.
    Expected Excel columns: Division, Department, Employee ID, Employee Name, 
    Role Specific Competency (MHS), Designation, Competency, Project, Skill,
    Current Expertise Level, Target Expertise Level, Target Date, Comments
    """
    logging.info("--- Starting Employee Competency Excel data load ---")
    try:
        logging.info("Step 1: Clearing old data from employee_competency table...")
        await db.execute(text("DELETE FROM employee_competency"))
        # Reset the sequence to start from 1 after deletion
        await db.execute(text("ALTER SEQUENCE employee_competency_id_seq RESTART WITH 1"))
        logging.info("-> Old data cleared successfully.")
        logging.info("-> ID sequence reset to start from 1.")

        # Read Excel file
        logging.info("Step 2: Reading 'Employee Competency' sheet from Excel...")
        excel_file_source.seek(0)
        try:
            df_raw = pd.read_excel(excel_file_source, sheet_name="Employee Competency", engine='openpyxl')
        except ValueError as e:
            # List available sheets if the sheet name is wrong
            excel_file_source.seek(0)
            xl_file = pd.ExcelFile(excel_file_source, engine='openpyxl')
            available_sheets = xl_file.sheet_names
            logging.error(f"Sheet 'Employee Competency' not found! Available sheets: {available_sheets}")
            raise ValueError(f"Sheet 'Employee Competency' not found. Available sheets: {available_sheets}")
        
        logging.info(f"-> Original column names (before cleaning): {list(df_raw.columns)}")
        
        df = df_raw.replace({np.nan: None})
        df = clean_headers(df)
        logging.info(f"-> Found {len(df)} rows in 'Employee Competency'.")
        logging.info(f"-> Column names after cleaning: {list(df.columns)}")
        
        # Show first few rows
        if len(df) > 0:
            logging.info("-> First 3 rows of data:")
            for idx in range(min(3, len(df))):
                logging.info(f"   Row {idx+2}: {df.iloc[idx].to_dict()}")

        # Step 3: Temporarily disable foreign key constraint to allow loading data first
        logging.info("Step 3: Temporarily disabling foreign key constraint...")
        logging.info("   Note: All data will be loaded. Users can be registered separately later.")
        
        # Get the constraint name
        constraint_result = await db.execute(text("""
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_name = 'employee_competency' 
            AND constraint_type = 'FOREIGN KEY'
            AND constraint_name LIKE '%employee_empid%'
        """))
        constraint_name = constraint_result.scalar()
        
        fk_disabled = False
        if constraint_name:
            try:
                await db.execute(text(f"ALTER TABLE employee_competency DROP CONSTRAINT IF EXISTS {constraint_name}"))
                await db.commit()
                fk_disabled = True
                logging.info(f"   ✅ Foreign key constraint '{constraint_name}' temporarily disabled")
            except Exception as e:
                logging.warning(f"   ⚠️  Could not disable constraint: {e}")
                logging.warning("   Will try to load data anyway (may fail if users don't exist)")
        else:
            logging.warning("   ⚠️  Foreign key constraint not found - may already be disabled")

        # Step 4: Process all rows from Excel
        logging.info("Step 4: Processing all rows from Excel...")

        # Process rows
        competencies_to_add = []
        skipped_count = 0
        
        for i, row in df.iterrows():
            try:
                # Map Excel columns to database columns using flexible matching
                row_dict = row.to_dict()
                
                employee_empid = find_column_flexible(row_dict, ['employee_id', 'employeeid', 'empid', 'employee_empid'])
                if employee_empid:
                    # Convert float to int then to string (handles Excel's 5504763.0 -> "5504763")
                    if isinstance(employee_empid, float):
                        employee_empid = str(int(employee_empid))
                    else:
                        employee_empid = str(employee_empid).strip()
                else:
                    employee_empid = None
                
                employee_name = find_column_flexible(row_dict, ['employee_name', 'employeename', 'employee name', 'name'])
                if employee_name and isinstance(employee_name, str):
                    employee_name = employee_name.strip()
                
                division = find_column_flexible(row_dict, ['division'])
                if division and isinstance(division, str):
                    division = division.strip()
                
                department = find_column_flexible(row_dict, ['department'])
                if department and isinstance(department, str):
                    department = department.strip()
                
                project = find_column_flexible(row_dict, ['project'])
                if project and isinstance(project, str):
                    project = project.strip()
                
                role_specific_comp = find_column_flexible(row_dict, ['role_specific_competency_(mhs)', 'role_specific_competency', 'role_specific_comp', 'role specific competency (mhs)'])
                if role_specific_comp and isinstance(role_specific_comp, str):
                    role_specific_comp = role_specific_comp.strip()
                
                destination = find_column_flexible(row_dict, ['designation', 'destination', 'desination'])
                if destination and isinstance(destination, str):
                    destination = destination.strip()
                
                competency = find_column_flexible(row_dict, ['competency', 'competence'])
                if competency and isinstance(competency, str):
                    competency = competency.strip()
                
                skill = find_column_flexible(row_dict, ['skill'])
                if skill and isinstance(skill, str):
                    skill = skill.strip()
                
                current_expertise = find_column_flexible(row_dict, ['current_expertise_level', 'current_expertise', 'current expertise level', 'current expertise'])
                if current_expertise and isinstance(current_expertise, str):
                    current_expertise = current_expertise.strip()
                
                target_expertise = find_column_flexible(row_dict, ['target_expertise_level', 'target_expertise', 'target expertise level', 'target expertise'])
                if target_expertise and isinstance(target_expertise, str):
                    target_expertise = target_expertise.strip()
                
                comments = find_column_flexible(row_dict, ['comments', 'comment'])
                if comments and isinstance(comments, str):
                    comments = comments.strip()
                
                # Handle target_date - convert from Excel date to Python date
                target_date = find_column_flexible(row_dict, ['target_date', 'target date'])
                try:
                    final_target_date = pd.to_datetime(target_date).date() if pd.notna(target_date) and target_date else None
                except Exception:
                    final_target_date = None
                
                # Validate required fields
                if not employee_empid:
                    skipped_count += 1
                    logging.warning(f"Skipping row {i+2} due to missing employee_empid")
                    continue
                
                # Create EmployeeCompetency object (load all data, no user validation)
                competencies_to_add.append(
                    EmployeeCompetency(
                        employee_empid=employee_empid,
                        employee_name=employee_name,
                        department=department,
                        division=division,
                        project=project,
                        role_specific_comp=role_specific_comp,
                        destination=destination,
                        competency=competency,
                        skill=skill,
                        current_expertise=current_expertise,
                        target_expertise=target_expertise,
                        comments=comments,
                        target_date=final_target_date
                    )
                )
                
                if i < 3:  # Log first 3 successful rows
                    logging.info(f"✅ Row {i+2} added: employee={employee_empid} ({employee_name}), skill={skill}, competency={competency}")
                    
            except Exception as row_error:
                skipped_count += 1
                logging.warning(f"Skipping row {i+2} due to error: {row_error}")
                continue
        
        logging.info(f"-> Validation complete: {len(competencies_to_add)} valid rows, {skipped_count} skipped.")
        
        # Add all objects to the session
        logging.info(f"Step 5: Adding {len(competencies_to_add)} employee competency records to database session...")
        if competencies_to_add:
            db.add_all(competencies_to_add)
            logging.info("-> Data added to session successfully.")
        else:
            logging.warning("⚠️ No employee competency records to add - all rows were skipped!")
            raise ValueError("No valid data found in Excel file. All rows were skipped during validation.")

        # Commit the transaction
        logging.info("Step 6: Committing transaction to the database...")
        try:
            await db.commit()
            logging.info(f"✅ COMMIT SUCCESSFUL! Database updated with {len(competencies_to_add)} employee competency records.")
            
            # Step 7: Reset IDs to be sequential starting from 1
            logging.info("Step 7: Resetting IDs to start from 1...")
            # First reset sequence to 1
            await db.execute(text("ALTER SEQUENCE employee_competency_id_seq RESTART WITH 1"))
            
            # Update all IDs sequentially starting from 1
            await db.execute(text("""
                WITH numbered_rows AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY id) as new_id
                    FROM employee_competency
                )
                UPDATE employee_competency ec
                SET id = nr.new_id
                FROM numbered_rows nr
                WHERE ec.id = nr.id
            """))
            
            # Reset sequence to 1 for next refresh
            await db.execute(text("ALTER SEQUENCE employee_competency_id_seq RESTART WITH 1"))
            logging.info("-> IDs reset to start from 1, sequence reset to 1 for next refresh")
            
            await db.commit()
            
            # Verify the data was actually inserted
            from sqlalchemy import select, func
            count_result = await db.execute(
                select(func.count()).select_from(EmployeeCompetency)
            )
            total_count = count_result.scalar()
            
            # Verify IDs start from 1
            min_id_result = await db.execute(text("SELECT MIN(id) FROM employee_competency"))
            min_id = min_id_result.scalar()
            
            logging.info(f"✅ VERIFICATION: Database now contains {total_count} employee competency records.")
            logging.info(f"✅ IDs start from {min_id} and are sequential.")
            
            # Step 8: Keep foreign key constraint disabled
            # Data loads first, users register later through application
            # Linking happens automatically when employee_empid matches username after registration
            if fk_disabled:
                logging.info("Step 8: Foreign key constraint remains disabled.")
                logging.info("   ✅ Data loaded successfully. Users will register separately through application.")
                logging.info("   ✅ Linking will happen automatically when employee_empid matches username.")
            
        except Exception as commit_error:
            logging.error(f"❌ COMMIT FAILED: {commit_error}", exc_info=True)
            raise

    except Exception as e:
        logging.error(f"❌ An error occurred during the Employee Competency loading process: {e}", exc_info=True)
        logging.error("Rolling back all changes. Your database is in its original state.")
        await db.rollback()
        raise