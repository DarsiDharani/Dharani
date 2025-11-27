#!/usr/bin/env python3
"""
Database migration script to create the training_requests table.
Run this script to add the new table to your existing database.
"""

import asyncio
import sys
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import DATABASE_URL

async def create_training_requests_table():
    """Create the training_requests table if it doesn't exist, or update it if it's missing columns."""
    
    # Create async engine
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.begin() as conn:
            # Check if table already exists
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'training_requests'
            """))
            
            table_exists = result.fetchone()
            
            if table_exists:
                print("✅ training_requests table exists, checking for missing columns...")
                
                # Check if manager_notes column exists
                columns_result = await conn.execute(text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'training_requests' AND table_schema = 'public'
                """))
                columns = columns_result.fetchall()
                column_names = [col[0] for col in columns]
                
                if 'manager_notes' not in column_names:
                    print("🔧 Adding missing manager_notes column...")
                    await conn.execute(text("""
                        ALTER TABLE training_requests ADD COLUMN manager_notes TEXT
                    """))
                    print("✅ Added manager_notes column!")
                
                if 'response_date' not in column_names:
                    print("🔧 Adding missing response_date column...")
                    await conn.execute(text("""
                        ALTER TABLE training_requests ADD COLUMN response_date TIMESTAMP
                    """))
                    print("✅ Added response_date column!")
                
                # Check if indexes exist and create them if they don't
                indexes_result = await conn.execute(text("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'training_requests' AND indexname LIKE 'idx_training_requests_%'
                """))
                existing_indexes = [row[0] for row in indexes_result.fetchall()]
                
                if 'idx_training_requests_employee' not in existing_indexes:
                    await conn.execute(text("""
                        CREATE INDEX idx_training_requests_employee ON training_requests (employee_empid)
                    """))
                    print("✅ Created employee index!")
                
                if 'idx_training_requests_manager' not in existing_indexes:
                    await conn.execute(text("""
                        CREATE INDEX idx_training_requests_manager ON training_requests (manager_empid)
                    """))
                    print("✅ Created manager index!")
                
                if 'idx_training_requests_status' not in existing_indexes:
                    await conn.execute(text("""
                        CREATE INDEX idx_training_requests_status ON training_requests (status)
                    """))
                    print("✅ Created status index!")
                
                print("✅ training_requests table is now up to date!")
            else:
                print("🔧 Creating training_requests table...")
                
                # Create the training_requests table
                await conn.execute(text("""
                    CREATE TABLE training_requests (
                        id SERIAL PRIMARY KEY,
                        training_id INTEGER NOT NULL,
                        employee_empid VARCHAR NOT NULL,
                        manager_empid VARCHAR NOT NULL,
                        request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR DEFAULT 'pending',
                        manager_notes TEXT,
                        response_date TIMESTAMP,
                        FOREIGN KEY (training_id) REFERENCES training_details (id),
                        FOREIGN KEY (employee_empid) REFERENCES users (username),
                        FOREIGN KEY (manager_empid) REFERENCES users (username)
                    )
                """))
                
                # Create indexes for better performance
                await conn.execute(text("""
                    CREATE INDEX idx_training_requests_employee ON training_requests (employee_empid)
                """))
                
                await conn.execute(text("""
                    CREATE INDEX idx_training_requests_manager ON training_requests (manager_empid)
                """))
                
                await conn.execute(text("""
                    CREATE INDEX idx_training_requests_status ON training_requests (status)
                """))
                
                print("✅ Successfully created training_requests table with indexes!")
            
    except Exception as e:
        print(f"❌ Error updating training_requests table: {e}")
        raise
    finally:
        await engine.dispose()

async def main():
    """Main function to run the migration."""
    print("🚀 Starting database migration for training_requests table...")
    await create_training_requests_table()
    print("🎉 Migration completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())

