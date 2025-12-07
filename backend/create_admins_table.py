#!/usr/bin/env python3
"""
Database migration script to create the admins table.
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

async def create_admins_table():
    """Create the admins table if it doesn't exist."""
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    try:
        async with engine.begin() as conn:
            # Check if table already exists
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'admins'
            """))
            
            table_exists = result.fetchone()
            
            if table_exists:
                print("ℹ️  Admins table already exists. Skipping creation.")
            else:
                # Create the admins table
                await conn.execute(text("""
                    CREATE TABLE admins (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR,
                        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                    )
                """))
                
                # Create index for better performance
                await conn.execute(text("""
                    CREATE INDEX idx_admins_username ON admins(username)
                """))
                
                print("✅ Admins table created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating admins table: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(create_admins_table())

