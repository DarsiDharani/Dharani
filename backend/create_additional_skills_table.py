#!/usr/bin/env python3
"""
Database migration script to create the additional_skills table.
Run this script to add the new table to your existing database.
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, ForeignKey
from datetime import datetime

# Database URL - PostgreSQL
DATABASE_URL = "postgresql+asyncpg://postgres:admin123@localhost:5432/skillorbit"

async def create_additional_skills_table():
    """Create the additional_skills table"""
    engine = create_async_engine(DATABASE_URL, echo=True)

    async with engine.begin() as conn:
        # Create the table using raw SQL to avoid import issues
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS additional_skills (
                id SERIAL PRIMARY KEY,
                employee_empid VARCHAR NOT NULL,
                skill_name VARCHAR NOT NULL,
                skill_level VARCHAR NOT NULL,
                skill_category VARCHAR NOT NULL,
                description VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_empid) REFERENCES users(username)
            )
        """)

        # Create an index for better performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_additional_skills_employee
            ON additional_skills(employee_empid)
        """)

    await engine.dispose()
    print("✅ Additional skills table created successfully!")

if __name__ == "__main__":
    asyncio.run(create_additional_skills_table())
