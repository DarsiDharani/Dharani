"""
Migration script to add action_url column to notifications table

Run this script once to update the existing notifications table.
This adds the action_url column that was missing.

Usage:
    python migrate_notifications_table.py
"""

import asyncio
from sqlalchemy import text
from app.database import async_engine

async def migrate():
    """Add action_url column to notifications table if it doesn't exist"""
    async with async_engine.begin() as conn:
        # Check if column exists
        check_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='notifications' AND column_name='action_url'
        """)
        result = await conn.execute(check_query)
        exists = result.scalar() is not None
        
        if not exists:
            print("Adding action_url column to notifications table...")
            # Add the column
            alter_query = text("""
                ALTER TABLE notifications 
                ADD COLUMN action_url VARCHAR
            """)
            await conn.execute(alter_query)
            print("✓ Successfully added action_url column")
        else:
            print("✓ action_url column already exists")

if __name__ == "__main__":
    asyncio.run(migrate())


