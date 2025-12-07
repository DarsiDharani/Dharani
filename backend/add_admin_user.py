#!/usr/bin/env python3
"""
Script to add a user as admin.
Usage: python add_admin_user.py <username> [created_by]
Example: python add_admin_user.py INT00137 system
"""

import asyncio
import sys
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import DATABASE_URL

async def add_admin_user(username: str, created_by: str = "system"):
    """Add a user to the admins table."""
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    try:
        async with engine.begin() as conn:
            # First, check if user exists in users table
            user_check = await conn.execute(
                text("SELECT username FROM users WHERE username = :username"),
                {"username": username}
            )
            user_exists = user_check.fetchone()
            
            if not user_exists:
                print(f"❌ Error: User '{username}' does not exist in users table.")
                print("   Please register the user first before making them an admin.")
                return False
            
            # Check if user is already an admin
            admin_check = await conn.execute(
                text("SELECT username FROM admins WHERE username = :username"),
                {"username": username}
            )
            already_admin = admin_check.fetchone()
            
            if already_admin:
                print(f"ℹ️  User '{username}' is already an admin.")
                return True
            
            # Add user to admins table
            await conn.execute(
                text("""
                    INSERT INTO admins (username, created_by) 
                    VALUES (:username, :created_by)
                """),
                {"username": username, "created_by": created_by}
            )
            
            print(f"✅ Successfully added '{username}' as admin!")
            print(f"   Created by: {created_by}")
            return True
        
    except Exception as e:
        print(f"❌ Error adding admin user: {e}")
        return False
    finally:
        await engine.dispose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python add_admin_user.py <username> [created_by]")
        print("Example: python add_admin_user.py INT00137 system")
        sys.exit(1)
    
    username = sys.argv[1]
    created_by = sys.argv[2] if len(sys.argv) > 2 else "system"
    
    # First ensure admins table exists
    print("📋 Checking if admins table exists...")
    from create_admins_table import create_admins_table
    await create_admins_table()
    
    print(f"\n👤 Adding '{username}' as admin...")
    success = await add_admin_user(username, created_by)
    
    if success:
        print(f"\n🎉 User '{username}' can now login as admin!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

