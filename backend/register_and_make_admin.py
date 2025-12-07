#!/usr/bin/env python3
"""
Script to register a user and make them admin in one step.
Usage: python register_and_make_admin.py <username> <password> [created_by]
Example: python register_and_make_admin.py INT00137 tempPassword123 system
"""

import asyncio
import sys
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import DATABASE_URL
from app.auth_utils import get_password_hash

async def register_and_make_admin(username: str, password: str, created_by: str = "system"):
    """Register a user and make them admin."""
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    try:
        async with engine.begin() as conn:
            # Check if user already exists
            user_check = await conn.execute(
                text("SELECT username FROM users WHERE username = :username"),
                {"username": username}
            )
            user_exists = user_check.fetchone()
            
            if not user_exists:
                # Register the user first
                print(f"📝 Registering user '{username}'...")
                hashed_password = get_password_hash(password)
                
                await conn.execute(
                    text("""
                        INSERT INTO users (username, hashed_password, created_at)
                        VALUES (:username, :hashed_password, CURRENT_TIMESTAMP)
                    """),
                    {"username": username, "hashed_password": hashed_password}
                )
                print(f"✅ User '{username}' registered successfully!")
            else:
                print(f"ℹ️  User '{username}' already exists. Skipping registration.")
            
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
            print(f"👑 Making '{username}' an admin...")
            await conn.execute(
                text("""
                    INSERT INTO admins (username, created_by) 
                    VALUES (:username, :created_by)
                """),
                {"username": username, "created_by": created_by}
            )
            
            print(f"✅ Successfully made '{username}' an admin!")
            if not user_exists:
                print(f"\n📋 Login Credentials:")
                print(f"   Username: {username}")
                print(f"   Password: {password}")
                print(f"\n⚠️  Please change the password after first login!")
            return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        await engine.dispose()

async def main():
    if len(sys.argv) < 3:
        print("Usage: python register_and_make_admin.py <username> <password> [created_by]")
        print("Example: python register_and_make_admin.py INT00137 tempPassword123 system")
        sys.exit(1)
    
    username = sys.argv[1]
    password = sys.argv[2]
    created_by = sys.argv[3] if len(sys.argv) > 3 else "system"
    
    # First ensure admins table exists
    print("📋 Checking if admins table exists...")
    from create_admins_table import create_admins_table
    await create_admins_table()
    
    print(f"\n🚀 Registering and making '{username}' an admin...\n")
    success = await register_and_make_admin(username, password, created_by)
    
    if success:
        print(f"\n🎉 User '{username}' is now registered and can login as admin!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

