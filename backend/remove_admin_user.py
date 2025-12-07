#!/usr/bin/env python3
"""
Script to remove a user from the admins table.
Usage: python remove_admin_user.py <username>
Example: python remove_admin_user.py 5500909
"""

import asyncio
import sys
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import DATABASE_URL

async def remove_admin_user(username: str):
    """Remove a user from the admins table."""
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    try:
        async with engine.begin() as conn:
            # Check if user is an admin
            admin_check = await conn.execute(
                text("SELECT username FROM admins WHERE username = :username"),
                {"username": username}
            )
            is_admin = admin_check.fetchone()
            
            if not is_admin:
                print(f"ℹ️  User '{username}' is not an admin. Nothing to remove.")
                return True
            
            # Remove user from admins table
            await conn.execute(
                text("DELETE FROM admins WHERE username = :username"),
                {"username": username}
            )
            
            print(f"✅ Successfully removed '{username}' from admins table!")
            print(f"   User '{username}' will no longer have admin privileges.")
            return True
        
    except Exception as e:
        print(f"❌ Error removing admin user: {e}")
        return False
    finally:
        await engine.dispose()

async def list_all_admins():
    """List all users in the admins table."""
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT username, created_at, created_by FROM admins ORDER BY username")
            )
            admins = result.fetchall()
            
            if not admins:
                print("📋 No admins found in the database.")
            else:
                print(f"\n📋 Current Admins ({len(admins)}):")
                print("-" * 60)
                for admin in admins:
                    print(f"  Username: {admin[0]}")
                    print(f"  Created: {admin[1]}")
                    print(f"  Created by: {admin[2] or 'N/A'}")
                    print("-" * 60)
            
            return True
        
    except Exception as e:
        print(f"❌ Error listing admins: {e}")
        return False
    finally:
        await engine.dispose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_admin_user.py <username>")
        print("       python remove_admin_user.py --list  (to list all admins)")
        print("Example: python remove_admin_user.py 5500909")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        await list_all_admins()
        return
    
    username = sys.argv[1]
    
    print(f"\n🔍 Removing '{username}' from admins table...\n")
    success = await remove_admin_user(username)
    
    if success:
        print(f"\n✅ User '{username}' has been removed from admins.")
        print(f"   They will need to log out and log back in for changes to take effect.")
        print(f"\n📋 Current admins:")
        await list_all_admins()
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

