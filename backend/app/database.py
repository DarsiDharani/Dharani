"""
Database Configuration Module

Purpose: Configure SQLAlchemy async database connection and session management
Features:
- Async database engine creation
- Async session factory
- Database table creation
- Dependency injection for database sessions

Database: PostgreSQL with asyncpg driver
Connection: Configured via DATABASE_URL environment variable or hardcoded for development

@author Orbit Skill Development Team
@date 2025
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Database connection URL
# Format: postgresql+asyncpg://username:password@host:port/database
# TODO: Move to environment variable for production
DATABASE_URL = "postgresql+asyncpg://postgres:admin123@localhost:5432/skillorbit"

# Create async database engine
# echo=True enables SQL query logging (disable in production)
# future=True enables SQLAlchemy 2.0 style
async_engine = create_async_engine(DATABASE_URL, echo=True, future=True)

# Create async session factory
# This factory is used to create database sessions throughout the application
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,  # Manual commit control
    autoflush=False    # Manual flush control
)

# Create all tables
async def create_db_and_tables():
    """
    Create all database tables defined in models.
    
    This function reads all SQLAlchemy models and creates corresponding tables
    in the database if they don't already exist.
    
    Note: Import models here to avoid circular import issues.
    """
    # Import here to avoid circular imports
    from app.models import Base
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Dependency to get DB session (async)
async def get_db_async() -> AsyncSession:
    """
    Dependency function for FastAPI route handlers.
    
    Provides an async database session that is automatically closed after use.
    Use this as a dependency in route handlers:
        @router.get("/endpoint")
        async def my_endpoint(db: AsyncSession = Depends(get_db_async)):
            ...
    
    Yields:
        AsyncSession: Database session object
        
    Note: Session is automatically closed when request completes
    """
    async with AsyncSessionLocal() as session:
        yield session

# Alias for backward compatibility
get_db = get_db_async