# Alembic Database Migrations Setup

This document describes the Alembic database migration system that has been set up for the LangGraph Agent Protocol Server.

## Overview

The project now uses **Alembic** with **SQLAlchemy** for database schema management and migrations. This provides:

- ✅ **Version-controlled database schema** 
- ✅ **Automatic migration generation** from model changes
- ✅ **Context field support** for JavaScript compliance  
- ✅ **Async PostgreSQL** support
- ✅ **Production-ready** migration system

## Key Files

### Database Models
- `server/db/models.py` - SQLAlchemy models (Assistant, Thread, Run, etc.)
- `server/db/base.py` - Database connection and session configuration
- `server/storage/sqlalchemy_storage.py` - SQLAlchemy-based storage implementation

### Migration System
- `migrations/` - Alembic migration files
- `alembic.ini` - Alembic configuration
- `run_migrations.py` - Migration runner script

## Usage

### 1. Running Migrations

```bash
# Run all pending migrations
python run_migrations.py

# Check migration status
python run_migrations.py --status

# Alternative: Use Alembic directly
uv run alembic upgrade head
```

### 2. Creating New Migrations

When you modify the SQLAlchemy models in `server/db/models.py`:

```bash
# Generate migration automatically
uv run alembic revision --autogenerate -m "Description of changes"

# Apply the migration
uv run alembic upgrade head
```

### 3. Database Configuration

Set the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql+asyncpg://user:password@host:port/database"
```

Default: `postgresql+asyncpg://postgres:postgres@localhost:5432/postgres`

## Assistant Context Field

The migration system includes the **context field** for JavaScript compliance:

### Database Schema
```sql
CREATE TABLE assistants (
    assistant_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    graph_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1,
    config JSONB NOT NULL DEFAULT '{}',
    context JSONB NOT NULL DEFAULT '{}',  -- ✅ New context field
    metadata JSONB NOT NULL DEFAULT '{}'
);
```

### API Models
```python
class AssistantPatch(BaseModel):
    graph_id: Optional[str] = None
    config: Optional[AssistantConfig] = None
    context: Optional[Dict[str, Any]] = None  # ✅ JavaScript-compliant
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

## Migration History

### Initial Migration (`e77318a0528c`)
- Creates all tables: `assistants`, `threads`, `runs`, `store_items`, etc.
- Includes **context field** in assistants table
- Sets up all indexes and constraints
- **100% JavaScript-compliant** assistant schema

## Integration Points

### Current Integration Status
- ✅ **SQLAlchemy models** created and tested
- ✅ **Migration system** set up and working
- ✅ **Context field** added to all relevant models
- ⏳ **Storage layer** ready (SQLAlchemy implementation available)
- ⏳ **FastAPI integration** (can be switched from raw SQL to SQLAlchemy)

### Switching to SQLAlchemy Storage

To use the SQLAlchemy storage instead of raw SQL:

```python
# In your router or startup code
from server.storage.sqlalchemy_storage import sqlalchemy_assistant_storage

# Replace postgres_assistant_storage with sqlalchemy_assistant_storage
assistant_data = await sqlalchemy_assistant_storage.patch(assistant_id, updates)
```

## Commands Reference

```bash
# Migration Management
uv run alembic upgrade head              # Apply all migrations
uv run alembic downgrade -1             # Rollback one migration
uv run alembic current                   # Show current revision
uv run alembic history                   # Show migration history

# Development
uv run alembic revision --autogenerate -m "Add new field"  # Generate migration
uv run alembic show e77318a0528c        # Show specific migration
uv run alembic edit e77318a0528c        # Edit migration file

# Utilities
python run_migrations.py --status       # Check status
python run_migrations.py               # Run migrations
```

## Benefits

1. **JavaScript Compliance**: Context field enables 1:1 API compatibility
2. **Version Control**: Database schema changes are tracked in git
3. **Team Collaboration**: Consistent database state across environments  
4. **Production Ready**: Alembic is battle-tested for production deployments
5. **Auto-generation**: Schema changes automatically create migrations
6. **Rollback Support**: Can undo migrations safely

## Next Steps

1. **Test migrations** with your database
2. **Switch storage layer** to SQLAlchemy when ready
3. **Add custom migrations** for data transformations if needed
4. **Set up CI/CD** to run migrations in deployment pipeline

The system is ready to use and provides a solid foundation for database schema management!