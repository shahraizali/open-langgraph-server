"""Add database constraints and triggers

Revision ID: 6850cca67e52
Revises: aeffbe57b8d9
Create Date: 2025-09-05 02:35:32.157172

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6850cca67e52'
down_revision: Union[str, Sequence[str], None] = 'aeffbe57b8d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add database constraints, default values, and triggers."""
    
    # Add default values to columns that need them
    op.alter_column('assistants', 'created_at', server_default=sa.text('NOW()'))
    op.alter_column('assistants', 'updated_at', server_default=sa.text('NOW()'))
    op.alter_column('assistants', 'version', server_default=sa.text('1'))
    op.alter_column('assistants', 'config', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('assistants', 'context', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('assistants', 'metadata', server_default=sa.text("'{}'::jsonb"))
    
    op.alter_column('assistant_versions', 'created_at', server_default=sa.text('NOW()'))
    op.alter_column('assistant_versions', 'updated_at', server_default=sa.text('NOW()'))
    op.alter_column('assistant_versions', 'config', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('assistant_versions', 'context', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('assistant_versions', 'metadata', server_default=sa.text("'{}'::jsonb"))
    
    op.alter_column('threads', 'created_at', server_default=sa.text('NOW()'))
    op.alter_column('threads', 'updated_at', server_default=sa.text('NOW()'))
    op.alter_column('threads', 'metadata', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('threads', 'status', server_default=sa.text("'idle'"))
    op.alter_column('threads', 'config', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('threads', 'interrupts', server_default=sa.text("'{}'::jsonb"))
    
    op.alter_column('runs', 'created_at', server_default=sa.text('NOW()'))
    op.alter_column('runs', 'updated_at', server_default=sa.text('NOW()'))
    op.alter_column('runs', 'status', server_default=sa.text("'pending'"))
    op.alter_column('runs', 'metadata', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('runs', 'kwargs', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('runs', 'multitask_strategy', server_default=sa.text("'reject'"))
    
    op.alter_column('background_runs', 'created_at', server_default=sa.text('NOW()'))
    op.alter_column('background_runs', 'updated_at', server_default=sa.text('NOW()'))
    op.alter_column('background_runs', 'status', server_default=sa.text("'pending'"))
    
    op.alter_column('store_items', 'created_at', server_default=sa.text('NOW()'))
    op.alter_column('store_items', 'updated_at', server_default=sa.text('NOW()'))
    op.alter_column('store_items', 'namespace', server_default=sa.text("''"))
    
    op.alter_column('thread_states', 'created_at', server_default=sa.text('NOW()'))
    op.alter_column('thread_states', 'checkpoint_id', server_default=sa.text('uuid_generate_v4()'))
    op.alter_column('thread_states', 'values', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('thread_states', 'next_steps', server_default=sa.text("'[]'::jsonb"))
    op.alter_column('thread_states', 'metadata', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('thread_states', 'tasks', server_default=sa.text("'[]'::jsonb"))
    
    # Create the updated_at trigger function
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Apply updated_at triggers to all relevant tables
    op.execute("DROP TRIGGER IF EXISTS update_threads_updated_at ON threads;")
    op.execute("""
        CREATE TRIGGER update_threads_updated_at BEFORE UPDATE ON threads
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("DROP TRIGGER IF EXISTS update_runs_updated_at ON runs;")
    op.execute("""
        CREATE TRIGGER update_runs_updated_at BEFORE UPDATE ON runs
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("DROP TRIGGER IF EXISTS update_assistants_updated_at ON assistants;")
    op.execute("""
        CREATE TRIGGER update_assistants_updated_at BEFORE UPDATE ON assistants
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("DROP TRIGGER IF EXISTS update_assistant_versions_updated_at ON assistant_versions;")
    op.execute("""
        CREATE TRIGGER update_assistant_versions_updated_at BEFORE UPDATE ON assistant_versions
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("DROP TRIGGER IF EXISTS update_store_items_updated_at ON store_items;")
    op.execute("""
        CREATE TRIGGER update_store_items_updated_at BEFORE UPDATE ON store_items
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("DROP TRIGGER IF EXISTS update_background_runs_updated_at ON background_runs;")
    op.execute("""
        CREATE TRIGGER update_background_runs_updated_at BEFORE UPDATE ON background_runs
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    """Remove database constraints and triggers."""
    
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS update_threads_updated_at ON threads;")
    op.execute("DROP TRIGGER IF EXISTS update_runs_updated_at ON runs;")
    op.execute("DROP TRIGGER IF EXISTS update_assistants_updated_at ON assistants;")
    op.execute("DROP TRIGGER IF EXISTS update_assistant_versions_updated_at ON assistant_versions;")
    op.execute("DROP TRIGGER IF EXISTS update_store_items_updated_at ON store_items;")
    op.execute("DROP TRIGGER IF EXISTS update_background_runs_updated_at ON background_runs;")
    
    # Drop trigger function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
    
    # Remove server defaults (reversing the alter_column operations)
    op.alter_column('assistants', 'created_at', server_default=None)
    op.alter_column('assistants', 'updated_at', server_default=None)
    op.alter_column('assistants', 'version', server_default=None)
    op.alter_column('assistants', 'config', server_default=None)
    op.alter_column('assistants', 'context', server_default=None)
    op.alter_column('assistants', 'metadata', server_default=None)
    
    op.alter_column('assistant_versions', 'created_at', server_default=None)
    op.alter_column('assistant_versions', 'updated_at', server_default=None)
    op.alter_column('assistant_versions', 'config', server_default=None)
    op.alter_column('assistant_versions', 'context', server_default=None)
    op.alter_column('assistant_versions', 'metadata', server_default=None)
    
    op.alter_column('threads', 'created_at', server_default=None)
    op.alter_column('threads', 'updated_at', server_default=None)
    op.alter_column('threads', 'metadata', server_default=None)
    op.alter_column('threads', 'status', server_default=None)
    op.alter_column('threads', 'config', server_default=None)
    op.alter_column('threads', 'interrupts', server_default=None)
    
    op.alter_column('runs', 'created_at', server_default=None)
    op.alter_column('runs', 'updated_at', server_default=None)
    op.alter_column('runs', 'status', server_default=None)
    op.alter_column('runs', 'metadata', server_default=None)
    op.alter_column('runs', 'kwargs', server_default=None)
    op.alter_column('runs', 'multitask_strategy', server_default=None)
    
    op.alter_column('background_runs', 'created_at', server_default=None)
    op.alter_column('background_runs', 'updated_at', server_default=None)
    op.alter_column('background_runs', 'status', server_default=None)
    
    op.alter_column('store_items', 'created_at', server_default=None)
    op.alter_column('store_items', 'updated_at', server_default=None)
    op.alter_column('store_items', 'namespace', server_default=None)
    
    op.alter_column('thread_states', 'created_at', server_default=None)
    op.alter_column('thread_states', 'checkpoint_id', server_default=None)
    op.alter_column('thread_states', 'values', server_default=None)
    op.alter_column('thread_states', 'next_steps', server_default=None)
    op.alter_column('thread_states', 'metadata', server_default=None)
    op.alter_column('thread_states', 'tasks', server_default=None)
