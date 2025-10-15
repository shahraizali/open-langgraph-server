-- Complete PostgreSQL schema for Agent Protocol implementation
-- This matches the JavaScript server's LangGraph database schema
-- Includes all tables, fields, indexes, and triggers

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Threads table
CREATE TABLE IF NOT EXISTS threads (
    thread_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'idle',
    values JSONB,
    config JSONB DEFAULT '{}',
    interrupts JSONB DEFAULT '{}'
);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID REFERENCES threads(thread_id) ON DELETE CASCADE,
    assistant_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    metadata JSONB NOT NULL DEFAULT '{}',
    kwargs JSONB NOT NULL DEFAULT '{}',
    multitask_strategy VARCHAR(50) NOT NULL DEFAULT 'reject'
);

-- Assistants table
CREATE TABLE IF NOT EXISTS assistants (
    assistant_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    graph_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1,
    config JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Assistant versions table (for version history)
CREATE TABLE IF NOT EXISTS assistant_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assistant_id VARCHAR(255) NOT NULL REFERENCES assistants(assistant_id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    graph_id VARCHAR(255) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    name VARCHAR(255),
    UNIQUE(assistant_id, version)
);

-- Store items table (key-value store with namespacing)
CREATE TABLE IF NOT EXISTS store_items (
    item_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(255) NOT NULL DEFAULT '',
    key VARCHAR(255) NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(namespace, key)
);

-- Thread states table (for LangGraph state management)
CREATE TABLE IF NOT EXISTS thread_states (
    state_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    checkpoint_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    values JSONB NOT NULL DEFAULT '{}',
    next_steps JSONB NOT NULL DEFAULT '[]',
    checkpoint JSONB,
    metadata JSONB DEFAULT '{}',
    parent_checkpoint_id UUID,
    tasks JSONB DEFAULT '[]'
);

-- Note: Checkpoint tables (checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations)
-- are created and managed automatically by LangGraph's PostgresSaver when setup() is called.
-- These tables handle LangGraph's checkpoint persistence and should not be created manually.

-- Background runs table
CREATE TABLE IF NOT EXISTS background_runs (
    background_run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    result JSONB,
    error_message TEXT
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_threads_created_at ON threads(created_at);
CREATE INDEX IF NOT EXISTS idx_threads_updated_at ON threads(updated_at);
CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_threads_metadata_gin ON threads USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_runs_thread_id ON runs(thread_id);
CREATE INDEX IF NOT EXISTS idx_runs_assistant_id ON runs(assistant_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs(updated_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_metadata_gin ON runs USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_assistants_graph_id ON assistants(graph_id);
CREATE INDEX IF NOT EXISTS idx_assistants_created_at ON assistants(created_at);

CREATE INDEX IF NOT EXISTS idx_assistant_versions_assistant_id ON assistant_versions(assistant_id);
CREATE INDEX IF NOT EXISTS idx_assistant_versions_version ON assistant_versions(version);
CREATE INDEX IF NOT EXISTS idx_assistant_versions_created_at ON assistant_versions(created_at);

CREATE INDEX IF NOT EXISTS idx_store_namespace ON store_items(namespace);
CREATE INDEX IF NOT EXISTS idx_store_key ON store_items(key);
CREATE INDEX IF NOT EXISTS idx_store_namespace_key ON store_items(namespace, key);
CREATE INDEX IF NOT EXISTS idx_store_value_gin ON store_items USING GIN(value);

CREATE INDEX IF NOT EXISTS idx_thread_states_thread_id ON thread_states(thread_id);
CREATE INDEX IF NOT EXISTS idx_thread_states_checkpoint_id ON thread_states(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_thread_states_created_at ON thread_states(created_at);

-- Note: Checkpoint table indexes are also created automatically by LangGraph's PostgresSaver

CREATE INDEX IF NOT EXISTS idx_background_runs_run_id ON background_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_background_runs_status ON background_runs(status);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
DROP TRIGGER IF EXISTS update_threads_updated_at ON threads;
CREATE TRIGGER update_threads_updated_at BEFORE UPDATE ON threads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_runs_updated_at ON runs;
CREATE TRIGGER update_runs_updated_at BEFORE UPDATE ON runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_assistants_updated_at ON assistants;
CREATE TRIGGER update_assistants_updated_at BEFORE UPDATE ON assistants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_assistant_versions_updated_at ON assistant_versions;
CREATE TRIGGER update_assistant_versions_updated_at BEFORE UPDATE ON assistant_versions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_store_items_updated_at ON store_items;
CREATE TRIGGER update_store_items_updated_at BEFORE UPDATE ON store_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_background_runs_updated_at ON background_runs;
CREATE TRIGGER update_background_runs_updated_at BEFORE UPDATE ON background_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Verify the schema creation
SELECT
    'Complete schema created successfully' as status,
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public') as total_tables,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'threads' AND column_name = 'interrupts') as threads_interrupts_exists,
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'assistant_versions') as assistant_versions_exists,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'assistant_versions' AND column_name = 'updated_at') as assistant_versions_updated_at_exists;
