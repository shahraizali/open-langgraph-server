from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from .routers import assistants, threads, meta, runs
from .middleware import setup_cors_middleware
from .graph.loader import register_graphs_from_config, GRAPHS
from .storage.database import get_database, close_database
from .storage.postgres_storage import postgres_assistant_storage
from .storage.checkpoint import checkpointer

app = FastAPI(
    title="LangGraph Agent Protocol Server",
    version="0.0.1",
    port=8003,
    description="""
    A lightweight, production-ready implementation of the Agent Protocol specification with native LangGraph integration.

    ## Features
    - **LangGraph Integration**: Native support for LangGraph-based agents and workflows
    - **Real-time Streaming**: Server-Sent Events (SSE) for live execution monitoring
    - **PostgreSQL Backend**: Robust data persistence with JSONB support
    - **Dynamic Graph Loading**: Automatic graph registration from langgraph.json
    - **Checkpoint Management**: State persistence and recovery for workflows

    ## Quick Links
    - [GitHub Repository](https://github.com/your-org/open-langgraph-server)
    - [Agent Protocol Specification](https://github.com/langchain-ai/agent-protocol)
    - [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
    """,
    contact={
        "name": "LangGraph Agent Protocol Server",
        "url": "https://github.com/your-org/open-langgraph-server",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable redoc
    openapi_tags=[
        {"name": "Meta", "description": "System metadata and health check endpoints"},
        {
            "name": "Assistants",
            "description": "Assistant management - create, update, and manage LangGraph-based assistants",
        },
        {
            "name": "Threads",
            "description": "Thread lifecycle management - conversation threads and state management",
        },
        {
            "name": "Runs",
            "description": "Run execution - create and monitor agent runs with real-time streaming",
        },
        {"name": "Debug", "description": "Development and debugging endpoints"},
    ],
)

# Setup CORS middleware (allow all for development)
setup_cors_middleware(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize database and assistants during FastAPI startup."""
    # Load environment variables from .env file
    load_dotenv()

    logger.info("=== Starting FastAPI application ===")

    try:
        # Initialize database
        logger.info("Initializing database...")
        db = await get_database()
        await db.migrate()
        logger.info("Database initialized and migrated")

        # Initialize checkpoint storage
        logger.info("Initializing checkpoint storage...")
        storage_type = os.getenv("CHECKPOINT_STORAGE", "database").lower()
        if storage_type == "database":
            await checkpointer.initialize()
        else:
            raise ValueError(f"Invalid checkpoint storage type: {storage_type}")

        logger.info(f"Checkpoint storage initialized ({storage_type})")

        # Load graphs from langgraph.json
        logger.info("Loading graphs from langgraph.json...")
        config_path = "langgraph.json"
        cwd = os.getcwd()

        if not os.path.exists(config_path):
            logger.warning(
                f"langgraph.json not found at {config_path}, skipping graph loading"
            )
        else:
            try:
                registered_graphs = await register_graphs_from_config(config_path, cwd)
                logger.info(f"Successfully loaded {len(registered_graphs)} graphs")

                # Create assistants in PostgreSQL for each loaded graph
                for graph_id, assistant_id in registered_graphs.items():
                    logger.info(
                        f"Creating assistant for graph '{graph_id}' -> '{assistant_id}'"
                    )
                    await postgres_assistant_storage.put(
                        assistant_id,
                        {
                            "graph_id": graph_id,
                            "name": graph_id,
                            "config": {},
                            "metadata": {
                                "created_by": "system",
                                "loaded_from": "langgraph.json",
                            },
                        },
                    )
                    logger.info(
                        f"Created assistant {assistant_id} for graph {graph_id}"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to load graphs from langgraph.json: {e}", exc_info=True
                )
                # Continue startup even if graph loading fails

        # Log the mapping for debugging
        logger.info(f"Available graphs: {list(GRAPHS.keys())}")

        # Test database connection
        health = await db.health_check()
        logger.info(f"Database health: {health}")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        # Don't raise - continue startup

    logger.info("=== FastAPI application startup completed ===")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources during FastAPI shutdown."""
    logger.info("=== Shutting down FastAPI application ===")

    try:
        # Close checkpoint storage
        storage_type = os.getenv("CHECKPOINT_STORAGE", "database").lower()
        if storage_type == "database":
            await checkpointer.close()
        logger.info("Checkpoint storage closed")

        # Close database connections
        await close_database()
        logger.info("Database connections closed")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

    logger.info("=== FastAPI application shutdown completed ===")


# Include working routers only
app.include_router(meta.router)
app.include_router(assistants.router)
app.include_router(threads.router)
# TODO: Implement other routers
# app.include_router(background_runs.router)
app.include_router(runs.router)
# app.include_router(store.router)


# Add Scalar API documentation
@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url, title=f"{app.title} - API Documentation"
    )


@app.get("/")
async def root():
    return {
        "message": "LangGraph Agent Protocol Server - Visit /docs for API documentation"
    }
