# LangGraph Agent Protocol Server

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5+-purple.svg)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)](https://www.docker.com/)

A lightweight, production-ready implementation of the [Agent Protocol](https://github.com/langchain-ai/agent-protocol) specification with native LangGraph integration. This server provides a standardized API for building, deploying, and managing AI agents with real-time streaming capabilities and robust data persistence.

## Features

### 🚀 **Core Capabilities**
- **Agent Protocol Compliance**: Full implementation of the standardized Agent Protocol API
- **LangGraph Integration**: Native support for LangGraph-based agents and workflows
- **Real-time Streaming**: Server-Sent Events (SSE) for live execution monitoring
- **Production Ready**: PostgreSQL backend with connection pooling and migrations

### 🔧 **Advanced Features**
- **Dynamic Graph Loading**: Automatic graph registration from `langgraph.json` configuration
- **Checkpoint Management**: State persistence and recovery for long-running workflows
- **Auth Utilities**: Helper functions for authentication implementation (not currently active as middleware)
- **PostgreSQL Backend**: Robust data persistence with JSONB support and full schema
- **Comprehensive Logging**: Structured logging with configurable levels

### 📊 **Developer Experience**
- **Modern API Docs**: Beautiful, interactive documentation powered by Scalar
- **Docker Support**: Container-ready with docker-compose setup
- **Hot Reload**: Development server with automatic code reloading
- **Extensive Documentation**: Detailed architecture and API documentation

## Quick Start

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 12+**
- **UV Package Manager** (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shahraizali/open-langgraph-server.git
   cd open-langgraph-server
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Configure your graphs**
   Create a `langgraph.json` file:
   ```json
   {
     "dependencies": ["."],
     "graphs": {
       "my_agent": "agents/my_agent/graph.py:agent_graph"
     },
     "env": ".env"
   }
   ```

5. **Start the server**
   ```bash
   uv run uvicorn server.main:app --reload
   ```

The server will start at `http://localhost:8000` with modern API documentation powered by Scalar at `http://localhost:8000/docs`.

## Environment Configuration

### Required Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/langgraph

# LLM API Keys (as needed for your graphs)
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key  # For search tools

# Optional Configuration
LOG_LEVEL=INFO
CHECKPOINT_STORAGE=database  # or "file"
DATABASE_POOL_SIZE=10
```

### Graph Configuration

The `langgraph.json` file defines available graphs:

```json
{
  "dependencies": ["."],
  "graphs": {
    "research_agent": "agents/research/graph.py:research_graph",
    "support_agent": "agents/support/graph.py:support_graph"
  },
  "env": ".env"
}
```

## API Endpoints

This server implements the complete Agent Protocol specification:

### Core Endpoints
- `GET /` - Health check and server status
- `GET /meta` - System metadata and available graphs

### Assistant Management
- `POST /assistants` - Create a new assistant
- `GET /assistants/{id}` - Get assistant details
- `POST /assistants/search` - Search assistants
- `PATCH /assistants/{id}` - Update assistant
- `DELETE /assistants/{id}` - Delete assistant

### Thread Management
- `POST /threads` - Create a new conversation thread
- `GET /threads/{id}` - Get thread details
- `PATCH /threads/{id}` - Update thread

### Run Execution
- `POST /runs` - Create and execute a run
- `GET /runs/{id}` - Get run status and results
- `POST /runs/{id}/stream` - Stream run execution in real-time

### Advanced Features
- `GET /assistants/{id}/schemas` - Get assistant input/output schemas
- `GET /assistants/{id}/graph` - Get graph visualization
- `POST /assistants/{id}/versions` - Manage assistant versions

## API Documentation

The server provides beautiful, interactive API documentation powered by **Scalar** at `/docs`. Features include:

- **Modern Interface**: Clean, professional design with dark/light theme support
- **Interactive Testing**: Test endpoints directly from the documentation
- **Real-time Examples**: Live request/response examples with syntax highlighting
- **Search & Navigation**: Quick search with keyboard shortcuts (Cmd/Ctrl + K)
- **Code Generation**: Auto-generated code samples in multiple languages
- **OpenAPI 3.0**: Full OpenAPI specification compliance

### Accessing Documentation

- **Main Documentation**: `http://localhost:8000/docs`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`
- **Health Check**: `http://localhost:8000/`

The documentation automatically updates when you modify your API endpoints and includes detailed descriptions for all Agent Protocol endpoints.

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Start all services**
   ```bash
   docker-compose up -d
   ```

   This starts:
   - **LangGraph Server** on port 8000
   - **PostgreSQL Database** on port 5432
   - **Adminer** (DB admin) on port 8080

2. **View logs**
   ```bash
   docker-compose logs -f langgraph_server
   ```

3. **Stop services**
   ```bash
   docker-compose down
   ```

### Manual Docker Build

```bash
# Build the image
docker build -t langgraph-server .

# Run with environment file
docker run -p 8000:8000 --env-file .env langgraph-server
```

## Development

### Development Commands

```bash
# Install dependencies
uv sync

# Run development server with hot reload
uv run uvicorn server.main:app --reload

# Run tests
uv run pytest

# Run specific test module
uv run pytest tests/test_assistants.py

# Linting and formatting
uv run ruff check
uv run ruff format
uv run black .
```

### Database Operations

```bash
# Check database health
curl http://localhost:8000/meta

# View database contents (if Adminer is running)
open http://localhost:8080
```

### Creating Custom Agents

1. **Create your agent graph**
   ```python
   # agents/my_agent/graph.py
   from langgraph.graph import StateGraph
   from typing import TypedDict

   class AgentState(TypedDict):
       messages: list

   def my_node(state: AgentState):
       # Your agent logic here
       return {"messages": state["messages"] + ["Response"]}

   workflow = StateGraph(AgentState)
   workflow.add_node("agent", my_node)
   workflow.set_entry_point("agent")
   workflow.set_finish_point("agent")

   my_agent = workflow.compile()
   ```

2. **Register in langgraph.json**
   ```json
   {
     "graphs": {
       "my_agent": "agents/my_agent/graph.py:my_agent"
     }
   }
   ```

3. **Restart the server** - Your agent will be automatically registered as an assistant.

## Architecture

### High-Level Overview

```mermaid
graph TB
    subgraph "Client Layer"
        C[Client Applications]
        B[Browser/Postman]
        SDK[Agent Protocol SDK]
    end

    subgraph "FastAPI Application"
        FA[FastAPI App]
        MW[CORS Middleware]

        subgraph "Active Routers"
            MR[Meta Router]
            AR[Assistants Router]
            TR[Threads Router]
            RR[Runs Router]
        end

        subgraph "Inactive Routers"
            SR[Store Router - Commented Out]
            BR[Background Runs Router - Commented Out]
        end
    end

    subgraph "LangGraph Integration"
        GL[Graph Loader]
        GR[Graph Resolver]
        GS[Graph Schema]
        GC[Graph Config]
        GRAPHS[(Loaded Graphs)]
    end

    subgraph "Storage & Utilities"
        SS[Streaming System]
        AUTH_UTILS[Auth Utilities - Not Middleware]
        SERDE[Serialization Utils]

        subgraph "Storage Classes"
            PAS[PostgreSQL Assistant Storage]
            PS[PostgreSQL Storage]
            CS[Checkpoint Storage]
        end
    end

    subgraph "PostgreSQL Database"
        AT[assistants]
        AV[assistant_versions]
        TT[threads]
        TS[thread_states]
        RT[runs]
        BRT[background_runs]
        SI[store_items]
        LGC[langgraph_checkpoints]
        LGCW[langgraph_checkpoint_writes]
    end

    subgraph "External APIs"
        OPENAI[ChatOpenAI]
        TAVILY[TavilySearch]
    end

    subgraph "Graph Implementation"
        subgraph "Single Graph Example"
            NODES[Nodes: call_model, tool_node]
            TOOLS[Tools: TavilySearch]
            STATE[AgentState]
        end
    end

    %% Connections
    C & B & SDK --> FA --> MW --> MR & AR & TR & RR
    AR --> PAS & GL & GS
    TR & RR --> PS
    RR --> SS
    MR --> GRAPHS
    GL --> GRAPHS & GR
    GRAPHS --> NODES --> TOOLS --> OPENAI & TAVILY
    PAS --> AT & AV
    PS --> TT & TS & RT
    CS --> LGC & LGCW
    GL -.-> FS[File System - langgraph.json]

    %% Styling
    classDef clientStyle fill:#e1f5fe
    classDef apiStyle fill:#f3e5f5
    classDef langraphStyle fill:#e8f5e8
    classDef storageStyle fill:#fff3e0
    classDef dbStyle fill:#ffebee
    classDef externalStyle fill:#ffe0e6
    classDef inactiveStyle fill:#f5f5f5,stroke-dasharray: 5 5

    class C,B,SDK clientStyle
    class FA,MW,MR,AR,TR,RR apiStyle
    class GL,GR,GS,GC,GRAPHS,NODES,TOOLS,STATE langraphStyle
    class SS,AUTH_UTILS,SERDE,PAS,PS,CS storageStyle
    class AT,AV,TT,TS,RT,BRT,SI,LGC,LGCW dbStyle
    class OPENAI,TAVILY externalStyle
    class SR,BR inactiveStyle
```

### High-Level Component Diagram

```mermaid
graph TD
    subgraph "User / Client"
        User[User/Client Application]
    end

    subgraph "Docker Environment"
        subgraph "Services"
            Server[LangGraph Server]
            Postgres[PostgreSQL Database]
            Adminer[Adminer]
        end
    end

    subgraph "External Services"
        LLM[LLM Provider e.g., OpenAI]
        Tools[External Tools e.g., Tavily]
    end

    User -- "API Requests" --> Server
    Server -- "DB Operations" --> Postgres
    Server -- "LLM Calls" --> LLM
    Server -- "Tool Calls" --> Tools
    User -- "DB Management" --> Adminer
    Adminer -- "DB Connection" --> Postgres
```

### API Request Sequence Diagram (Create & Run Thread)

```mermaid
sequenceDiagram
    participant C as Client
    participant S as LangGraph Server
    participant G as Graph Execution
    participant DB as PostgreSQL

    C->>S: POST /threads (Create Thread)
    S->>DB: INSERT INTO threads
    DB-->>S: Return new thread_id
    S-->>C: 200 OK (thread_id)

    C->>S: POST /threads/{thread_id}/runs (Start Run)
    S->>DB: INSERT INTO runs (status: pending)
    DB-->>S: Return new run_id
    S-->>C: 200 OK (run_id)

    Note over S,G: Server starts background execution

    S->>G: astream_events(input, config)
    G-->>S: Stream event (e.g., values, messages)
    S-->>C: SSE Event
    G-->>S: Stream event
    S-->>C: SSE Event

    Note over G: Graph executes nodes, tools, etc.

    G->>DB: Read/Write Checkpoints
    DB-->>G: Return State

    G-->>S: Final State
    S->>DB: UPDATE runs (status: success)
    S->>DB: UPDATE threads (values)
    S-->>C: SSE Event (end)
```

### Storage Layer ERD

```mermaid
erDiagram
    threads {
        UUID thread_id PK
        jsonb metadata
        varchar status
        jsonb values
        jsonb config
        jsonb interrupts
        timestamp created_at
        timestamp updated_at
    }

    runs {
        UUID run_id PK
        UUID thread_id FK
        varchar assistant_id
        varchar status
        jsonb metadata
        jsonb kwargs
        varchar multitask_strategy
        timestamp created_at
        timestamp updated_at
    }

    assistants {
        varchar assistant_id PK
        varchar name
        varchar graph_id
        integer version
        jsonb config
        jsonb metadata
        timestamp created_at
        timestamp updated_at
    }

    assistant_versions {
        varchar assistant_id PK, FK
        integer version PK
        varchar name
        varchar graph_id
        jsonb config
        jsonb metadata
        timestamp created_at
    }

    thread_states {
        UUID state_id PK
        UUID thread_id FK
        UUID checkpoint_id
        jsonb values
        jsonb next_steps
        jsonb checkpoint
        jsonb metadata
        UUID parent_checkpoint_id
        jsonb tasks
        timestamp created_at
    }

    langgraph_checkpoints {
        varchar thread_id PK
        varchar checkpoint_ns PK
        UUID checkpoint_id PK
        UUID parent_checkpoint_id
        jsonb checkpoint
        jsonb metadata
        timestamp created_at
    }

    langgraph_checkpoint_writes {
        varchar thread_id PK, FK
        varchar checkpoint_ns PK, FK
        UUID checkpoint_id PK, FK
        varchar task_id PK
        jsonb write
    }

    store_items {
        UUID item_id PK
        varchar namespace
        varchar key
        jsonb value
        timestamp created_at
        timestamp updated_at
        UNIQUE(namespace key)
    }

    background_runs {
        UUID run_id PK
        UUID thread_id
        varchar assistant_id
        varchar status
        jsonb output
        text error
        timestamp created_at
        timestamp updated_at
    }

    threads ||--o{ runs : "has"
    threads ||--o{ thread_states : "has"
    runs }o--|| assistants : "executed_by"
    assistants ||--o{ assistant_versions : "has"
    langgraph_checkpoints ||--o{ langgraph_checkpoint_writes : "has"

```

### Graph Loading Flowchart

```mermaid
graph TD
    A[Start Server] --> B{Load langgraph.json};
    B --> C{For each graph_id, spec};
    C --> D[Resolve Graph Spec];
    D --> E[Load Module from Path];
    E --> F{Compile Graph?};
    F -- Yes --> G[Compile Graph];
    F -- No --> H[Use as-is];
    G --> I[Register Graph];
    H --> I;
    I --> J[Generate Assistant ID];
    J --> K[Store Assistant in DB];
    K --> C;
    C -- Done --> L[Ready for Requests];
```

### Key Components

- **FastAPI Application**: RESTful API server with CORS middleware and automatic OpenAPI documentation
- **Active Routers**: Meta, Assistants, Threads, and Runs APIs (Store and Background Runs are inactive)
- **LangGraph Integration**: Graph loader, resolver, and schema extraction for dynamic graph execution
- **PostgreSQL Storage**: 9 database tables including assistants, threads, runs, and LangGraph checkpoints
- **Storage Classes**: Specialized storage handlers for different data types with direct router integration
- **Streaming System**: Real-time execution monitoring via Server-Sent Events in runs router
- **Auth Utilities**: Authentication helper functions (not implemented as middleware)

## Production Considerations

### Performance
- **Connection Pooling**: Configurable PostgreSQL connection pools
- **Async Operations**: Full async/await support throughout the stack
- **Caching**: Graph and schema caching for improved performance
- **Streaming**: Efficient real-time data streaming with backpressure handling

### Security
- **Input Validation**: Comprehensive Pydantic model validation
- **SQL Injection Protection**: Parameterized queries and ORM usage
- **CORS Configuration**: Configurable cross-origin request policies
- **Auth Utilities**: Helper functions for future authentication implementation (not currently active)

### Monitoring
- **Health Checks**: Built-in health monitoring endpoints
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Metrics**: Ready for Prometheus integration
- **Error Tracking**: Comprehensive error handling and reporting

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

### Code Style

- Follow PEP 8 conventions
- Use type hints throughout
- Add comprehensive docstrings for LangGraph tools
- Ensure test coverage for new features

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[API Documentation](docs/api/)** - Detailed endpoint documentation
- **[Architecture Guide](docs/arch/)** - System design and architecture
- **[Development Guide](docs/README.md)** - Setup and development workflows

## Troubleshooting

### Common Issues

**Database Connection Errors**
- Verify PostgreSQL is running and accessible
- Check `DATABASE_URL` environment variable
- Ensure database exists and credentials are correct

**Graph Loading Failures**
- Validate `langgraph.json` syntax
- Check that all dependencies are installed
- Verify graph export symbols match function names

**Environment Variable Issues**
- Ensure `.env` file exists and is properly formatted
- Check that required API keys are set
- Verify file permissions and location

### Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/your-org/repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/repo/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the foundational AI frameworks
- [LangGraph](https://github.com/langchain-ai/langgraph) for the graph-based agent architecture
- [Agent Protocol](https://github.com/langchain-ai/agent-protocol) for the standardized agent API specification
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance web framework

## Collaborators

<table>
<tr>
<td align="center"><a href="https://github.com/shahraizali"><img src="https://github.com/shahraizali.png" width="100px;" alt="Shahraiz Ali"/><br /><sub><b>Shahraiz Ali</b></sub></a><br /><a href="https://github.com/shahraizali" title="Code">💻</a></td>
</tr>
</table>
