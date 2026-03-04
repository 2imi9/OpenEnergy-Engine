"""FastAPI application for AI Earth Models."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env file BEFORE anything reads os.environ
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.mcp.tools import ToolHandlers
from src.mcp.config import MCPConfig
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)

# Global singleton — persists models across requests
handlers: ToolHandlers | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global handlers
    config = MCPConfig.from_env()
    handlers = ToolHandlers(config)
    logger.info("ToolHandlers initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="AI Earth Models API",
    version="0.1.0",
    description="Renewable energy detection, climate risk, and valuation API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include route modules
from api.routes import detection, climate, valuation, eia, llm  # noqa: E402

app.include_router(detection.router, prefix="/api", tags=["detection"])
app.include_router(climate.router, prefix="/api", tags=["climate"])
app.include_router(valuation.router, prefix="/api", tags=["valuation"])
app.include_router(eia.router, prefix="/api", tags=["eia"])
app.include_router(llm.router, prefix="/api", tags=["llm"])


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Check API health and module availability."""
    modules = {
        "detection": True,
        "climate": True,
        "valuation": True,
        "eia": False,
        "satellite": False,
        "llm": False,
    }

    # Check EIA availability
    try:
        import os
        modules["eia"] = bool(os.environ.get("EIA_API_KEY"))
    except Exception:
        pass

    # Check satellite deps
    try:
        import planetary_computer  # noqa: F401
        modules["satellite"] = True
    except ImportError:
        pass

    # Check LLM availability (NVIDIA NIM)
    try:
        import os as _os
        if _os.environ.get("NVIDIA_API_KEY"):
            modules["llm"] = True
    except Exception:
        pass

    return HealthResponse(status="ok", modules=modules)
