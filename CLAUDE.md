# CLAUDE.md - OpenEnergy Engine

## Project Overview

AI-powered Earth observation platform for renewable energy verification and NEMS-based valuation. Part of the "AI Earth Observation as Infrastructure for Democratic Sustainable Energy Investment" research project (Millennium Fellowship Research, Northeastern University).

## Quick Start

### Docker (recommended)

```bash
# Copy and fill in your API keys
cp .env.example .env

# Start API + UI
docker compose up --build

# Open http://localhost:8501 in your browser
```

### Local development

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt -r requirements-api.txt -r requirements-ui.txt

# Optional satellite imagery support
pip install planetary-computer pystac-client rasterio shapely

# Start API backend (terminal 1)
uvicorn api.main:app --reload

# Start Streamlit UI (terminal 2)
streamlit run ui/app.py
```

## Architecture

```
openenergy-engine/
├── src/                          # Core business logic
│   ├── models/
│   │   ├── olmo_earth.py        # Vision transformer for renewable energy detection
│   │   └── climate_risk.py      # Climate risk assessment model (ACE2-inspired)
│   ├── data/
│   │   └── satellite.py         # Microsoft Planetary Computer client
│   ├── eia/
│   │   └── client.py            # EIA API client (EIA-860/923, AEO/NEMS)
│   ├── valuation/
│   │   └── engine.py            # NPV/IRR valuation with NEMS projections
│   ├── training/
│   │   └── trainer.py           # Multi-task training pipeline
│   ├── llm/                     # Local LLM integration (vLLM)
│   │   ├── config.py            # VLLMConfig dataclass
│   │   ├── client.py            # VLLMClient wrapper
│   │   └── prompts.py           # Prompt templates
│   └── mcp/                     # MCP Server
│       ├── config.py            # MCPConfig dataclass
│       ├── tools.py             # Tool definitions & handlers
│       └── server.py            # FastMCP server
├── api/                          # FastAPI REST backend
│   ├── main.py                  # App entry point, CORS, health check
│   ├── schemas.py               # Pydantic request/response models
│   └── routes/
│       ├── detection.py         # POST /api/detect
│       ├── climate.py           # POST /api/climate-risk
│       ├── valuation.py         # POST /api/value-asset, /api/tokenize
│       └── eia.py               # GET /api/eia/*
├── ui/                           # Streamlit web dashboard
│   ├── app.py                   # Entry point + landing page
│   ├── pages/
│   │   ├── 1_Dashboard.py       # KPI overview, health status
│   │   ├── 2_Site_Selection.py  # Interactive map, click-to-select
│   │   ├── 3_Climate_Risk.py    # Risk assessment + radar charts
│   │   ├── 4_Asset_Valuation.py # NPV/IRR/LCOE + cash flow charts
│   │   └── 5_Detection.py       # Satellite detection results
│   ├── components/
│   │   ├── map_widget.py        # Folium map with click support
│   │   ├── charts.py            # Plotly chart helpers
│   │   └── sidebar.py           # Shared sidebar
│   └── utils/
│       ├── api_client.py        # HTTP client for FastAPI backend
│       └── state.py             # Session state management
├── docker/
│   ├── Dockerfile.api           # API container (Python + torch CPU)
│   ├── Dockerfile.ui            # UI container (lightweight, no torch)
│   └── Dockerfile.gpu           # GPU container (CUDA + vLLM)
├── docker-compose.yml           # API + UI services
├── docker-compose.gpu.yml       # GPU override for LLM support
├── .dockerignore
├── .env.example
├── requirements.txt             # Core Python deps
├── requirements-api.txt         # FastAPI deps
├── requirements-ui.txt          # Streamlit deps
└── README.md
```

## Core Components

### 1. OlmoEarth Model (`src/models/olmo_earth.py`)
- Vision transformer for satellite imagery (Sentinel-2, 12 bands)
- Multi-task heads: detection, classification, capacity estimation, segmentation
- Factory: `create_model(task="multi", model_size="300m")`
- Input shape: `(B, 12, 224, 224)` for 12-band Sentinel-2

### 2. Climate Risk Model (`src/models/climate_risk.py`)
- SSP scenario projections (SSP126, SSP245, SSP370, SSP585)
- Extreme event prediction (wildfire, flood, drought, etc.)
- Solar GHI and wind speed resource assessment
- Entry point: `create_climate_model().assess_risk(lat, lon, ...)`

### 3. EIA Client (`src/eia/client.py`)
- Requires `EIA_API_KEY` environment variable
- EIA-860: Generator inventory with coordinates
- EIA-923: Monthly generation data
- AEO/NEMS: Price and capacity projections through 2050
- Register free: https://www.eia.gov/opendata/register.php

### 4. Valuation Engine (`src/valuation/engine.py`)
- NPV, IRR, LCOE calculations
- Risk-adjusted valuations with climate and verification discounts
- Tokenization metrics for RWA applications
- Default discount rate: 8%

### 5. Satellite Client (`src/data/satellite.py`)
- Microsoft Planetary Computer STAC API
- Sentinel-2 L2A imagery (10m resolution)
- Time series for change detection
- Falls back to `MockSatelliteClient` without geo dependencies

### 6. LLM Integration (`src/llm/`)
- Local LLM inference using vLLM with Qwen3-8B default
- Natural language queries, automated analysis, report generation
- Lazy model loading to avoid startup delay
- Factory: `create_vllm_client()`
- Falls back to `MockVLLMClient` without GPU/vLLM

### 7. MCP Server (`src/mcp/`)
- Model Context Protocol server for tool exposure
- Tools: detect_renewable, assess_climate_risk, value_asset, query_eia, analyze
- LLM-powered analysis and report generation
- Factory: `create_mcp_server()`
- Run: `python -m src.mcp.server`

## Key Data Classes

```python
# Model configuration
ModelConfig(input_channels=12, input_size=224, hidden_dim=768, num_layers=12)

# Asset for valuation
AssetCharacteristics(
    asset_id, asset_type, latitude, longitude, state, capacity_mw,
    verification_status, verification_confidence
)

# Climate scenarios
ClimateScenario.SSP126 | SSP245 | SSP370 | SSP585

# Asset types
AssetType.SOLAR_UTILITY | SOLAR_DISTRIBUTED | WIND_ONSHORE | WIND_OFFSHORE | HYDRO
```

## Environment Variables

```bash
# Data Access
EIA_API_KEY=your_key_here              # Required for EIA data access
PLANETARY_COMPUTER_KEY=your_key        # Optional, for higher rate limits

# Web UI / API
API_BASE_URL=http://api:8000/api       # Used by Streamlit UI in Docker

# vLLM Settings (Optional)
VLLM_MODEL=Qwen/Qwen3-8B               # Default LLM model
VLLM_DTYPE=float16                     # Model dtype
VLLM_GPU_MEMORY=0.9                    # GPU memory utilization
VLLM_MAX_TOKENS=2048                   # Max generation length

# MCP Server Settings (Optional)
MCP_PORT=3000                          # Server port
MCP_HOST=localhost                     # Server host
MCP_ENABLE_LLM=true                    # Enable LLM features
```

## Common Tasks

### Run model inference
```python
from src.models import create_model
import torch

model = create_model(task="multi")
images = torch.randn(2, 12, 224, 224)
outputs = model(images)
# outputs: {detection, classification, capacity_mw, segmentation}
```

### Value a renewable asset
```python
from src.valuation import ValuationEngine, AssetCharacteristics, AssetType

engine = ValuationEngine(discount_rate=0.08)
asset = AssetCharacteristics(
    asset_id="solar_001", asset_type=AssetType.SOLAR_UTILITY,
    latitude=35.0, longitude=-119.9, state="CA", capacity_mw=100,
    verification_status="verified", verification_confidence=0.92
)
valuation = engine.value_asset(asset)
```

### Assess climate risk
```python
from src.models import create_climate_model, ClimateScenario

model = create_climate_model()
risk = model.assess_risk(
    latitude=35.0, longitude=-119.9, elevation=500,
    asset_type="solar", scenario=ClimateScenario.SSP245, target_year=2050
)
```

### Use local LLM for analysis
```python
from src.llm import create_vllm_client

# Create client (loads model on first use)
client = create_vllm_client()

# Analyze an asset
analysis = client.analyze_asset({
    "asset_type": "solar_utility",
    "capacity_mw": 100,
    "npv": 5000000,
    "irr": 0.12
})

# Generate a report
report = client.generate_report(data, report_type="valuation")

# Ask questions
answer = client.query("What factors affect solar capacity factors?")
```

### Run MCP server
```python
from src.mcp import create_mcp_server

# Start server (exposes tools via MCP protocol)
server = create_mcp_server()
server.run()
```

Or via CLI:
```bash
python -m src.mcp.server
python -m src.mcp.server --port 8080 --no-llm
```

## Dependencies

**Core:** numpy, pandas, torch, torchvision, transformers, timm, requests, scipy, tqdm

**Optional (satellite):** planetary-computer, pystac-client, rasterio, shapely

**Optional (LLM):** vllm (requires NVIDIA GPU with CUDA)

**Optional (MCP):** fastmcp

**API:** fastapi, uvicorn, pydantic

**UI:** streamlit, streamlit-folium, folium, plotly

**Dev:** pytest, black, isort

## Hardware Requirements (for LLM features)

- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended for Qwen3-8B
- **RAM**: 32GB+ system RAM
- **CUDA**: 11.8+ or 12.x

## Testing

```bash
pytest tests/
```

## Code Style

- Python 3.9+
- Type hints throughout
- Dataclasses for configuration and data structures
- Factory functions (`create_model()`, `create_climate_model()`, etc.)
- Enums for categoricals (TaskType, AssetType, ClimateScenario)

## MCP Tools Reference

| Tool | Description |
|------|-------------|
| `detect_renewable` | Detect solar/wind installations from satellite imagery |
| `assess_climate_risk` | Assess climate risk and resource availability |
| `value_asset` | Calculate NPV, IRR, LCOE for an asset |
| `query_eia` | Query EIA database (generators, prices, capacity) |
| `analyze` | LLM-powered natural language analysis |
| `generate_report` | Generate formatted reports |

## Web UI (Streamlit)

5-page dashboard accessible at `http://localhost:8501`:

| Page | Purpose |
|------|---------|
| Dashboard | Health status, module availability, recent analyses |
| Site Selection | Interactive Folium map, click-to-select lat/lon, quick-analyze |
| Climate Risk | SSP scenario assessment, extreme event radar, resource P10/P50/P90 |
| Asset Valuation | Full 25-year cash flow charts, NPV/IRR/LCOE, tokenization |
| Detection | Satellite-based renewable energy detection results |

### Run Streamlit locally
```bash
streamlit run ui/app.py
```

## REST API (FastAPI)

The API layer wraps `src/mcp/tools.ToolHandlers` and `src/valuation/engine.ValuationEngine`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Module availability check |
| `/api/detect` | POST | Satellite detection |
| `/api/climate-risk` | POST | Climate risk assessment |
| `/api/value-asset` | POST | Full 25-year valuation |
| `/api/tokenize` | POST | Tokenization metrics |
| `/api/eia/generators` | GET | EIA generator inventory |
| `/api/eia/generation/{state}` | GET | State generation data |
| `/api/eia/prices` | GET | Price forecasts |
| `/api/eia/capacity/{source}` | GET | Capacity forecasts |
| `/api/eia/summary/{state}` | GET | State renewable summary |

### Run API locally
```bash
uvicorn api.main:app --reload
# Swagger docs: http://localhost:8000/docs
```

## Docker

```bash
# CPU-only (default)
docker compose up --build

# With GPU/LLM support
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Services:
- **api** (port 8000): FastAPI + PyTorch, holds models in memory
- **ui** (port 8501): Streamlit, lightweight (no torch), calls API via HTTP

## External References

- **NEMS**: https://github.com/EIAgov/NEMS
- **EIA API**: https://www.eia.gov/opendata/
- **Planetary Computer**: https://planetarycomputer.microsoft.com/
- **OlmoEarth**: https://allenai.org/blog/olmoearth-models
- **vLLM**: https://docs.vllm.ai/
- **FastMCP**: https://github.com/jlowin/fastmcp
