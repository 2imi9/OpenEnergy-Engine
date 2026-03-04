# OpenEnergy Engine

**AI-Powered Earth Observation for Renewable Energy Verification and NEMS-Based Valuation**

> OpenEnergy Ledger & EO-AI Verification Engine
>
> Millennium Fellowship Research | Northeastern University
>
> NCAR/NLR/NOAA Open Hackathon 2026 -- Team eleveno

---

## Overview

End-to-end platform for satellite-based renewable energy verification and climate-risk-adjusted asset valuation:

1. **Renewable Energy Detection** -- OlmoEarth vision transformer for detecting solar/wind installations from 12-band Sentinel-2 imagery
2. **Climate Risk Assessment** -- ACE2-inspired model with SSP scenario projections, extreme event prediction, and resource assessment
3. **NEMS-Based Valuation** -- NPV/IRR/LCOE calculations using EIA Annual Energy Outlook projections
4. **EIA Data Pipeline** -- EIA-860 generator inventory, EIA-923 generation data, AEO/NEMS price and capacity forecasts
5. **Satellite Data Pipeline** -- Microsoft Planetary Computer STAC API for Sentinel-2 L2A imagery
6. **LLM Integration** -- Local inference (vLLM/Qwen3-8B) or cloud (NVIDIA NIM) for natural language analysis
7. **MCP Server** -- Model Context Protocol server exposing all tools for agent workflows
8. **Web Dashboard** -- Streamlit UI with interactive maps, charts, and AI chat
9. **REST API** -- FastAPI backend wrapping all core functionality

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env   # fill in API keys
docker compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

### Local development

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt -r requirements-api.txt -r requirements-ui.txt

# Optional: satellite imagery support
pip install planetary-computer pystac-client rasterio shapely

# Terminal 1: API
uvicorn api.main:app --reload

# Terminal 2: UI
streamlit run ui/app.py
```

### GPU / LLM support

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Requires NVIDIA GPU with 16GB+ VRAM and CUDA 11.8+.

---

## Architecture

```
openenergy-engine/
├── src/                           # Core business logic
│   ├── models/
│   │   ├── olmo_earth.py         # Vision transformer (Sentinel-2, 12 bands)
│   │   └── climate_risk.py       # Climate risk model (SSP scenarios)
│   ├── data/
│   │   └── satellite.py          # Planetary Computer STAC client
│   ├── eia/
│   │   └── client.py             # EIA API v2 (860/923/AEO)
│   ├── valuation/
│   │   └── engine.py             # NPV/IRR/LCOE valuation engine
│   ├── training/
│   │   └── trainer.py            # Multi-task training pipeline
│   ├── llm/
│   │   ├── config.py             # VLLMConfig dataclass
│   │   ├── client.py             # vLLM local inference client
│   │   ├── cloud_client.py       # NVIDIA NIM cloud client
│   │   └── prompts.py            # Prompt templates
│   └── mcp/
│       ├── config.py             # MCPConfig dataclass
│       ├── tools.py              # Tool definitions & handlers
│       └── server.py             # FastMCP server
├── api/                           # FastAPI REST backend
│   ├── main.py                   # App entry, CORS, health check
│   ├── schemas.py                # Pydantic request/response models
│   └── routes/
│       ├── detection.py          # POST /api/detect
│       ├── climate.py            # POST /api/climate-risk
│       ├── valuation.py          # POST /api/value-asset, /api/tokenize
│       ├── eia.py                # GET /api/eia/*
│       └── llm.py                # POST /api/analyze, /api/report
├── ui/                            # Streamlit dashboard
│   ├── app.py                    # Entry point
│   ├── pages/                    # Dashboard, Site Selection, Climate Risk,
│   │                             # Asset Valuation, Detection, AI Chat
│   ├── components/               # Map widget, charts, sidebar
│   └── utils/                    # API client, session state
├── docker/
│   ├── Dockerfile.api            # API container (Python + torch CPU)
│   ├── Dockerfile.ui             # UI container (lightweight)
│   └── Dockerfile.gpu            # GPU container (CUDA + vLLM)
├── tests/                         # Test suite
│   ├── test_models.py
│   ├── test_valuation.py
│   └── test_climate.py
├── benchmarks/
│   └── benchmark_gpu.py          # GPU vs CPU performance benchmarks
├── docker-compose.yml
├── docker-compose.gpu.yml
├── .env.example
├── requirements.txt               # Core deps
├── requirements-api.txt           # FastAPI deps
├── requirements-ui.txt            # Streamlit deps
└── README.md
```

---

## Core Components

### OlmoEarth Detection Model

```python
from src.models import create_model
import torch

model = create_model(task="multi", model_size="base")
images = torch.randn(2, 12, 224, 224)  # 12-band Sentinel-2
outputs = model(images)
# outputs: {detection, classification, capacity_mw, segmentation}
```

- Spectral attention for multi-band satellite imagery
- Multi-task heads: detection, classification, capacity estimation, segmentation
- Supports real OlmoEarth pretrained backbone (falls back to custom ViT)

### Climate Risk Model

```python
from src.models import create_climate_model, ClimateScenario

model = create_climate_model()
risk = model.assess_risk(
    latitude=35.0, longitude=-119.9, elevation=500,
    asset_type="solar", scenario=ClimateScenario.SSP245, target_year=2050
)
print(f"Risk: {risk.risk_score:.2f}, GHI: {risk.solar_ghi_kwh_m2_year['p50']:.0f}")
```

### Asset Valuation (NEMS-Based)

```python
from src.valuation import ValuationEngine, AssetCharacteristics, AssetType

engine = ValuationEngine(discount_rate=0.08)
asset = AssetCharacteristics(
    asset_id="solar_001", asset_type=AssetType.SOLAR_UTILITY,
    latitude=35.0, longitude=-119.9, state="CA", capacity_mw=100,
    verification_status="verified", verification_confidence=0.92
)
v = engine.value_asset(asset)
print(f"NPV: ${v.npv_usd:,.0f} | IRR: {v.irr:.1%} | LCOE: ${v.lcoe_per_mwh:.2f}/MWh")
```

### EIA Data Access

```python
from src.eia import EIAClient  # requires EIA_API_KEY env var

client = EIAClient()
solar = client.get_solar_generators(state="CA", min_capacity_mw=10)
prices = client.get_electricity_price_forecast(scenario="ref2025")
```

Register for a free API key: https://www.eia.gov/opendata/register.php

---

## Real vs Mock Data

The platform gracefully degrades when optional dependencies are missing. This table shows what runs with real data and what falls back to mock/synthetic:

| Component | Real | Mock Fallback | What Triggers Real |
|-----------|------|---------------|-------------------|
| **OlmoEarth backbone** | Pretrained weights from Allen Institute (LatentMIM, 3.5M–1.4B params) | Custom ViT (randomly initialized, same API) | `olmoearth_pretrain` installed |
| **Satellite imagery** | Sentinel-2 L2A from Microsoft Planetary Computer (10m, 12 bands) | Synthetic spectral arrays with realistic band ranges | `planetary-computer`, `pystac-client`, `rasterio`, `shapely` installed |
| **EIA data** | Live EIA API v2 (860/923 generator inventory, AEO price forecasts) | Returns error (no mock) | `EIA_API_KEY` env var set |
| **LLM analysis** | NVIDIA NIM cloud or local vLLM inference | Placeholder template responses | `NVIDIA_API_KEY` or GPU + `vllm` |
| **Climate risk model** | Always runs (pure PyTorch) | — | Weights are randomly initialized; architecture is real but not yet trained on climate data |
| **Valuation engine** | Always runs (pure Python math) | — | NPV/IRR/LCOE calculations are fully functional |
| **Task heads** (detection, classification, capacity, segmentation) | Architecture is real | — | Weights are randomly initialized; require fine-tuning on labeled renewable energy data |

**Docker (CPU)** uses mock satellite + fallback ViT backbone. **Local with GPU** uses real OlmoEarth + real satellite imagery when geo deps are installed.

> **Note:** The climate model and task heads produce structurally valid outputs but with random weights. Classification outputs will appear near-uniform until fine-tuned on labeled data.

---

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Module availability check |
| `/api/detect` | POST | Satellite-based renewable detection |
| `/api/climate-risk` | POST | Climate risk assessment |
| `/api/value-asset` | POST | Full 25-year valuation |
| `/api/tokenize` | POST | Tokenization metrics |
| `/api/eia/generators` | GET | EIA generator inventory |
| `/api/eia/generation/{state}` | GET | State generation data |
| `/api/eia/prices` | GET | Price forecasts |
| `/api/eia/capacity/{source}` | GET | Capacity forecasts |

Swagger docs at `http://localhost:8000/docs` when running.

---

## Web Dashboard

5-page Streamlit app at `http://localhost:8501`:

| Page | Purpose |
|------|---------|
| Dashboard | Health status, module availability, recent analyses |
| Site Selection | Interactive Folium map, click-to-select lat/lon |
| Climate Risk | SSP scenario assessment, extreme event radar, resource P10/P50/P90 |
| Asset Valuation | 25-year cash flow charts, NPV/IRR/LCOE, tokenization |
| Detection | Satellite-based detection results |
| AI Chat | LLM-powered natural language analysis |

---

## MCP Server

Exposes tools via Model Context Protocol for agent workflows:

```bash
python -m src.mcp.server
python -m src.mcp.server --port 8080 --no-llm
```

| Tool | Description |
|------|-------------|
| `detect_renewable` | Detect solar/wind from satellite imagery |
| `assess_climate_risk` | Climate risk and resource assessment |
| `value_asset` | NPV, IRR, LCOE calculations |
| `query_eia` | EIA database queries |
| `analyze` | LLM-powered analysis |
| `generate_report` | Formatted report generation |

---

## GPU Acceleration (Hackathon Focus)

Key workloads targeted for GPU/HPC optimization:

1. **Vision Transformer Inference** -- batch satellite image processing through OlmoEarth backbone
2. **Satellite Band Normalization** -- replace per-sample CPU loops with vectorized GPU ops
3. **Climate Ensemble Processing** -- batch SSP scenario evaluation across continental grid
4. **Mixed Precision Training** -- AMP integration for 2x training speedup

Run benchmarks:

```bash
python benchmarks/benchmark_gpu.py
python benchmarks/benchmark_gpu.py --batch-size 64 --device cuda
```

---

## Environment Variables

```bash
# Data Access
EIA_API_KEY=your_key               # Required for EIA data
PLANETARY_COMPUTER_KEY=your_key    # Optional, higher rate limits

# LLM (choose one)
NVIDIA_API_KEY=your_key            # NVIDIA NIM cloud inference
VLLM_MODEL=Qwen/Qwen3-8B          # Local vLLM model

# Docker
API_BASE_URL=http://api:8000/api   # Used by Streamlit in Docker
```

---

## Testing

```bash
pytest tests/ -v
```

---

## References

| Source | Description |
|--------|-------------|
| [NEMS](https://github.com/EIAgov/NEMS) | National Energy Modeling System |
| [EIA API](https://www.eia.gov/opendata/) | Energy Information Administration |
| [Planetary Computer](https://planetarycomputer.microsoft.com/) | Microsoft satellite data |
| [OlmoEarth](https://allenai.org/blog/olmoearth-models) | Allen Institute foundation model |
| [vLLM](https://docs.vllm.ai/) | Local LLM inference |
| [FastMCP](https://github.com/jlowin/fastmcp) | MCP server framework |

---

## License

MIT License -- see [LICENSE](LICENSE).

## Acknowledgments

- Allen Institute for AI (OlmoEarth, ACE2)
- U.S. Energy Information Administration (NEMS, EIA-860/923)
- Microsoft Planetary Computer
- NASA/IBM (Prithvi)
- NCAR/NLR/NOAA Open Hackathon
