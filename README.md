# OpenEnergy Engine

**AI-Powered Earth Observation for Renewable Energy Verification and NEMS-Based Valuation**

Research-stage framework for satellite-based renewable-energy verification and asset valuation. Built for the NCAR/NLR/NOAA Open Hackathon 2026 as Millennium Fellowship research at Northeastern University.

> **Read this first.** The valuation, data, API, dashboard, MCP and LLM layers run on real logic. The vision detector and the climate-risk module are untrained scaffolding: their architectures are real, their weights are random, and no checkpoint ships with this repository. Nothing here verifies a site or predicts climate risk today.

## What works today

| Component | What it does |
|---|---|
| **Valuation engine** | NPV, IRR and LCOE over a 25-year project life, driven by EIA AEO price and capacity projections. Pure Python math, no model involved. [`src/valuation/engine.py`](src/valuation/engine.py) |
| **EIA data client** | Live EIA API v2: EIA-860 generator inventory, EIA-923 generation, AEO forecasts. Returns an error rather than mock data when `EIA_API_KEY` is unset. [`src/eia/client.py`](src/eia/client.py) |
| **Satellite pipeline** | Microsoft Planetary Computer STAC client for Sentinel-2 L2A. Falls back to synthetic spectral arrays when the geo dependencies are absent. [`src/data/satellite.py`](src/data/satellite.py) |
| **REST API** | FastAPI backend over the above. [`api/routes/`](api/routes) |
| **Web dashboard** | 6-page Streamlit app: Dashboard, Site Selection, Climate Risk, Asset Valuation, Detection, AI Chat. [`ui/pages/`](ui/pages) |
| **MCP server** | 6 tools exposed over Model Context Protocol for agent workflows. [`src/mcp/`](src/mcp) |
| **LLM integration** | Local vLLM (`Qwen/Qwen3-8B`) or NVIDIA NIM cloud. Placeholder responses without a key or GPU. [`src/llm/`](src/llm) |

## What is scaffolding

Two components are architecture only.

- **OlmoEarth ViT detector** on 12-band Sentinel-2, with multi-task heads for detection, classification, capacity estimation and segmentation. It can load an `olmoearth_pretrain` backbone when that package is installed, but the task heads are randomly initialized and no fine-tuned checkpoint exists here. Classification outputs are near-uniform.
- **Climate-risk module** with SSP scenario projections and extreme-event prediction. Real PyTorch encoders, random weights, never trained on climate data.

Training the detector would need labeled solar and wind installation footprints over Sentinel-2 tiles: segmentation masks or bounding boxes paired with nameplate capacity, spread across enough regions and array geometries to generalize beyond the training area.

## Quick start

```bash
cp .env.example .env          # fill in API keys
docker compose up --build     # API on :8000/docs, UI on :8501
```

Local install:

```bash
pip install -r requirements.txt -r requirements-api.txt -r requirements-ui.txt
pip install planetary-computer pystac-client rasterio shapely   # optional: real imagery
uvicorn api.main:app --reload      # terminal 1
streamlit run ui/app.py            # terminal 2
```

GPU stack (CUDA 12.4 base image): `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`

## Usage

Valuation, the part that runs end to end:

```python
from src.valuation import ValuationEngine, AssetCharacteristics, AssetType

engine = ValuationEngine(discount_rate=0.08)
asset = AssetCharacteristics(
    asset_id="solar_001", asset_type=AssetType.SOLAR_UTILITY,
    latitude=35.0, longitude=-119.9, state="CA", capacity_mw=100,
    verification_status="verified", verification_confidence=0.92,
)
v = engine.value_asset(asset)
print(f"NPV: ${v.npv_usd:,.0f} | IRR: {v.irr:.1%} | LCOE: ${v.lcoe_per_mwh:.2f}/MWh")
```

EIA data:

```python
from src.eia import EIAClient           # requires EIA_API_KEY
client = EIAClient()
solar = client.get_solar_generators(state="CA", min_capacity_mw=10)
```

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Module availability |
| `/api/detect` | POST | Detection (scaffolding, untrained) |
| `/api/climate-risk` | POST | Climate risk (scaffolding, untrained) |
| `/api/value-asset` | POST | 25-year valuation |
| `/api/tokenize` | POST | Tokenization metrics |
| `/api/eia/generators` | GET | EIA generator inventory |
| `/api/eia/generation/{state}` | GET | State generation data |
| `/api/eia/prices` | GET | Price forecasts |
| `/api/eia/capacity/{source}` | GET | Capacity forecasts |

Swagger at `http://localhost:8000/docs`.

## Repository layout

```
src/models/      ViT detector + climate-risk model (untrained)
src/valuation/   NPV / IRR / LCOE engine
src/eia/         EIA API v2 client
src/data/        Planetary Computer STAC client
src/llm/         vLLM + NVIDIA NIM clients
src/mcp/         MCP server and tools
api/             FastAPI backend
ui/              Streamlit dashboard
tests/           Test suite
benchmarks/      GPU vs CPU benchmark harness
```

## Environment variables

```bash
EIA_API_KEY=your_key             # required for EIA data
PLANETARY_COMPUTER_KEY=your_key  # optional, higher rate limits
NVIDIA_API_KEY=your_key          # NVIDIA NIM cloud inference
VLLM_MODEL=Qwen/Qwen3-8B         # local vLLM model
```

## Testing

```bash
pytest tests/ -v
```

## References

| Source | Description |
|---|---|
| [NEMS](https://github.com/EIAgov/NEMS) | National Energy Modeling System |
| [EIA API](https://www.eia.gov/opendata/) | Energy Information Administration |
| [Planetary Computer](https://planetarycomputer.microsoft.com/) | Microsoft satellite data |
| [OlmoEarth](https://allenai.org/blog/olmoearth-models) | Allen Institute foundation model |
| [vLLM](https://docs.vllm.ai/) | Local LLM inference |
| [FastMCP](https://github.com/jlowin/fastmcp) | MCP server framework |

## License

MIT License, see [LICENSE](LICENSE).

## Acknowledgments

Allen Institute for AI (OlmoEarth, ACE2), U.S. Energy Information Administration (NEMS, EIA-860/923), Microsoft Planetary Computer, NASA/IBM (Prithvi), NCAR/NLR/NOAA Open Hackathon.
