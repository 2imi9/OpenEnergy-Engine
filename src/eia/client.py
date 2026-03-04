"""
EIA API Client for Renewable Energy Data

Provides access to:
- EIA-860: Generator inventory (locations, capacity, technology)
- EIA-923: Monthly generation data
- AEO/NEMS: Annual Energy Outlook projections

API Documentation: https://www.eia.gov/opendata/documentation.php
NEMS Reference: https://github.com/EIAgov/NEMS

Author: Zim (Millennium Fellowship Research)
Project: AI Earth Observation for Democratic Sustainable Energy Investment
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import logging
import json

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2"


@dataclass
class EIAConfig:
    """Configuration for EIA API client."""
    api_key: str
    base_url: str = EIA_BASE_URL
    max_rows: int = 5000
    rate_limit_delay: float = 0.5
    cache_dir: Optional[str] = None
    cache_ttl_hours: int = 24


class EIAClient:
    """
    Client for EIA API v2.
    
    Provides access to:
    - EIA-860: Generator inventory
    - EIA-923: Monthly generation
    - AEO: Annual Energy Outlook projections (NEMS outputs)
    - RTO: Real-time grid operations
    
    Usage:
        client = EIAClient()
        
        # Get solar generators in California
        solar = client.get_solar_generators(state="CA")
        
        # Get AEO projections
        projections = client.get_aeo_projections(scenario="ref2025")
    """
    
    # Energy source codes
    ENERGY_SOURCES = {
        "SUN": "Solar Photovoltaic",
        "STH": "Solar Thermal",
        "WND": "Wind",
        "WAT": "Conventional Hydroelectric",
        "GEO": "Geothermal",
        "WDS": "Wood/Wood Waste",
        "BIO": "Biomass",
        "MWH": "Battery Storage",
        "NG": "Natural Gas",
        "NUC": "Nuclear"
    }
    
    # Prime mover codes
    PRIME_MOVERS = {
        "PV": "Photovoltaic",
        "CP": "Concentrated Solar",
        "WT": "Wind Turbine (onshore)",
        "WS": "Wind Turbine (offshore)",
        "HY": "Hydraulic Turbine",
        "BA": "Battery"
    }
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[EIAConfig] = None):
        """
        Initialize EIA client.
        
        Args:
            api_key: EIA API key. If None, reads from EIA_API_KEY env var.
            config: Optional EIAConfig for advanced settings.
        """
        key = api_key or os.environ.get("EIA_API_KEY")
        if not key:
            raise ValueError(
                "EIA API key required. Set EIA_API_KEY env var or pass api_key. "
                "Register free at https://www.eia.gov/opendata/register.php"
            )
        
        if config:
            self.config = config
            self.config.api_key = key
        else:
            self.config = EIAConfig(api_key=key)
            
        self.session = requests.Session()
        self._last_request_time = 0
        
        # Setup cache if configured
        if self.config.cache_dir:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
            
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
        
    def _get_cache_path(self, cache_key: str) -> Optional[Path]:
        """Get cache file path for a given key."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{cache_key}.json"
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if valid cached data exists."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path or not cache_path.exists():
            return None
            
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            
            # Check TTL
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if (datetime.now() - cached_time).total_seconds() > self.config.cache_ttl_hours * 3600:
                return None
                
            return cached["data"]
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
            
    def _write_cache(self, cache_key: str, data: Dict):
        """Write data to cache."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path:
            return
            
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _request(
        self,
        route: str,
        params: Optional[Dict[str, Any]] = None,
        facets: Optional[Dict[str, List[str]]] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Make API request with rate limiting and optional caching.
        
        Args:
            route: API route
            params: Query parameters
            facets: Filter facets
            use_cache: Whether to use cache
            
        Returns:
            JSON response as dict
        """
        # Build cache key
        cache_key = f"{route}_{hash(str(params))}_{hash(str(facets))}"
        
        # Check cache
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached:
                return cached
        
        self._rate_limit()
        
        url = f"{self.config.base_url}/{route}"
        
        request_params = {"api_key": self.config.api_key}
        if params:
            request_params.update(params)
            
        # Handle facets
        if facets:
            for key, values in facets.items():
                for value in values:
                    request_params[f"facets[{key}][]"] = value
        
        try:
            response = self.session.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache successful response
            if use_cache:
                self._write_cache(cache_key, data)
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"EIA API error: {e}")
            raise
            
    # =========================================================================
    # EIA-860: Generator Inventory
    # =========================================================================
    
    def get_operating_generators(
        self,
        state: Optional[str] = None,
        energy_source: Optional[str] = None,
        status: str = "operating",
        min_capacity_mw: float = 1.0,
        year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get operating generator inventory from EIA-860.
        
        Args:
            state: Two-letter state code (e.g., "CA")
            energy_source: Energy source code ("SUN", "WND", etc.)
            status: Generator status
            min_capacity_mw: Minimum capacity filter
            year: Data year
            
        Returns:
            DataFrame with generator data
        """
        route = "electricity/operating-generator-capacity/data"
        
        facets = {}
        if state:
            facets["stateid"] = [state]
        if energy_source:
            facets["energy_source_code"] = [energy_source]
        if status:
            facets["status"] = [status]
            
        params = {
            "frequency": "annual",
            "data[0]": "nameplate-capacity-mw",
            "data[1]": "net-summer-capacity-mw",
            "data[2]": "latitude",
            "data[3]": "longitude",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": self.config.max_rows
        }
        
        if year:
            params["start"] = f"{year}-01"
            params["end"] = f"{year}-12"
            
        result = self._request(route, params=params, facets=facets)
        
        if "response" in result and "data" in result["response"]:
            df = pd.DataFrame(result["response"]["data"])
            
            # Filter by capacity
            if not df.empty and "nameplate-capacity-mw" in df.columns:
                df["nameplate-capacity-mw"] = pd.to_numeric(
                    df["nameplate-capacity-mw"], errors="coerce"
                )
                df = df[df["nameplate-capacity-mw"] >= min_capacity_mw]
                
            return df
            
        return pd.DataFrame()
    
    def get_solar_generators(
        self,
        state: Optional[str] = None,
        min_capacity_mw: float = 1.0
    ) -> pd.DataFrame:
        """Get solar PV generators."""
        return self.get_operating_generators(
            state=state,
            energy_source="SUN",
            min_capacity_mw=min_capacity_mw
        )
    
    def get_wind_generators(
        self,
        state: Optional[str] = None,
        min_capacity_mw: float = 1.0
    ) -> pd.DataFrame:
        """Get wind generators."""
        return self.get_operating_generators(
            state=state,
            energy_source="WND",
            min_capacity_mw=min_capacity_mw
        )
        
    def get_plant_details(self, plant_id: Union[str, int]) -> Dict:
        """Get detailed information for a specific plant."""
        route = "electricity/operating-generator-capacity/data"
        
        params = {
            "frequency": "annual",
            "data[0]": "nameplate-capacity-mw",
            "data[1]": "net-summer-capacity-mw",
            "data[2]": "latitude",
            "data[3]": "longitude",
            "length": 100
        }
        facets = {"plantid": [str(plant_id)]}
        
        result = self._request(route, params=params, facets=facets)
        
        if "response" in result and "data" in result["response"]:
            data = result["response"]["data"]
            return data[0] if data else {}
        return {}
    
    # =========================================================================
    # EIA-923: Generation Data
    # =========================================================================
    
    def get_plant_generation(
        self,
        plant_id: Union[str, int],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get monthly generation data for a plant.
        
        Args:
            plant_id: EIA plant ID
            start_date: Start date (YYYY-MM)
            end_date: End date (YYYY-MM)
            
        Returns:
            DataFrame with monthly generation (MWh)
        """
        route = "electricity/facility-fuel/data"
        
        params = {
            "frequency": "monthly",
            "data[0]": "generation",
            "data[1]": "gross-generation",
            "start": start_date,
            "end": end_date,
            "length": self.config.max_rows
        }
        facets = {"plantid": [str(plant_id)]}
        
        result = self._request(route, params=params, facets=facets)
        
        if "response" in result and "data" in result["response"]:
            return pd.DataFrame(result["response"]["data"])
        return pd.DataFrame()
    
    def get_state_generation(
        self,
        state: str,
        fuel_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get state-level generation by fuel type."""
        route = "electricity/state-generation/data"
        
        params = {
            "frequency": "monthly",
            "data[0]": "generation",
            "length": self.config.max_rows
        }
        
        facets = {"stateid": [state]}
        if fuel_type:
            facets["fueltypeid"] = [fuel_type]
            
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
            
        result = self._request(route, params=params, facets=facets)
        
        if "response" in result and "data" in result["response"]:
            return pd.DataFrame(result["response"]["data"])
        return pd.DataFrame()
    
    # =========================================================================
    # AEO/NEMS: Energy Projections
    # =========================================================================
    
    def get_aeo_projections(
        self,
        scenario: str = "ref2025",
        series_id: Optional[str] = None,
        start_year: int = 2025,
        end_year: int = 2050
    ) -> pd.DataFrame:
        """
        Get Annual Energy Outlook projections (NEMS outputs).
        
        These projections come from EIA's NEMS model:
        https://github.com/EIAgov/NEMS
        
        Args:
            scenario: AEO scenario
                - "ref2025": Reference case
                - "lowprice": Low oil price
                - "highprice": High oil price  
                - "lowzerocarbon": Low zero-carbon tech cost
                - "highzerocarbon": High zero-carbon tech cost
            series_id: Specific AEO series ID
            start_year: Start year
            end_year: End year
            
        Returns:
            DataFrame with projections
        """
        route = "aeo/data"
        
        params = {
            "frequency": "annual",
            "data[0]": "value",
            "start": str(start_year),
            "end": str(end_year),
            "length": self.config.max_rows
        }
        
        facets = {"scenario": [scenario]}
        if series_id:
            facets["seriesId"] = [series_id]
            
        result = self._request(route, params=params, facets=facets)
        
        if "response" in result and "data" in result["response"]:
            return pd.DataFrame(result["response"]["data"])
        return pd.DataFrame()
    
    def get_electricity_price_forecast(
        self,
        scenario: str = "ref2025",
        sector: str = "residential",
        region: str = "USA"
    ) -> pd.DataFrame:
        """
        Get electricity price projections through 2050.
        
        Critical for NPV calculations in asset valuation.
        
        Args:
            scenario: AEO scenario
            sector: "residential", "commercial", "industrial", "transportation"
            region: Region code
            
        Returns:
            DataFrame with price projections (cents/kWh)
        """
        route = "aeo/data"
        
        params = {
            "frequency": "annual",
            "data[0]": "value",
            "start": "2025",
            "end": "2050",
            "length": 200
        }
        
        facets = {
            "scenario": [scenario],
            "seriesId": [f"prce_nom_elep_{sector}_na_{region}_y"]
        }
        
        result = self._request(route, params=params, facets=facets)
        
        if "response" in result and "data" in result["response"]:
            df = pd.DataFrame(result["response"]["data"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df
        return pd.DataFrame()
    
    def get_renewable_capacity_forecast(
        self,
        energy_source: str = "solar",
        scenario: str = "ref2025"
    ) -> pd.DataFrame:
        """
        Get renewable capacity projections through 2050.
        
        Args:
            energy_source: "solar", "wind", "hydro"
            scenario: AEO scenario
            
        Returns:
            DataFrame with capacity projections (GW)
        """
        route = "aeo/data"
        
        # Map to AEO series IDs
        series_map = {
            "solar": "cap_elep_solpv_na_na_usa_gw",
            "wind": "cap_elep_wind_na_na_usa_gw",
            "hydro": "cap_elep_hydr_na_na_usa_gw"
        }
        
        series_id = series_map.get(energy_source.lower())
        if not series_id:
            raise ValueError(f"Unknown energy source: {energy_source}")
            
        params = {
            "frequency": "annual",
            "data[0]": "value",
            "start": "2025",
            "end": "2050",
            "length": 200
        }
        
        facets = {
            "scenario": [scenario],
            "seriesId": [series_id]
        }
        
        result = self._request(route, params=params, facets=facets)
        
        if "response" in result and "data" in result["response"]:
            df = pd.DataFrame(result["response"]["data"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df
        return pd.DataFrame()
    
    # =========================================================================
    # RTO: Real-time Grid Data
    # =========================================================================
    
    def get_grid_demand(
        self,
        balancing_authority: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get hourly grid demand data.
        
        Args:
            balancing_authority: BA code (e.g., "CISO", "PJM", "ERCOT")
            start_date: Start datetime (ISO format)
            end_date: End datetime (ISO format)
            
        Returns:
            DataFrame with hourly demand (MW)
        """
        route = "electricity/rto/region-data/data"
        
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "start": start_date,
            "end": end_date,
            "length": self.config.max_rows
        }
        
        facets = {
            "respondent": [balancing_authority],
            "type": ["D"]  # Demand
        }
        
        result = self._request(route, params=params, facets=facets)
        
        if "response" in result and "data" in result["response"]:
            return pd.DataFrame(result["response"]["data"])
        return pd.DataFrame()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_state_renewable_summary(self, state: str) -> Dict[str, Any]:
        """Get summary of renewable capacity for a state."""
        solar = self.get_solar_generators(state)
        wind = self.get_wind_generators(state)
        
        def safe_sum(df, col):
            if df.empty or col not in df.columns:
                return 0.0
            return pd.to_numeric(df[col], errors="coerce").sum()
        
        return {
            "state": state,
            "solar_mw": safe_sum(solar, "nameplate-capacity-mw"),
            "wind_mw": safe_sum(wind, "nameplate-capacity-mw"),
            "solar_count": len(solar),
            "wind_count": len(wind),
            "retrieved_at": datetime.utcnow().isoformat()
        }
    
    def get_plants_with_coordinates(
        self,
        state: Optional[str] = None,
        energy_source: Optional[str] = None,
        min_capacity_mw: float = 1.0
    ) -> pd.DataFrame:
        """
        Get plants with valid latitude/longitude for mapping.
        
        Returns:
            DataFrame filtered to plants with valid coordinates
        """
        df = self.get_operating_generators(
            state=state,
            energy_source=energy_source,
            min_capacity_mw=min_capacity_mw
        )
        
        if df.empty:
            return df
            
        # Filter for valid coordinates
        for col in ["latitude", "longitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.dropna(subset=["latitude", "longitude"])
            df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
            
        return df


def create_eia_client(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> EIAClient:
    """Factory function to create EIA client."""
    config = None
    if cache_dir:
        config = EIAConfig(
            api_key=api_key or "",
            cache_dir=cache_dir
        )
    return EIAClient(api_key=api_key, config=config)


if __name__ == "__main__":
    print("Testing EIA Client...")
    
    try:
        client = EIAClient()
        
        # Test state summary
        summary = client.get_state_renewable_summary("CA")
        print(f"\nCalifornia Renewable Summary:")
        print(f"  Solar: {summary['solar_mw']:.0f} MW ({summary['solar_count']} plants)")
        print(f"  Wind: {summary['wind_mw']:.0f} MW ({summary['wind_count']} plants)")
        
        # Test AEO projections
        projections = client.get_renewable_capacity_forecast("solar", "ref2025")
        if not projections.empty:
            print(f"\nSolar Capacity Projections (ref2025):")
            print(projections.head())
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Set EIA_API_KEY environment variable to test")
