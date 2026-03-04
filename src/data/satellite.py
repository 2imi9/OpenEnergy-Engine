"""
Satellite Data Pipeline for Earth Observation

Fetches and processes satellite imagery from:
- Sentinel-2 (10m optical, 5-day revisit)
- Landsat-8/9 (30m optical, 16-day revisit)
- Sentinel-1 SAR (10m radar, 6-day revisit)

Uses Microsoft Planetary Computer STAC API.

Author: Zim (Millennium Fellowship Research)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Optional imports - gracefully handle missing dependencies
try:
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    HAS_GEO_DEPS = True
except ImportError:
    HAS_GEO_DEPS = False
    logger.warning("Geospatial dependencies not installed. Install with: pip install planetary-computer pystac-client rasterio shapely")


@dataclass
class ImageryConfig:
    """Configuration for satellite imagery retrieval."""
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    max_cloud_cover: float = 10.0
    default_chip_size_meters: int = 1000
    default_bands: List[str] = None
    
    def __post_init__(self):
        if self.default_bands is None:
            self.default_bands = ["B04", "B03", "B02", "B08"]  # RGB + NIR


@dataclass
class ImageChip:
    """A satellite image chip with metadata."""
    data: np.ndarray  # Shape: (C, H, W)
    bands: List[str]
    bbox: List[float]
    datetime: str
    source: str
    cloud_cover: float
    item_id: str
    crs: str = "EPSG:4326"
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    def to_rgb(self, bands: List[str] = ["B04", "B03", "B02"]) -> np.ndarray:
        """Extract RGB channels."""
        indices = [self.bands.index(b) for b in bands if b in self.bands]
        if len(indices) == 3:
            return self.data[indices].transpose(1, 2, 0)
        return self.data[:3].transpose(1, 2, 0)
    
    def normalize(self, percentile: float = 98) -> np.ndarray:
        """Normalize to 0-1 range."""
        data = self.data.astype(np.float32)
        for i in range(data.shape[0]):
            p_low, p_high = np.percentile(data[i], [2, percentile])
            data[i] = np.clip((data[i] - p_low) / (p_high - p_low + 1e-8), 0, 1)
        return data


class SatelliteClient:
    """
    Client for retrieving satellite imagery from Planetary Computer.
    
    Usage:
        client = SatelliteClient()
        chip = client.get_sentinel2_chip(lat=35.0, lon=-119.9)
        
        # For time series
        chips = client.get_time_series(lat, lon, "2024-01-01", "2024-12-31")
    """
    
    SENTINEL2_BANDS = {
        "B02": {"name": "Blue", "wavelength": 490, "resolution": 10},
        "B03": {"name": "Green", "wavelength": 560, "resolution": 10},
        "B04": {"name": "Red", "wavelength": 665, "resolution": 10},
        "B05": {"name": "Red Edge 1", "wavelength": 705, "resolution": 20},
        "B06": {"name": "Red Edge 2", "wavelength": 740, "resolution": 20},
        "B07": {"name": "Red Edge 3", "wavelength": 783, "resolution": 20},
        "B08": {"name": "NIR", "wavelength": 842, "resolution": 10},
        "B8A": {"name": "NIR Narrow", "wavelength": 865, "resolution": 20},
        "B11": {"name": "SWIR 1", "wavelength": 1610, "resolution": 20},
        "B12": {"name": "SWIR 2", "wavelength": 2190, "resolution": 20},
    }
    
    def __init__(self, config: Optional[ImageryConfig] = None):
        if not HAS_GEO_DEPS:
            raise ImportError(
                "Geospatial dependencies required. Install with:\n"
                "pip install planetary-computer pystac-client rasterio shapely"
            )
        
        self.config = config or ImageryConfig()
        self.catalog = pystac_client.Client.open(
            self.config.stac_url,
            modifier=planetary_computer.sign_inplace,
        )
    
    def _create_bbox(self, lat: float, lon: float, size_meters: int) -> List[float]:
        """Create bounding box around point."""
        buffer = size_meters / 111000  # Approximate degrees
        return [lon - buffer, lat - buffer, lon + buffer, lat + buffer]
    
    def _load_bands(
        self,
        item,
        bbox: List[float],
        bands: List[str],
        target_size: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Load bands from STAC item, handling CRS transformation.

        All bands are resampled to the same pixel dimensions so they can
        be stacked even when native resolutions differ (10m vs 20m).
        """
        chip_data = []
        loaded_bands = []
        ref_shape = None

        for band in bands:
            if band not in item.assets:
                continue

            href = item.assets[band].href
            try:
                with rasterio.open(href) as src:
                    # Transform bbox from EPSG:4326 to raster CRS
                    raster_bbox = transform_bounds(
                        "EPSG:4326", src.crs, *bbox
                    )
                    window = from_bounds(*raster_bbox, transform=src.transform)
                    out_image = src.read(1, window=window)

                    # Use first band's shape as reference, resize others to match
                    if ref_shape is None:
                        ref_shape = out_image.shape
                    elif out_image.shape != ref_shape:
                        from PIL import Image
                        out_image = np.array(
                            Image.fromarray(out_image).resize(
                                (ref_shape[1], ref_shape[0]), Image.BILINEAR
                            )
                        )

                    chip_data.append(out_image)
                    loaded_bands.append(band)
            except Exception as e:
                logger.warning(f"Error loading band {band}: {e}")
                continue

        if not chip_data:
            return None, []

        return np.stack(chip_data, axis=0), loaded_bands
    
    def get_sentinel2_chip(
        self,
        lat: float,
        lon: float,
        chip_size_meters: Optional[int] = None,
        date_range: str = "2024-01-01/2024-12-31",
        max_cloud_cover: Optional[float] = None,
        bands: Optional[List[str]] = None
    ) -> Optional[ImageChip]:
        """
        Fetch Sentinel-2 image chip.
        
        Args:
            lat, lon: Center coordinates
            chip_size_meters: Size of chip
            date_range: Date range for search
            max_cloud_cover: Cloud cover threshold
            bands: List of band names
            
        Returns:
            ImageChip or None
        """
        chip_size = chip_size_meters or self.config.default_chip_size_meters
        cloud_thresh = max_cloud_cover or self.config.max_cloud_cover
        bands = bands or self.config.default_bands
        
        bbox = self._create_bbox(lat, lon, chip_size)
        
        # Search
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": cloud_thresh}}
        )
        
        items = list(search.items())
        if not items:
            logger.warning(f"No imagery found for ({lat}, {lon})")
            return None
        
        # Get least cloudy
        best_item = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
        
        # Load bands
        data, loaded_bands = self._load_bands(best_item, bbox, bands)
        if data is None:
            return None
        
        return ImageChip(
            data=data,
            bands=loaded_bands,
            bbox=bbox,
            datetime=best_item.datetime.isoformat(),
            source="sentinel-2-l2a",
            cloud_cover=best_item.properties.get("eo:cloud_cover", 0),
            item_id=best_item.id
        )
    
    def get_time_series(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        max_images: int = 12,
        max_cloud_cover: float = 20.0
    ) -> List[ImageChip]:
        """Get time series of images for change detection."""
        bbox = self._create_bbox(lat, lon, self.config.default_chip_size_meters)
        
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            sortby=[{"field": "datetime", "direction": "asc"}],
            limit=max_images
        )
        
        chips = []
        for item in search.items():
            data, bands = self._load_bands(item, bbox, self.config.default_bands)
            if data is not None:
                chips.append(ImageChip(
                    data=data,
                    bands=bands,
                    bbox=bbox,
                    datetime=item.datetime.isoformat(),
                    source="sentinel-2-l2a",
                    cloud_cover=item.properties.get("eo:cloud_cover", 0),
                    item_id=item.id
                ))
            if len(chips) >= max_images:
                break
        
        return chips


class MockSatelliteClient:
    """Mock client for testing without API access."""
    
    def __init__(self, config: Optional[ImageryConfig] = None):
        self.config = config or ImageryConfig()
    
    def get_sentinel2_chip(self, lat: float, lon: float, **kwargs) -> ImageChip:
        """Return synthetic test data."""
        size = 224
        data = np.random.rand(4, size, size).astype(np.float32) * 10000
        
        return ImageChip(
            data=data,
            bands=["B04", "B03", "B02", "B08"],
            bbox=[lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01],
            datetime=datetime.now().isoformat(),
            source="mock",
            cloud_cover=5.0,
            item_id="mock_001"
        )
    
    def get_time_series(self, lat: float, lon: float, **kwargs) -> List[ImageChip]:
        """Return synthetic time series."""
        return [self.get_sentinel2_chip(lat, lon) for _ in range(6)]


def create_satellite_client(use_mock: bool = False) -> SatelliteClient:
    """Factory function."""
    if use_mock or not HAS_GEO_DEPS:
        return MockSatelliteClient()
    return SatelliteClient()


if __name__ == "__main__":
    print("Testing Satellite Client...")
    
    if HAS_GEO_DEPS:
        client = SatelliteClient()
        chip = client.get_sentinel2_chip(lat=35.0381, lon=-119.9397)
        if chip:
            print(f"Retrieved chip: {chip.shape}")
            print(f"Bands: {chip.bands}")
            print(f"Date: {chip.datetime}")
    else:
        print("Using mock client (geospatial deps not installed)")
        client = MockSatelliteClient()
        chip = client.get_sentinel2_chip(lat=35.0, lon=-119.9)
        print(f"Mock chip: {chip.shape}")
