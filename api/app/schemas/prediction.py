"""Pydantic schemas for the prediction API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PointPredictionRequest(BaseModel):
    """Request body for single-point prediction."""

    lat: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    lng: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    zoom: int = Field(16, ge=1, le=22, description="Mapbox zoom level (default 16)")


class PointPredictionResponse(BaseModel):
    """Response body for single-point prediction."""

    lat: float
    lng: float
    prediction: Literal["agri", "non-agri"]
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    image_url: str


class AreaPredictionRequest(BaseModel):
    """Request body for area (polygon) prediction."""

    polygon: list[list[float]] = Field(
        ...,
        description="List of [lng, lat] coordinates forming a closed polygon",
    )
    zoom: int = Field(16, ge=1, le=22, description="Mapbox zoom level (default 16)")
    grid_size: Optional[int] = Field(
        None,
        ge=2,
        le=20,
        description="Deprecated: grid density is now computed automatically.",
    )


class TileBounds(BaseModel):
    """Geographic bounds of a single 64×64 tile."""

    north: float
    south: float
    east: float
    west: float


class GridPoint(BaseModel):
    """A single grid point prediction result."""

    lat: float
    lng: float
    prediction: Literal["agri", "non-agri"]
    confidence: float
    probabilities: dict[str, float]
    tile_bounds: TileBounds


class AreaPredictionResponse(BaseModel):
    """Response body for area (polygon) prediction."""

    total_points: int
    agri_points: int
    non_agri_points: int
    agri_percentage: float
    avg_confidence: float
    grid_points: list[GridPoint]
    bounding_box: dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    gpu: bool
    model_loaded: bool
    device: str
