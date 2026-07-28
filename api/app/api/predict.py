"""Prediction API routes."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from api.app.schemas import (
    AreaPredictionRequest,
    AreaPredictionResponse,
    PointPredictionRequest,
    PointPredictionResponse,
)
from api.app.services import ModelService, SatelliteService
from api.app.services.tiling import AreaTooLargeError, TilingService

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])

model_service = ModelService()
satellite_service = SatelliteService()

MAX_AREA_PATCHES = int(os.getenv("MAX_AREA_PATCHES", "200"))


@lru_cache
def _tiling_service() -> TilingService:
    token = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError("MAPBOX_ACCESS_TOKEN not set in environment")
    return TilingService(token)


@router.post("/point", response_model=PointPredictionResponse)
async def predict_point(request: PointPredictionRequest) -> dict:
    """Classify a single point on the map as agricultural or non-agricultural land.

    - Fetches a 64×64 satellite tile from Mapbox
    - Runs CNN-ViT inference on GPU
    - Returns prediction with confidence scores
    """
    try:
        # 1. Fetch satellite image
        image = satellite_service.fetch_image(request.lat, request.lng, request.zoom)

        # 2. Preprocess
        preprocessed = satellite_service.preprocess(image)
        tensor = preprocessed["tensor"]

        # 3. Inference
        result = model_service.predict(tensor)

        # 4. Build image URL for frontend
        image_url = satellite_service.get_image_url(request.lat, request.lng, request.zoom)

        return {
            "lat": request.lat,
            "lng": request.lng,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "image_url": image_url,
        }

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@router.post("/area", response_model=AreaPredictionResponse)
async def predict_area(request: AreaPredictionRequest) -> dict:
    """Classify an area (polygon) on the map as agricultural or non-agricultural land.

    - Generates a dense 64×64 tile grid covering the polygon
    - Fetches satellite tiles from Mapbox and slices them locally
    - Runs CNN-ViT inference on GPU in batches
    - Returns aggregated statistics with per-tile results
    """
    tiling_service = _tiling_service()
    zoom = request.zoom

    try:
        # 1. Generate 64×64 patches covering the polygon
        patches = tiling_service.generate_patches(
            request.polygon, zoom=zoom, max_patches=MAX_AREA_PATCHES
        )

        if not patches:
            raise HTTPException(status_code=400, detail="No grid points generated from polygon")

        # 2. Fetch tiles and crop patches
        tile_images = await tiling_service.fetch_tiles(patches, zoom=zoom)
        batch_tensor = tiling_service.crop_and_preprocess(patches, tile_images)

        # 3. Batch inference
        results = model_service.predict_batch(batch_tensor)

        # 4. Build response
        grid_points = []
        for patch, result in zip(patches, results):
            grid_points.append(
                {
                    "lat": patch.lat,
                    "lng": patch.lng,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "tile_bounds": patch.bounds,
                }
            )

        agri_count = sum(1 for r in results if r["prediction"] == "agri")
        non_agri_count = len(results) - agri_count
        agri_percentage = (agri_count / len(results)) * 100 if results else 0
        avg_confidence = sum(r["confidence"] for r in results) / len(results)

        lats = [p[1] for p in request.polygon]
        lngs = [p[0] for p in request.polygon]
        bounding_box = {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lng": min(lngs),
            "max_lng": max(lngs),
        }

        return {
            "total_points": len(results),
            "agri_points": agri_count,
            "non_agri_points": non_agri_count,
            "agri_percentage": round(agri_percentage, 2),
            "avg_confidence": round(avg_confidence, 4),
            "grid_points": grid_points,
            "bounding_box": bounding_box,
        }

    except AreaTooLargeError as exc:
        logger.warning("Area too large: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Area too large. Please draw a smaller area.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Area prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Area prediction failed: {exc}")
