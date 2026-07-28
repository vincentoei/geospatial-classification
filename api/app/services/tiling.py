"""Polygon tiling service for dense satellite-image coverage."""

import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Tuple

import httpx
import mercantile
import torch
from PIL import Image

from geospatial_classification.data.transforms import get_inference_transform

logger = logging.getLogger(__name__)

TILE_SIZE = 256  # Mapbox standard raster tile size in CSS pixels
PATCH_SIZE = 64  # Model input size in CSS pixels

# Half the world width in Web Mercator meters.
_WORLD_HALF = 20037508.342789244


class AreaTooLargeError(Exception):
    """Raised when a polygon would require more patches than the allowed limit."""

    pass


@dataclass
class Patch:
    """A single 64×64 patch inside a polygon."""

    lat: float
    lng: float
    bounds: Dict[str, float]
    tile_x: int
    tile_y: int
    offset_x: int
    offset_y: int


class TilingService:
    """Generate 64×64 patch coverage for a polygon using Mapbox raster tiles."""

    def __init__(self, access_token: str) -> None:
        self.access_token = access_token
        self.tile_url_template = (
            "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png" "?access_token={token}"
        )

    def _point_to_global_pixel(self, lng: float, lat: float, zoom: int) -> Tuple[float, float]:
        """Convert a lat/lng to global Web Mercator pixel coordinates."""
        tile = mercantile.tile(lng, lat, zoom)
        upper_left = mercantile.ul(tile)
        tile_left, tile_top = mercantile.xy(upper_left.lng, upper_left.lat)
        point_x, point_y = mercantile.xy(lng, lat)

        world_pixels = (2**zoom) * TILE_SIZE
        meters_per_pixel = (2 * _WORLD_HALF) / world_pixels

        offset_x = (point_x - tile_left) / meters_per_pixel
        offset_y = (tile_top - point_y) / meters_per_pixel

        global_x = tile.x * TILE_SIZE + offset_x
        global_y = tile.y * TILE_SIZE + offset_y
        return global_x, global_y

    def _global_pixel_to_latlng(self, px: float, py: float, zoom: int) -> Tuple[float, float]:
        """Convert global Web Mercator pixel coordinates back to lat/lng."""
        tile_x = int(px // TILE_SIZE)
        tile_y = int(py // TILE_SIZE)
        offset_x = px - tile_x * TILE_SIZE
        offset_y = py - tile_y * TILE_SIZE

        tile = mercantile.Tile(x=tile_x, y=tile_y, z=zoom)
        upper_left = mercantile.ul(tile)
        tile_left, tile_top = mercantile.xy(upper_left.lng, upper_left.lat)

        world_pixels = (2**zoom) * TILE_SIZE
        meters_per_pixel = (2 * _WORLD_HALF) / world_pixels

        point_x = tile_left + offset_x * meters_per_pixel
        point_y = tile_top - offset_y * meters_per_pixel
        lng, lat = mercantile.lnglat(point_x, point_y)
        return lng, lat

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0][0], polygon[0][1]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n][0], polygon[i % n][1]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def generate_patches(
        self,
        polygon: List[List[float]],
        zoom: int,
        max_patches: int,
    ) -> List[Patch]:
        """Generate all 64×64 patches whose centers fall inside the polygon.

        Args:
            polygon: List of [lng, lat] coordinates.
            zoom: Mapbox zoom level.
            max_patches: Maximum number of patches allowed.

        Returns:
            List of Patch objects.

        Raises:
            AreaTooLargeError: If the polygon would require more than max_patches.
        """
        global_pixels = [self._point_to_global_pixel(lng, lat, zoom) for lng, lat in polygon]
        min_px = int(min(p[0] for p in global_pixels))
        max_px = int(max(p[0] for p in global_pixels))
        min_py = int(min(p[1] for p in global_pixels))
        max_py = int(max(p[1] for p in global_pixels))

        start_px = (min_px // PATCH_SIZE) * PATCH_SIZE
        start_py = (min_py // PATCH_SIZE) * PATCH_SIZE
        end_px = ((max_px // PATCH_SIZE) + 1) * PATCH_SIZE
        end_py = ((max_py // PATCH_SIZE) + 1) * PATCH_SIZE

        cols = (end_px - start_px) // PATCH_SIZE
        rows = (end_py - start_py) // PATCH_SIZE
        bbox_cell_count = cols * rows

        # If the bounding box alone is far larger than the limit, reject early.
        if bbox_cell_count > max_patches * 4:
            raise AreaTooLargeError(
                f"Area too large ({bbox_cell_count} candidate tiles). "
                f"Please draw a smaller area."
            )

        patches: List[Patch] = []
        for px in range(start_px, end_px, PATCH_SIZE):
            for py in range(start_py, end_py, PATCH_SIZE):
                center_px = px + PATCH_SIZE / 2
                center_py = py + PATCH_SIZE / 2
                center_lng, center_lat = self._global_pixel_to_latlng(center_px, center_py, zoom)

                if self._point_in_polygon(center_lng, center_lat, polygon):
                    nw_lng, nw_lat = self._global_pixel_to_latlng(px, py, zoom)
                    se_lng, se_lat = self._global_pixel_to_latlng(
                        px + PATCH_SIZE, py + PATCH_SIZE, zoom
                    )

                    patches.append(
                        Patch(
                            lat=center_lat,
                            lng=center_lng,
                            bounds={
                                "north": nw_lat,
                                "south": se_lat,
                                "east": se_lng,
                                "west": nw_lng,
                            },
                            tile_x=int(px // TILE_SIZE),
                            tile_y=int(py // TILE_SIZE),
                            offset_x=int(px - (px // TILE_SIZE) * TILE_SIZE),
                            offset_y=int(py - (py // TILE_SIZE) * TILE_SIZE),
                        )
                    )

                    if len(patches) > max_patches:
                        raise AreaTooLargeError(
                            f"Area too large ({len(patches)} tiles). "
                            f"Please draw a smaller area."
                        )

        return patches

    async def fetch_tiles(
        self, patches: List[Patch], zoom: int
    ) -> Dict[Tuple[int, int], Image.Image]:
        """Fetch the unique Mapbox tiles needed for the given patches."""
        unique_tiles = {(p.tile_x, p.tile_y) for p in patches}

        async def _fetch(
            client: httpx.AsyncClient, tile_x: int, tile_y: int
        ) -> Tuple[Tuple[int, int], Image.Image]:
            url = self.tile_url_template.format(x=tile_x, y=tile_y, z=zoom, token=self.access_token)
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return (tile_x, tile_y), image

        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *[_fetch(client, tx, ty) for tx, ty in unique_tiles],
                return_exceptions=True,
            )

        tile_images: Dict[Tuple[int, int], Image.Image] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error("Failed to fetch tile: %s", result)
                raise RuntimeError(f"Failed to fetch satellite tile: {result}")
            (tile_x, tile_y), image = result
            tile_images[(tile_x, tile_y)] = image

        return tile_images

    def crop_and_preprocess(
        self,
        patches: List[Patch],
        tile_images: Dict[Tuple[int, int], Image.Image],
        img_size: Tuple[int, int] = (PATCH_SIZE, PATCH_SIZE),
    ) -> torch.Tensor:
        """Crop patches from tiles and stack into a model-ready tensor batch."""
        transform = get_inference_transform(img_size)
        tensors = []

        for patch in patches:
            tile = tile_images[(patch.tile_x, patch.tile_y)]
            cropped = tile.crop(
                (
                    patch.offset_x,
                    patch.offset_y,
                    patch.offset_x + PATCH_SIZE,
                    patch.offset_y + PATCH_SIZE,
                )
            )
            tensor = transform(cropped)
            tensors.append(tensor)

        return torch.stack(tensors)
