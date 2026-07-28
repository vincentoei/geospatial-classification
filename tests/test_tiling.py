"""Tests for the polygon tiling service."""

import pytest

from api.app.services.tiling import AreaTooLargeError, TilingService


@pytest.fixture
def tiling_service() -> TilingService:
    return TilingService("dummy-token")


def test_generate_patches_small_polygon(tiling_service: TilingService) -> None:
    """A small polygon should produce a non-empty set of patches."""
    polygon = [
        [-93.10, 41.87],
        [-93.09, 41.87],
        [-93.09, 41.88],
        [-93.10, 41.88],
        [-93.10, 41.87],
    ]
    patches = tiling_service.generate_patches(polygon, zoom=16, max_patches=200)
    assert len(patches) > 0
    assert len(patches) <= 200
    for patch in patches:
        assert "north" in patch.bounds
        assert "south" in patch.bounds
        assert "east" in patch.bounds
        assert "west" in patch.bounds
        assert patch.bounds["north"] > patch.bounds["south"]
        assert patch.bounds["east"] > patch.bounds["west"]


def test_generate_patches_large_polygon_rejected(tiling_service: TilingService) -> None:
    """A huge polygon should exceed the patch limit and raise AreaTooLargeError."""
    polygon = [
        [-93.20, 41.80],
        [-93.00, 41.80],
        [-93.00, 42.00],
        [-93.20, 42.00],
        [-93.20, 41.80],
    ]
    with pytest.raises(AreaTooLargeError):
        tiling_service.generate_patches(polygon, zoom=16, max_patches=200)


def test_generate_patches_exceeds_limit(tiling_service: TilingService) -> None:
    """A polygon that fits in the bounding box but exceeds max_patches is rejected."""
    polygon = [
        [-93.10, 41.87],
        [-93.09, 41.87],
        [-93.09, 41.88],
        [-93.10, 41.88],
        [-93.10, 41.87],
    ]
    with pytest.raises(AreaTooLargeError):
        tiling_service.generate_patches(polygon, zoom=16, max_patches=5)


def test_point_in_polygon(tiling_service: TilingService) -> None:
    """Ray-casting point-in-polygon test for a square."""
    polygon = [
        [-93.10, 41.87],
        [-93.09, 41.87],
        [-93.09, 41.88],
        [-93.10, 41.88],
        [-93.10, 41.87],
    ]
    assert tiling_service._point_in_polygon(-93.095, 41.875, polygon)
    assert not tiling_service._point_in_polygon(-93.20, 41.90, polygon)
