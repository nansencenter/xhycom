"""Tests for xhycom._boundaries: tp2_sections and TP2Sections."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xhycom
from xhycom._boundaries import (
    TP2Sections,
    _extend_transect,
    _glorys_path_from_vface_row,
    _unwrap_lons,
)
from xhycom._transect import ResolvedTransect, Transect

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tp2_grid() -> xr.Dataset:
    """Minimal 5×8 grid with all variables required by from_vface_row."""
    jdm, idm = 5, 8
    j, i = np.meshgrid(np.arange(jdm), np.arange(idm), indexing="ij")
    return xr.Dataset(
        {
            "plon": (("y", "x"), i.astype(float)),
            "plat": (("y", "x"), (45.0 + j).astype(float)),
            "scvx": (("y", "x"), np.full((jdm, idm), 74_000.0)),
        }
    )


@pytest.fixture()
def tp2_bathy(tp2_grid: xr.Dataset) -> xr.Dataset:
    """Bathymetry with all ocean cells wet."""
    jdm = tp2_grid.sizes["y"]
    idm = tp2_grid.sizes["x"]
    return xr.Dataset({"depth": (("y", "x"), np.full((jdm, idm), 500.0))})


@pytest.fixture()
def glorys_grid() -> xr.Dataset:
    """Tiny GLORYS-style rectilinear grid around the synthetic domain."""
    lats = np.arange(44.0, 52.0, 0.5)
    lons = np.arange(-1.0, 9.0, 0.5)
    return xr.Dataset(
        {
            "uo": (("latitude", "longitude"), np.ones((len(lats), len(lons)))),
            "vo": (("latitude", "longitude"), np.zeros((len(lats), len(lons)))),
        },
        coords={"latitude": lats, "longitude": lons},
    )


@pytest.fixture()
def aleutian_arc_in_domain() -> Transect:
    """Aleutian Arc transect placed inside the synthetic GLORYS domain for testing."""
    return Transect(lons=[2.0, 4.0, 6.0], lats=[46.5, 47.5, 48.5], name="aleutian_arc")


# ---------------------------------------------------------------------------
# _unwrap_lons
# ---------------------------------------------------------------------------


def test_unwrap_lons_no_jump() -> None:
    """Lons without jumps are returned unchanged."""
    lons = np.array([10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(_unwrap_lons(lons), lons)


def test_unwrap_lons_positive_jump() -> None:
    """A jump from -179° to +179° (diff=+358, westward crossing) shifts by -360°."""
    lons = np.array([-179.0, 179.0])
    result = _unwrap_lons(lons)
    assert result[1] == pytest.approx(179.0 - 360.0)  # = -181


def test_unwrap_lons_negative_jump() -> None:
    """A jump from +179° to -179° (diff=-358, eastward crossing) shifts by +360°."""
    lons = np.array([179.0, -179.0])
    result = _unwrap_lons(lons)
    assert result[1] == pytest.approx(-179.0 + 360.0)  # = 181


def test_unwrap_lons_does_not_mutate() -> None:
    """_unwrap_lons does not modify the input array."""
    lons = np.array([179.0, -179.0])
    original = lons.copy()
    _unwrap_lons(lons)
    np.testing.assert_array_equal(lons, original)


# ---------------------------------------------------------------------------
# _extend_transect
# ---------------------------------------------------------------------------


def test_extend_transect_start_adds_point() -> None:
    """deg_start > 0 prepends a point before the first waypoint."""
    t = Transect(lons=[0.0, 10.0], lats=[45.0, 45.0])
    ext = _extend_transect(t, deg_start=1.0)
    assert len(ext.lons) == 3
    assert ext.lons[1] == pytest.approx(0.0)


def test_extend_transect_end_adds_point() -> None:
    """deg_end > 0 appends a point after the last waypoint."""
    t = Transect(lons=[0.0, 10.0], lats=[45.0, 45.0])
    ext = _extend_transect(t, deg_end=1.0)
    assert len(ext.lons) == 3
    assert ext.lons[-2] == pytest.approx(10.0)


def test_extend_transect_preserves_name() -> None:
    """_extend_transect keeps the original transect name."""
    t = Transect(lons=[0.0, 10.0], lats=[45.0, 45.0], name="test_section")
    assert _extend_transect(t, deg_start=1.0).name == "test_section"


def test_extend_transect_zero_noop() -> None:
    """deg_start=0 and deg_end=0 returns a transect with the same waypoints."""
    t = Transect(lons=[0.0, 5.0, 10.0], lats=[45.0, 46.0, 47.0])
    ext = _extend_transect(t)
    assert len(ext.lons) == len(t.lons)


# ---------------------------------------------------------------------------
# tp2_sections — HYCOM only
# ---------------------------------------------------------------------------


def test_tp2_sections_returns_tp2sections(
    tp2_grid: xr.Dataset, tp2_bathy: xr.Dataset
) -> None:
    """tp2_sections returns a TP2Sections instance."""
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy)
    assert isinstance(sec, TP2Sections)


def test_tp2_sections_hycom_are_resolved_transects(
    tp2_grid: xr.Dataset, tp2_bathy: xr.Dataset
) -> None:
    """Atlantic and Pacific HYCOM sections are ResolvedTransect objects."""
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy)
    assert isinstance(sec.hycom_atlantic_boundary, ResolvedTransect)
    assert isinstance(sec.hycom_pacific_boundary, ResolvedTransect)


def test_tp2_sections_glorys_none_without_data(
    tp2_grid: xr.Dataset, tp2_bathy: xr.Dataset
) -> None:
    """GLORYS sections are None when glorys_data is not provided."""
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy)
    assert sec.glorys_atlantic_boundary is None
    assert sec.glorys_pacific_boundary is None
    assert sec.glorys_aleutian_boundary is None


def test_tp2_sections_atlantic_sign(
    tp2_grid: xr.Dataset, tp2_bathy: xr.Dataset
) -> None:
    """Atlantic section has positive face_sign (inflow = positive)."""
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy)
    assert np.all(sec.hycom_atlantic_boundary.face_sign > 0)


def test_tp2_sections_pacific_sign(tp2_grid: xr.Dataset, tp2_bathy: xr.Dataset) -> None:
    """Pacific section has negative face_sign (positive transport = inflow)."""
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy)
    assert np.all(sec.hycom_pacific_boundary.face_sign < 0)


def test_tp2_sections_boundary_rows(
    tp2_grid: xr.Dataset, tp2_bathy: xr.Dataset
) -> None:
    """Atlantic section is at j=2 and Pacific at j=jdm-2 (one row inward from ghost)."""
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy)
    jdm = tp2_grid.sizes["y"]
    assert np.all(sec.hycom_atlantic_boundary.face_j == 2)
    assert np.all(sec.hycom_pacific_boundary.face_j == jdm - 2)


# ---------------------------------------------------------------------------
# tp2_sections — with GLORYS data
# ---------------------------------------------------------------------------


def test_tp2_sections_glorys_resolved_transects(
    tp2_grid: xr.Dataset,
    tp2_bathy: xr.Dataset,
    glorys_grid: xr.Dataset,
    aleutian_arc_in_domain: Transect,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLORYS sections are ResolvedTransects when glorys_data is provided."""
    pytest.importorskip("scipy")
    import xhycom._boundaries as _b

    monkeypatch.setattr(_b, "_GLORYS_ALEUTIAN_ARC", aleutian_arc_in_domain)
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy, glorys_grid)
    assert isinstance(sec.glorys_atlantic_boundary, ResolvedTransect)
    assert isinstance(sec.glorys_pacific_boundary, ResolvedTransect)
    assert isinstance(sec.glorys_aleutian_boundary, ResolvedTransect)


def test_tp2_sections_glorys_no_face_data(
    tp2_grid: xr.Dataset,
    tp2_bathy: xr.Dataset,
    glorys_grid: xr.Dataset,
    aleutian_arc_in_domain: Transect,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLORYS sections are T-point resolved (no face data)."""
    pytest.importorskip("scipy")
    import xhycom._boundaries as _b

    monkeypatch.setattr(_b, "_GLORYS_ALEUTIAN_ARC", aleutian_arc_in_domain)
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy, glorys_grid)
    assert not sec.glorys_atlantic_boundary.has_face_data
    assert not sec.glorys_pacific_boundary.has_face_data
    assert not sec.glorys_aleutian_boundary.has_face_data


def test_tp2_sections_glorys_have_cells(
    tp2_grid: xr.Dataset,
    tp2_bathy: xr.Dataset,
    glorys_grid: xr.Dataset,
    aleutian_arc_in_domain: Transect,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GLORYS sections have at least one T-cell."""
    pytest.importorskip("scipy")
    import xhycom._boundaries as _b

    monkeypatch.setattr(_b, "_GLORYS_ALEUTIAN_ARC", aleutian_arc_in_domain)
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy, glorys_grid)
    assert sec.glorys_atlantic_boundary.n_cells >= 1
    assert sec.glorys_pacific_boundary.n_cells >= 1
    assert sec.glorys_aleutian_boundary.n_cells >= 1


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------


def test_glorys_path_from_vface_row_reverse(
    tp2_grid: xr.Dataset, tp2_bathy: xr.Dataset
) -> None:
    """reverse=True returns waypoints in the opposite order to the default."""
    sec = xhycom.tp2_sections(tp2_grid, tp2_bathy)
    fwd = _glorys_path_from_vface_row(sec.hycom_atlantic_boundary, name="s")
    rev = _glorys_path_from_vface_row(
        sec.hycom_atlantic_boundary, name="s", reverse=True
    )
    np.testing.assert_array_almost_equal(list(rev.lons), list(reversed(fwd.lons)))
    np.testing.assert_array_almost_equal(list(rev.lats), list(reversed(fwd.lats)))


def test_aleutian_arc_oriented_into_domain() -> None:
    """_GLORYS_ALEUTIAN_ARC goes east→west so rightward (northwestward) is into the domain."""
    from xhycom._boundaries import _GLORYS_ALEUTIAN_ARC

    # First waypoint should be east of the last waypoint (i.e. NE→SW orientation).
    assert _GLORYS_ALEUTIAN_ARC.lons[0] > _GLORYS_ALEUTIAN_ARC.lons[-1]
