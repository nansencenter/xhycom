"""Tests for xhycom._transect: geometry helpers, Transect, and ResolvedTransect."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

from xhycom._transect import (
    ResolvedTransect,
    Transect,
    _break_diagonals,
    _cell_widths_km,
    _cumulative_distance_km,
    _forward_bearing,
    _haversine_km,
    _sample_polyline,
    _section_bearings,
)

# ---------------------------------------------------------------------------
# _haversine_km
# ---------------------------------------------------------------------------


def test_haversine_same_point_is_zero() -> None:
    """Distance from a point to itself is 0."""
    assert _haversine_km(10.0, 45.0, 10.0, 45.0) == pytest.approx(0.0)


def test_haversine_quarter_meridian() -> None:
    """Quarter-meridian distance is ~10 008 km."""
    d = _haversine_km(0.0, 0.0, 0.0, 90.0)
    assert d == pytest.approx(10007.5, rel=1e-3)


def test_haversine_equatorial_quarter_equals_meridional() -> None:
    """Quarter-equator and quarter-meridian are the same length."""
    assert _haversine_km(0.0, 0.0, 90.0, 0.0) == pytest.approx(
        _haversine_km(0.0, 0.0, 0.0, 90.0), rel=1e-6
    )


def test_haversine_vectorised_shape() -> None:
    """Vectorised input returns the same shape and all-positive values."""
    d = _haversine_km(np.array([0.0, 10.0]), 45.0, np.array([10.0, 20.0]), 45.0)
    assert d.shape == (2,)
    assert np.all(d > 0)


# ---------------------------------------------------------------------------
# _forward_bearing
# ---------------------------------------------------------------------------


def test_bearing_due_north() -> None:
    """Bearing from (0,0) to (0°N+10°) is 0° (due north)."""
    assert _forward_bearing(0.0, 0.0, 0.0, 10.0) == pytest.approx(0.0, abs=0.01)


def test_bearing_due_east() -> None:
    """Bearing from (0°E,0°N) to (10°E,0°N) on the equator is 90°."""
    assert _forward_bearing(0.0, 0.0, 10.0, 0.0) == pytest.approx(90.0, abs=0.01)


def test_bearing_due_west() -> None:
    """Bearing from (10°E,0) to (0°E,0) is 270°."""
    assert _forward_bearing(10.0, 0.0, 0.0, 0.0) == pytest.approx(270.0, abs=0.01)


def test_bearing_due_south() -> None:
    """Bearing from (0°E,10°N) to (0°E,0°N) is 180°."""
    assert _forward_bearing(0.0, 10.0, 0.0, 0.0) == pytest.approx(180.0, abs=0.01)


# ---------------------------------------------------------------------------
# _sample_polyline
# ---------------------------------------------------------------------------


def test_sample_polyline_starts_at_first_waypoint() -> None:
    """First output point matches the first waypoint exactly."""
    lons, lats = _sample_polyline([0.0, 10.0], [0.0, 0.0])
    assert lons[0] == pytest.approx(0.0)
    assert lats[0] == pytest.approx(0.0)


def test_sample_polyline_ends_at_last_waypoint() -> None:
    """Last output point matches the last waypoint exactly."""
    lons, lats = _sample_polyline([0.0, 10.0], [0.0, 0.0])
    assert lons[-1] == pytest.approx(10.0)
    assert lats[-1] == pytest.approx(0.0)


def test_sample_polyline_smaller_step_gives_more_points() -> None:
    """A finer step_km produces a denser sampling."""
    lo, _ = _sample_polyline([0.0, 5.0], [45.0, 45.0], step_km=20.0)
    hi, _ = _sample_polyline([0.0, 5.0], [45.0, 45.0], step_km=2.0)
    assert len(hi) > len(lo)


# ---------------------------------------------------------------------------
# _cumulative_distance_km
# ---------------------------------------------------------------------------


def test_cumulative_distance_starts_at_zero() -> None:
    """Cumulative distance always begins at 0."""
    d = _cumulative_distance_km(
        np.array([0.0, 5.0, 10.0]), np.array([45.0, 45.0, 45.0])
    )
    assert d[0] == 0.0


def test_cumulative_distance_is_monotone() -> None:
    """Cumulative distance is strictly increasing for distinct points."""
    d = _cumulative_distance_km(
        np.array([0.0, 5.0, 10.0]), np.array([45.0, 45.0, 45.0])
    )
    assert np.all(np.diff(d) > 0)


def test_cumulative_distance_single_point() -> None:
    """Single waypoint returns [0]."""
    d = _cumulative_distance_km(np.array([5.0]), np.array([45.0]))
    np.testing.assert_array_equal(d, [0.0])


# ---------------------------------------------------------------------------
# _cell_widths_km
# ---------------------------------------------------------------------------


def test_cell_widths_sum_equals_total_distance() -> None:
    """Sum of cell widths equals the last cumulative distance."""
    dist = np.array([0.0, 10.0, 30.0, 50.0])
    w = _cell_widths_km(dist)
    assert w.sum() == pytest.approx(dist[-1])


def test_cell_widths_single_cell_is_zero() -> None:
    """Single cell has zero width (no neighbours to average from)."""
    assert _cell_widths_km(np.array([0.0]))[0] == 0.0


def test_cell_widths_two_cells_equal_halves() -> None:
    """Two equidistant cells each own half the segment."""
    np.testing.assert_allclose(_cell_widths_km(np.array([0.0, 10.0])), [5.0, 5.0])


def test_cell_widths_interior_larger_than_ends() -> None:
    """Interior cells are wider than end cells for uniform spacing."""
    w = _cell_widths_km(np.array([0.0, 10.0, 20.0, 30.0]))
    assert w[1] > w[0]
    assert w[2] > w[3]


# ---------------------------------------------------------------------------
# _section_bearings
# ---------------------------------------------------------------------------


def test_section_bearings_zonal_near_ninety_degrees() -> None:
    """A purely zonal path has bearing within 5° of 90° at all cells."""
    lons = np.array([0.0, 2.0, 4.0, 6.0])
    lats = np.full(4, 45.0)
    b = _section_bearings(lons, lats)
    np.testing.assert_allclose(b, 90.0, atol=5.0)


def test_section_bearings_meridional_near_zero_degrees() -> None:
    """A purely meridional path (northward) has bearing within 1° of 0°."""
    lons = np.full(4, 5.0)
    lats = np.array([45.0, 46.0, 47.0, 48.0])
    b = _section_bearings(lons, lats)
    np.testing.assert_allclose(b, 0.0, atol=1.0)


# ---------------------------------------------------------------------------
# _break_diagonals
# ---------------------------------------------------------------------------


def test_break_diagonals_straight_zonal_unchanged() -> None:
    """A purely zonal path is returned without insertions."""
    j = np.array([0, 0, 0, 0])
    i = np.array([0, 1, 2, 3])
    j2, i2 = _break_diagonals(j, i)
    np.testing.assert_array_equal(j2, j)
    np.testing.assert_array_equal(i2, i)


def test_break_diagonals_straight_meridional_unchanged() -> None:
    """A purely meridional path is returned without insertions."""
    j = np.array([0, 1, 2, 3])
    i = np.array([0, 0, 0, 0])
    j2, i2 = _break_diagonals(j, i)
    np.testing.assert_array_equal(j2, j)
    np.testing.assert_array_equal(i2, i)


def test_break_diagonals_inserts_i_step_first() -> None:
    """A diagonal step (Δj=1, Δi=1) gets an i-step inserted before the j-step."""
    j2, i2 = _break_diagonals(np.array([0, 1]), np.array([0, 1]))
    assert len(j2) == 3
    np.testing.assert_array_equal(j2, [0, 0, 1])
    np.testing.assert_array_equal(i2, [0, 1, 1])


def test_break_diagonals_negative_diagonal() -> None:
    """A diagonal step (Δj=-1, Δi=1) also gets an intermediate i-step."""
    j2, i2 = _break_diagonals(np.array([1, 0]), np.array([0, 1]))
    assert len(j2) == 3
    np.testing.assert_array_equal(j2, [1, 1, 0])
    np.testing.assert_array_equal(i2, [0, 1, 1])


# ---------------------------------------------------------------------------
# Transect construction
# ---------------------------------------------------------------------------


def test_transect_stores_waypoints() -> None:
    """Transect stores lons and lats as float64 arrays."""
    t = Transect(lons=[0.0, 10.0], lats=[45.0, 45.0])
    np.testing.assert_array_equal(t.lons, [0.0, 10.0])
    np.testing.assert_array_equal(t.lats, [45.0, 45.0])


def test_transect_too_few_waypoints_raises() -> None:
    """Single-waypoint Transect raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        Transect(lons=[0.0], lats=[0.0])


def test_transect_mismatched_lengths_raises() -> None:
    """Mismatched lons/lats lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        Transect(lons=[0.0, 1.0], lats=[0.0])


def test_transect_stores_name() -> None:
    """Name attribute is stored verbatim."""
    t = Transect(lons=[0.0, 1.0], lats=[0.0, 0.0], name="test")
    assert t.name == "test"


def test_transect_named_returns_transect() -> None:
    """Transect.named returns a Transect with at least two waypoints."""
    t = Transect.named("fram_strait")
    assert isinstance(t, Transect)
    assert len(t.lons) >= 2


def test_transect_named_unknown_raises() -> None:
    """Unknown section name raises ValueError with helpful message."""
    with pytest.raises(ValueError, match="Unknown section"):
        Transect.named("made_up_section_xyz")


def test_transect_available_names_sorted() -> None:
    """available_names() returns a sorted list containing known sections."""
    names = Transect.available_names()
    assert names == sorted(names)
    assert "fram_strait" in names
    assert "bering_strait" in names
    assert "fsc" in names
    assert "topaz_southern" not in names
    assert "topaz_northern" not in names


# ---------------------------------------------------------------------------
# Transect.from_boundary_row
# ---------------------------------------------------------------------------


@pytest.fixture()
def boundary_grid_bathy() -> tuple[xr.Dataset, xr.Dataset]:
    """5×10 synthetic grid; row 0 all land, row 1 partially wet, row 4 wet."""
    jdm, idm = 5, 10
    j, i = np.meshgrid(np.arange(jdm), np.arange(idm), indexing="ij")
    grid = xr.Dataset(
        {
            "plon": (("y", "x"), i.astype(float)),
            "plat": (("y", "x"), (45.0 + j).astype(float)),
            "scuy": (("y", "x"), np.full((jdm, idm), 111_000.0)),
            "scvx": (("y", "x"), np.full((jdm, idm), 74_000.0)),
        }
    )
    depth = np.full((jdm, idm), 1000.0)
    depth[0, :] = np.nan  # southern edge: all land
    depth[1, :3] = np.nan  # partial land in first wet row
    bathy = xr.Dataset({"depth": (("y", "x"), depth)})
    return grid, bathy


def test_from_boundary_row_returns_transect(
    boundary_grid_bathy: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """from_boundary_row returns a Transect with at least 2 waypoints."""
    grid, bathy = boundary_grid_bathy
    t = Transect.from_boundary_row(grid, j=1, bathy=bathy)
    assert isinstance(t, Transect)
    assert len(t.lons) >= 2


def test_from_boundary_row_skips_land(
    boundary_grid_bathy: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """from_boundary_row only uses wet cells as waypoints."""
    grid, bathy = boundary_grid_bathy
    t = Transect.from_boundary_row(grid, j=1, bathy=bathy)
    # Columns 0-2 are land at j=1; wet starts at i=3
    assert t.lons[0] == pytest.approx(3.0)


def test_from_boundary_row_name_propagated(
    boundary_grid_bathy: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """from_boundary_row propagates the name argument."""
    grid, bathy = boundary_grid_bathy
    t = Transect.from_boundary_row(grid, j=1, bathy=bathy, name="my_boundary")
    assert t.name == "my_boundary"


def test_from_boundary_row_i_range(
    boundary_grid_bathy: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """from_boundary_row with i_range restricts columns."""
    grid, bathy = boundary_grid_bathy
    t = Transect.from_boundary_row(grid, j=4, bathy=bathy, i_range=(3, 6))
    assert t.lons[0] == pytest.approx(3.0)
    assert t.lons[-1] == pytest.approx(6.0)
    assert len(t.lons) == 4


def test_from_boundary_row_all_land_raises(
    boundary_grid_bathy: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """from_boundary_row raises ValueError when the row is entirely land."""
    grid, bathy = boundary_grid_bathy
    with pytest.raises(ValueError, match="Fewer than 2 wet cells"):
        Transect.from_boundary_row(grid, j=0, bathy=bathy)


def test_transect_reverse_flips_waypoints() -> None:
    """reverse() returns a new Transect with waypoints in opposite order."""
    t = Transect(lons=[0.0, 5.0, 10.0], lats=[45.0, 46.0, 47.0])
    r = t.reverse()
    np.testing.assert_array_equal(r.lons, [10.0, 5.0, 0.0])
    np.testing.assert_array_equal(r.lats, [47.0, 46.0, 45.0])


def test_transect_reverse_preserves_name() -> None:
    """reverse() keeps the original name attribute."""
    t = Transect(lons=[0.0, 10.0], lats=[45.0, 45.0], name="my_section")
    assert t.reverse().name == "my_section"


def test_transect_reverse_does_not_mutate_original() -> None:
    """reverse() returns a new object without modifying the original."""
    t = Transect(lons=[0.0, 10.0], lats=[45.0, 45.0])
    _ = t.reverse()
    assert t.lons[0] == 0.0


# ---------------------------------------------------------------------------
# Transect.resolve — requires scipy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_grid() -> xr.Dataset:
    """10×15 synthetic regular HYCOM grid, lon 0–14°E, lat 45–54°N."""
    jdm, idm = 10, 15
    j, i = np.meshgrid(np.arange(jdm), np.arange(idm), indexing="ij")
    return xr.Dataset(
        {
            "plon": (("y", "x"), i.astype(float)),
            "plat": (("y", "x"), (45.0 + j).astype(float)),
            "scuy": (("y", "x"), np.full((jdm, idm), 111_000.0)),
            "scvx": (("y", "x"), np.full((jdm, idm), 74_000.0)),
        }
    )


def test_resolve_returns_resolved_transect(simple_grid: xr.Dataset) -> None:
    """resolve() against a valid grid returns a ResolvedTransect."""
    pytest.importorskip("scipy")
    t = Transect(lons=[2.0, 8.0], lats=[49.0, 49.0])
    r = t.resolve(simple_grid)
    assert isinstance(r, ResolvedTransect)
    assert r.has_face_data


def test_resolve_ew_section_gives_v_faces(simple_grid: xr.Dataset) -> None:
    """An east–west section produces V-faces as the dominant cross-section faces.

    Transport through an east-west section is carried by north-south velocity
    on V-faces.  A small number of endpoint U-faces may appear at the section
    corners where the masked region meets the section endpoints (these carry
    near-zero u-velocity in real ocean grids where endpoints are at coastlines).
    """
    pytest.importorskip("scipy")
    r = Transect(lons=[2.0, 8.0], lats=[49.0, 49.0]).resolve(simple_grid)
    assert np.any(r.face_type == 1)
    assert np.sum(r.face_type == 1) > np.sum(r.face_type == 0)


def test_resolve_ns_section_gives_u_faces(simple_grid: xr.Dataset) -> None:
    """A north–south section produces only U-faces (face_type == 0).

    Transport through a north-south section is carried by east-west velocity
    on U-faces.
    """
    pytest.importorskip("scipy")
    r = Transect(lons=[5.0, 5.0], lats=[46.0, 52.0]).resolve(simple_grid)
    assert np.all(r.face_type == 0)


def test_resolve_eastward_section_sign_negative(simple_grid: xr.Dataset) -> None:
    """Walking east, rightward = south; V-face flag is -1 (south mask, north boundary).

    For an east-going section the right-hand side is south, so the hemisphere
    mask covers the south half.  The V-faces are on the NORTH boundary of that
    mask, where flagv = mask[j] - mask[j-1] = 0 - 1 = -1.  Positive v (northward)
    then contributes -1 * v < 0, meaning southward flow is positive — consistent
    with "rightward when walking east = south".  Endpoint U-faces (at the section
    corners) may have mixed signs and are excluded from this check.
    """
    pytest.importorskip("scipy")
    r = Transect(lons=[2.0, 8.0], lats=[49.0, 49.0]).resolve(simple_grid)
    v_signs = r.face_sign[r.face_type == 1]
    assert len(v_signs) > 0
    assert np.all(v_signs == -1.0)


def test_resolve_westward_section_sign_positive(simple_grid: xr.Dataset) -> None:
    """Walking west, rightward = north; V-face flag is +1 (north mask, south boundary).

    Endpoint U-faces at the section corners may have mixed signs; this check
    considers only the V-faces that carry the cross-section transport.
    """
    pytest.importorskip("scipy")
    r = Transect(lons=[8.0, 2.0], lats=[49.0, 49.0]).resolve(simple_grid)
    v_signs = r.face_sign[r.face_type == 1]
    assert len(v_signs) > 0
    assert np.all(v_signs == 1.0)


def test_resolve_distance_km_monotone(simple_grid: xr.Dataset) -> None:
    """distance_km along the resolved path is non-decreasing."""
    pytest.importorskip("scipy")
    r = Transect(lons=[2.0, 8.0], lats=[49.0, 49.0]).resolve(simple_grid)
    assert np.all(np.diff(r.distance_km) >= 0)


def test_resolve_face_dist_in_bounds(simple_grid: xr.Dataset) -> None:
    """face_dist_km values are non-negative and close to the section length.

    Face centres are the midpoints of two straddling T-cells, so they lie
    roughly half a grid spacing off the section line; their great-circle
    distance from the section start can slightly exceed the section's own
    T-cell-path length.  We allow a 20 % margin.
    """
    pytest.importorskip("scipy")
    r = Transect(lons=[2.0, 8.0], lats=[49.0, 49.0]).resolve(simple_grid)
    assert np.all(r.face_dist_km >= 0)
    assert np.all(r.face_dist_km <= r.distance_km[-1] * 1.2)


def test_resolve_too_few_cells_raises(simple_grid: xr.Dataset) -> None:
    """A transect spanning much less than one grid cell raises ValueError."""
    pytest.importorskip("scipy")
    # 0.001° at 49°N ≈ 70 m — well within a single ~74 km grid cell
    t = Transect(lons=[5.000, 5.001], lats=[49.0, 49.0])
    with pytest.raises(ValueError, match="fewer than 2"):
        t.resolve(simple_grid)


def test_resolve_missing_grid_var_raises(simple_grid: xr.Dataset) -> None:
    """Missing required grid variable raises ValueError."""
    pytest.importorskip("scipy")
    grid_no_scuy = simple_grid.drop_vars("scuy")
    t = Transect(lons=[2.0, 8.0], lats=[49.0, 49.0])
    with pytest.raises(ValueError, match="missing 'scuy'"):
        t.resolve(grid_no_scuy)


# ---------------------------------------------------------------------------
# ResolvedTransect properties
# ---------------------------------------------------------------------------


def test_resolved_has_face_data_false_without_faces() -> None:
    """has_face_data is False when face arrays are None."""
    t = Transect(lons=[0.0, 1.0], lats=[0.0, 0.0])
    r = ResolvedTransect(
        transect=t,
        j=np.array([0, 1], dtype=np.intp),
        i=np.array([0, 1], dtype=np.intp),
        cell_lon=np.array([0.0, 1.0]),
        cell_lat=np.array([0.0, 0.0]),
        distance_km=np.array([0.0, 100.0]),
        cell_width_km=np.array([50.0, 50.0]),
        bearing_deg=np.array([90.0, 90.0]),
    )
    assert not r.has_face_data
    assert r.n_faces == 0
    assert r.n_cells == 2


# ---------------------------------------------------------------------------
# Transect.__repr__
# ---------------------------------------------------------------------------


def test_transect_repr_with_name() -> None:
    """Repr includes the name and waypoint count."""
    t = Transect(lons=[0.0, 1.0], lats=[0.0, 0.0], name="my_section")
    r = repr(t)
    assert "my_section" in r
    assert "2" in r


def test_transect_repr_unnamed() -> None:
    """Repr for a nameless Transect contains 'unnamed'."""
    t = Transect(lons=[0.0, 1.0, 2.0], lats=[0.0, 0.0, 0.0])
    assert "unnamed" in repr(t)


# ---------------------------------------------------------------------------
# V-face signs from resolve()
# ---------------------------------------------------------------------------


def test_resolve_northward_section_sign_positive(simple_grid: xr.Dataset) -> None:
    """Walking north, rightward = east; U-face flag is +1 (east mask, west boundary).

    For a north-going section the right-hand side is east.  The mask covers the
    east half; the U-faces are on the WEST boundary of that mask, where
    flagu = mask[j,i] - mask[j,i-1] = 1 - 0 = +1.  Positive u (eastward) then
    contributes +1 * u > 0 — consistent with "rightward = east".
    """
    pytest.importorskip("scipy")
    r = Transect(lons=[5.0, 5.0], lats=[46.0, 52.0]).resolve(simple_grid)
    assert np.all(r.face_sign == 1.0)


def test_resolve_southward_section_sign_negative(simple_grid: xr.Dataset) -> None:
    """Walking south, rightward = west; U-face flag is -1 (west mask, east boundary)."""
    pytest.importorskip("scipy")
    r = Transect(lons=[5.0, 5.0], lats=[52.0, 46.0]).resolve(simple_grid)
    assert np.all(r.face_sign == -1.0)


def test_resolve_nfaces_nonzero(simple_grid: xr.Dataset) -> None:
    """A resolved transect spanning several grid cells has at least one face."""
    pytest.importorskip("scipy")
    r = Transect(lons=[2.0, 8.0], lats=[49.0, 49.0]).resolve(simple_grid)
    assert r.n_faces >= 1


# ---------------------------------------------------------------------------
# ResolvedTransect.plot — smoke tests
# ---------------------------------------------------------------------------


matplotlib.use("Agg")


def _make_bare_resolved() -> ResolvedTransect:
    """Minimal ResolvedTransect without face data for plot smoke tests."""
    t = Transect(lons=[0.0, 5.0], lats=[45.0, 45.0])
    return ResolvedTransect(
        transect=t,
        j=np.array([0, 1, 2], dtype=np.intp),
        i=np.array([0, 1, 2], dtype=np.intp),
        cell_lon=np.array([0.0, 2.5, 5.0]),
        cell_lat=np.full(3, 45.0),
        distance_km=np.array([0.0, 200.0, 400.0]),
        cell_width_km=np.array([100.0, 200.0, 100.0]),
        bearing_deg=np.full(3, 90.0),
    )


def test_resolved_plot_basic_smoke() -> None:
    """plot() without a grid runs without error."""
    import matplotlib.pyplot as plt

    _make_bare_resolved().plot()
    plt.close("all")


def test_resolved_plot_bathy_without_grid_raises() -> None:
    """plot(bathy=...) without grid raises ValueError."""
    with pytest.raises(ValueError, match="grid="):
        _make_bare_resolved().plot(bathy="some_path")


def test_resolved_plot_with_face_data_smoke(simple_grid: xr.Dataset) -> None:
    """plot() on a fully resolved transect (with face data) runs without error."""
    import matplotlib.pyplot as plt

    pytest.importorskip("scipy")
    r = Transect(lons=[2.0, 8.0], lats=[49.0, 49.0]).resolve(simple_grid)
    r.plot()
    plt.close("all")
