"""End-to-end tests against the bundled real TP0 sample (see conftest.tp0).

Unlike the synthetic fixtures in ``conftest.py``, these run the readers and the
regridding pipeline against a real (subset) HYCOM archive — a coupled
physics/BGC run on a 100x110 curvilinear grid.  The archive uses a hybrid
vertical coordinate: the upper layers carry tiny target densities to force
fixed z-levels near the surface, transitioning to isopycnal layers below.

The ``.b`` header stores per-field min/max computed independently of the
``.a`` binary, so asserting the read array's min/max against the header is a
strong check that the binary record layout is decoded correctly.
"""
import re

import numpy as np
import pytest
import xarray as xr

import xhycom
from xhycom._reader import detect_filetype


# ---------------------------------------------------------------------------
# .b header parsing helpers (independent of the reader under test)
# ---------------------------------------------------------------------------
def _grid_minmax(basename, field):
    """(min, max) for a grid field, e.g. 'plon:  min,max = -179.5 179.9'."""
    with open(basename + ".b") as f:
        for line in f:
            m = re.match(rf"{re.escape(field)}:\s*min,max\s*=\s*(\S+)\s+(\S+)", line)
            if m:
                return float(m.group(1)), float(m.group(2))
    raise KeyError(field)


def _bathy_minmax(basename):
    with open(basename + ".b") as f:
        for line in f:
            m = re.match(r"min,max\s+depth\s*=\s*(\S+)\s+(\S+)", line)
            if m:
                return float(m.group(1)), float(m.group(2))
    raise KeyError("depth")


def _archive_minmax(basename, field, k):
    """(min, max) for an archive field at level k from its .b field line."""
    with open(basename + ".b") as f:
        for line in f:
            if line[:8].strip() == field and re.search(r"=", line):
                parts = re.split(r"[ =]+", line.strip())
                # field = step day k dens min max
                if int(parts[3]) == k:
                    return float(parts[-2]), float(parts[-1])
    raise KeyError((field, k))


# ---------------------------------------------------------------------------
# Reader / open_dataset
# ---------------------------------------------------------------------------
def test_detect_filetype(tp0):
    assert detect_filetype(tp0.grid) == "grid"
    assert detect_filetype(tp0.bathy) == "bathy"
    assert detect_filetype(tp0.archive) == "archv"


def test_open_grid(tp0):
    grid = xhycom.open_dataset(tp0.grid)
    # all 19 grid variables present on (y, x)
    for name in ("plon", "plat", "qlon", "qlat", "pang", "scpx", "scpy", "cori"):
        assert name in grid, name
        assert grid[name].dims == ("y", "x")
    assert grid["plon"].shape == (tp0.jdm, tp0.idm)
    # array min/max match the independently-stored header stats
    for field in ("plon", "plat", "pang"):
        lo, hi = _grid_minmax(tp0.grid, field)
        np.testing.assert_allclose(float(grid[field].min()), lo, atol=1e-4)
        np.testing.assert_allclose(float(grid[field].max()), hi, atol=1e-4)


def test_open_bathy_requires_grid(tp0):
    with pytest.raises(ValueError, match="grid="):
        xhycom.open_dataset(tp0.bathy)


def test_open_bathy(tp0):
    grid = xhycom.open_dataset(tp0.grid)
    bathy = xhycom.open_dataset(tp0.bathy, grid=grid)
    assert bathy["depth"].shape == (tp0.jdm, tp0.idm)
    assert "lon" in bathy.coords and "lat" in bathy.coords
    # land is masked to NaN; this run is ~58% land
    assert np.isnan(bathy["depth"].values).any()
    lo, hi = _bathy_minmax(tp0.bathy)
    np.testing.assert_allclose(np.nanmin(bathy["depth"].values), lo, atol=1e-3)
    np.testing.assert_allclose(np.nanmax(bathy["depth"].values), hi, atol=1e-3)


def test_open_archive_structure(tp0):
    grid = xhycom.open_dataset(tp0.grid)
    ds = xhycom.open_dataset(tp0.archive, grid=grid)
    assert set(ds.data_vars) == {"montg1", "srfhgt", "temp", "salin",
                                 "thknss", "u-vel.", "v-vel."}
    assert set(ds["temp"].dims) == {"time", "k", "y", "x"}
    assert ds["srfhgt"].dims == ("time", "y", "x")
    assert ds.sizes["k"] == tp0.nlayers
    assert ds.sizes["time"] == 1
    # lon/lat on T-point vars; u/v get their own staggered coords
    assert "lon" in ds["temp"].coords and "lat" in ds["temp"].coords
    assert "lon_u" in ds["u-vel."].coords
    assert "lon_v" in ds["v-vel."].coords


def test_open_archive_time_decoded(tp0):
    grid = xhycom.open_dataset(tp0.grid)
    ds = xhycom.open_dataset(tp0.archive, grid=grid)
    t = ds["time"].values[0]
    assert (t.year, t.month, t.day) == (2006, 7, 9)


def test_open_archive_hybrid_density_profile(tp0):
    """Upper layers are forced z-levels (tiny dens), lower layers isopycnal."""
    grid = xhycom.open_dataset(tp0.grid)
    ds = xhycom.open_dataset(tp0.archive, grid=grid)
    dens = ds["dens"].values
    assert dens.shape == (tp0.nlayers,)
    assert np.all(dens[:5] < 1.0)           # z-coordinate cap layers
    assert np.all(dens[5:] > 20.0)          # isopycnal interior
    assert np.all(np.diff(dens) > 0)        # monotonically increasing


def test_open_archive_values_match_header(tp0):
    grid = xhycom.open_dataset(tp0.grid)
    ds = xhycom.open_dataset(tp0.archive, grid=grid)
    for k in (1, 14, 28):
        lo, hi = _archive_minmax(tp0.archive, "temp", k)
        t = ds["temp"].isel(time=0, k=k - 1).values
        np.testing.assert_allclose(np.nanmin(t), lo, rtol=1e-5)
        np.testing.assert_allclose(np.nanmax(t), hi, rtol=1e-5)


def test_open_archive_variables_filter(tp0):
    grid = xhycom.open_dataset(tp0.grid)
    ds = xhycom.open_dataset(tp0.archive, grid=grid, variables=["temp", "thknss"])
    assert set(ds.data_vars) == {"temp", "thknss"}


def test_lazy_matches_eager(tp0):
    pytest.importorskip("dask")
    grid = xhycom.open_dataset(tp0.grid)
    eager = xhycom.open_dataset(tp0.archive, grid=grid)
    lazy = xhycom.open_dataset(tp0.archive, grid=grid, chunks={"k": 1})
    assert lazy["temp"].chunks is not None
    np.testing.assert_allclose(
        lazy["temp"].values, eager["temp"].values, equal_nan=True,
    )


def test_postprocess(tp0):
    grid = xhycom.open_dataset(tp0.grid)
    ds = xhycom.open_dataset(tp0.archive, grid=grid, postprocess=True)
    assert ds["srfhgt"].attrs["units"] == "m"
    assert ds["thknss"].attrs["units"] == "m"
    # SSH in metres is physically O(1) m, not the raw geopotential (~tens)
    assert np.nanmax(np.abs(ds["srfhgt"].values)) < 5.0
    # layer thicknesses are non-negative metres
    assert np.nanmin(ds["thknss"].values) >= 0.0


# ---------------------------------------------------------------------------
# Regridding (real data)
# ---------------------------------------------------------------------------
# A target window comfortably inside the model domain (Nordic/Arctic sector).
_TLON = np.linspace(-20.0, 20.0, 21)
_TLAT = np.linspace(60.0, 80.0, 21)


@pytest.fixture
def real_ds(tp0):
    grid = xhycom.open_dataset(tp0.grid)
    ds = xhycom.open_dataset(tp0.archive, grid=grid, postprocess=True)
    return ds, grid


def test_regrid_vertical(real_ds):
    pytest.importorskip("xgcm")
    ds, _ = real_ds
    depth = [0, 10, 50, 100, 500, 1000]
    out = xhycom.regrid_vertical(ds, depth=depth)
    assert "k" not in out.dims
    assert list(out["depth"].values) == depth
    assert out["depth"].attrs["positive"] == "down"
    assert "thknss" not in out
    # interpolated temps stay within the global source range (no overshoot)
    tmin, tmax = float(np.nanmin(ds["temp"])), float(np.nanmax(ds["temp"]))
    vals = out["temp"].values
    assert np.nanmin(vals) >= tmin - 1e-3
    assert np.nanmax(vals) <= tmax + 1e-3


def test_regrid_horizontal(real_ds):
    pytest.importorskip("xesmf")
    ds, grid = real_ds
    out = xhycom.regrid_horizontal(ds, lon=_TLON, lat=_TLAT, grid=grid)
    assert out["temp"].dims == ("time", "k", "lat", "lon")
    assert out.sizes["lon"] == _TLON.size and out.sizes["lat"] == _TLAT.size
    # surface temp recovered somewhere in the target window
    assert np.isfinite(out["temp"].isel(time=0, k=0).values).any()
    # velocities de-staggered & rotated to geographic axes
    assert out["u-vel."].attrs["standard_name"] == "eastward_sea_water_velocity"
    assert out["v-vel."].attrs["standard_name"] == "northward_sea_water_velocity"
    assert "lon_u" not in out.coords and "lon_v" not in out.coords


def test_regrid_end_to_end(real_ds):
    pytest.importorskip("xesmf")
    pytest.importorskip("xgcm")
    ds, grid = real_ds
    out = xhycom.regrid(
        ds, lon=_TLON, lat=_TLAT, depth=[0, 50, 200, 1000], grid=grid,
    )
    assert set(out["temp"].dims) == {"time", "depth", "lat", "lon"}
    assert np.isfinite(out["temp"].values).any()


# ---------------------------------------------------------------------------
# Regridding onto a real GLORYS target grid (conftest.glorys)
# ---------------------------------------------------------------------------
def test_regrid_to_glorys_target(real_ds, glorys):
    pytest.importorskip("xesmf")
    pytest.importorskip("xgcm")
    ds, grid = real_ds
    out = xhycom.regrid(ds, target=glorys, grid=grid)   # conservative default
    # lands exactly on the GLORYS lon/lat/depth grid
    assert set(out["temp"].dims) == {"time", "depth", "lat", "lon"}
    np.testing.assert_array_equal(out["lon"].values, glorys["longitude"].values)
    np.testing.assert_array_equal(out["lat"].values, glorys["latitude"].values)
    np.testing.assert_array_equal(out["depth"].values, glorys["depth"].values)
    # GLORYS land mask is applied: every land cell is NaN.
    land = glorys["mask"].values == 0
    temp = out["temp"].isel(time=0).transpose("depth", "lat", "lon").values
    assert np.all(np.isnan(temp[land]))


def test_regrid_to_glorys_no_overshoot(real_ds, glorys):
    pytest.importorskip("xesmf")
    pytest.importorskip("xgcm")
    ds, grid = real_ds
    out = xhycom.regrid(ds, target=glorys, grid=grid)
    tmin, tmax = float(np.nanmin(ds["temp"])), float(np.nanmax(ds["temp"]))
    vals = out["temp"].values
    # conservative (thickness-weighted) means cannot exceed the source range.
    assert np.nanmin(vals) >= tmin - 1e-3
    assert np.nanmax(vals) <= tmax + 1e-3


def test_horizontal_conservative_to_glorys_preserves_constant(real_ds, glorys):
    """Conservative remapping to GLORYS reproduces a constant field exactly."""
    pytest.importorskip("xesmf")
    ds, grid = real_ds
    const = ds.copy()
    const["temp"] = xr.full_like(ds["temp"], 4.0)
    out = xhycom.regrid_horizontal(const, target=glorys, grid=grid,
                                   method="conservative", apply_target_mask=False)
    v = out["temp"].isel(time=0, k=0).values
    finite = np.isfinite(v)
    assert finite.any()
    np.testing.assert_allclose(v[finite], 4.0, atol=1e-5)


def test_horizontal_conservative_to_glorys_conserves_area_integral(real_ds, glorys):
    """Surface area integral over the target window is conserved vs bilinear."""
    pytest.importorskip("xesmf")
    ds, grid = real_ds
    lat = glorys["latitude"].values
    lon = glorys["longitude"].values
    # area weights ~ cos(lat) * dlon * dlat on the regular target grid
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]

    def area_mean(method):
        o = xhycom.regrid_horizontal(ds, target=glorys, grid=grid,
                                     method=method, apply_target_mask=False)
        t = o["temp"].isel(time=0, k=0).values
        m = np.isfinite(t)
        return np.sum(t[m] * w[m]) / np.sum(w[m])

    cons = area_mean("conservative")
    bil = area_mean("bilinear")
    # both estimate the same surface-temperature integral over the window
    np.testing.assert_allclose(cons, bil, rtol=0.05)
