"""Reader tests against synthetic .ab fixtures (see conftest.py)."""
import numpy as np
import pytest

import xhycom
from xhycom._reader import detect_filetype


# ---------------------------------------------------------------------------
# detect_filetype
# ---------------------------------------------------------------------------
def test_detect_filetype(grid_file, bathy_file, archive_file):
    assert detect_filetype(grid_file[0]) == "grid"
    assert detect_filetype(bathy_file[0]) == "bathy"
    assert detect_filetype(archive_file) == "archv"


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def test_open_grid(grid_file):
    base, fields = grid_file
    ds = xhycom.open_dataset(base)
    for name, arr in fields.items():
        assert name in ds, name
        np.testing.assert_allclose(ds[name].values, arr, rtol=1e-5)
    assert ds["plon"].dims == ("y", "x")


# ---------------------------------------------------------------------------
# Bathymetry
# ---------------------------------------------------------------------------
def test_open_bathy_requires_grid(bathy_file):
    with pytest.raises(ValueError, match="grid="):
        xhycom.open_dataset(bathy_file[0])


def test_open_bathy(bathy_file, grid_file):
    base, depth = bathy_file
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_dataset(base, grid=grid)
    wet = depth < 1e30
    np.testing.assert_allclose(ds["depth"].values[wet], depth[wet], rtol=1e-5)
    assert np.isnan(ds["depth"].values[0, 0])         # land point -> NaN
    assert "lon" in ds.coords and "lat" in ds.coords


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
def test_open_archive_structure(archive_file, grid_file):
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_dataset(archive_file, grid=grid)
    assert set(ds["temp"].dims) == {"time", "k", "y", "x"}
    assert ds["srfhgt"].dims == ("time", "y", "x")
    np.testing.assert_array_equal(ds["k"].values, [1, 2, 3])
    np.testing.assert_allclose(ds["dens"].values, [28.0, 29.0, 30.0])
    # temp[k] == 10 - k
    np.testing.assert_allclose(
        ds["temp"].isel(time=0, y=0, x=0).values, [9.0, 8.0, 7.0], rtol=1e-5,
    )
    assert ds["time"].size == 1


def test_open_archive_variables_filter(archive_file, grid_file):
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_dataset(archive_file, grid=grid, variables=["temp", "thknss"])
    assert "temp" in ds and "thknss" in ds
    assert "salin" not in ds


def test_open_mfdataset_concats_time(archive_pair, grid_file):
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_mfdataset(archive_pair, grid=grid)
    assert ds["time"].size == 2
    assert ds["temp"].sizes["time"] == 2


def test_open_mfdataset_lazy_chunks(archive_pair, grid_file):
    pytest.importorskip("dask")
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_mfdataset(archive_pair, grid=grid, chunks={"time": 1})
    assert ds["temp"].chunks is not None
    np.testing.assert_allclose(
        ds["temp"].isel(time=0, y=0, x=0).values, [9.0, 8.0, 7.0], rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# postprocess threaded through the real read path
# ---------------------------------------------------------------------------
def test_postprocess_through_open_archive(archive_file, grid_file):
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_dataset(archive_file, grid=grid, postprocess=True)
    assert ds["srfhgt"].attrs["units"] == "m"
    np.testing.assert_allclose(ds["srfhgt"].isel(time=0, y=0, x=0).values, 0.5,
                               rtol=1e-4)
    assert ds["thknss"].attrs["units"] == "m"
    np.testing.assert_allclose(ds["thknss"].isel(time=0, y=0, x=0).values,
                               [10.0, 10.0, 10.0], rtol=1e-4)


def test_postprocess_through_open_grid(grid_file):
    base, _ = grid_file
    ds = xhycom.open_dataset(base, postprocess=True)
    assert "area" in ds
    np.testing.assert_allclose(ds["area"].values, 100.0 * 200.0, rtol=1e-5)


def test_postprocess_through_open_bathy(bathy_file, grid_file):
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_dataset(bathy_file[0], grid=grid, postprocess=True)
    assert "landmask" in ds
    assert ds["landmask"].values[0, 0] == 0          # land
    assert ds["landmask"].values[-1, -1] == 1        # ocean
