"""Reader tests against synthetic .ab fixtures (see conftest.py)."""

import numpy as np
import pytest

import xhycom
from xhycom._reader import detect_filetype, read_ave


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
    assert np.isnan(ds["depth"].values[0, 0])  # land point -> NaN
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
        ds["temp"].isel(time=0, y=0, x=0).values,
        [9.0, 8.0, 7.0],
        rtol=1e-5,
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
        ds["temp"].isel(time=0, y=0, x=0).values,
        [9.0, 8.0, 7.0],
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# postprocess threaded through the real read path
# ---------------------------------------------------------------------------
def test_postprocess_through_open_archive(archive_file, grid_file):
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_dataset(archive_file, grid=grid, postprocess=True)
    assert ds["srfhgt"].attrs["units"] == "m"
    np.testing.assert_allclose(
        ds["srfhgt"].isel(time=0, y=0, x=0).values, 0.5, rtol=1e-4
    )
    assert ds["thknss"].attrs["units"] == "m"
    np.testing.assert_allclose(
        ds["thknss"].isel(time=0, y=0, x=0).values, [10.0, 10.0, 10.0], rtol=1e-4
    )


def test_postprocess_through_open_grid(grid_file):
    base, _ = grid_file
    ds = xhycom.open_dataset(base, postprocess=True)
    assert "area" in ds
    np.testing.assert_allclose(ds["area"].values, 100.0 * 200.0, rtol=1e-5)


def test_postprocess_through_open_bathy(bathy_file, grid_file):
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_dataset(bathy_file[0], grid=grid, postprocess=True)
    assert "landmask" in ds
    assert ds["landmask"].values[0, 0] == 0  # land
    assert ds["landmask"].values[-1, -1] == 1  # ocean


def test_archive_type_instantaneous(archive_file):
    # The synthetic fixture writes a "... model day" header -> instantaneous archv.
    ds = xhycom.open_dataset(archive_file)
    assert ds.attrs["archive_type"] == "instantaneous"


# ---------------------------------------------------------------------------
# AVE (hycave/ensave monthly average)
# ---------------------------------------------------------------------------
def test_detect_filetype_ave(ave_file: str) -> None:
    """AVE header (has 'iversn' + 'kdm   ') is distinguished from archv."""
    assert detect_filetype(ave_file) == "ave"


def test_open_ave_structure(ave_file: str, grid_file: tuple) -> None:
    """2-D and 3-D AVE fields land on the expected dims; archive_type is set."""
    grid = xhycom.open_dataset(grid_file[0])
    ds = xhycom.open_dataset(ave_file, grid=grid)
    assert "ssh" in ds
    assert "temp" in ds
    assert ds["ssh"].dims == ("time", "y", "x")
    assert set(ds["temp"].dims) == {"time", "k", "y", "x"}
    np.testing.assert_array_equal(ds["k"].values, [1, 2, 3])
    assert ds["time"].size == 1
    assert ds.attrs["archive_type"] == "time_average"


def test_open_ave_time_from_filename(ave_file: str) -> None:
    """Time coordinate is parsed from _YYYY_MM suffix, not from model-day (always 0)."""
    import cftime

    ds = xhycom.open_dataset(ave_file)
    t = ds["time"].values[0]
    assert isinstance(t, cftime.datetime)
    assert t.year == 1991
    assert t.month == 1
    assert t.day == 1


def test_open_ave_variables_filter(ave_file: str) -> None:
    """variables= kwarg restricts which AVE fields are loaded."""
    ds = xhycom.open_dataset(ave_file, variables=["temp"])
    assert "temp" in ds
    assert "ssh" not in ds


def test_open_ave_no_time_without_pattern(tmp_path, grid_file: tuple) -> None:
    """Basenames without _YYYY_MM suffix produce a Dataset with no time dim."""
    from conftest import _write_ave

    base = str(tmp_path / "EXPAVE_nodate")
    _write_ave(base)
    ds = xhycom.open_dataset(base)
    assert "time" not in ds.dims


def test_read_ave_public_api(ave_file: str) -> None:
    """read_ave() is importable from xhycom._reader and returns a Dataset."""
    ds = read_ave(ave_file)
    assert "ssh" in ds and "temp" in ds


def test_archive_type_mean(tp0):
    # The bundled archm fixture writes a "... mean day" header -> mean archive,
    # whose u-vel./v-vel. are already the total current.
    ds = xhycom.open_dataset(tp0.archive, postprocess=True)
    assert ds.attrs["archive_type"] == "mean"
    assert ds["u-vel."].attrs["hycom_velocity"] == "total"


def test_open_ave_chunks(ave_file: str) -> None:
    """read_ave with chunks= loads lazily via Dask."""
    pytest.importorskip("dask")
    ds = read_ave(ave_file, chunks={"k": 1})
    assert ds["temp"].chunks is not None
    np.testing.assert_allclose(
        ds["temp"].isel(time=0, y=0, x=0).values,
        [9.0, 8.0, 7.0],
        rtol=1e-5,
    )
