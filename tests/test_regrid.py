"""Tests for xhycom.regrid (vertical + lateral) using synthetic datasets."""
import numpy as np
import pytest
import xarray as xr

import xhycom
from xhycom._regrid import _ONEM, _uv_to_east_north, layer_centre_depth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _column_ds(thickness_m=10.0, nlayers=5, ny=3, nx=4):
    """Dataset with uniform-thickness layers and temp = 20 - 0.1 * z_centre."""
    thk_pa = np.full((nlayers, ny, nx), thickness_m * _ONEM)
    thknss = xr.DataArray(thk_pa, dims=("k", "y", "x"),
                          coords={"k": np.arange(1, nlayers + 1)})
    z_centre = layer_centre_depth(thknss).isel(y=0, x=0).values  # 1-D
    temp = 20.0 - 0.1 * z_centre
    temp = xr.DataArray(
        np.broadcast_to(temp[:, None, None], (nlayers, ny, nx)).copy(),
        dims=("k", "y", "x"), coords={"k": np.arange(1, nlayers + 1)},
    )
    return xr.Dataset({"thknss": thknss, "temp": temp}), z_centre


# ---------------------------------------------------------------------------
# layer_centre_depth
# ---------------------------------------------------------------------------
def test_layer_centre_depth():
    ds, _ = _column_ds(thickness_m=10.0, nlayers=5)
    z = layer_centre_depth(ds["thknss"]).isel(y=0, x=0).values
    # centres at 5, 15, 25, 35, 45 (+ negligible monotonic ramp)
    np.testing.assert_allclose(z, [5, 15, 25, 35, 45], atol=1e-3)


def test_layer_centre_depth_strictly_increasing_with_massless_layers():
    # bottom two layers massless (zero thickness) -> would duplicate depth
    thk = np.array([10.0, 10.0, 10.0, 0.0, 0.0]) * _ONEM
    thknss = xr.DataArray(thk[:, None, None], dims=("k", "y", "x"),
                          coords={"k": np.arange(1, 6)})
    z = layer_centre_depth(thknss).isel(y=0, x=0).values
    assert np.all(np.diff(z) > 0), "depths must be strictly increasing"


# ---------------------------------------------------------------------------
# regrid_vertical
# ---------------------------------------------------------------------------
def test_regrid_vertical_recovers_layer_centres():
    pytest.importorskip("xgcm")
    ds, z_centre = _column_ds()
    out = xhycom.regrid_vertical(ds, depth=z_centre)
    expected = 20.0 - 0.1 * z_centre
    got = out["temp"].isel(y=0, x=0).values
    np.testing.assert_allclose(got, expected, atol=1e-3)
    assert out["depth"].attrs["positive"] == "down"
    assert "k" not in out.dims


def test_regrid_vertical_linear_interpolation():
    pytest.importorskip("xgcm")
    ds, _ = _column_ds()  # centres 5,15,25,35,45 ; temp linear in z
    out = xhycom.regrid_vertical(ds, depth=[10.0, 20.0, 30.0])
    # linear field -> interpolated values lie exactly on the line
    np.testing.assert_allclose(
        out["temp"].isel(y=0, x=0).values, [19.0, 18.0, 17.0], atol=1e-2,
    )


def test_regrid_vertical_masks_below_bottom():
    pytest.importorskip("xgcm")
    ds, _ = _column_ds()  # bottom centre ~45 m
    out = xhycom.regrid_vertical(ds, depth=[100.0], mask_edges=True)
    assert np.isnan(out["temp"].isel(y=0, x=0).values).all()


def test_regrid_vertical_drops_thknss_and_keeps_2d():
    pytest.importorskip("xgcm")
    ds, _ = _column_ds()
    ds["srfhgt"] = xr.DataArray(np.ones((3, 4)), dims=("y", "x"))
    out = xhycom.regrid_vertical(ds, depth=[5, 15])
    assert "thknss" not in out
    assert "srfhgt" in out and "depth" not in out["srfhgt"].dims


# ---------------------------------------------------------------------------
# velocity de-stagger + rotation
# ---------------------------------------------------------------------------
def _uv_ds(u_val, v_val, ny=4, nx=5):
    u = xr.DataArray(np.full((ny, nx), u_val), dims=("y", "x"), name="u-vel.")
    v = xr.DataArray(np.full((ny, nx), v_val), dims=("y", "x"), name="v-vel.")
    return xr.Dataset({"u-vel.": u, "v-vel.": v})


def test_rotation_identity_when_pang_zero():
    ds = _uv_ds(2.0, -3.0)
    pang = xr.DataArray(np.zeros((4, 5)), dims=("y", "x"))
    out = _uv_to_east_north(ds, pang)
    # interior point (avoid de-stagger boundary NaNs at last col/row)
    assert out["u-vel."].isel(y=1, x=1).item() == pytest.approx(2.0)
    assert out["v-vel."].isel(y=1, x=1).item() == pytest.approx(-3.0)


def test_rotation_quarter_turn():
    ds = _uv_ds(1.0, 0.0)
    pang = xr.DataArray(np.full((4, 5), np.pi / 2), dims=("y", "x"))
    out = _uv_to_east_north(ds, pang)
    # east = u*cos - v*sin = 0 ; north = u*sin + v*cos = 1
    assert out["u-vel."].isel(y=1, x=1).item() == pytest.approx(0.0, abs=1e-12)
    assert out["v-vel."].isel(y=1, x=1).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# horizontal regrid (xESMF) + full wrapper
# ---------------------------------------------------------------------------
def _curvilinear_ds(nlayers=3):
    """A regular grid expressed as a 2-D curvilinear grid; temp = lat."""
    lon1d = np.linspace(0, 10, 12)
    lat1d = np.linspace(40, 50, 11)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    ny, nx = lat2d.shape
    thknss = xr.DataArray(
        np.full((nlayers, ny, nx), 10.0 * _ONEM), dims=("k", "y", "x"),
        coords={"k": np.arange(1, nlayers + 1)},
    )
    temp = xr.DataArray(
        np.broadcast_to(lat2d, (nlayers, ny, nx)).copy(), dims=("k", "y", "x"),
        coords={"k": np.arange(1, nlayers + 1)},
    )
    ds = xr.Dataset({"thknss": thknss, "temp": temp})
    ds = ds.assign_coords(
        lon=(("y", "x"), lon2d), lat=(("y", "x"), lat2d),
    )
    return ds


def test_regrid_horizontal_recovers_field():
    pytest.importorskip("xesmf")
    ds = _curvilinear_ds()
    tgt_lon = np.linspace(2, 8, 7)
    tgt_lat = np.linspace(42, 48, 7)
    out = xhycom.regrid_horizontal(ds, lon=tgt_lon, lat=tgt_lat)
    # temp == lat everywhere; bilinear must recover lat at interior points
    expected = np.broadcast_to(tgt_lat[:, None], (7, 7))
    # ESMF bilinear uses great-circle weighting, so recovery is ~1e-3, not exact
    np.testing.assert_allclose(
        out["temp"].isel(k=0).values, expected, atol=2e-3,
    )


def test_regrid_wrapper_end_to_end():
    pytest.importorskip("xesmf")
    ds = _curvilinear_ds()
    out = xhycom.regrid(
        ds, lon=np.linspace(2, 8, 5), lat=np.linspace(42, 48, 5),
        depth=[5.0, 15.0, 25.0],
    )
    assert set(out["temp"].dims) == {"depth", "lat", "lon"}
    # temp independent of depth (== lat) within the column range
    col = out["temp"].isel(lat=2, lon=2).values
    assert np.allclose(col, col[0], atol=1e-3)
