"""Tests for xhycom.postprocess (unit conversions + derived fields)."""
import numpy as np
import xarray as xr

import xhycom
from xhycom._postprocess import _G, _ONEM, postprocess
from xhycom._regrid import layer_centre_depth


def test_ssh_converted_to_metres():
    # srfhgt is geopotential g*eta; 1.5 m of SSH -> stored as g*1.5
    srfhgt = xr.DataArray(np.full((2, 3), _G * 1.5), dims=("y", "x"),
                          name="srfhgt", attrs={"units": "Pa"})
    out = postprocess(xr.Dataset({"srfhgt": srfhgt}))
    np.testing.assert_allclose(out["srfhgt"].values, 1.5)
    assert out["srfhgt"].attrs["units"] == "m"
    assert "converted from" in out["srfhgt"].attrs["comment"]


def test_thknss_converted_with_onem_not_g():
    thknss = xr.DataArray(np.full((1, 2, 2), 10.0 * _ONEM), dims=("k", "y", "x"),
                          name="thknss", attrs={"units": "Pa"})
    out = postprocess(xr.Dataset({"thknss": thknss}))
    np.testing.assert_allclose(out["thknss"].values, 10.0)
    assert out["thknss"].attrs["units"] == "m"


def test_area_added_from_scales():
    ds = xr.Dataset({
        "scpx": (("y", "x"), np.full((2, 2), 100.0)),
        "scpy": (("y", "x"), np.full((2, 2), 200.0)),
    })
    out = postprocess(ds)
    assert "area" in out
    np.testing.assert_allclose(out["area"].values, 20000.0)
    assert out["area"].attrs["units"] == "m2"


def test_landmask_from_depth():
    depth = xr.DataArray(
        np.array([[10.0, np.nan], [5.0, 3.0]]), dims=("y", "x"), name="depth",
        attrs={"units": "m"},
    )
    out = postprocess(xr.Dataset({"depth": depth}))
    assert "landmask" in out
    np.testing.assert_array_equal(out["landmask"].values, [[1, 0], [1, 1]])


def test_idempotent_does_not_double_convert():
    srfhgt = xr.DataArray(np.full((2, 2), _G), dims=("y", "x"), name="srfhgt",
                          attrs={"units": "Pa"})
    once = postprocess(xr.Dataset({"srfhgt": srfhgt}))
    twice = postprocess(once)
    np.testing.assert_allclose(twice["srfhgt"].values, once["srfhgt"].values)


def test_postprocess_exported():
    assert xhycom.postprocess is postprocess


def test_regrid_vertical_unit_aware_after_postprocess():
    # thknss in Pa vs converted to m must yield identical layer-centre depths
    thk_pa = xr.DataArray(np.full((4, 1, 1), 10.0 * _ONEM), dims=("k", "y", "x"),
                          name="thknss", attrs={"units": "Pa"})
    thk_m = postprocess(xr.Dataset({"thknss": thk_pa}))["thknss"]
    z_pa = layer_centre_depth(thk_pa).values
    z_m = layer_centre_depth(thk_m).values
    np.testing.assert_allclose(z_pa, z_m)
    np.testing.assert_allclose(z_pa[:, 0, 0], [5, 15, 25, 35], atol=1e-3)
