"""Tests for xhycom.postprocess (unit conversions + derived fields)."""

import numpy as np
import pytest
import xarray as xr

import xhycom
from xhycom._postprocess import _G, _ONEM, _reconcile_velocities, postprocess
from xhycom._regrid import layer_centre_depth


def test_ssh_converted_to_metres():
    # srfhgt is geopotential g*eta; 1.5 m of SSH -> stored as g*1.5
    srfhgt = xr.DataArray(
        np.full((2, 3), _G * 1.5), dims=("y", "x"), name="srfhgt", attrs={"units": "Pa"}
    )
    out = postprocess(xr.Dataset({"srfhgt": srfhgt}))
    np.testing.assert_allclose(out["srfhgt"].values, 1.5)
    assert out["srfhgt"].attrs["units"] == "m"
    assert "converted from" in out["srfhgt"].attrs["comment"]


def test_thknss_converted_with_onem_not_g():
    thknss = xr.DataArray(
        np.full((1, 2, 2), 10.0 * _ONEM),
        dims=("k", "y", "x"),
        name="thknss",
        attrs={"units": "Pa"},
    )
    out = postprocess(xr.Dataset({"thknss": thknss}))
    np.testing.assert_allclose(out["thknss"].values, 10.0)
    assert out["thknss"].attrs["units"] == "m"


def test_area_added_from_scales():
    ds = xr.Dataset(
        {
            "scpx": (("y", "x"), np.full((2, 2), 100.0)),
            "scpy": (("y", "x"), np.full((2, 2), 200.0)),
        }
    )
    out = postprocess(ds)
    assert "area" in out
    np.testing.assert_allclose(out["area"].values, 20000.0)
    assert out["area"].attrs["units"] == "m2"


def test_landmask_from_depth():
    depth = xr.DataArray(
        np.array([[10.0, np.nan], [5.0, 3.0]]),
        dims=("y", "x"),
        name="depth",
        attrs={"units": "m"},
    )
    out = postprocess(xr.Dataset({"depth": depth}))
    assert "landmask" in out
    np.testing.assert_array_equal(out["landmask"].values, [[1, 0], [1, 1]])


def test_idempotent_does_not_double_convert():
    srfhgt = xr.DataArray(
        np.full((2, 2), _G), dims=("y", "x"), name="srfhgt", attrs={"units": "Pa"}
    )
    once = postprocess(xr.Dataset({"srfhgt": srfhgt}))
    twice = postprocess(once)
    np.testing.assert_allclose(twice["srfhgt"].values, once["srfhgt"].values)


def test_postprocess_exported():
    assert xhycom.postprocess is postprocess


def test_regrid_vertical_unit_aware_after_postprocess():
    # thknss in Pa vs converted to m must yield identical layer-centre depths
    thk_pa = xr.DataArray(
        np.full((4, 1, 1), 10.0 * _ONEM),
        dims=("k", "y", "x"),
        name="thknss",
        attrs={"units": "Pa"},
    )
    thk_m = postprocess(xr.Dataset({"thknss": thk_pa}))["thknss"]
    z_pa = layer_centre_depth(thk_pa).values
    z_m = layer_centre_depth(thk_m).values
    np.testing.assert_allclose(z_pa, z_m)
    np.testing.assert_allclose(z_pa[:, 0, 0], [5, 15, 25, 35], atol=1e-3)


# ---------------------------------------------------------------------------
# Velocity convention: archv stores baroclinic, archm stores total
# ---------------------------------------------------------------------------
def _vel_ds(archive_type, with_barotropic=True):
    data = {
        "u-vel.": xr.DataArray(np.full((2, 2, 2), 0.3), dims=("k", "y", "x")),
        "v-vel.": xr.DataArray(np.full((2, 2, 2), -0.2), dims=("k", "y", "x")),
    }
    if with_barotropic:
        data["u_btrop"] = xr.DataArray(np.full((2, 2), 0.1), dims=("y", "x"))
        data["v_btrop"] = xr.DataArray(np.full((2, 2), 0.05), dims=("y", "x"))
    return xr.Dataset(data, attrs={"archive_type": archive_type})


def test_velocity_total_for_instantaneous_archive():
    out = postprocess(_vel_ds("instantaneous"))
    np.testing.assert_allclose(
        out["u-vel."].values, 0.4
    )  # 0.3 baroclinic + 0.1 barotropic
    np.testing.assert_allclose(out["v-vel."].values, -0.15)
    assert out["u-vel."].attrs["hycom_velocity"] == "total"
    assert "barotropic" in out["u-vel."].attrs["comment"]


def test_velocity_total_for_mean_archive():
    out = postprocess(_vel_ds("mean"))
    np.testing.assert_allclose(out["u-vel."].values, 0.3)  # already total, unchanged
    assert out["u-vel."].attrs["hycom_velocity"] == "total"


def test_velocity_left_baroclinic_without_barotropic():
    with pytest.warns(UserWarning, match="baroclinic"):
        out = postprocess(_vel_ds("instantaneous", with_barotropic=False))
    np.testing.assert_allclose(out["u-vel."].values, 0.3)  # unchanged
    assert out["u-vel."].attrs["hycom_velocity"] == "baroclinic"


def test_velocity_reconcile_idempotent():
    out = postprocess(postprocess(_vel_ds("instantaneous")))
    np.testing.assert_allclose(out["u-vel."].values, 0.4)  # not 0.5


def test_velocity_untouched_without_archive_type():
    ds = xr.Dataset({"u-vel.": xr.DataArray(np.full((2, 2), 0.3), dims=("y", "x"))})
    out = _reconcile_velocities(ds)
    assert "hycom_velocity" not in out["u-vel."].attrs


def test_augment_velocity_vars():
    from xhycom import _augment_velocity_vars

    aug, auto = _augment_velocity_vars(["temp", "u-vel.", "v-vel."], True)
    assert set(auto) == {"u_btrop", "v_btrop"}
    assert {"temp", "u-vel.", "v-vel.", "u_btrop", "v_btrop"} == set(aug)
    assert _augment_velocity_vars(["u-vel."], False) == (["u-vel."], [])
    assert _augment_velocity_vars(None, True) == (None, [])
    assert _augment_velocity_vars(["temp"], True) == (["temp"], [])
    aug2, auto2 = _augment_velocity_vars(["u-vel.", "u_btrop"], True)
    assert auto2 == ["v_btrop"]
