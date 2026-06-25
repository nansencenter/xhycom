"""Tests for xhycom.regrid (vertical + lateral) using synthetic datasets."""
import numpy as np
import pytest
import xarray as xr

import xhycom
from xhycom._regrid import (
    _ONEM, _uv_to_east_north, layer_centre_depth, layer_interface_depth,
)


def _explode_on_compute(shape, chunks):
    """A dask-backed array that raises if any block is ever computed.

    Building the graph (cumsum, pad, transform, ...) is fine; only an
    accidental eager materialization — a stray ``.values`` / ``.compute()`` —
    trips the guard.  Lets a test assert that an operation stays lazy without
    needing large data: a regridded HYCOM year is tens of GB, so eagerly
    loading every time step at construction time (the bug this guards against)
    crashes the kernel.
    """
    dka = pytest.importorskip("dask.array")

    def _boom(_block):
        raise AssertionError("array was computed — expected a lazy graph")

    base = dka.zeros(shape, chunks=chunks)
    return base.map_blocks(_boom, dtype=base.dtype,
                           meta=np.empty((0,), base.dtype))


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
# Laziness: construction must not materialize the data (kernel-OOM guard)
# ---------------------------------------------------------------------------
def test_layer_interface_depth_stays_lazy():
    """A dask-backed thknss yields a dask-backed result — never a numpy load.

    Regression guard: the conservative path once forced ``thknss.values`` here,
    which pulls every time step into RAM at once.
    """
    pytest.importorskip("dask")
    thk = xr.DataArray(
        _explode_on_compute((6, 5, 4, 4), (1, 5, 4, 4)),
        dims=("time", "k", "y", "x"), attrs={"units": "m"},
    )
    iface = layer_interface_depth(thk)              # must not raise (no compute)
    assert iface.chunks is not None                 # still lazy
    assert iface.sizes["z_i"] == 6                  # n_layers + 1


def test_regrid_vertical_conservative_builds_lazily():
    """Constructing a conservative vertical regrid triggers no eager compute."""
    pytest.importorskip("xgcm")
    pytest.importorskip("dask")
    ds = xr.Dataset({
        "thknss": xr.DataArray(
            _explode_on_compute((6, 5, 4, 4), (1, 5, 4, 4)),
            dims=("time", "k", "y", "x"), attrs={"units": "m"}),
        "temp": xr.DataArray(
            _explode_on_compute((6, 5, 4, 4), (1, 5, 4, 4)),
            dims=("time", "k", "y", "x")),
    })
    # Building the graph must not compute anything; the result stays lazy.
    out = xhycom.regrid_vertical(ds, depth=[5.0, 15.0, 25.0],
                                 method="conservative")
    assert out["temp"].chunks is not None


# ---------------------------------------------------------------------------
# regrid_vertical
# ---------------------------------------------------------------------------
def test_regrid_vertical_recovers_layer_centres():
    pytest.importorskip("xgcm")
    ds, z_centre = _column_ds()
    out = xhycom.regrid_vertical(ds, depth=z_centre, method="linear")
    expected = 20.0 - 0.1 * z_centre
    got = out["temp"].isel(y=0, x=0).values
    np.testing.assert_allclose(got, expected, atol=1e-3)
    assert out["depth"].attrs["positive"] == "down"
    assert "k" not in out.dims


def test_regrid_vertical_linear_interpolation():
    pytest.importorskip("xgcm")
    ds, _ = _column_ds()  # centres 5,15,25,35,45 ; temp linear in z
    out = xhycom.regrid_vertical(ds, depth=[10.0, 20.0, 30.0], method="linear")
    # linear field -> interpolated values lie exactly on the line
    np.testing.assert_allclose(
        out["temp"].isel(y=0, x=0).values, [19.0, 18.0, 17.0], atol=1e-2,
    )


def test_regrid_vertical_masks_below_bottom():
    pytest.importorskip("xgcm")
    ds, _ = _column_ds()  # bottom centre ~45 m
    out = xhycom.regrid_vertical(ds, depth=[100.0], method="linear",
                                 mask_edges=True)
    assert np.isnan(out["temp"].isel(y=0, x=0).values).all()


def test_regrid_vertical_drops_thknss_and_keeps_2d():
    pytest.importorskip("xgcm")
    ds, _ = _column_ds()
    ds["srfhgt"] = xr.DataArray(np.ones((3, 4)), dims=("y", "x"))
    out = xhycom.regrid_vertical(ds, depth=[5, 15], method="linear")
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
    out = xhycom.regrid_horizontal(ds, lon=tgt_lon, lat=tgt_lat,
                                   method="bilinear")
    # temp == lat everywhere; bilinear must recover lat at interior points
    expected = np.broadcast_to(tgt_lat[:, None], (7, 7))
    # ESMF bilinear uses great-circle weighting, so recovery is ~1e-3, not exact
    np.testing.assert_allclose(
        out["temp"].isel(k=0).values, expected, atol=2e-3,
    )


def test_regrid_horizontal_builds_mask_from_one_timestep():
    """The source ocean mask must come from a single time step.

    Guards two bugs the single-step fixtures above can't see, both on a
    multi-step source:

    1. *Performance* — reducing the mask over ``time`` (``isfinite(temp).any``)
       forces xESMF to read **every** step when it materializes the mask at
       construction (tens of GB for a HYCOM year), which is what made the weight
       cache look useless.
    2. *Correctness* — slicing ``time=0`` without dropping the scalar coord
       makes ``assign_coords(mask=...)`` raise "dimension 'time' already exists
       as a scalar variable".

    Here only ``time=0`` is real; later steps explode if ever computed, so a
    mask that touches them fails loudly.
    """
    pytest.importorskip("xesmf")
    dka = pytest.importorskip("dask.array")
    base = _curvilinear_ds(nlayers=2)
    nk, ny, nx = (base.sizes[d] for d in ("k", "y", "x"))
    good = dka.from_array(base["temp"].values[None], chunks=(1, nk, ny, nx))
    boom = _explode_on_compute((3, nk, ny, nx), (1, nk, ny, nx))
    temp = dka.concatenate([good, boom], axis=0)          # time=4, only t=0 safe
    ds = base.assign(temp=(("time", "k", "y", "x"), temp))

    out = xhycom.regrid_horizontal(
        ds, lon=np.linspace(2, 8, 7), lat=np.linspace(42, 48, 7),
        method="bilinear",
    )
    # Built without the scalar-time crash; the mask read only t=0 (no boom).
    assert "time" in out["temp"].dims
    assert np.isfinite(out["temp"].isel(time=0, k=0).values).any()


def test_subset_target_trims_to_source_bbox():
    from xhycom._regrid import _subset_target
    ds = _curvilinear_ds()                              # lon 0..10, lat 40..50
    tlon = np.linspace(-180, 179, 720)
    tlat = np.linspace(-80, 89, 680)
    tgt = xr.Dataset(
        {"mask": (("latitude", "longitude"), np.ones((tlat.size, tlon.size), "i1"))},
        coords={"longitude": tlon, "latitude": tlat},
    )
    sub = _subset_target(tgt, ds, pad=1.0)
    assert sub.sizes["longitude"] < tlon.size and sub.sizes["latitude"] < tlat.size
    # Kept points reach the source edges (within the pad) and exclude the far field.
    assert sub.longitude.min() < 1 and sub.longitude.max() > 9
    assert sub.latitude.min() < 41 and sub.latitude.max() > 49
    assert sub.longitude.max() < 20 and sub.latitude.max() < 60


def test_weights_cache_reused_and_consistent(tmp_path):
    pytest.importorskip("xesmf")
    ds = _curvilinear_ds()
    lon = np.linspace(2, 8, 7)
    lat = np.linspace(42, 48, 7)
    wfile = tmp_path / "w.nc"
    first = xhycom.regrid_horizontal(ds, lon=lon, lat=lat, method="bilinear",
                                     weights=wfile)
    assert wfile.exists()
    second = xhycom.regrid_horizontal(ds, lon=lon, lat=lat, method="bilinear",
                                      weights=wfile)              # reuse path
    np.testing.assert_array_equal(first["temp"].values, second["temp"].values)


def test_weights_cache_auto_keyed_by_geometry(tmp_path, monkeypatch):
    pytest.importorskip("xesmf")
    monkeypatch.setenv("XHYCOM_CACHE_DIR", str(tmp_path))
    ds = _curvilinear_ds()
    lon = np.linspace(2, 8, 7)
    lat = np.linspace(42, 48, 7)
    xhycom.regrid_horizontal(ds, lon=lon, lat=lat, method="bilinear", weights=True)
    files = [p.name for p in tmp_path.glob("weights_*.nc")]
    assert len(files) == 1 and files[0].startswith("weights_12x11_to_7x7_bilinear_")
    assert (tmp_path / "manifest.json").exists()


def test_nan_pole_blanks_lat_90_row_by_default():
    pytest.importorskip("xesmf")
    ds = _curvilinear_ds()
    lat = np.array([42.0, 60.0, 90.0])          # includes the exact pole row
    lon = np.linspace(2, 8, 7)
    # Default (nan_pole=True): the grid is unchanged but the pole row is NaN.
    out = xhycom.regrid_horizontal(ds, lon=lon, lat=lat, method="bilinear")
    np.testing.assert_array_equal(out["lat"].values, lat)   # row kept
    assert bool(np.isnan(out["temp"].sel(lat=90.0)).all())  # but blanked
    assert not bool(np.isnan(out["temp"].sel(lat=60.0)).all())

    # nan_pole=False keeps the raw remapped pole value.
    raw = xhycom.regrid_horizontal(ds, lon=lon, lat=lat, method="bilinear",
                                   nan_pole=False)
    np.testing.assert_array_equal(raw["lat"].values, lat)
    assert bool(np.isfinite(raw["temp"].sel(lat=90.0)).any())


def test_regrid_wrapper_end_to_end():
    pytest.importorskip("xesmf")
    ds = _curvilinear_ds()
    out = xhycom.regrid(
        ds, lon=np.linspace(2, 8, 5), lat=np.linspace(42, 48, 5),
        depth=[5.0, 15.0, 25.0], method="bilinear", z_method="linear",
    )
    assert set(out["temp"].dims) == {"depth", "lat", "lon"}
    # temp independent of depth (== lat) within the column range
    col = out["temp"].isel(lat=2, lon=2).values
    assert np.allclose(col, col[0], atol=1e-3)


# ---------------------------------------------------------------------------
# Conservative regridding (the default) — conservation properties
# ---------------------------------------------------------------------------
def test_regrid_vertical_conservative_preserves_column_integral():
    pytest.importorskip("xgcm")
    ds, z_centre = _column_ds()                       # 5 layers x 10 m
    h = 10.0
    src_integral = float((ds["temp"].isel(y=0, x=0).values * h).sum())
    # coarser target bins centred at 12.5 and 37.5 -> edges [0, 25, 50]
    out = xhycom.regrid_vertical(ds, depth=[12.5, 37.5], method="conservative",
                                 mask_edges=False)
    prof = out["temp"].isel(y=0, x=0).values
    tgt_integral = float((prof * 25.0).sum())
    np.testing.assert_allclose(tgt_integral, src_integral, rtol=1e-4)


def test_regrid_vertical_conservative_constant():
    pytest.importorskip("xgcm")
    ds, _ = _column_ds()
    ds["temp"] = xr.full_like(ds["temp"], 7.0)
    out = xhycom.regrid_vertical(ds, depth=[10.0, 30.0], method="conservative",
                                 mask_edges=False)
    np.testing.assert_allclose(out["temp"].isel(y=0, x=0).values, 7.0, atol=1e-6)


def _curvilinear_grid(dx=0.5, nx=20, ny=20, lon0=0.0, lat0=40.0):
    """Regular grid as 2-D centres (plon/plat) + SW corners (qlon/qlat)."""
    lonc = lon0 + (np.arange(nx) + 0.5) * dx
    latc = lat0 + (np.arange(ny) + 0.5) * dx
    lon2d, lat2d = np.meshgrid(lonc, latc)
    lonq = lon0 + np.arange(nx) * dx
    latq = lat0 + np.arange(ny) * dx
    qlon2d, qlat2d = np.meshgrid(lonq, latq)
    grid = xr.Dataset({"qlon": (("y", "x"), qlon2d),
                       "qlat": (("y", "x"), qlat2d),
                       "plon": (("y", "x"), lon2d),
                       "plat": (("y", "x"), lat2d)})
    return grid, lon2d, lat2d


def test_regrid_horizontal_conservative_preserves_constant():
    pytest.importorskip("xesmf")
    grid, lon2d, lat2d = _curvilinear_grid()
    ny, nx = lat2d.shape
    temp = xr.DataArray(np.full((1, ny, nx), 5.0), dims=("k", "y", "x"),
                        coords={"k": [1]})
    ds = xr.Dataset({"temp": temp}).assign_coords(
        lon=(("y", "x"), lon2d), lat=(("y", "x"), lat2d))
    out = xhycom.regrid_horizontal(
        ds, lon=np.linspace(2, 8, 9), lat=np.linspace(42, 48, 9),
        grid=grid, method="conservative",
    )
    v = out["temp"].isel(k=0).values
    np.testing.assert_allclose(v[np.isfinite(v)], 5.0, atol=1e-6)


def test_regrid_horizontal_conservative_is_thickness_weighted():
    """A constant tracer is preserved even where layer thickness varies."""
    pytest.importorskip("xesmf")
    grid, lon2d, lat2d = _curvilinear_grid()
    ny, nx = lat2d.shape
    # thickness ramps across the domain; tracer is uniform.
    h = np.broadcast_to((10.0 + (lat2d - 40.0)) * _ONEM, (1, ny, nx)).copy()
    thknss = xr.DataArray(h, dims=("k", "y", "x"), coords={"k": [1]})
    temp = xr.DataArray(np.full((1, ny, nx), 3.0), dims=("k", "y", "x"),
                        coords={"k": [1]})
    ds = xr.Dataset({"temp": temp, "thknss": thknss}).assign_coords(
        lon=(("y", "x"), lon2d), lat=(("y", "x"), lat2d))
    out = xhycom.regrid_horizontal(
        ds, lon=np.linspace(2, 8, 9), lat=np.linspace(42, 48, 9),
        grid=grid, method="conservative",
    )
    v = out["temp"].isel(k=0).values
    np.testing.assert_allclose(v[np.isfinite(v)], 3.0, atol=1e-6)


def test_regrid_horizontal_accepts_depth_levels():
    """On fixed depth levels (no thknss) a constant field is preserved, and a
    NaN below-bottom patch does not contaminate a wet neighbour."""
    pytest.importorskip("xesmf")
    grid, lon2d, lat2d = _curvilinear_grid()
    ny, nx = lat2d.shape
    temp = np.full((2, ny, nx), 4.0)
    temp[1, : ny // 2, :] = np.nan          # deep level dry over half the domain
    ds = xr.Dataset(
        {"temp": (("depth", "y", "x"), temp)},
    ).assign_coords(lon=(("y", "x"), lon2d), lat=(("y", "x"), lat2d),
                    depth=("depth", [5.0, 50.0]))
    out = xhycom.regrid_horizontal(
        ds, lon=np.linspace(2, 8, 9), lat=np.linspace(42, 48, 9),
        grid=grid, method="conservative",
    )
    assert set(out["temp"].dims) == {"depth", "lat", "lon"}
    v = out["temp"].values
    np.testing.assert_allclose(v[np.isfinite(v)], 4.0, atol=1e-6)
    # the wet half of the deep level still carries the constant (not NaN'd).
    assert np.isfinite(out["temp"].isel(depth=1).sel(lat=47, method="nearest")).any()


@pytest.mark.parametrize("order", ["horizontal_first", "vertical_first"])
def test_regrid_wrapper_conservative_end_to_end(order):
    """Either order, both steps conservative, preserves a constant field."""
    pytest.importorskip("xesmf")
    pytest.importorskip("xgcm")
    grid, lon2d, lat2d = _curvilinear_grid()
    ny, nx = lat2d.shape
    nlayers = 4
    thknss = xr.DataArray(
        np.full((nlayers, ny, nx), 10.0 * _ONEM), dims=("k", "y", "x"),
        coords={"k": np.arange(1, nlayers + 1)})
    temp = xr.DataArray(
        np.full((nlayers, ny, nx), 6.0), dims=("k", "y", "x"),
        coords={"k": np.arange(1, nlayers + 1)})
    ds = xr.Dataset({"thknss": thknss, "temp": temp}).assign_coords(
        lon=(("y", "x"), lon2d), lat=(("y", "x"), lat2d))
    out = xhycom.regrid(
        ds, lon=np.linspace(2, 8, 9), lat=np.linspace(42, 48, 9),
        depth=[5.0, 15.0, 25.0], grid=grid, order=order,
    )
    assert set(out["temp"].dims) == {"depth", "lat", "lon"}
    v = out["temp"].values
    np.testing.assert_allclose(v[np.isfinite(v)], 6.0, atol=1e-6)


def test_regrid_invalid_order():
    ds = _curvilinear_ds()
    with pytest.raises(ValueError, match="order"):
        xhycom.regrid(ds, lon=[3.0], lat=[45.0], depth=[5.0], order="sideways")


def test_regrid_horizontal_conservative_requires_grid():
    pytest.importorskip("xesmf")
    ds = _curvilinear_ds()
    with pytest.raises(ValueError, match="qlon"):
        xhycom.regrid_horizontal(ds, lon=np.linspace(2, 8, 5),
                                 lat=np.linspace(42, 48, 5),
                                 method="conservative")


# ---------------------------------------------------------------------------
# regrid_to_hycom: regular lon/lat product -> HYCOM curvilinear grid
# ---------------------------------------------------------------------------
def _regular_product(value):
    """A regular lon/lat product (GLORYS-style names) spanning the grid below."""
    lon = np.linspace(-2, 12, 30)
    lat = np.linspace(38, 52, 30)
    field = value(lat, lon) if callable(value) else np.full((30, 30), value)
    temp = xr.DataArray(field, dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})
    return xr.Dataset({"temp": temp})


def test_regrid_to_hycom_recovers_field():
    pytest.importorskip("xesmf")
    grid, lon2d, lat2d = _curvilinear_grid()           # centres inside [38, 52]
    product = _regular_product(lambda lat, lon: np.broadcast_to(lat[:, None], (30, 30)))
    out = xhycom.regrid_to_hycom(product, grid=grid, method="bilinear")
    # temp == latitude; bilinear must recover the HYCOM cell latitudes.
    assert set(out["temp"].dims) == {"y", "x"}
    np.testing.assert_allclose(out["temp"].values, lat2d, atol=1e-3)


def test_regrid_to_hycom_conservative_preserves_constant():
    pytest.importorskip("xesmf")
    grid, _, _ = _curvilinear_grid()
    out = xhycom.regrid_to_hycom(_regular_product(7.0), grid=grid,
                                 method="conservative")
    v = out["temp"].values
    np.testing.assert_allclose(v[np.isfinite(v)], 7.0, atol=1e-6)


def test_regrid_to_hycom_applies_like_mask():
    pytest.importorskip("xesmf")
    grid, _, lat2d = _curvilinear_grid()
    ny, nx = lat2d.shape
    like_temp = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
    like_temp[0, 0] = np.nan                           # one land cell
    like = xr.Dataset({"temp": like_temp})
    out = xhycom.regrid_to_hycom(_regular_product(3.0), grid=grid,
                                 method="bilinear", like=like)
    assert np.isnan(out["temp"].values[0, 0])          # masked to land
    assert np.isfinite(out["temp"].values[ny // 2, nx // 2])
