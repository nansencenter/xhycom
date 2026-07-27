"""Tests for xhycom._transport: transport, section_data, section_flux_density, transport_tpoint."""

import numpy as np
import pytest
import xarray as xr

from xhycom._transect import ResolvedTransect, Transect
from xhycom._transport import (
    _CP,
    _ONEM,
    _RHO0,
    _SREF,
    _TREF,
    _check_velocity_complete,
    _thknss_m,
    section_data,
    section_flux_density,
    transport,
    transport_tpoint,
)

# ---------------------------------------------------------------------------
# Synthetic test fixtures
# ---------------------------------------------------------------------------

N_FACES = 4
N_K = 3
NY = 3
NX = 8
U_VAL = 1.0  # m/s
THK_M = 10.0  # m per layer
WIDTH_M = 1_000.0  # m per face
TEMP_VAL = 5.0  # °C
SAL_VAL = 35.0  # PSU

# Expected volume = N_FACES × N_K × U_VAL × THK_M × WIDTH_M × 1e-6 Sv
_EXPECTED_VOL_SV = N_FACES * N_K * U_VAL * THK_M * WIDTH_M * 1e-6


def _make_resolved(
    n_faces: int = N_FACES,
    ny: int = NY,
    nx: int = NX,
    width_m: float = WIDTH_M,
) -> ResolvedTransect:
    """Synthetic ResolvedTransect: n_faces eastward U-faces at j=1."""
    t = Transect(lons=np.arange(n_faces + 1, dtype=float), lats=np.ones(n_faces + 1))
    fj = np.ones(n_faces, dtype=np.intp)
    fi = np.arange(1, n_faces + 1, dtype=np.intp)
    return ResolvedTransect(
        transect=t,
        j=np.ones(n_faces + 1, dtype=np.intp),
        i=np.arange(n_faces + 1, dtype=np.intp),
        cell_lon=np.arange(n_faces + 1, dtype=float),
        cell_lat=np.ones(n_faces + 1),
        distance_km=np.arange(n_faces + 1, dtype=float),
        cell_width_km=np.ones(n_faces + 1),
        bearing_deg=np.full(n_faces + 1, 90.0),
        face_type=np.zeros(n_faces, dtype=np.uint8),  # 0 = U-face
        face_j=fj,
        face_i=fi,
        face_sign=np.ones(n_faces),
        face_t1_j=np.ones(n_faces, dtype=np.intp),
        face_t1_i=np.arange(n_faces, dtype=np.intp),  # left T-cells: i=0..3
        face_t2_j=np.ones(n_faces, dtype=np.intp),
        face_t2_i=np.arange(1, n_faces + 1, dtype=np.intp),  # right T-cells: i=1..4
        face_width_m=np.full(n_faces, width_m),
        face_dist_km=np.arange(n_faces, dtype=float) + 0.5,
    )


def _make_ds(
    ny: int = NY,
    nx: int = NX,
    nk: int = N_K,
    u_val: float = U_VAL,
    thk_m: float = THK_M,
    temp_val: float = TEMP_VAL,
    sal_val: float = SAL_VAL,
    thk_units: str = "m",
) -> xr.Dataset:
    """Uniform synthetic HYCOM dataset with dims (k, y, x)."""
    return xr.Dataset(
        {
            "u-vel.": (("k", "y", "x"), np.full((nk, ny, nx), u_val)),
            "v-vel.": (("k", "y", "x"), np.zeros((nk, ny, nx))),
            "thknss": (
                ("k", "y", "x"),
                np.full((nk, ny, nx), thk_m),
                {"units": thk_units},
            ),
            "temp": (("k", "y", "x"), np.full((nk, ny, nx), temp_val)),
            "salin": (("k", "y", "x"), np.full((nk, ny, nx), sal_val)),
        }
    )


def _make_tcell_resolved(n_cells: int = 5) -> ResolvedTransect:
    """ResolvedTransect with T-cells only (no face data)."""
    t = Transect(lons=np.arange(n_cells, dtype=float), lats=np.zeros(n_cells))
    return ResolvedTransect(
        transect=t,
        j=np.zeros(n_cells, dtype=np.intp),
        i=np.arange(n_cells, dtype=np.intp),
        cell_lon=np.arange(n_cells, dtype=float),
        cell_lat=np.zeros(n_cells),
        distance_km=np.arange(n_cells, dtype=float),
        cell_width_km=np.ones(n_cells),
        bearing_deg=np.full(n_cells, 90.0),
    )


# ---------------------------------------------------------------------------
# _thknss_m
# ---------------------------------------------------------------------------


def test_thknss_m_converts_pa_to_metres() -> None:
    """Pa thickness divided by _ONEM gives metres."""
    thk = xr.DataArray(np.full((N_K, NY, NX), 10.0 * _ONEM), dims=("k", "y", "x"))
    result = _thknss_m(xr.Dataset({"thknss": thk}), "thknss")
    np.testing.assert_allclose(result.values, 10.0)


def test_thknss_m_passes_through_metres() -> None:
    """Thickness with units='m' is returned unchanged."""
    thk = xr.DataArray(
        np.full((N_K, NY, NX), 10.0), dims=("k", "y", "x"), attrs={"units": "m"}
    )
    result = _thknss_m(xr.Dataset({"thknss": thk}), "thknss")
    np.testing.assert_allclose(result.values, 10.0)


def test_thknss_m_pa_and_m_give_same_result() -> None:
    """Pa and m input with identical physical thickness give identical output."""
    ds_pa = _make_ds(thk_m=10.0 * _ONEM, thk_units="Pa")
    ds_m = _make_ds(thk_m=10.0, thk_units="m")
    np.testing.assert_allclose(
        _thknss_m(ds_pa, "thknss").values,
        _thknss_m(ds_m, "thknss").values,
    )


# ---------------------------------------------------------------------------
# _check_velocity_complete
# ---------------------------------------------------------------------------


def test_check_velocity_baroclinic_attr_raises() -> None:
    """Velocity marked baroclinic raises ValueError."""
    ds = _make_ds()
    ds["u-vel."].attrs["hycom_velocity"] = "baroclinic"
    with pytest.raises(ValueError, match="baroclinic"):
        _check_velocity_complete(ds, "u-vel.", "v-vel.")


def test_check_velocity_btrop_present_raises() -> None:
    """Presence of u_btrop alongside u-vel. without postprocess raises ValueError."""
    ds = _make_ds()
    ds["u_btrop"] = ds["u-vel."]
    with pytest.raises(ValueError, match="baroclinic"):
        _check_velocity_complete(ds, "u-vel.", "v-vel.")


def test_check_velocity_clean_passes() -> None:
    """Dataset with no barotropic artifact passes silently."""
    _check_velocity_complete(_make_ds(), "u-vel.", "v-vel.")  # no exception


# ---------------------------------------------------------------------------
# transport — volume
# ---------------------------------------------------------------------------


def test_transport_volume_exact() -> None:
    """Volume transport matches analytical calculation for uniform u."""
    tr = transport(_make_ds(), _make_resolved())
    np.testing.assert_allclose(tr["volume"].item(), _EXPECTED_VOL_SV, rtol=1e-10)


def test_transport_volume_zero_velocity() -> None:
    """Zero velocity gives zero volume transport."""
    tr = transport(_make_ds(u_val=0.0), _make_resolved())
    assert tr["volume"].item() == pytest.approx(0.0)


def test_transport_volume_sign_negative_for_reversed() -> None:
    """Reversing face_sign negates the volume transport."""
    r = _make_resolved()
    r_rev = ResolvedTransect(
        transect=r.transect,
        j=r.j,
        i=r.i,
        cell_lon=r.cell_lon,
        cell_lat=r.cell_lat,
        distance_km=r.distance_km,
        cell_width_km=r.cell_width_km,
        bearing_deg=r.bearing_deg,
        face_type=r.face_type,
        face_j=r.face_j,
        face_i=r.face_i,
        face_sign=-r.face_sign,  # flipped
        face_t1_j=r.face_t1_j,
        face_t1_i=r.face_t1_i,
        face_t2_j=r.face_t2_j,
        face_t2_i=r.face_t2_i,
        face_width_m=r.face_width_m,
        face_dist_km=r.face_dist_km,
    )
    tr = transport(_make_ds(), r_rev)
    np.testing.assert_allclose(tr["volume"].item(), -_EXPECTED_VOL_SV, rtol=1e-10)


def test_transport_volume_scales_with_thickness() -> None:
    """Doubling the layer thickness doubles the volume transport."""
    tr1 = transport(_make_ds(thk_m=THK_M), _make_resolved())
    tr2 = transport(_make_ds(thk_m=2 * THK_M), _make_resolved())
    np.testing.assert_allclose(
        tr2["volume"].item(), 2 * tr1["volume"].item(), rtol=1e-10
    )


def test_transport_volume_pa_thickness_matches_m() -> None:
    """Pa and m thickness inputs give the same volume transport."""
    tr_pa = transport(_make_ds(thk_m=THK_M * _ONEM, thk_units="Pa"), _make_resolved())
    tr_m = transport(_make_ds(thk_m=THK_M, thk_units="m"), _make_resolved())
    np.testing.assert_allclose(
        tr_pa["volume"].item(), tr_m["volume"].item(), rtol=1e-10
    )


def test_transport_raises_no_face_data() -> None:
    """ResolvedTransect without face data raises ValueError."""
    with pytest.raises(ValueError, match="no face data"):
        transport(_make_ds(), _make_tcell_resolved())


def test_transport_raises_missing_variable() -> None:
    """Missing required velocity variable raises ValueError."""
    ds = _make_ds()
    ds_no_u = ds.drop_vars("u-vel.")
    with pytest.raises(ValueError, match="u-vel"):
        transport(ds_no_u, _make_resolved())


# ---------------------------------------------------------------------------
# transport — heat
# ---------------------------------------------------------------------------


def test_transport_heat_exact() -> None:
    """Heat transport matches analytical formula for uniform fields."""
    expected_tw = (
        N_FACES
        * N_K
        * U_VAL
        * (TEMP_VAL - _TREF)
        * THK_M
        * WIDTH_M
        * _RHO0
        * _CP
        * 1e-12
    )
    tr = transport(_make_ds(), _make_resolved())
    np.testing.assert_allclose(tr["heat"].item(), expected_tw, rtol=1e-10)


def test_transport_heat_zero_at_tref() -> None:
    """Heat transport is zero when temperature equals t_ref."""
    tr = transport(_make_ds(temp_val=0.0), _make_resolved(), t_ref=0.0)
    np.testing.assert_allclose(tr["heat"].item(), 0.0, atol=1e-15)


def test_transport_heat_skipped_when_temp_absent() -> None:
    """Heat variable is absent from output when temp is not in the dataset."""
    ds = _make_ds().drop_vars("temp")
    tr = transport(ds, _make_resolved())
    assert "heat" not in tr


# ---------------------------------------------------------------------------
# transport — salt and freshwater
# ---------------------------------------------------------------------------


def test_transport_salt_exact() -> None:
    """Salt transport matches analytical calculation."""
    expected_kgs = N_FACES * N_K * U_VAL * SAL_VAL * THK_M * WIDTH_M * _RHO0 / 1000.0
    tr = transport(_make_ds(), _make_resolved())
    np.testing.assert_allclose(tr["salt"].item(), expected_kgs, rtol=1e-10)


def test_transport_fw_exact() -> None:
    """Freshwater transport matches analytical calculation."""
    expected_sv = (
        N_FACES * N_K * U_VAL * ((_SREF - SAL_VAL) / _SREF) * THK_M * WIDTH_M * 1e-6
    )
    tr = transport(_make_ds(), _make_resolved())
    np.testing.assert_allclose(tr["fw"].item(), expected_sv, rtol=1e-10)


def test_transport_fw_positive_when_sal_below_sref() -> None:
    """FW transport is positive when salinity is below s_ref."""
    tr = transport(_make_ds(sal_val=_SREF - 1.0), _make_resolved())
    assert tr["fw"].item() > 0


def test_transport_salt_skipped_when_salin_absent() -> None:
    """Salt and fw absent from output when salin is not in the dataset."""
    ds = _make_ds().drop_vars("salin")
    tr = transport(ds, _make_resolved())
    assert "salt" not in tr
    assert "fw" not in tr


# ---------------------------------------------------------------------------
# transport — constraints
# ---------------------------------------------------------------------------


def test_transport_constraint_excludes_all_cells() -> None:
    """Constraint that no cell satisfies gives zero volume."""
    # temp=5, constraint > 10: nothing passes
    tr = transport(
        _make_ds(temp_val=5.0),
        _make_resolved(),
        constraints={"temp": ("gt", 10.0)},
    )
    np.testing.assert_allclose(tr["volume"].item(), 0.0, atol=1e-15)


def test_transport_constraint_passes_all_cells() -> None:
    """Constraint that all cells satisfy returns same result as unconstrained."""
    # temp=5, constraint > 0: everything passes
    tr_all = transport(_make_ds(), _make_resolved())
    tr_con = transport(
        _make_ds(),
        _make_resolved(),
        constraints={"temp": ("gt", 0.0)},
    )
    np.testing.assert_allclose(
        tr_con["volume"].item(), tr_all["volume"].item(), rtol=1e-10
    )


def test_transport_constraint_unknown_op_raises() -> None:
    """Unknown constraint operator raises ValueError."""
    with pytest.raises(ValueError, match="Unknown constraint operator"):
        transport(_make_ds(), _make_resolved(), constraints={"temp": ("gg", 0.0)})


def test_transport_constraint_missing_var_raises() -> None:
    """Constraint variable not in dataset raises ValueError."""
    with pytest.raises(ValueError, match="Constraint variable"):
        transport(_make_ds(), _make_resolved(), constraints={"oxygen": ("gt", 0.0)})


# ---------------------------------------------------------------------------
# transport — output attributes
# ---------------------------------------------------------------------------


def test_transport_output_has_units_attrs() -> None:
    """Output variables carry long_name and units attributes."""
    tr = transport(_make_ds(), _make_resolved())
    for var in ("volume", "heat", "salt", "fw"):
        assert "units" in tr[var].attrs, f"Missing units attr on {var!r}"
        assert "long_name" in tr[var].attrs, f"Missing long_name attr on {var!r}"


# ---------------------------------------------------------------------------
# section_data
# ---------------------------------------------------------------------------


def test_section_data_extracts_correct_values() -> None:
    """section_data selects the right T-cell values from the dataset."""
    r = _make_tcell_resolved(n_cells=5)
    ds = _make_ds(temp_val=TEMP_VAL)
    sec = section_data(ds, r)
    np.testing.assert_allclose(sec["temp"].values, TEMP_VAL)


def test_section_data_has_distance_km_coord() -> None:
    """Output dataset carries distance_km as a section coordinate."""
    r = _make_tcell_resolved(n_cells=5)
    sec = section_data(_make_ds(), r)
    assert "distance_km" in sec.coords


def test_section_data_depth_m_from_uniform_thknss() -> None:
    """depth_m is cumsum of thknss minus half-layer for uniform thickness."""
    r = _make_tcell_resolved(n_cells=5)
    sec = section_data(_make_ds(thk_m=10.0, thk_units="m"), r)
    expected = np.array([5.0, 15.0, 25.0])  # 3 layers of 10 m
    np.testing.assert_allclose(sec["depth_m"].isel(section=0).values, expected)


def test_section_data_omits_depth_m_when_no_thknss() -> None:
    """depth_m is absent when thknss is not in the dataset."""
    r = _make_tcell_resolved(n_cells=5)
    ds = _make_ds().drop_vars("thknss")
    sec = section_data(ds, r, thknss_var="thknss")
    assert "depth_m" not in sec


def test_section_data_explicit_variables() -> None:
    """Only variables listed in variables= are included."""
    r = _make_tcell_resolved(n_cells=5)
    sec = section_data(_make_ds(), r, variables=["temp"])
    assert "temp" in sec
    assert "salin" not in sec


def test_section_data_section_dim_length() -> None:
    """Section dimension equals the number of T-cells in the resolved transect."""
    n = 5
    r = _make_tcell_resolved(n_cells=n)
    sec = section_data(_make_ds(), r)
    assert sec.sizes["section"] == n


# ---------------------------------------------------------------------------
# section_flux_density
# ---------------------------------------------------------------------------


def test_section_flux_density_has_required_vars() -> None:
    """Output contains flux_density and depth_m for any valid input."""
    fd = section_flux_density(_make_ds(), _make_resolved())
    assert "flux_density" in fd
    assert "depth_m" in fd


def test_section_flux_density_uniform_value() -> None:
    """Uniform u=1, thk=10 m gives flux_density = 10 m² s⁻¹ everywhere."""
    fd = section_flux_density(_make_ds(), _make_resolved())
    np.testing.assert_allclose(fd["flux_density"].values, THK_M * U_VAL)


def test_section_flux_density_integral_matches_transport() -> None:
    """Integral of flux_density × face_width_m equals transport volume (m³ s⁻¹)."""
    r = _make_resolved()
    ds = _make_ds()
    fd = section_flux_density(ds, r)
    tr = transport(ds, r)
    integral_m3s = float((fd["flux_density"] * fd["face_width_m"]).sum())
    np.testing.assert_allclose(integral_m3s, tr["volume"].item() * 1e6, rtol=1e-10)


def test_section_flux_density_heat_matches_transport() -> None:
    """Integral of heat_flux_density × face_width_m matches heat transport (W)."""
    r = _make_resolved()
    ds = _make_ds()
    fd = section_flux_density(ds, r)
    tr = transport(ds, r)
    integral_w = float((fd["heat_flux_density"] * fd["face_width_m"]).sum())
    np.testing.assert_allclose(integral_w, tr["heat"].item() * 1e12, rtol=1e-10)


def test_section_flux_density_faces_sorted_by_distance() -> None:
    """Face distance_km coordinate is non-decreasing."""
    fd = section_flux_density(_make_ds(), _make_resolved())
    dist = fd.coords["distance_km"].values
    assert np.all(np.diff(dist) >= 0)


def test_section_flux_density_raises_no_face_data() -> None:
    """ResolvedTransect without face data raises ValueError."""
    with pytest.raises(ValueError, match="no face data"):
        section_flux_density(_make_ds(), _make_tcell_resolved())


# ---------------------------------------------------------------------------
# transport_tpoint — requires scipy
# ---------------------------------------------------------------------------


def _make_glorys_ds(
    ny: int = 8,
    nx: int = 10,
    nz: int = 3,
    u_val: float = 1.0,
    v_val: float = 0.0,
    temp_val: float = TEMP_VAL,
    sal_val: float = SAL_VAL,
) -> xr.Dataset:
    """Simple rectilinear dataset mimicking GLORYS structure."""
    lat = np.arange(44, 44 + ny, dtype=float)
    lon = np.arange(0, nx, dtype=float)
    depth = np.array([5.0, 15.0, 25.0])[:nz]
    shape = (nz, ny, nx)
    return xr.Dataset(
        {
            "uo": (("depth", "latitude", "longitude"), np.full(shape, u_val)),
            "vo": (("depth", "latitude", "longitude"), np.full(shape, v_val)),
            "thetao": (("depth", "latitude", "longitude"), np.full(shape, temp_val)),
            "so": (("depth", "latitude", "longitude"), np.full(shape, sal_val)),
        },
        coords={"latitude": lat, "longitude": lon, "depth": depth},
    )


def test_transport_tpoint_returns_volume() -> None:
    """transport_tpoint returns a Dataset with a volume variable."""
    pytest.importorskip("scipy")
    ds = _make_glorys_ds()
    t = Transect(lons=[5.0, 5.0], lats=[44.0, 51.0])
    tr = transport_tpoint(ds, t, lat_dim="latitude", lon_dim="longitude", z_dim="depth")
    assert "volume" in tr


def test_transport_tpoint_ns_section_uo_positive() -> None:
    """N–S section with uo=1 gives positive volume (rightward = eastward)."""
    pytest.importorskip("scipy")
    ds = _make_glorys_ds(u_val=1.0, v_val=0.0)
    t = Transect(lons=[5.0, 5.0], lats=[44.0, 51.0])
    tr = transport_tpoint(ds, t, lat_dim="latitude", lon_dim="longitude", z_dim="depth")
    assert tr["volume"].item() > 0


def test_transport_tpoint_ew_section_vo_negative() -> None:
    """E–W section with vo=1 gives negative volume (rightward = southward)."""
    pytest.importorskip("scipy")
    ds = _make_glorys_ds(u_val=0.0, v_val=1.0)
    t = Transect(lons=[2.0, 8.0], lats=[47.0, 47.0])
    tr = transport_tpoint(ds, t, lat_dim="latitude", lon_dim="longitude", z_dim="depth")
    assert tr["volume"].item() < 0


def test_transport_tpoint_constraint_zeros_all() -> None:
    """Constraint that no cell passes gives zero volume."""
    pytest.importorskip("scipy")
    ds = _make_glorys_ds(temp_val=2.0)
    t = Transect(lons=[5.0, 5.0], lats=[44.0, 51.0])
    tr = transport_tpoint(
        ds,
        t,
        lat_dim="latitude",
        lon_dim="longitude",
        z_dim="depth",
        constraints={"thetao": ("gt", 10.0)},
    )
    np.testing.assert_allclose(tr["volume"].item(), 0.0, atol=1e-15)


def test_transport_tpoint_missing_scipy_raises() -> None:
    """ImportError is raised when scipy is unavailable (mocked via patching)."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "scipy.spatial":
            raise ImportError("mocked missing scipy")
        return real_import(name, *args, **kwargs)

    import unittest.mock as mock

    ds = _make_glorys_ds()
    t = Transect(lons=[5.0, 5.0], lats=[44.0, 51.0])
    with mock.patch("builtins.__import__", side_effect=mock_import):
        # Re-importing the module forces re-evaluation of the import guard
        import xhycom._transport as _tr

        importlib.reload(_tr)
        with pytest.raises(ImportError, match="scipy"):
            _tr.transport_tpoint(
                ds,
                t,
                lat_dim="latitude",
                lon_dim="longitude",
                z_dim="depth",
            )
    importlib.reload(_tr)  # restore


def test_transport_tpoint_output_units_sv() -> None:
    """Volume output carries units='Sv'."""
    pytest.importorskip("scipy")
    ds = _make_glorys_ds()
    t = Transect(lons=[5.0, 5.0], lats=[44.0, 51.0])
    tr = transport_tpoint(ds, t, lat_dim="latitude", lon_dim="longitude", z_dim="depth")
    assert tr["volume"].attrs.get("units") == "Sv"
