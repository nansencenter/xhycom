"""Regrid HYCOM output to a regular lon/lat/depth grid.

HYCOM output lives on two non-standard grids:

* **Horizontal** — a curvilinear grid whose ``lon`` / ``lat`` are 2-D arrays,
  with velocities on a staggered Arakawa C-grid and oriented along the
  (rotated) model axes.
* **Vertical** — a hybrid isopycnal coordinate whose layer thicknesses are
  stored as pressure (``thknss``, in Pa) and whose layer positions vary in
  space and time.

To compare against products such as GLORYS (regular lon/lat, fixed depth
levels, eastward/northward velocities) we need to map onto that grid.  This
module provides three composable functions:

* :func:`regrid_horizontal` — curvilinear → regular lon/lat (xESMF), including
  C-grid de-staggering and rotation of velocities to east/north.
* :func:`regrid_vertical`   — hybrid layers → fixed depth levels (xgcm).
* :func:`regrid`            — convenience wrapper that chains both.

The heavy dependencies are imported lazily so that ``import xhycom`` works
without them:

* Vertical regridding needs only ``xgcm`` (pip): ``pip install xhycom[regrid]``.
* Lateral regridding additionally needs ``xesmf``, whose ESMF/esmpy backend is
  conda-only (no PyPI wheels): ``conda env create -f ci/environment-regrid.yml``.
"""
import numpy as np
import xarray as xr

# Pa per metre of water column: rho0 * g = 1000 * 9.806.  HYCOM's "onem".
_ONEM = 9806.0

# Natural (u, v) pairs on the C-grid.  u-vars carry lon_u/lat_u, v-vars
# lon_v/lat_v (see xhycom._reader._h_coords).
_UV_PAIRS = (
    ("u-vel.", "v-vel."),
    ("u_btrop", "v_btrop"),
    ("umix", "vmix"),
    ("si_u", "si_v"),
)
_U_VARS = frozenset(u for u, _ in _UV_PAIRS)
_V_VARS = frozenset(v for _, v in _UV_PAIRS)


# ---------------------------------------------------------------------------
# Velocities: de-stagger to T-points and rotate to east/north
# ---------------------------------------------------------------------------
def _uv_to_east_north(ds, pang):
    """Move C-grid velocities to T-points and rotate to true east/north.

    HYCOM stores ``u`` on the western cell edge and ``v`` on the southern
    edge.  For each pair we average the two surrounding edge values onto the
    centre (p-point), then rotate the grid-relative components onto the
    geographic axes using ``pang`` (angle of the model x-axis from true east)::

        east  = u * cos(pang) - v * sin(pang)
        north = u * sin(pang) + v * cos(pang)

    The rotated fields replace the originals *in place* on the returned
    Dataset, keep their HYCOM names, and are re-attached to the T-point
    ``lon`` / ``lat`` coordinates so the single T-grid regridder can handle
    them.  Velocity attrs are updated to record the rotation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that may contain one or more (u, v) pairs.
    pang : xr.DataArray
        Grid rotation angle on the T-point grid, dims ``(y, x)``, radians.
    """
    cos_a = np.cos(pang)
    sin_a = np.sin(pang)

    out = ds
    for u_name, v_name in _UV_PAIRS:
        if u_name not in ds or v_name not in ds:
            continue

        u = ds[u_name]
        v = ds[v_name]
        # Edge -> centre.  p(i) sits between u(i) and u(i+1); likewise in y.
        # The last column/row (no neighbour) becomes NaN — a boundary cell.
        u_p = 0.5 * (u + u.shift(x=-1))
        v_p = 0.5 * (v + v.shift(y=-1))

        east = u_p * cos_a - v_p * sin_a
        north = u_p * sin_a + v_p * cos_a

        east = _move_to_tpoint(east, u, "eastward")
        north = _move_to_tpoint(north, v, "northward")
        out = out.drop_vars([u_name, v_name]).assign(**{u_name: east, v_name: north})

    return out


def _move_to_tpoint(da, like, direction):
    """Re-home a de-staggered velocity onto the T-point lon/lat coords."""
    da = da.drop_vars([c for c in ("lon_u", "lat_u", "lon_v", "lat_v") if c in da.coords])
    attrs = dict(like.attrs)
    base = attrs.get("long_name", like.name)
    attrs["long_name"] = f"{direction} component of {base}"
    attrs["standard_name"] = f"{direction}_sea_water_velocity"
    attrs["comment"] = "de-staggered to T-points and rotated to geographic axes"
    da.attrs = attrs
    return da.rename(like.name)


# ---------------------------------------------------------------------------
# Horizontal: curvilinear -> regular lon/lat (xESMF)
# ---------------------------------------------------------------------------
def regrid_horizontal(ds, lon, lat, grid=None, method="bilinear",
                      periodic=False, mask_var=None):
    """Regrid a HYCOM Dataset from its curvilinear grid to a regular lon/lat grid.

    Velocities (if present) are first de-staggered to T-points and rotated to
    true east/north (requires the grid angle ``pang``); everything is then
    interpolated with a single T-grid xESMF regridder.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from :func:`xhycom.open_dataset` / ``open_mfdataset``, opened
        **with** a ``grid=`` so that ``lon`` / ``lat`` 2-D coords are attached.
    lon, lat : array-like
        1-D target longitudes and latitudes (degrees).
    grid : xr.Dataset, optional
        Grid Dataset (from ``open_dataset`` on ``regional.grid``).  Required to
        rotate velocities — it supplies ``pang``.  If ``ds`` already carries a
        ``pang`` coordinate, this may be omitted.
    method : str
        xESMF interpolation method (``"bilinear"``, ``"conservative"``,
        ``"patch"``, ...).  Default ``"bilinear"``.
    periodic : bool
        Whether the source grid is periodic in longitude.  Default ``False``.
    mask_var : str, optional
        Name of the variable used to derive the land/sea mask.  By default the
        first available of ``temp`` / ``thknss`` is used (finite = ocean).

    Returns
    -------
    xr.Dataset
        Dataset on dims ``(time, k, lat, lon)`` with 1-D ``lon`` / ``lat``
        dimension coordinates.  ``thknss`` is retained for the vertical step.
    """
    try:
        import xesmf as xe
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "regrid_horizontal requires xESMF. Install with the 'regrid' extra "
            "via conda-forge (xESMF needs ESMF/esmpy):\n"
            "    conda env create -f ci/environment-regrid.yml"
        ) from exc

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError(
            "ds has no 'lon'/'lat' coordinates — open it with a grid=, e.g. "
            "open_dataset(path, grid='regional.grid')."
        )

    # Rotate velocities to east/north if any are present.
    if any(v in ds for v in (_U_VARS | _V_VARS)):
        pang = _get_pang(ds, grid)
        ds = _uv_to_east_north(ds, pang)

    # Build the source grid description xESMF expects (2-D lon/lat + mask).
    src = ds
    mask2d = _ocean_mask(ds, mask_var)
    if mask2d is not None:
        src = ds.assign_coords(mask=mask2d)

    target = xr.Dataset(
        {"lat": (["lat"], np.asarray(lat)), "lon": (["lon"], np.asarray(lon))}
    )

    regridder = xe.Regridder(
        src, target, method=method, periodic=periodic,
        ignore_degenerate=True, unmapped_to_nan=True,
    )
    out = regridder(src, keep_attrs=True)

    out["lon"].attrs.setdefault("standard_name", "longitude")
    out["lon"].attrs.setdefault("units", "degrees_east")
    out["lat"].attrs.setdefault("standard_name", "latitude")
    out["lat"].attrs.setdefault("units", "degrees_north")
    return out


def _get_pang(ds, grid):
    """Locate the grid rotation angle (radians) on the T-grid."""
    if "pang" in ds.coords or "pang" in ds:
        return ds["pang"]
    if grid is not None and "pang" in grid:
        pang = grid["pang"]
        # Align onto ds's (y, x) — grid and ds share the native shape.
        return xr.DataArray(np.asarray(pang.values), dims=("y", "x"))
    raise ValueError(
        "Velocity variables are present but 'pang' (grid rotation angle) was "
        "not found. Pass grid=<grid Dataset> so velocities can be rotated to "
        "east/north."
    )


def _ocean_mask(ds, mask_var):
    """2-D (y, x) ocean mask (1 ocean, 0 land) from a representative field."""
    if mask_var is None:
        for cand in ("temp", "thknss"):
            if cand in ds:
                mask_var = cand
                break
    if mask_var is None or mask_var not in ds:
        return None
    da = ds[mask_var]
    reduce_dims = [d for d in da.dims if d not in ("y", "x")]
    finite = np.isfinite(da)
    if reduce_dims:
        finite = finite.any(reduce_dims)
    return finite.astype("int8")


# ---------------------------------------------------------------------------
# Vertical: hybrid layers -> fixed depth levels (xgcm)
# ---------------------------------------------------------------------------
def regrid_vertical(ds, depth, method="linear", mask_edges=True,
                    layer_dim="k", variables=None):
    """Regrid HYCOM layered variables onto fixed depth levels.

    Layer-centre depths are reconstructed from ``thknss`` (Pa -> m via
    ``thknss / 9806``, cumulative sum to interfaces, minus half-thickness to
    centres) and used as the source coordinate for an ``xgcm`` vertical
    transform onto the requested ``depth`` levels.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``thknss`` and one or more variables on
        ``layer_dim``.  May be on the native or a regular horizontal grid.
    depth : array-like
        1-D target depths in metres, positive down (e.g. GLORYS levels).
    method : str
        xgcm transform method (``"linear"`` or ``"conservative"``).
    mask_edges : bool
        If True, target depths outside the source column range are NaN.
    layer_dim : str
        Name of the HYCOM layer dimension. Default ``"k"``.
    variables : list of str, optional
        Which layered variables to regrid. Default: all variables that have
        ``layer_dim`` (except ``thknss`` itself).

    Returns
    -------
    xr.Dataset
        Dataset with ``layer_dim`` replaced by a ``depth`` dimension
        coordinate (``positive='down'``).  2-D fields are carried through
        unchanged.
    """
    try:
        import xgcm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "regrid_vertical requires xgcm. Install with: pip install xgcm "
            "(or use the 'regrid' extra)."
        ) from exc

    if "thknss" not in ds:
        raise ValueError(
            "regrid_vertical needs 'thknss' to reconstruct layer depths; it is "
            "not in the Dataset."
        )

    z_centre = layer_centre_depth(ds["thknss"], layer_dim=layer_dim)

    depth = np.asarray(depth, dtype="float64")
    target = xr.DataArray(depth, dims=["depth"], coords={"depth": depth}, name="depth")

    grid = xgcm.Grid(
        ds, coords={"Z": {"center": layer_dim}}, periodic=False,
        autoparse_metadata=False,
    )

    if variables is None:
        variables = [
            name for name, da in ds.data_vars.items()
            if layer_dim in da.dims and name != "thknss"
        ]

    out_vars = {}
    for name in variables:
        da = ds[name]
        transformed = grid.transform(
            da, "Z", target=depth, target_data=z_centre,
            method=method, mask_edges=mask_edges,
        )
        # xgcm names the new dim after target_data ("depth"); be defensive.
        if "depth" not in transformed.dims:
            new_dim = [d for d in transformed.dims if d not in da.dims]
            if new_dim:
                transformed = transformed.rename({new_dim[0]: "depth"})
        transformed.attrs = dict(da.attrs)
        out_vars[name] = transformed

    # Carry through everything that didn't have the layer dimension
    # (2-D diagnostics, surface fields), excluding thknss.
    passthrough = {
        name: da for name, da in ds.data_vars.items()
        if layer_dim not in da.dims and name != "thknss"
    }

    out = xr.Dataset({**passthrough, **out_vars}, attrs=ds.attrs)
    out = out.assign_coords(depth=xr.Variable(
        "depth", depth,
        {"long_name": "depth", "units": "m", "positive": "down", "axis": "Z"},
    ))
    return out


def layer_centre_depth(thknss, layer_dim="k"):
    """Layer-centre depths (metres, positive down) from HYCOM ``thknss`` (Pa).

    A tiny strictly-increasing ramp (0.1 mm per layer) is added so that
    massless / zero-thickness hybrid layers — which would otherwise share an
    identical depth and make the column non-monotonic — do not break the
    vertical interpolation.

    Unit-aware: if ``thknss`` already carries ``units='m'`` (e.g. after
    ``open_dataset(..., postprocess=True)``) it is used as-is; otherwise it is
    treated as pressure in Pa and divided by ``onem`` (9806).
    """
    thknss_m = thknss if thknss.attrs.get("units") == "m" else thknss / _ONEM
    z_interface = thknss_m.cumsum(layer_dim)
    z_centre = z_interface - thknss_m / 2

    n = thknss.sizes[layer_dim]
    ramp = xr.DataArray(np.arange(n) * 1e-4, dims=[layer_dim])
    z_centre = z_centre + ramp

    z_centre.attrs = {"long_name": "layer centre depth", "units": "m",
                      "positive": "down"}
    return z_centre.rename("depth")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def regrid(ds, lon, lat, depth, grid=None, method="bilinear",
           z_method="linear", periodic=False, mask_edges=True, variables=None):
    """Regrid HYCOM output to a regular lon/lat/depth grid (lateral then vertical).

    Equivalent to :func:`regrid_horizontal` followed by :func:`regrid_vertical`.
    The lateral step runs first so that ``thknss`` and all fields share the
    regular grid before depths are reconstructed and interpolated.

    Parameters
    ----------
    ds : xr.Dataset
        HYCOM Dataset opened with a ``grid=`` (so ``lon`` / ``lat`` exist).
    lon, lat : array-like
        Target 1-D longitudes / latitudes (degrees).
    depth : array-like
        Target 1-D depths (metres, positive down).
    grid : xr.Dataset, optional
        Grid Dataset, needed to rotate velocities (supplies ``pang``).
    method : str
        Horizontal interpolation method (xESMF). Default ``"bilinear"``.
    z_method : str
        Vertical interpolation method (xgcm). Default ``"linear"``.
    periodic : bool
        Source grid periodic in longitude. Default ``False``.
    mask_edges : bool
        Mask target depths outside the source column range. Default ``True``.
    variables : list of str, optional
        Restrict the vertical step to these layered variables.

    Returns
    -------
    xr.Dataset
        Dataset on dims ``(time, depth, lat, lon)``.
    """
    ds = regrid_horizontal(ds, lon, lat, grid=grid, method=method,
                           periodic=periodic)
    ds = regrid_vertical(ds, depth, method=z_method, mask_edges=mask_edges,
                         variables=variables)
    return ds
