"""Section transports from HYCOM output using exact C-grid face velocities.

:func:`transport` computes volume, heat, salt, and freshwater transports
through a :class:`~xhycom.Transect` that has been resolved against a HYCOM
grid via :meth:`~xhycom.Transect.resolve`.  The calculation uses native
``u-vel.`` and ``v-vel.`` directly at the staggered U- and V-face locations —
no rotation or interpolation to T-points is needed.  Layer thickness and
tracer values are averaged from the two T-cells that straddle each face.

:func:`section_data` extracts hydrographic fields (temperature, salinity, ...)
along the T-cell path as a ``(section, k)`` Dataset suitable for
cross-section plots.

:func:`section_flux_density` returns the per-face, per-layer signed transport
density (m² s⁻¹ = v_normal × dz) along the section, indexed by
``distance_km`` and layer, for visualising where positive and negative
transport contributions come from.

:func:`transport_tpoint` computes the same four transports for any
dataset on a regular or rectilinear grid (e.g. GLORYS, EN4, observations)
by projecting T-point velocities onto the section-normal direction and
integrating over depth and along-section width.

Sign convention
---------------
Positive transport is rightward when walking from the first transect waypoint
to the last (see :class:`~xhycom.Transect`).

Units
-----

| Quantity        | Output variable  | Unit    |
|-----------------|------------------|---------|
| Volume          | ``volume``    | Sv      |
| Heat            | ``heat``      | TW      |
| Salt            | ``salt``      | kg s⁻¹  |
| Freshwater      | ``fw``        | Sv      |

Heat transport is computed relative to *t_ref* (default 0 °C).  Freshwater
transport is relative to *s_ref* (default 34.8 PSU).

Constrained transports
----------------------
Pass ``constraints`` to zero out face contributions where tracer conditions
are not met.  This lets you separate, e.g., Atlantic Water inflow
(``temp > 2``) from polar water outflow (``temp < 2``)::

    tr_aw  = xhycom.transport(ds, r, constraints={"temp": ("gt", 2.0)})
    tr_pw  = xhycom.transport(ds, r, constraints={"temp": ("le", 2.0)})
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from ._transect import ResolvedTransect, Transect, _load_grid

# ---------------------------------------------------------------------------
# NOTE on boundary-row transport
# ---------------------------------------------------------------------------
# ``Transect.from_boundary_row`` creates waypoints along the boundary at
# constant j, then ``resolve(grid)`` walks west→east, producing U-faces
# (east-west velocity).  For a south/north boundary we want V-faces
# (cross-boundary velocity).  Use ``boundary_transport()`` instead of
# ``transport()`` for boundary rows.
# ---------------------------------------------------------------------------

# Pa per metre of water column: rho0 * g = 1000 * 9.806 (HYCOM's ``onem``)
_ONEM: float = 9806.0

_RHO0: float = 1025.0  # reference density  kg m⁻³
_CP: float = 3996.0  # specific heat       J kg⁻¹ K⁻¹
_SREF: float = 34.8  # freshwater reference salinity  PSU
_TREF: float = 0.0  # heat-transport reference temperature  °C

# Operator strings accepted in *constraints*
_OPS = frozenset({"lt", "le", "gt", "ge", "eq"})


def transport(
    ds: xr.Dataset,
    transect: Transect | ResolvedTransect,
    grid: xr.Dataset | str | None = None,
    *,
    u_var: str = "u-vel.",
    v_var: str = "v-vel.",
    t_var: str = "temp",
    s_var: str | None = None,
    thknss_var: str = "thknss",
    k_dim: str = "k",
    z_dim: str | None = None,
    s_ref: float = _SREF,
    t_ref: float = _TREF,
    rho0: float = _RHO0,
    cp: float = _CP,
    constraints: dict[str, tuple[Literal["lt", "le", "gt", "ge", "eq"], float]]
    | None = None,
) -> xr.Dataset:
    """Compute section transports for HYCOM or any regular-grid dataset.

    For HYCOM, pass a transect resolved against the C-grid (face-exact
    integration).  For other datasets (e.g. GLORYS), resolve the transect
    with ``lat_var``/``lon_var`` and pass ``z_dim``; transport is then
    computed via nearest-cell T-point projection.

    Parameters
    ----------
    ds:
        Dataset containing velocity, thickness / depth, and optionally tracers.
        For HYCOM: ``u-vel.``, ``v-vel.``, ``thknss``, ``temp``, ``salin``.
        For generic grids: true east/north velocity components (m s⁻¹) and a
        1-D depth coordinate.
    transect:
        A :class:`~xhycom.Transect` or a pre-resolved
        :class:`~xhycom.ResolvedTransect`.  For generic grids, resolve first
        with :meth:`~xhycom.Transect.resolve` passing ``lat_var``/``lon_var``.
    grid:
        Required when *transect* is an unresolved :class:`~xhycom.Transect`
        being resolved against a HYCOM grid.
    u_var:
        Eastward velocity variable name (HYCOM default ``"u-vel."``).
    v_var:
        Northward velocity variable name (HYCOM default ``"v-vel."``).
    t_var:
        Temperature variable name.  Heat transport is skipped if absent.
    s_var:
        Salinity variable name.  Auto-detected from common names when ``None``.
        Salt and FW transport are skipped if absent.
    thknss_var:
        Layer-thickness variable (HYCOM path only).
    k_dim:
        Vertical layer dimension name (HYCOM path only).
    z_dim:
        Depth coordinate name for generic grids (e.g. ``"depth"``).  Required
        when the resolved transect has no HYCOM face data.
    s_ref:
        Freshwater reference salinity in PSU.  Default 34.8.
    t_ref:
        Heat-transport reference temperature in °C.  Default 0.0.
    rho0:
        Reference density in kg m⁻³.  Default 1025.
    cp:
        Specific heat capacity in J kg⁻¹ K⁻¹.  Default 3996.
    constraints:
        Optional ``{variable: (operator, threshold)}`` pairs that zero out
        contributions where the condition is not met.

    Returns
    -------
    xr.Dataset
        Variables ``volume`` (Sv), ``heat`` (TW), ``salt`` (kg s⁻¹), ``fw``
        (Sv), with a ``time`` dimension when present.

    Examples
    --------
    HYCOM::

        >>> fs  = xhycom.Transect.named("fram_strait")
        >>> tr  = xhycom.transport(ds, fs, grid=grid)

    GLORYS::

        >>> resolved = fs.resolve(glorys, lat_var="latitude", lon_var="longitude")
        >>> tr = xhycom.transport(glorys, resolved,
        ...          u_var="uo", v_var="vo", t_var="thetao", s_var="so",
        ...          z_dim="depth")
    """
    # ------------------------------------------------------------------
    # Resolve transect if needed
    # ------------------------------------------------------------------
    resolved = _ensure_resolved(transect, grid)

    # ------------------------------------------------------------------
    # Salinity auto-detection (shared by both paths)
    # ------------------------------------------------------------------
    if s_var is None:
        for _name in ("salin", "saln", "so", "salinity", "sal"):
            if _name in ds:
                s_var = _name
                break

    # ------------------------------------------------------------------
    # Generic T-point path (non-HYCOM resolved transect)
    # ------------------------------------------------------------------
    if not resolved.has_face_data:
        if z_dim is None:
            raise ValueError(
                "z_dim is required when transect was resolved against a "
                "non-HYCOM grid.  Pass z_dim=<depth coordinate name>."
            )
        for var in (u_var, v_var):
            if var not in ds:
                raise ValueError(f"Required variable {var!r} not found in dataset.")
        if constraints:
            for cvar, (op, _) in constraints.items():
                if cvar not in ds:
                    raise ValueError(f"Constraint variable {cvar!r} not found.")
                if op not in _OPS:
                    raise ValueError(f"Unknown operator {op!r}. Use: {sorted(_OPS)}")

        # For regular (1-D coordinate) grids use coordinate-value selection so the
        # result is correct even when ds is a spatial subset of the grid that was
        # originally passed to resolve() (e.g. HYCOM regridded to a clipped GLORYS).
        if resolved.y_dim in ds.coords and ds[resolved.y_dim].ndim == 1:
            _lat_idx = xr.DataArray(resolved.cell_lat, dims="section")
            _lon_idx = xr.DataArray(resolved.cell_lon, dims="section")
            sel = {resolved.y_dim: _lat_idx, resolved.x_dim: _lon_idx}
            _sel_kw: dict = {"method": "nearest"}
        else:
            j_da = xr.DataArray(resolved.j, dims="section")
            i_da = xr.DataArray(resolved.i, dims="section")
            sel = {resolved.y_dim: j_da, resolved.x_dim: i_da}
            _sel_kw = {}

        theta = np.radians(resolved.bearing_deg)
        cos_t = xr.DataArray(np.cos(theta), dims="section")
        sin_t = xr.DataArray(np.sin(theta), dims="section")
        w = xr.DataArray(resolved.cell_width_km * 1e3, dims="section")

        u = ds[u_var].sel(**sel, **_sel_kw)
        v = ds[v_var].sel(**sel, **_sel_kw)
        v_normal = u * cos_t - v * sin_t  # positive = rightward

        z_vals = ds[z_dim].values.astype(float)
        z_edges = np.empty(len(z_vals) + 1)
        z_edges[1:-1] = (z_vals[:-1] + z_vals[1:]) / 2.0
        z_edges[0] = z_vals[0] - (z_vals[1] - z_vals[0]) / 2.0
        z_edges[-1] = z_vals[-1] + (z_vals[-1] - z_vals[-2]) / 2.0
        dz = xr.DataArray(np.diff(z_edges), dims=z_dim)

        if constraints:
            cmask: xr.DataArray | None = None
            for cvar, (op, threshold) in constraints.items():
                val = ds[cvar].sel(**sel, **_sel_kw)
                cond: xr.DataArray = {
                    "lt": val < threshold,
                    "le": val <= threshold,
                    "gt": val > threshold,
                    "ge": val >= threshold,
                    "eq": val == threshold,
                }[op]
                cmask = cond if cmask is None else (cmask & cond)
            v_normal = v_normal.where(cmask, 0.0)

        def _tp_integrate(da: xr.DataArray) -> xr.DataArray:
            return (da * dz * w).sum(dim=[z_dim, "section"])

        compute_heat = t_var in ds
        compute_salt = s_var is not None and s_var in ds
        out_vars: dict[str, xr.DataArray] = {}
        out_vars["volume"] = _attach_attrs(
            _tp_integrate(v_normal) * 1e-6, "volume transport", "Sv"
        )
        if compute_heat:
            t = ds[t_var].sel(**sel, **_sel_kw)
            out_vars["heat"] = _attach_attrs(
                _tp_integrate((t - t_ref) * v_normal) * rho0 * cp * 1e-12,
                "heat transport",
                "TW",
            )
        if compute_salt:
            s = ds[s_var].sel(**sel, **_sel_kw)
            out_vars["salt"] = _attach_attrs(
                _tp_integrate(s * v_normal) * rho0 / 1000.0, "salt transport", "kg s-1"
            )
            out_vars["fw"] = _attach_attrs(
                _tp_integrate((s_ref - s) / s_ref * v_normal) * 1e-6,
                "freshwater transport",
                "Sv",
            )
        return xr.Dataset(out_vars)

    # ------------------------------------------------------------------
    # HYCOM C-grid path
    # ------------------------------------------------------------------
    for var in (u_var, v_var, thknss_var):
        if var not in ds:
            raise ValueError(f"Required variable {var!r} not found in dataset.")

    _check_velocity_complete(ds, u_var, v_var)

    compute_heat = t_var in ds
    compute_salt = s_var is not None and s_var in ds

    if constraints:
        for cvar in constraints:
            if cvar not in ds:
                raise ValueError(f"Constraint variable {cvar!r} not found in dataset.")
        for cvar, (op, _) in constraints.items():
            if op not in _OPS:
                raise ValueError(
                    f"Unknown constraint operator {op!r}. Use one of {sorted(_OPS)}."
                )

    # ------------------------------------------------------------------
    # Separate U-faces and V-faces
    # ------------------------------------------------------------------
    u_mask = resolved.face_type == 0  # _FACE_U
    v_mask = resolved.face_type == 1  # _FACE_V

    out_vars: dict[str, xr.DataArray] = {}

    vol_u = _face_volume_flux(
        ds, resolved, u_mask, "uf", u_var, thknss_var, k_dim, constraints
    )
    vol_v = _face_volume_flux(
        ds, resolved, v_mask, "vf", v_var, thknss_var, k_dim, constraints
    )

    vol = vol_u + vol_v  # (time?,) in m³ s⁻¹
    out_vars["volume"] = _attach_attrs(vol * 1e-6, "volume transport", "Sv")

    if compute_heat:
        heat_u = _face_tracer_flux(
            ds,
            resolved,
            u_mask,
            "uf",
            u_var,
            thknss_var,
            k_dim,
            t_var,
            t_ref,
            constraints,
        )
        heat_v = _face_tracer_flux(
            ds,
            resolved,
            v_mask,
            "vf",
            v_var,
            thknss_var,
            k_dim,
            t_var,
            t_ref,
            constraints,
        )
        heat = (heat_u + heat_v) * rho0 * cp  # W
        out_vars["heat"] = _attach_attrs(heat * 1e-12, "heat transport", "TW")

    if compute_salt:
        salt_u = _face_tracer_flux(
            ds,
            resolved,
            u_mask,
            "uf",
            u_var,
            thknss_var,
            k_dim,
            s_var,
            0.0,
            constraints,
        )
        salt_v = _face_tracer_flux(
            ds,
            resolved,
            v_mask,
            "vf",
            v_var,
            thknss_var,
            k_dim,
            s_var,
            0.0,
            constraints,
        )
        # Salinity in PSU (g kg⁻¹): multiply by rho0 and convert PSU→kg/kg (÷1000)
        salt = (salt_u + salt_v) * rho0 / 1000.0  # kg s⁻¹
        out_vars["salt"] = _attach_attrs(salt, "salt transport", "kg s-1")

        fw_u = _face_tracer_flux(
            ds,
            resolved,
            u_mask,
            "uf",
            u_var,
            thknss_var,
            k_dim,
            s_var,
            0.0,
            constraints,
            fw_sref=s_ref,
        )
        fw_v = _face_tracer_flux(
            ds,
            resolved,
            v_mask,
            "vf",
            v_var,
            thknss_var,
            k_dim,
            s_var,
            0.0,
            constraints,
            fw_sref=s_ref,
        )
        fw = fw_u + fw_v  # m³ s⁻¹
        out_vars["fw"] = _attach_attrs(fw * 1e-6, "freshwater transport", "Sv")

    return xr.Dataset(out_vars)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_BTROP_PAIRS = (("u-vel.", "u_btrop"), ("v-vel.", "v_btrop"))


def _check_velocity_complete(ds: xr.Dataset, u_var: str, v_var: str) -> None:
    """Raise if velocity looks baroclinic-only (postprocess not applied on archv)."""
    for vel_var, btrop_var in _BTROP_PAIRS:
        if vel_var not in (u_var, v_var):
            continue
        vel_status = ds[vel_var].attrs.get("hycom_velocity")
        if vel_status == "baroclinic":
            raise ValueError(
                f"{vel_var!r} is baroclinic-only (the barotropic component "
                f"{btrop_var!r} was missing when the dataset was opened). "
                "Re-open with the barotropic variable included, then apply "
                "xhycom.postprocess(), or use open_dataset(..., postprocess=True)."
            )
        if vel_status is None and btrop_var in ds:
            raise ValueError(
                f"{vel_var!r} appears to be baroclinic (archv file opened "
                f"without postprocess=True, and {btrop_var!r} is still present "
                "as a separate variable). Apply xhycom.postprocess() or re-open "
                "with open_dataset(..., postprocess=True) to form the total current."
            )


def _ensure_resolved(
    transect: Transect | ResolvedTransect,
    grid: xr.Dataset | str | None,
) -> ResolvedTransect:
    """Resolve *transect* against *grid* if it is not yet a ResolvedTransect."""
    if isinstance(transect, ResolvedTransect):
        return transect
    if grid is None:
        raise ValueError("grid= is required when transect is an unresolved Transect.")
    return transect.resolve(_load_grid(grid))


def _thknss_m(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Return layer thickness in metres, converting from Pa if necessary."""
    thk = ds[var]
    if thk.attrs.get("units") == "m":
        return thk
    return thk / _ONEM


def _face_arrays(
    resolved: ResolvedTransect,
    mask: np.ndarray,
    face_dim: str,
) -> tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
]:
    """Extract DataArrays for the subset of faces given by *mask*."""
    fj = xr.DataArray(resolved.face_j[mask], dims=face_dim)
    fi = xr.DataArray(resolved.face_i[mask], dims=face_dim)
    sgn = xr.DataArray(resolved.face_sign[mask], dims=face_dim)
    w = xr.DataArray(resolved.face_width_m[mask], dims=face_dim)
    t1j = xr.DataArray(resolved.face_t1_j[mask], dims=face_dim)
    t1i = xr.DataArray(resolved.face_t1_i[mask], dims=face_dim)
    t2j = xr.DataArray(resolved.face_t2_j[mask], dims=face_dim)
    t2i = xr.DataArray(resolved.face_t2_i[mask], dims=face_dim)
    return fj, fi, sgn, w, t1j, t1i, t2j, t2i


def _face_volume_flux(
    ds: xr.Dataset,
    resolved: ResolvedTransect,
    mask: np.ndarray,
    face_dim: str,
    vel_var: str,
    thknss_var: str,
    k_dim: str,
    constraints: dict | None,
) -> xr.DataArray:
    """Signed volume flux (m³ s⁻¹) summed over all layers and faces in *mask*."""
    if not mask.any():
        return xr.DataArray(0.0)

    fj, fi, sgn, w, t1j, t1i, t2j, t2i = _face_arrays(resolved, mask, face_dim)

    vel = ds[vel_var].isel(y=fj, x=fi)  # (…, k, fd)
    thk = _thknss_m(ds, thknss_var)
    thk1 = thk.isel(y=t1j, x=t1i)  # (…, k, fd)
    thk2 = thk.isel(y=t2j, x=t2i)
    thk_face = 0.5 * (thk1 + thk2)

    flux = sgn * vel * thk_face * w  # (…, k, fd)

    if constraints:
        flux = flux.where(
            _constraint_mask(ds, constraints, t1j, t1i, t2j, t2i, face_dim), 0.0
        )

    return flux.sum(dim=[k_dim, face_dim])


def _face_tracer_flux(
    ds: xr.Dataset,
    resolved: ResolvedTransect,
    mask: np.ndarray,
    face_dim: str,
    vel_var: str,
    thknss_var: str,
    k_dim: str,
    tracer_var: str,
    tracer_ref: float,
    constraints: dict | None,
    fw_sref: float | None = None,
) -> xr.DataArray:
    """Signed tracer-weighted volume flux (m³ s⁻¹ × tracer_units) summed over all layers/faces.

    When *fw_sref* is provided the tracer anomaly ``(sref - S)/sref`` is used
    instead of ``S - tracer_ref``, giving the freshwater flux contribution.
    """
    if not mask.any():
        return xr.DataArray(0.0)

    fj, fi, sgn, w, t1j, t1i, t2j, t2i = _face_arrays(resolved, mask, face_dim)

    vel = ds[vel_var].isel(y=fj, x=fi)
    thk = _thknss_m(ds, thknss_var)
    thk_face = 0.5 * (thk.isel(y=t1j, x=t1i) + thk.isel(y=t2j, x=t2i))

    tr1 = ds[tracer_var].isel(y=t1j, x=t1i)
    tr2 = ds[tracer_var].isel(y=t2j, x=t2i)
    tr_face = 0.5 * (tr1 + tr2)

    if fw_sref is not None:
        anom = (fw_sref - tr_face) / fw_sref
    else:
        anom = tr_face - tracer_ref

    flux = sgn * vel * anom * thk_face * w

    if constraints:
        flux = flux.where(
            _constraint_mask(ds, constraints, t1j, t1i, t2j, t2i, face_dim), 0.0
        )

    return flux.sum(dim=[k_dim, face_dim])


def _constraint_mask(
    ds: xr.Dataset,
    constraints: dict[str, tuple[str, float]],
    t1j: xr.DataArray,
    t1i: xr.DataArray,
    t2j: xr.DataArray,
    t2i: xr.DataArray,
    face_dim: str,
) -> xr.DataArray:
    """Boolean mask: True where all constraints are satisfied at the face."""
    mask: xr.DataArray | None = None
    for var, (op, threshold) in constraints.items():
        tr1 = ds[var].isel(y=t1j, x=t1i)
        tr2 = ds[var].isel(y=t2j, x=t2i)
        val = 0.5 * (tr1 + tr2)
        cond: xr.DataArray = {
            "lt": val < threshold,
            "le": val <= threshold,
            "gt": val > threshold,
            "ge": val >= threshold,
            "eq": val == threshold,
        }[op]
        mask = cond if mask is None else (mask & cond)
    return mask  # type: ignore[return-value]


def _attach_attrs(da: xr.DataArray, long_name: str, units: str) -> xr.DataArray:
    """Return *da* with long_name and units attributes set."""
    da.attrs["long_name"] = long_name
    da.attrs["units"] = units
    return da


# ---------------------------------------------------------------------------
# Boundary-row transport (V-face integration)
# ---------------------------------------------------------------------------


def boundary_transport(
    ds: xr.Dataset,
    grid: xr.Dataset | str,
    bathy: xr.Dataset | str,
    j: int,
    *,
    sign: float = 1.0,
    t_var: str = "temp",
    s_var: str | None = None,
    thknss_var: str = "thknss",
    k_dim: str = "k",
    s_ref: float = _SREF,
    t_ref: float = _TREF,
    rho0: float = _RHO0,
    cp: float = _CP,
) -> xr.Dataset:
    """Transport through a domain open boundary via direct V-face integration.

    .. note::
        The preferred workflow is now::

            rt = xhycom.ResolvedTransect.from_vface_row(grid, bathy, j=1, sign=+1)
            tr = xhycom.transport(ds, rt)

        :meth:`~xhycom.ResolvedTransect.from_vface_row` builds the V-face
        transect once and reuses it across many datasets; this function is a
        convenience shortcut that constructs the face geometry on every call.

    Do **not** use the standard transect workflow
    (``Transect.from_boundary_row`` → ``resolve`` → ``transport``) for boundary
    rows.  That path walks along the boundary at constant *j*, picking up
    **U-faces** (east-west velocity) — the wrong component for a south/north
    boundary.  This function integrates ``v-vel.`` at V-face row *j* directly,
    which carries the actual cross-boundary flux.

    Parameters
    ----------
    ds : xr.Dataset
        HYCOM dataset with ``v-vel.``, ``thknss``, and optionally ``temp`` and
        salinity.  Apply :func:`postprocess` (or open with ``postprocess=True``)
        so that ``thknss`` is in metres and ``v-vel.`` is the total velocity.
    grid : xr.Dataset or str
        HYCOM grid dataset or path to ``regional.grid``.  Must contain ``scvx``
        (V-face width in the x-direction).
    bathy : xr.Dataset or str
        Bathymetry dataset from ``xhycom.open_dataset(bathy_path, grid=...)``.
        Used to identify wet columns.
    j : int
        V-face row index.  ``v-vel.[y=j, x=i]`` is the northward velocity
        between T-cells ``(j-1, i)`` and ``(j, i)``.

        - **Southern boundary**: ``j=1`` — face between exterior row 0 and
          the first interior row 1.  Positive v-vel = northward = into domain.
          Use the default ``sign=+1``.
        - **Northern boundary**: ``j=grid.sizes["y"] - 1`` — face between the
          last interior row and the exterior.  Positive v-vel = northward =
          out of domain.  Pass ``sign=-1`` to adopt "positive = into domain".

    sign : float
        Applied to the velocity before integration.  ``+1`` (default) when
        positive v-vel means flow into the domain; ``-1`` to negate.

    Returns
    -------
    xr.Dataset
        ``volume`` (Sv), ``heat`` (TW), ``salt`` (kg s⁻¹), ``fw`` (Sv).
        Sign: positive = into the domain when *sign* is set correctly.

    Examples
    --------
    >>> jdm = grid.sizes["y"]
    >>> tr_s = xhycom.boundary_transport(ds, grid, bathy, j=1)
    >>> tr_n = xhycom.boundary_transport(ds, grid, bathy, j=jdm - 1, sign=-1)
    """
    grid_ds = _load_grid(grid)
    bathy_ds = _load_grid(bathy)

    # Interior T-cell row (ocean side of the V-face).
    # V-face j is between T(j-1) and T(j).
    # Southern boundary (sign > 0): T(j-1)=exterior, T(j)=interior → j_int = j
    # Northern boundary (sign < 0): T(j-1)=interior, T(j)=exterior → j_int = j-1
    j_int = j if sign >= 0 else j - 1

    if s_var is None:
        for _name in ("salin", "saln", "so", "salinity", "sal"):
            if _name in ds:
                s_var = _name
                break

    thk = _thknss_m(ds, thknss_var)
    wet = xr.DataArray(np.isfinite(bathy_ds["depth"].isel(y=j_int).values), dims=["x"])
    scvx = xr.DataArray(grid_ds["scvx"].isel(y=j).values, dims=["x"])

    v_vel = ds["v-vel."].isel(y=j)  # (…, k, x): northward velocity at V-face j
    thk_i = thk.isel(y=j_int)  # interior T-cell layer thicknesses

    # Signed cross-boundary volume flux [m³ s⁻¹ per column per layer]
    flux = (sign * v_vel * thk_i * scvx).where(wet)

    dims = [k_dim, "x"]
    out: dict[str, xr.DataArray] = {
        "volume": _attach_attrs(flux.sum(dim=dims) * 1e-6, "volume transport", "Sv"),
    }

    if t_var in ds:
        temp = ds[t_var].isel(y=j_int)
        out["heat"] = _attach_attrs(
            (flux * (temp - t_ref)).sum(dim=dims) * rho0 * cp * 1e-12,
            "heat transport",
            "TW",
        )

    if s_var is not None and s_var in ds:
        sal = ds[s_var].isel(y=j_int)
        out["salt"] = _attach_attrs(
            (flux * sal).sum(dim=dims) * rho0 / 1000.0,
            "salt transport",
            "kg s-1",
        )
        out["fw"] = _attach_attrs(
            (flux * (s_ref - sal) / s_ref).sum(dim=dims) * 1e-6,
            "freshwater transport",
            "Sv",
        )

    return xr.Dataset(out)


# ---------------------------------------------------------------------------
# Section data extraction
# ---------------------------------------------------------------------------


def section_data(
    ds: xr.Dataset,
    transect: Transect | ResolvedTransect,
    grid: xr.Dataset | str | None = None,
    *,
    variables: list[str] | None = None,
    thknss_var: str = "thknss",
    k_dim: str = "k",
    z_dim: str | None = None,
) -> xr.Dataset:
    """Extract hydrographic data along the section T-cell path.

    Selects every variable with the section's spatial dimensions from *ds* at
    the T-cell positions identified by :meth:`~xhycom.Transect.resolve`,
    producing a ``(k, section)`` Dataset suitable for cross-section plots.  A
    ``distance_km`` coordinate is attached along the ``section`` dimension, and
    ``depth_m`` is added from layer thickness (HYCOM) or the *z_dim* coordinate
    (z-level grids such as GLORYS).

    Parameters
    ----------
    ds:
        Dataset containing the fields to extract (HYCOM or any regular grid).
    transect:
        A :class:`~xhycom.Transect` or pre-resolved
        :class:`~xhycom.ResolvedTransect`.
    grid:
        Required when *transect* is an unresolved :class:`~xhycom.Transect`
        and no *lat_var*/*lon_var* were passed to :meth:`~xhycom.Transect.resolve`.
    variables:
        Explicit list of variable names to extract.  When ``None``, every
        variable whose dimensions include the section's spatial axes is used.
    thknss_var:
        Layer-thickness variable used to derive ``depth_m`` for HYCOM datasets.
    k_dim:
        Name of the vertical layer dimension (HYCOM).
    z_dim:
        Name of the depth coordinate for z-level datasets (e.g. ``"depth"``
        for GLORYS).  Used as ``depth_m`` when *thknss_var* is absent.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``section`` (and a vertical / ``time``
        dimension when present) and coordinate ``distance_km``.  ``depth_m``
        is included as a data variable when depth information is available.

    Examples
    --------
    >>> resolved = xhycom.Transect.named("fram_strait").resolve(grid)
    >>> sec = xhycom.section_data(ds, resolved)
    >>> sec["temp"].plot(x="distance_km", y="depth_m")

    >>> resolved_g = southern.resolve(glorys, lat_var="latitude", lon_var="longitude")
    >>> sec_g = xhycom.section_data(glorys, resolved_g, z_dim="depth")
    >>> xhycom.section_plot(sec_g, "thetao", flip_x=True, depth_max=3000)
    """
    resolved = _ensure_resolved(transect, grid)

    y_dim = resolved.y_dim
    x_dim = resolved.x_dim

    sel_vars = (
        variables
        if variables is not None
        else [v for v in ds.data_vars if y_dim in ds[v].dims and x_dim in ds[v].dims]
    )

    # For regular (1-D coordinate) grids — e.g. HYCOM regridded to GLORYS — the
    # stored integer indices j/i were computed against the dataset that was passed
    # to resolve().  If ds is a spatial subset of that dataset (as when regrid()
    # trims the target to the source bounding box), the positional offsets are
    # wrong.  Use label-based selection on the coordinate values instead: the
    # actual lat/lon of each T-cell is stored in cell_lat/cell_lon and is
    # unambiguous regardless of which subset of the grid ds covers.
    if y_dim in ds.coords and ds[y_dim].ndim == 1:
        lat_idx = xr.DataArray(resolved.cell_lat, dims="section")
        lon_idx = xr.DataArray(resolved.cell_lon, dims="section")
        out = xr.Dataset(
            {
                v: ds[v].sel({y_dim: lat_idx, x_dim: lon_idx}, method="nearest")
                for v in sel_vars
                if v in ds
            }
        )
    else:
        j = xr.DataArray(resolved.j, dims="section")
        i = xr.DataArray(resolved.i, dims="section")
        out = xr.Dataset(
            {v: ds[v].isel({y_dim: j, x_dim: i}) for v in sel_vars if v in ds}
        )
    # Normalise dim order: section must be last so section_plot sees (k, section).
    # Datasets where horizontal dims precede vertical (e.g. regridded on depth levels)
    # otherwise come out (section, depth) instead of (depth, section).
    for v in list(out.data_vars):
        da = out[v]
        if "section" in da.dims and da.dims[-1] != "section":
            out[v] = da.transpose(..., "section")
    out = out.assign_coords(distance_km=("section", resolved.distance_km))
    out["distance_km"].attrs = {
        "long_name": "distance along section",
        "units": "km",
    }

    if thknss_var in ds:
        if y_dim in ds.coords and ds[y_dim].ndim == 1:
            thk = ds[thknss_var].sel({y_dim: lat_idx, x_dim: lon_idx}, method="nearest")
        else:
            thk = ds[thknss_var].isel({y_dim: j, x_dim: i})
        if thk.attrs.get("units") != "m":
            thk = thk / _ONEM
        depth_m = thk.cumsum(dim=k_dim) - thk / 2.0
        depth_m.attrs = {"long_name": "depth of layer centre", "units": "m"}
        out["depth_m"] = depth_m
    elif z_dim is not None and z_dim in ds.coords:
        depth_m = ds[z_dim].astype(float)
        depth_m.attrs = {"long_name": "depth", "units": "m"}
        out["depth_m"] = depth_m

    return out


def section_flux_density(
    ds: xr.Dataset,
    transect: Transect | ResolvedTransect,
    grid: xr.Dataset | str | None = None,
    *,
    u_var: str = "u-vel.",
    v_var: str = "v-vel.",
    t_var: str = "temp",
    s_var: str | None = None,
    thknss_var: str = "thknss",
    k_dim: str = "k",
    s_ref: float = _SREF,
    t_ref: float = _TREF,
    rho0: float = _RHO0,
    cp: float = _CP,
    constraints: dict[str, tuple[Literal["lt", "le", "gt", "ge", "eq"], float]]
    | None = None,
) -> xr.Dataset:
    """Per-face, per-layer signed transport density along the section.

    Returns ``v_normal × dz`` and its tracer-weighted variants at each C-grid
    face crossed by the section, for each vertical layer.  Multiplying any
    density variable by ``face_width_m`` and summing over ``k`` and ``face``
    recovers the corresponding scalar transport from :func:`transport`.
    Plotting these quantities as colour-filled sections reveals where positive
    and negative contributions arise.

    Parameters
    ----------
    ds:
        HYCOM Dataset containing velocity, thickness, and optionally tracers.
    transect:
        :class:`~xhycom.Transect` or pre-resolved
        :class:`~xhycom.ResolvedTransect`.
    grid:
        Required when *transect* is an unresolved :class:`~xhycom.Transect`.
    u_var, v_var:
        Velocity variable names (total current — call
        :func:`xhycom.postprocess` first if needed).
    t_var:
        Temperature variable.  ``heat_flux_density`` is omitted if absent.
    s_var:
        Salinity variable.  Auto-detected (``salin`` then ``saln``) when
        ``None``.  ``salt_flux_density`` and ``fw_flux_density`` are omitted
        if absent.
    thknss_var:
        Layer-thickness variable (metres or Pa; auto-detected).
    k_dim:
        Name of the vertical layer dimension.
    s_ref:
        Freshwater reference salinity [PSU].  Default 34.8.
    t_ref:
        Heat-transport reference temperature [°C].  Default 0.0.
    rho0:
        Reference density [kg m⁻³].  Default 1025.
    cp:
        Specific heat [J kg⁻¹ K⁻¹].  Default 3996.
    constraints:
        Optional tracer constraints (same syntax as :func:`transport`) to
        mask specific water masses.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(k, face)`` and coordinates:

        * ``distance_km``     — position of each face along the section [km]
        * ``face_width_m``    — face width perpendicular to transport [m]

        Data variables (always present):

        * ``flux_density``      — v_normal × dz  [m² s⁻¹]
        * ``depth_m``           — depth of layer centre  [m]

        Data variables (present when the corresponding tracer is found):

        * ``heat_flux_density`` — (T − T_ref) × v_normal × dz × ρ₀ × cp  [W m⁻¹]
        * ``salt_flux_density`` — S × v_normal × dz × ρ₀ / 1000  [kg m⁻¹ s⁻¹]
        * ``fw_flux_density``   — (s_ref − S)/s_ref × v_normal × dz  [m² s⁻¹]

        Multiply any density variable by ``face_width_m`` and sum over ``k``
        and ``face`` to recover the total transport (in W, kg s⁻¹, or m³ s⁻¹
        respectively; apply the same scale factors as :func:`transport` for Sv
        or TW).

    Examples
    --------
    >>> sec_flux = xhycom.section_flux_density(ds, resolved)
    >>> xhycom.section_plot(sec_flux, "flux_density", center_zero=True)
    >>> xhycom.section_plot(sec_flux, "heat_flux_density", center_zero=True)
    """
    resolved = _ensure_resolved(transect, grid)
    if not resolved.has_face_data:
        raise ValueError(
            "ResolvedTransect has no face data. Use Transect.resolve(grid) first."
        )

    for var in (u_var, v_var, thknss_var):
        if var not in ds:
            raise ValueError(f"Required variable {var!r} not found in dataset.")

    _check_velocity_complete(ds, u_var, v_var)

    if s_var is None:
        s_var = "salin" if "salin" in ds else ("saln" if "saln" in ds else None)

    compute_heat = t_var in ds
    compute_salt = s_var is not None and s_var in ds

    u_mask = resolved.face_type == 0
    v_mask = resolved.face_type == 1
    thk = _thknss_m(ds, thknss_var)

    fd_parts: list[xr.DataArray] = []
    heat_parts: list[xr.DataArray] = []
    salt_parts: list[xr.DataArray] = []
    fw_parts: list[xr.DataArray] = []
    depth_parts: list[xr.DataArray] = []
    dist_parts: list[np.ndarray] = []
    width_parts: list[np.ndarray] = []

    for mask, vel_var in ((u_mask, u_var), (v_mask, v_var)):
        if not mask.any():
            continue
        fj, fi, sgn, w, t1j, t1i, t2j, t2i = _face_arrays(resolved, mask, "face")
        vel = ds[vel_var].isel(y=fj, x=fi) * sgn  # (..., k, face)
        thk_face = 0.5 * (thk.isel(y=t1j, x=t1i) + thk.isel(y=t2j, x=t2i))
        fd = vel * thk_face  # m² s⁻¹
        if constraints:
            cmask = _constraint_mask(ds, constraints, t1j, t1i, t2j, t2i, "face")
            fd = fd.where(cmask, 0.0)
        depth = thk_face.cumsum(dim=k_dim) - thk_face / 2.0

        fd_parts.append(fd)
        depth_parts.append(depth)
        dist_parts.append(resolved.face_dist_km[mask])
        width_parts.append(resolved.face_width_m[mask])

        if compute_heat:
            t_face = 0.5 * (ds[t_var].isel(y=t1j, x=t1i) + ds[t_var].isel(y=t2j, x=t2i))
            heat_parts.append(fd * (t_face - t_ref) * rho0 * cp)  # W m⁻¹

        if compute_salt:
            s_face = 0.5 * (ds[s_var].isel(y=t1j, x=t1i) + ds[s_var].isel(y=t2j, x=t2i))
            salt_parts.append(fd * s_face * rho0 / 1000.0)  # kg m⁻¹ s⁻¹
            fw_parts.append(fd * (s_ref - s_face) / s_ref)  # m² s⁻¹

    flux_all = xr.concat(fd_parts, dim="face")
    depth_all = xr.concat(depth_parts, dim="face")
    dist_all = np.concatenate(dist_parts)
    width_all = np.concatenate(width_parts)

    order = np.argsort(dist_all)
    idx = xr.DataArray(order, dims="face")
    flux_all = flux_all.isel(face=idx)
    depth_all = depth_all.isel(face=idx)
    dist_sorted = dist_all[order]
    width_sorted = width_all[order]

    out = xr.Dataset(
        {
            "flux_density": _attach_attrs(
                flux_all, "volume transport density", "m2 s-1"
            ),
            "depth_m": _attach_attrs(depth_all, "depth of layer centre", "m"),
        }
    )

    if compute_heat:
        out["heat_flux_density"] = _attach_attrs(
            xr.concat(heat_parts, dim="face").isel(face=idx),
            "heat transport density",
            "W m-1",
        )
    if compute_salt:
        out["salt_flux_density"] = _attach_attrs(
            xr.concat(salt_parts, dim="face").isel(face=idx),
            "salt transport density",
            "kg m-1 s-1",
        )
        out["fw_flux_density"] = _attach_attrs(
            xr.concat(fw_parts, dim="face").isel(face=idx),
            "freshwater transport density",
            "m2 s-1",
        )

    out = out.assign_coords(
        distance_km=("face", dist_sorted),
        face_width_m=("face", width_sorted),
    )
    out["distance_km"].attrs = {"long_name": "distance along section", "units": "km"}
    out["face_width_m"].attrs = {"long_name": "face width", "units": "m"}
    return out
