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

# Pa per metre of water column: rho0 * g = 1000 * 9.806 (HYCOM's ``onem``)
_ONEM: float = 9806.0

_RHO0: float = 1025.0   # reference density  kg m⁻³
_CP:   float = 3996.0   # specific heat       J kg⁻¹ K⁻¹
_SREF: float = 34.8     # freshwater reference salinity  PSU
_TREF: float = 0.0      # heat-transport reference temperature  °C

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
    s_ref: float = _SREF,
    t_ref: float = _TREF,
    rho0: float = _RHO0,
    cp: float = _CP,
    constraints: dict[str, tuple[Literal["lt", "le", "gt", "ge", "eq"], float]] | None = None,
) -> xr.Dataset:
    """Compute section transports through a HYCOM C-grid transect.

    Parameters
    ----------
    ds:
        Dataset containing ``u-vel.``, ``v-vel.``, ``thknss``, and optionally
        ``temp`` / ``salin``.  Velocities must be the **total** current (call
        :func:`xhycom.postprocess` first if needed).  Thickness must be in
        metres or Pa (auto-detected from the ``units`` attribute).
    transect:
        A :class:`~xhycom.Transect` or a pre-resolved
        :class:`~xhycom.ResolvedTransect`.  If a bare ``Transect`` is passed,
        *grid* must also be provided.
    grid:
        Required when *transect* is an unresolved :class:`~xhycom.Transect`.
        Ignored when *transect* is already a :class:`~xhycom.ResolvedTransect`.
    u_var:
        Name of the eastward (model +i) velocity variable.
    v_var:
        Name of the northward (model +j) velocity variable.
    t_var:
        Temperature variable name.  Heat transport is skipped if absent.
    s_var:
        Salinity variable name.  Auto-detected (``salin`` then ``saln``) when
        ``None``.  Salt and FW transport are skipped if absent.
    thknss_var:
        Layer-thickness variable name.
    k_dim:
        Name of the vertical layer dimension.
    s_ref:
        Freshwater reference salinity in PSU.  Default 34.8.
    t_ref:
        Heat-transport reference temperature in °C.  Default 0.0.
    rho0:
        Reference density in kg m⁻³.  Default 1025.
    cp:
        Specific heat capacity in J kg⁻¹ K⁻¹.  Default 3996.
    constraints:
        Optional dict of ``{variable: (operator, threshold)}`` pairs that
        restrict which face-layer cells contribute to transport.  Operator is
        one of ``"lt"``, ``"le"``, ``"gt"``, ``"ge"``, ``"eq"``.  Multiple
        constraints are AND-ed.  Tracer values at each face are the average of
        the two neighbouring T-cells.

    Returns
    -------
    xr.Dataset
        Dataset with a ``time`` dimension (if present in *ds*) and variables:

        * ``volume``  — volume transport in Sv
        * ``heat``    — heat transport in TW  (only if *t_var* is in *ds*)
        * ``salt``   — salt transport in kg s⁻¹  (only if salinity found)
        * ``fw``      — freshwater transport in Sv  (only if salinity found)

        Each variable carries ``long_name`` and ``units`` attributes.

    Raises
    ------
    ValueError
        If *transect* has no face data (i.e. was not resolved with
        :meth:`~xhycom.Transect.resolve`), or if required variables are absent.

    Examples
    --------
    >>> grid = xhycom.open_dataset("regional.grid")
    >>> ds   = xhycom.open_mfdataset("data/", grid=grid, postprocess=True,
    ...            variables=["u-vel.", "v-vel.", "temp", "salin", "thknss"])
    >>> fs   = xhycom.Transect.named("fram_strait")
    >>> tr   = xhycom.transport(ds, fs, grid=grid)
    >>> tr_aw = xhycom.transport(ds, fs, grid=grid,
    ...             constraints={"temp": ("gt", 2.0)})
    """
    # ------------------------------------------------------------------
    # Resolve transect if needed
    # ------------------------------------------------------------------
    resolved = _ensure_resolved(transect, grid)
    if not resolved.has_face_data:
        raise ValueError(
            "ResolvedTransect has no face data. Use Transect.resolve(grid) "
            "to obtain exact C-grid faces before calling transport()."
        )

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for var in (u_var, v_var, thknss_var):
        if var not in ds:
            raise ValueError(f"Required variable {var!r} not found in dataset.")

    _check_velocity_complete(ds, u_var, v_var)

    if s_var is None:
        s_var = "salin" if "salin" in ds else ("saln" if "saln" in ds else None)

    compute_heat = t_var in ds
    compute_salt = s_var is not None and s_var in ds

    if constraints:
        for cvar in constraints:
            if cvar not in ds:
                raise ValueError(
                    f"Constraint variable {cvar!r} not found in dataset."
                )
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

    vol_u = _face_volume_flux(ds, resolved, u_mask, "uf", u_var, thknss_var, k_dim, constraints)
    vol_v = _face_volume_flux(ds, resolved, v_mask, "vf", v_var, thknss_var, k_dim, constraints)

    vol = vol_u + vol_v  # (time?,) in m³ s⁻¹
    out_vars["volume"] = _attach_attrs(
        vol * 1e-6, "volume transport", "Sv"
    )

    if compute_heat:
        heat_u = _face_tracer_flux(ds, resolved, u_mask, "uf", u_var, thknss_var, k_dim, t_var, t_ref, constraints)
        heat_v = _face_tracer_flux(ds, resolved, v_mask, "vf", v_var, thknss_var, k_dim, t_var, t_ref, constraints)
        heat = (heat_u + heat_v) * rho0 * cp  # W
        out_vars["heat"] = _attach_attrs(heat * 1e-12, "heat transport", "TW")

    if compute_salt:
        salt_u = _face_tracer_flux(ds, resolved, u_mask, "uf", u_var, thknss_var, k_dim, s_var, 0.0, constraints)
        salt_v = _face_tracer_flux(ds, resolved, v_mask, "vf", v_var, thknss_var, k_dim, s_var, 0.0, constraints)
        # Salinity in PSU (g kg⁻¹): multiply by rho0 and convert PSU→kg/kg (÷1000)
        salt = (salt_u + salt_v) * rho0 / 1000.0  # kg s⁻¹
        out_vars["salt"] = _attach_attrs(salt, "salt transport", "kg s-1")

        fw_u = _face_tracer_flux(ds, resolved, u_mask, "uf", u_var, thknss_var, k_dim, s_var, 0.0, constraints, fw_sref=s_ref)
        fw_v = _face_tracer_flux(ds, resolved, v_mask, "vf", v_var, thknss_var, k_dim, s_var, 0.0, constraints, fw_sref=s_ref)
        fw = fw_u + fw_v  # m³ s⁻¹
        out_vars["fw"] = _attach_attrs(fw * 1e-6, "freshwater transport", "Sv")

    return xr.Dataset(out_vars)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_BTROP_PAIRS = (("u-vel.", "u_btrop"), ("v-vel.", "v_btrop"))


def _check_velocity_complete(ds: xr.Dataset, u_var: str, v_var: str) -> None:
    """Raise if velocity looks baroclinic-only (postprocess not applied on archv)."""
    import warnings as _warnings
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
        raise ValueError(
            "grid= is required when transect is an unresolved Transect."
        )
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
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Extract DataArrays for the subset of faces given by *mask*."""
    fj  = xr.DataArray(resolved.face_j[mask],     dims=face_dim)
    fi  = xr.DataArray(resolved.face_i[mask],     dims=face_dim)
    sgn = xr.DataArray(resolved.face_sign[mask],  dims=face_dim)
    w   = xr.DataArray(resolved.face_width_m[mask], dims=face_dim)
    t1j = xr.DataArray(resolved.face_t1_j[mask],  dims=face_dim)
    t1i = xr.DataArray(resolved.face_t1_i[mask],  dims=face_dim)
    t2j = xr.DataArray(resolved.face_t2_j[mask],  dims=face_dim)
    t2i = xr.DataArray(resolved.face_t2_i[mask],  dims=face_dim)
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

    vel  = ds[vel_var].isel(y=fj, x=fi)                          # (…, k, fd)
    thk  = _thknss_m(ds, thknss_var)
    thk1 = thk.isel(y=t1j, x=t1i)                                # (…, k, fd)
    thk2 = thk.isel(y=t2j, x=t2i)
    thk_face = 0.5 * (thk1 + thk2)

    flux = sgn * vel * thk_face * w                               # (…, k, fd)

    if constraints:
        flux = flux.where(_constraint_mask(ds, constraints, t1j, t1i, t2j, t2i, face_dim), 0.0)

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

    vel  = ds[vel_var].isel(y=fj, x=fi)
    thk  = _thknss_m(ds, thknss_var)
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
        flux = flux.where(_constraint_mask(ds, constraints, t1j, t1i, t2j, t2i, face_dim), 0.0)

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
            "lt": val <  threshold,
            "le": val <= threshold,
            "gt": val >  threshold,
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
) -> xr.Dataset:
    """Extract hydrographic data along the section T-cell path.

    Selects every variable with ``(y, x)`` dimensions from *ds* at the T-cell
    positions identified by :meth:`~xhycom.Transect.resolve`, producing a
    ``(section, k)`` Dataset suitable for cross-section plots.  A
    ``distance_km`` coordinate is attached along the ``section`` dimension,
    and ``depth_m`` (depth of each layer centre from the sea surface) is added
    when *thknss_var* is present.

    Parameters
    ----------
    ds:
        HYCOM Dataset (archive or multi-file) containing the fields to extract.
    transect:
        A :class:`~xhycom.Transect` or pre-resolved
        :class:`~xhycom.ResolvedTransect`.
    grid:
        Required when *transect* is an unresolved :class:`~xhycom.Transect`.
    variables:
        Explicit list of variable names to extract.  When ``None`` every
        variable with ``y`` and ``x`` dimensions is included.
    thknss_var:
        Layer-thickness variable used to derive ``depth_m``.
    k_dim:
        Name of the vertical layer dimension.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``section`` (and ``k`` / ``time`` when
        present) and coordinate ``distance_km`` along the section.
        ``depth_m`` is included as a data variable when *thknss_var* is
        present in *ds*.

    Examples
    --------
    >>> resolved = xhycom.Transect.named("fram_strait").resolve(grid)
    >>> sec = xhycom.section_data(ds, resolved)
    >>> sec["temp"].plot(x="distance_km", y="depth_m")
    """
    resolved = _ensure_resolved(transect, grid)

    j = xr.DataArray(resolved.j, dims="section")
    i = xr.DataArray(resolved.i, dims="section")

    sel_vars = variables if variables is not None else [
        v for v in ds.data_vars if "y" in ds[v].dims and "x" in ds[v].dims
    ]

    out = xr.Dataset(
        {v: ds[v].isel(y=j, x=i) for v in sel_vars if v in ds}
    )
    out = out.assign_coords(
        distance_km=("section", resolved.distance_km)
    )
    out["distance_km"].attrs = {
        "long_name": "distance along section",
        "units": "km",
    }

    if thknss_var in ds:
        thk = ds[thknss_var].isel(y=j, x=i)
        if thk.attrs.get("units") != "m":
            thk = thk / _ONEM
        depth_m = thk.cumsum(dim=k_dim) - thk / 2.0
        depth_m.attrs = {"long_name": "depth of layer centre", "units": "m"}
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
    constraints: dict[str, tuple[Literal["lt", "le", "gt", "ge", "eq"], float]] | None = None,
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
        vel = ds[vel_var].isel(y=fj, x=fi) * sgn           # (..., k, face)
        thk_face = 0.5 * (thk.isel(y=t1j, x=t1i) + thk.isel(y=t2j, x=t2i))
        fd = vel * thk_face                                  # m² s⁻¹
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
            heat_parts.append(fd * (t_face - t_ref) * rho0 * cp)   # W m⁻¹

        if compute_salt:
            s_face = 0.5 * (ds[s_var].isel(y=t1j, x=t1i) + ds[s_var].isel(y=t2j, x=t2i))
            salt_parts.append(fd * s_face * rho0 / 1000.0)          # kg m⁻¹ s⁻¹
            fw_parts.append(fd * (s_ref - s_face) / s_ref)          # m² s⁻¹

    flux_all = xr.concat(fd_parts, dim="face")
    depth_all = xr.concat(depth_parts, dim="face")
    dist_all = np.concatenate(dist_parts)
    width_all = np.concatenate(width_parts)

    order = np.argsort(dist_all)
    idx = xr.DataArray(order, dims="face")
    flux_all  = flux_all.isel(face=idx)
    depth_all = depth_all.isel(face=idx)
    dist_sorted  = dist_all[order]
    width_sorted = width_all[order]

    out = xr.Dataset(
        {
            "flux_density": _attach_attrs(flux_all, "volume transport density", "m2 s-1"),
            "depth_m": _attach_attrs(depth_all, "depth of layer centre", "m"),
        }
    )

    if compute_heat:
        out["heat_flux_density"] = _attach_attrs(
            xr.concat(heat_parts, dim="face").isel(face=idx),
            "heat transport density", "W m-1",
        )
    if compute_salt:
        out["salt_flux_density"] = _attach_attrs(
            xr.concat(salt_parts, dim="face").isel(face=idx),
            "salt transport density", "kg m-1 s-1",
        )
        out["fw_flux_density"] = _attach_attrs(
            xr.concat(fw_parts, dim="face").isel(face=idx),
            "freshwater transport density", "m2 s-1",
        )

    out = out.assign_coords(
        distance_km=("face", dist_sorted),
        face_width_m=("face", width_sorted),
    )
    out["distance_km"].attrs = {"long_name": "distance along section", "units": "km"}
    out["face_width_m"].attrs = {"long_name": "face width", "units": "m"}
    return out


# ---------------------------------------------------------------------------
# Regular-grid transport
# ---------------------------------------------------------------------------

def transport_tpoint(
    ds: xr.Dataset,
    transect: Transect,
    *,
    u_var: str = "uo",
    v_var: str = "vo",
    t_var: str = "thetao",
    s_var: str | None = None,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    z_dim: str = "depth",
    s_ref: float = _SREF,
    t_ref: float = _TREF,
    rho0: float = _RHO0,
    cp: float = _CP,
    constraints: dict[str, tuple[Literal["lt", "le", "gt", "ge", "eq"], float]] | None = None,
) -> xr.Dataset:
    """Compute section transports for a regular or rectilinear grid dataset.

    Suitable for non-HYCOM products such as GLORYS, EN4 or observations that
    carry eastward/northward velocity components on a latitude–longitude grid.
    The section is sampled by finding the nearest grid cell to each point on
    the transect polyline; velocities are then projected onto the
    section-normal direction and integrated over depth and along-section width.

    Sign convention matches :func:`transport`: positive = rightward when
    walking from the first transect waypoint to the last.

    Parameters
    ----------
    ds:
        Dataset with ``u_var``, ``v_var`` and a depth coordinate ``z_dim``.
        Velocities must be in true east/north geographic components (m s⁻¹).
    transect:
        An unresolved :class:`~xhycom.Transect` (grid-independent geometry).
    u_var:
        Eastward velocity variable name.  Default ``"uo"`` (GLORYS convention).
    v_var:
        Northward velocity variable name.  Default ``"vo"``.
    t_var:
        Temperature variable.  Heat transport is skipped if absent.
        Default ``"thetao"`` (GLORYS convention).
    s_var:
        Salinity variable.  Auto-detected from common names when ``None``.
    lat_dim:
        Name of the latitude dimension in *ds*.
    lon_dim:
        Name of the longitude dimension in *ds*.
    z_dim:
        Name of the depth dimension in *ds*.  Layer thicknesses are derived
        from finite differences of the coordinate values.
    s_ref:
        Freshwater reference salinity [PSU].  Default 34.8.
    t_ref:
        Heat-transport reference temperature [°C].  Default 0.0.
    rho0:
        Reference density [kg m⁻³].  Default 1025.
    cp:
        Specific heat [J kg⁻¹ K⁻¹].  Default 3996.
    constraints:
        Optional ``{variable: (operator, threshold)}`` pairs that zero out
        contributions where the condition is not met (same syntax as
        :func:`transport`).

    Returns
    -------
    xr.Dataset
        Same structure as :func:`transport`: ``volume``, ``heat``
        (when *t_var* found), ``salt`` and ``fw`` (when salinity
        found), optionally with a ``time`` dimension.

    Raises
    ------
    ImportError
        If ``scipy`` is not installed.
    ValueError
        If required variables are absent or the section misses the grid.

    Examples
    --------
    >>> glorys = xr.open_dataset("glorys.nc")
    >>> fs = xhycom.Transect.named("fram_strait")
    >>> tr_g = xhycom.transport_tpoint(glorys, fs, lat_dim="latitude",
    ...                                 lon_dim="longitude", z_dim="depth")
    """
    try:
        from scipy.spatial import KDTree
    except ImportError as exc:
        raise ImportError(
            "scipy is required for transport_tpoint.\n"
            "Install it with: pip install scipy"
        ) from exc

    for var in (u_var, v_var):
        if var not in ds:
            raise ValueError(f"Required variable {var!r} not found in dataset.")

    if s_var is None:
        for name in ("so", "salin", "saln", "salinity", "sal"):
            if name in ds:
                s_var = name
                break

    compute_heat = t_var in ds
    compute_salt = s_var is not None and s_var in ds

    if constraints:
        for cvar, (op, _) in constraints.items():
            if cvar not in ds:
                raise ValueError(f"Constraint variable {cvar!r} not found.")
            if op not in _OPS:
                raise ValueError(f"Unknown operator {op!r}. Use: {sorted(_OPS)}")

    # ------------------------------------------------------------------
    # Build KDTree from the dataset's lat/lon coordinates
    # ------------------------------------------------------------------
    from ._transect import (
        _sample_polyline,
        _cumulative_distance_km,
        _cell_widths_km,
        _section_bearings,
        _to_xyz,
    )

    lat_vals = ds[lat_dim].values  # 1-D (ny,) or 2-D (ny, nx)
    lon_vals = ds[lon_dim].values

    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
        lon2d, lat2d = lon_vals, lat_vals
    else:
        raise ValueError(
            f"lat_dim {lat_dim!r} and lon_dim {lon_dim!r} must both be 1-D "
            "(rectilinear) or both 2-D (curvilinear)."
        )

    ny, nx = lon2d.shape
    tree = KDTree(_to_xyz(lon2d.ravel(), lat2d.ravel()))

    # ------------------------------------------------------------------
    # Section cells on this grid
    # ------------------------------------------------------------------
    slons, slats = _sample_polyline(transect.lons, transect.lats)
    _, flat_idx = tree.query(_to_xyz(slons, slats))
    j_samp = (flat_idx // nx).astype(np.intp)
    i_samp = (flat_idx % nx).astype(np.intp)

    pairs = np.column_stack([j_samp, i_samp])
    keep = np.concatenate([[True], np.any(pairs[1:] != pairs[:-1], axis=1)])
    j_cells = j_samp[keep]
    i_cells = i_samp[keep]

    if len(j_cells) < 2:
        raise ValueError(
            "Transect intersects fewer than 2 grid cells in the dataset. "
            "Check that the waypoints lie within the dataset domain."
        )

    cell_lons = lon2d[j_cells, i_cells]
    cell_lats = lat2d[j_cells, i_cells]
    widths_m = _cell_widths_km(
        _cumulative_distance_km(cell_lons, cell_lats)
    ) * 1e3
    bearings = _section_bearings(cell_lons, cell_lats)

    # Section-normal direction (rightward): n = (cos θ, -sin θ) in (E, N).
    theta = np.radians(bearings)
    cos_t = xr.DataArray(np.cos(theta), dims="section")
    sin_t = xr.DataArray(np.sin(theta), dims="section")
    w = xr.DataArray(widths_m, dims="section")

    # ------------------------------------------------------------------
    # Index into dataset (handles 1-D or 2-D lat/lon)
    # ------------------------------------------------------------------
    if lat_vals.ndim == 1:
        lat_idx = xr.DataArray(j_cells, dims="section")
        lon_idx = xr.DataArray(i_cells, dims="section")
        sel = {lat_dim: lat_idx, lon_dim: lon_idx}
    else:
        # For 2-D coordinates, use a flat index broadcast to the 2-D grid
        # by wrapping j, i into multi-dim indexing via DataArrays
        lat_idx = xr.DataArray(j_cells, dims="section")
        lon_idx = xr.DataArray(i_cells, dims="section")
        # Both dims of the 2-D array share the same dim names; use vectorised
        # indexing: pass both as DataArrays with matching "section" dim so
        # xarray aligns them.
        dim_y, dim_x = ds[u_var].dims[-2], ds[u_var].dims[-1]
        sel = {dim_y: lat_idx, dim_x: lon_idx}

    u = ds[u_var].isel(**sel)   # (..., z, section)
    v = ds[v_var].isel(**sel)

    v_normal = u * cos_t - v * sin_t   # positive = rightward

    # ------------------------------------------------------------------
    # Layer thicknesses from the depth coordinate
    # ------------------------------------------------------------------
    z_vals = ds[z_dim].values.astype(float)
    z_edges = np.empty(len(z_vals) + 1)
    z_edges[1:-1] = (z_vals[:-1] + z_vals[1:]) / 2.0
    z_edges[0] = z_vals[0] - (z_vals[1] - z_vals[0]) / 2.0
    z_edges[-1] = z_vals[-1] + (z_vals[-1] - z_vals[-2]) / 2.0
    dz = xr.DataArray(np.diff(z_edges), dims=z_dim)

    # ------------------------------------------------------------------
    # Integrate
    # ------------------------------------------------------------------
    def _integrate(da: xr.DataArray) -> xr.DataArray:
        return (da * dz * w).sum(dim=[z_dim, "section"])

    def _tpoint_constraint_mask(t1j: xr.DataArray, t1i: xr.DataArray) -> xr.DataArray | None:
        if not constraints:
            return None
        mask: xr.DataArray | None = None
        for cvar, (op, threshold) in constraints.items():
            val = ds[cvar].isel(**{
                (lat_dim if lat_vals.ndim == 1 else ds[cvar].dims[-2]): t1j,
                (lon_dim if lat_vals.ndim == 1 else ds[cvar].dims[-1]): t1i,
            })
            cond: xr.DataArray = {
                "lt": val < threshold, "le": val <= threshold,
                "gt": val > threshold, "ge": val >= threshold,
                "eq": val == threshold,
            }[op]
            mask = cond if mask is None else (mask & cond)
        return mask

    cmask = _tpoint_constraint_mask(
        xr.DataArray(j_cells, dims="section"),
        xr.DataArray(i_cells, dims="section"),
    )

    vn_masked = v_normal if cmask is None else v_normal.where(cmask, 0.0)

    out_vars: dict[str, xr.DataArray] = {}

    vol = _integrate(vn_masked)
    out_vars["volume"] = _attach_attrs(vol * 1e-6, "volume transport", "Sv")

    if compute_heat:
        t = ds[t_var].isel(**sel)
        heat = _integrate((t - t_ref) * vn_masked) * rho0 * cp
        out_vars["heat"] = _attach_attrs(heat * 1e-12, "heat transport", "TW")

    if compute_salt:
        s = ds[s_var].isel(**sel)
        salt = _integrate(s * vn_masked) * rho0 / 1000.0
        out_vars["salt"] = _attach_attrs(salt, "salt transport", "kg s-1")
        fw = _integrate((s_ref - s) / s_ref * vn_masked)
        out_vars["fw"] = _attach_attrs(fw * 1e-6, "freshwater transport", "Sv")

    return xr.Dataset(out_vars)
