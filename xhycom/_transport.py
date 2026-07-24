"""Section transports from HYCOM output using exact C-grid face velocities.

:func:`transport` computes volume, heat, salt, and freshwater transports
through a :class:`~xhycom.Transect` that has been resolved against a HYCOM
grid via :meth:`~xhycom.Transect.resolve`.  The calculation uses native
``u-vel.`` and ``v-vel.`` directly at the staggered U- and V-face locations —
no rotation or interpolation to T-points is needed.  Layer thickness and
tracer values are averaged from the two T-cells that straddle each face.

Sign convention
---------------
Positive transport is rightward when walking from the first transect waypoint
to the last (see :class:`~xhycom.Transect`).

Units
-----

| Quantity        | Output variable  | Unit    |
|-----------------|------------------|---------|
| Volume          | ``volume_sv``    | Sv      |
| Heat            | ``heat_tw``      | TW      |
| Salt            | ``salt_kgs``     | kg s⁻¹  |
| Freshwater      | ``fw_sv``        | Sv      |

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

        * ``volume_sv``  — volume transport in Sv
        * ``heat_tw``    — heat transport in TW  (only if *t_var* is in *ds*)
        * ``salt_kgs``   — salt transport in kg s⁻¹  (only if salinity found)
        * ``fw_sv``      — freshwater transport in Sv  (only if salinity found)

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
    out_vars["volume_sv"] = _attach_attrs(
        vol * 1e-6, "volume transport", "Sv"
    )

    if compute_heat:
        heat_u = _face_tracer_flux(ds, resolved, u_mask, "uf", u_var, thknss_var, k_dim, t_var, t_ref, constraints)
        heat_v = _face_tracer_flux(ds, resolved, v_mask, "vf", v_var, thknss_var, k_dim, t_var, t_ref, constraints)
        heat = (heat_u + heat_v) * rho0 * cp  # W
        out_vars["heat_tw"] = _attach_attrs(heat * 1e-12, "heat transport", "TW")

    if compute_salt:
        salt_u = _face_tracer_flux(ds, resolved, u_mask, "uf", u_var, thknss_var, k_dim, s_var, 0.0, constraints)
        salt_v = _face_tracer_flux(ds, resolved, v_mask, "vf", v_var, thknss_var, k_dim, s_var, 0.0, constraints)
        # Salinity in PSU (g kg⁻¹): multiply by rho0 and convert PSU→kg/kg (÷1000)
        salt = (salt_u + salt_v) * rho0 / 1000.0  # kg s⁻¹
        out_vars["salt_kgs"] = _attach_attrs(salt, "salt transport", "kg s-1")

        fw_u = _face_tracer_flux(ds, resolved, u_mask, "uf", u_var, thknss_var, k_dim, s_var, 0.0, constraints, fw_sref=s_ref)
        fw_v = _face_tracer_flux(ds, resolved, v_mask, "vf", v_var, thknss_var, k_dim, s_var, 0.0, constraints, fw_sref=s_ref)
        fw = fw_u + fw_v  # m³ s⁻¹
        out_vars["fw_sv"] = _attach_attrs(fw * 1e-6, "freshwater transport", "Sv")

    return xr.Dataset(out_vars)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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
