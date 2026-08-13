"""Pre-defined open-boundary sections for HYCOM model domains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ._transect import ResolvedTransect, Transect, _load_grid

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _unwrap_lons(lons: np.ndarray) -> np.ndarray:
    """Unwrap longitude jumps across ±180°."""
    lons = np.asarray(lons, dtype=float).copy()
    diff = np.diff(lons)
    shifts = np.cumsum(np.where(diff > 180, -360.0, np.where(diff < -180, 360.0, 0.0)))
    lons[1:] += shifts
    return lons


def _extend_transect(
    t: Transect, deg_start: float = 0.0, deg_end: float = 0.0
) -> Transect:
    """Extend a Transect by pushing each endpoint along its terminal bearing."""

    def _nudge(
        lon: float, lat: float, lon_ref: float, lat_ref: float, d: float
    ) -> tuple[float, float]:
        brg = np.arctan2(np.radians(lon - lon_ref), np.radians(lat - lat_ref))
        return lon + d * np.sin(brg), lat + d * np.cos(brg)

    lons: list[float] = list(t.lons)
    lats: list[float] = list(t.lats)
    if deg_start:
        lon0, lat0 = _nudge(t.lons[0], t.lats[0], t.lons[1], t.lats[1], deg_start)
        lons, lats = [lon0, *lons], [lat0, *lats]
    if deg_end:
        lon1, lat1 = _nudge(t.lons[-1], t.lats[-1], t.lons[-2], t.lats[-2], deg_end)
        lons, lats = [*lons, lon1], [*lats, lat1]
    return Transect(lons=lons, lats=lats, name=t.name)


def _glorys_path_from_vface_row(
    hycom_section: ResolvedTransect,
    *,
    name: str = "",
    deg_start: float = 0.0,
    deg_end: float = 0.0,
    reverse: bool = False,
) -> Transect:
    """Build a GLORYS-ready ``Transect`` from a HYCOM V-face section.

    Unwraps longitudes across ±180° so the path is continuous, then
    extends each endpoint to ensure the KDTree snap always lands on a
    GLORYS ocean cell even where the two land masks diverge.

    Sign convention: transport is positive in the rightward direction when
    walking along the transect in waypoint order.  Set *reverse=True* to flip
    the path so that positive transport means flow into the domain:

    * **Atlantic boundary** (HYCOM j=2, positive into domain = northward):
      use ``reverse=True``.  Walking east→west makes rightward = north.
    * **Pacific boundary** (HYCOM j=jdm-2, positive into domain = southward):
      use the default ``reverse=False``.  Walking west→east makes
      rightward = south = into the domain.

    Use this **once per new domain** to generate the waypoint coordinates,
    then hardcode the result as a module-level ``Transect`` constant::

        sec = xhycom.tp2_sections(grid, bathy)   # HYCOM sections only
        # Atlantic — reversed so positive = northward = into domain:
        t = _glorys_path_from_vface_row(sec.hycom_atlantic_boundary, deg_start=1, deg_end=1, reverse=True)
        print(list(t.lons))
        print(list(t.lats))
    """
    lons = _unwrap_lons(hycom_section.cell_lon)
    lats = np.asarray(hycom_section.cell_lat, dtype=float)
    if reverse:
        lons, lats = lons[::-1], lats[::-1]
        deg_start, deg_end = deg_end, deg_start
    return _extend_transect(
        Transect(lons=lons, lats=lats, name=name),
        deg_start=deg_start,
        deg_end=deg_end,
    )


# ---------------------------------------------------------------------------
# Fixed GLORYS section paths — shared across domains, independent of HYCOM grid
# ---------------------------------------------------------------------------
#
# Derived from the TP2 HYCOM boundary rows via ``_glorys_path_from_vface_row``
# and verified in docs/boundaries.ipynb.  Because the GLORYS grid covers both
# the TP2 and TP5 domains, the same paths are reused for both — regenerate only
# if the TP2 topo or the GLORYS product changes.

_GLORYS_ATLANTIC: Transect | None = None  # TODO: hardcode after extraction
_GLORYS_PACIFIC: Transect | None = None  # TODO: hardcode after extraction
_GLORYS_ALEUTIAN_ARC = Transect(
    # Reversed (NE→SW) so that rightward = NW = into the domain.
    # Original SW→NE orientation would give positive = SE = out of domain.
    lons=[-160.0, -165.0, -173.74995422],
    lats=[56.0, 54.0, 52.08423615],
    name="aleutian_arc",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class DomainSections:
    """Pre-resolved boundary sections for a HYCOM domain.

    Obtain via :func:`~xhycom.tp2_sections` or :func:`~xhycom.tp5_sections`.

    Attributes
    ----------
    hycom_atlantic_boundary : ResolvedTransect
        Atlantic (southern) V-face section for HYCOM transport
        (face-exact C-grid integration, sign=+1, positive = into domain).
    hycom_pacific_boundary : ResolvedTransect
        Pacific (northern) V-face section for HYCOM transport
        (sign=-1, positive = into domain).
    glorys_atlantic_boundary : ResolvedTransect or None
        Atlantic boundary resolved on the GLORYS grid.
        ``None`` when the factory is called without *glorys_data*.
    glorys_pacific_boundary : ResolvedTransect or None
        Pacific boundary resolved on the GLORYS grid.
    glorys_aleutian_boundary : ResolvedTransect or None
        Short GLORYS-only section that closes a land-mask gap at the Pacific
        corner of the domain.  Present for TP2; ``None`` for domains where no
        such gap exists.
    """

    hycom_atlantic_boundary: ResolvedTransect
    hycom_pacific_boundary: ResolvedTransect
    glorys_atlantic_boundary: ResolvedTransect | None = None
    glorys_pacific_boundary: ResolvedTransect | None = None
    glorys_aleutian_boundary: ResolvedTransect | None = None


#: Backward-compatible alias.
TP2Sections = DomainSections


def _resolve_domain(
    grid: xr.Dataset | str,
    bathy: xr.Dataset | str,
    atlantic_name: str,
    pacific_name: str,
    glorys_atlantic_path: Transect | None,
    glorys_pacific_path: Transect | None,
    glorys_aleutian_path: Transect | None,
    glorys_data: xr.Dataset | None,
    *,
    atlantic_deg_start: float = 1.0,
    atlantic_deg_end: float = 1.0,
    pacific_deg_start: float = 0.0,
    pacific_deg_end: float = 2.0,
) -> DomainSections:
    """Shared builder for any two-boundary HYCOM domain."""
    grid_ds = _load_grid(grid)
    jdm = grid_ds.sizes["y"]

    # j=2 / j=jdm-2: one row inward from the open-boundary ghost face.
    # HYCOM archives store zero layer thickness (thknss) at the boundary
    # ghost row (j=1 / j=jdm-1), so the transport at those faces is always
    # zero.  The first V-face with real prognostic thickness is one row
    # interior; by continuity its transport equals the boundary transport.
    hycom_atlantic = ResolvedTransect.from_vface_row(
        grid_ds, bathy, j=2, sign=+1, name=atlantic_name
    )
    hycom_pacific = ResolvedTransect.from_vface_row(
        grid_ds, bathy, j=jdm - 2, sign=-1, name=pacific_name
    )

    if glorys_data is None:
        return DomainSections(
            hycom_atlantic_boundary=hycom_atlantic,
            hycom_pacific_boundary=hycom_pacific,
        )

    # Fall back to deriving from the HYCOM grid when fixed paths are not yet
    # hardcoded (i.e. the module-level constant is None).
    if glorys_atlantic_path is None:
        # Reversed so walking E→W makes rightward = N = into domain.
        glorys_atlantic_path = _glorys_path_from_vface_row(
            hycom_atlantic,
            name=atlantic_name,
            deg_start=atlantic_deg_start,
            deg_end=atlantic_deg_end,
            reverse=True,
        )
    if glorys_pacific_path is None:
        # Default W→E orientation: rightward = S = into domain from the north.
        glorys_pacific_path = _glorys_path_from_vface_row(
            hycom_pacific,
            name=pacific_name,
            deg_start=pacific_deg_start,
            deg_end=pacific_deg_end,
        )

    resolve_kw: dict[str, str] = dict(lat_var="latitude", lon_var="longitude")
    return DomainSections(
        hycom_atlantic_boundary=hycom_atlantic,
        hycom_pacific_boundary=hycom_pacific,
        glorys_atlantic_boundary=glorys_atlantic_path.resolve(
            glorys_data, **resolve_kw
        ),
        glorys_pacific_boundary=glorys_pacific_path.resolve(glorys_data, **resolve_kw),
        glorys_aleutian_boundary=(
            glorys_aleutian_path.resolve(glorys_data, **resolve_kw)
            if glorys_aleutian_path is not None
            else None
        ),
    )


def tp2_sections(
    grid: xr.Dataset | str,
    bathy: xr.Dataset | str,
    glorys_data: xr.Dataset | None = None,
) -> DomainSections:
    """Build pre-defined open-boundary sections for the TP2 domain.

    Parameters
    ----------
    grid : Dataset or str
        HYCOM ``regional.grid`` Dataset or path.
    bathy : Dataset or str
        Bathymetry Dataset (from :func:`~xhycom.open_dataset`) or path.
    glorys_data : Dataset, optional
        GLORYS dataset with ``latitude`` and ``longitude`` coordinates.
        When provided, the three GLORYS sections are resolved and returned;
        otherwise :attr:`DomainSections.glorys_atlantic_boundary`,
        :attr:`~DomainSections.glorys_pacific_boundary`, and
        :attr:`~DomainSections.glorys_aleutian_boundary` are ``None``.

    Returns
    -------
    DomainSections
        Named boundary sections ready for use with :func:`~xhycom.transport`.

    Examples
    --------
    HYCOM transport only:

    >>> sec = xhycom.tp2_sections(grid, bathy)
    >>> tr_atlantic = xhycom.transport(ds, sec.hycom_atlantic_boundary)
    >>> tr_pacific  = xhycom.transport(ds, sec.hycom_pacific_boundary)

    HYCOM + GLORYS transport:

    >>> sec = xhycom.tp2_sections(grid, bathy, glorys_data)
    >>> tpoint_kw = dict(u_var="uo", v_var="vo", t_var="thetao", s_var="so", z_dim="depth")
    >>> tr_atlantic_g = xhycom.transport(glorys_data, sec.glorys_atlantic_boundary, **tpoint_kw)
    >>> tr_pacific_g  = xhycom.transport(glorys_data, sec.glorys_pacific_boundary,  **tpoint_kw)
    >>> tr_aleutian_g = xhycom.transport(glorys_data, sec.glorys_aleutian_boundary, **tpoint_kw)
    """
    return _resolve_domain(
        grid,
        bathy,
        "tp2_atlantic",
        "tp2_pacific",
        _GLORYS_ATLANTIC,
        _GLORYS_PACIFIC,
        _GLORYS_ALEUTIAN_ARC,
        glorys_data,
        atlantic_deg_start=1.0,
        atlantic_deg_end=1.0,
        pacific_deg_start=0.0,
        pacific_deg_end=2.0,
    )


def tp5_sections(
    grid: xr.Dataset | str,
    bathy: xr.Dataset | str,
    glorys_data: xr.Dataset | None = None,
) -> DomainSections:
    """Build pre-defined open-boundary sections for the TP5 domain.

    Parameters
    ----------
    grid : Dataset or str
        HYCOM ``regional.grid`` Dataset or path.
    bathy : Dataset or str
        Bathymetry Dataset (from :func:`~xhycom.open_dataset`) or path.
    glorys_data : Dataset, optional
        GLORYS dataset with ``latitude`` and ``longitude`` coordinates.

    Returns
    -------
    DomainSections
        Named boundary sections ready for use with :func:`~xhycom.transport`.
    """
    return _resolve_domain(
        grid,
        bathy,
        "tp5_atlantic",
        "tp5_pacific",
        _GLORYS_ATLANTIC,
        _GLORYS_PACIFIC,
        _GLORYS_ALEUTIAN_ARC,
        glorys_data,
        atlantic_deg_start=1.0,
        atlantic_deg_end=1.0,
        pacific_deg_start=0.0,
        pacific_deg_end=2.0,
    )
