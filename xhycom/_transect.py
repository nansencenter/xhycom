"""Section geometry and HYCOM C-grid face resolution for transect transport and plots.

A :class:`Transect` is defined by an ordered sequence of (lon, lat) waypoints and
knows nothing about any particular model grid.  Calling :meth:`Transect.resolve`
against a HYCOM ``regional.grid`` Dataset walks the section polyline through the
curvilinear grid and records:

* The ordered T-cells the section passes through — used by
  :func:`xhycom.section_data` to extract hydrographic sections.
* The exact U- and V-faces the section crosses — used by
  :func:`xhycom.transport` to compute transport without rotating or
  interpolating velocities to T-points.

Sign convention
---------------
Positive transport is defined as flow to the **right** when walking from the
first waypoint to the last.  For a section defined west-to-east at Fram Strait,
positive = eastward-directed flow; to flip the sign reverse the waypoint order
or negate the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    import matplotlib.axes

_R_EARTH_KM: float = 6371.0

# Approximate waypoints for common Arctic / sub-Arctic sections.
# Convention: positive transport is rightward when walking from first to last
# waypoint.
# note: the section definitions are verified for the TP2 and TP5 domains only
_NAMED_SECTIONS: dict[str, tuple[list[float], list[float]]] = {
    "fram_strait": ([15.0, -20.0], [79.0, 79.0]),
    "denmark_strait": ([-22.68, -37.0], [66.0, 66.1]),
    "bering_strait": ([-166.0, -171.0], [66.0, 66.0]),
    "lancaster_sound": ([-82.0, -82.0], [73.4, 74.9]),
    "jones_strait": ([-80.5, -80.1], [75.2, 76.6]),
    "robeson_channel": ([-71.0, -77.0], [78.0, 78.1]),
    "hudson_strait": ([-64.7, -65.0], [60.0, 63.0]),
    "kola_section": ([33.5, 33.5], [69.0, 74.0]),
    "kara_gate": ([58.8, 57.4], [70.4, 70.6]),
    "barents_opening": ([26.5, 15.5], [69.5, 77.9]),
    "svinoy": ([5.5, -7.0], [62.0, 62.3]),
    "gimsoy": ([5.0, 16.2], [70.5, 68.7]),
    "fsc": ([-1.4, -7.0], [60.2, 62.3]),
    "barents_kara_northern_boundary": ([108.5, 99.5, 15.0], [76.0, 79.0, 79.0]),
    "davis_strait": ([-53.0, -63.0], [67.0, 66.5]),
}

# Face-type sentinel values stored in ResolvedTransect.face_type
_FACE_U: np.uint8 = np.uint8(0)
_FACE_V: np.uint8 = np.uint8(1)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _haversine_km(
    lon1: float | np.ndarray,
    lat1: float | np.ndarray,
    lon2: float | np.ndarray,
    lat2: float | np.ndarray,
) -> float | np.ndarray:
    """Great-circle distance in km between (lon1, lat1) and (lon2, lat2)."""
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dr = r2 - r1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dr / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dl / 2) ** 2
    return 2.0 * _R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _forward_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Forward bearing in degrees clockwise from north."""
    dl = np.radians(lon2 - lon1)
    r1, r2 = np.radians(lat1), np.radians(lat2)
    x = np.sin(dl) * np.cos(r2)
    y = np.cos(r1) * np.sin(r2) - np.sin(r1) * np.cos(r2) * np.cos(dl)
    return float(np.degrees(np.arctan2(x, y)) % 360.0)


def _to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convert lon/lat (degrees) to unit-sphere Cartesian, shape (..., 3)."""
    lr = np.radians(np.asarray(lat, dtype=float))
    nr = np.radians(np.asarray(lon, dtype=float))
    return np.stack(
        [np.cos(lr) * np.cos(nr), np.cos(lr) * np.sin(nr), np.sin(lr)], axis=-1
    )


def _sample_polyline(
    lons: Sequence[float],
    lats: Sequence[float],
    step_km: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Densely oversample a multi-segment polyline at *step_km* intervals.

    Parameters
    ----------
    lons, lats:
        Waypoint coordinates in degrees.
    step_km:
        Target spacing between output samples in km.  Default 0.5 km is fine
        for HYCOM grids down to 1/12° resolution (~9 km).

    Returns
    -------
    tuple of two 1-D arrays: (lons_out, lats_out).
    """
    la = np.asarray(lons, dtype=float)
    lb = np.asarray(lats, dtype=float)
    out_lon: list[np.ndarray] = [la[:1]]
    out_lat: list[np.ndarray] = [lb[:1]]
    for k in range(len(la) - 1):
        d_km = _haversine_km(la[k], lb[k], la[k + 1], lb[k + 1])
        n = max(2, int(np.ceil(d_km / step_km)) + 1)
        t = np.linspace(0.0, 1.0, n)[1:]  # skip first (already in list)
        out_lon.append(la[k] + t * (la[k + 1] - la[k]))
        out_lat.append(lb[k] + t * (lb[k + 1] - lb[k]))
    return np.concatenate(out_lon), np.concatenate(out_lat)


def _cumulative_distance_km(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance in km from the first point."""
    if len(lons) == 1:
        return np.zeros(1)
    segs = _haversine_km(lons[:-1], lats[:-1], lons[1:], lats[1:])
    return np.concatenate([[0.0], np.cumsum(segs)])


def _cell_widths_km(distances: np.ndarray) -> np.ndarray:
    """Trapezoidal cell widths: each cell owns the segment between midpoints to neighbours."""
    n = len(distances)
    if n == 1:
        return np.zeros(1)
    d = np.diff(distances)
    w = np.empty(n)
    w[0] = d[0] / 2.0
    w[-1] = d[-1] / 2.0
    if n > 2:
        w[1:-1] = (d[:-1] + d[1:]) / 2.0
    return w


def _section_bearings(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Local section bearing (°CW from N) at each cell via central differences."""
    n = len(lons)
    b = np.empty(n)
    for k in range(n):
        k0 = max(0, k - 1)
        k1 = min(n - 1, k + 1)
        b[k] = (
            _forward_bearing(lons[k0], lats[k0], lons[k1], lats[k1])
            if k0 != k1
            else 0.0
        )
    return b


def _faces_from_waypoints(
    plon: np.ndarray,
    plat: np.ndarray,
    scuy: np.ndarray,
    scvx: np.ndarray,
    wp_lons: np.ndarray,
    wp_lats: np.ndarray,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[float],
    list[int],
    list[int],
    list[int],
    list[int],
    list[float],
    list[float],
]:
    """Identify C-grid faces crossed by a section via hemisphere mask.

    This is the MSCPROGS method (Liseter 2008): for each waypoint-pair segment
    the grid is divided into two hemispheres by the great-circle plane through
    the endpoints.  A vectorised telescopic sum of the resulting Boolean mask
    produces ±1 flags at the U- and V-faces that lie on the segment boundary.
    This correctly identifies V-faces (north-south velocity) for east-west
    segments and U-faces (east-west velocity) for north-south segments — the
    T-cell-step approach used previously produced the wrong face type for
    axis-aligned segments.

    Sign convention: positive flag when the face's positive-normal direction
    (+i for U, +j for V) points to the right-hand side of the section (i.e.
    flow in that direction contributes positively to the transport, matching
    xhycom's "rightward when walking from first to last waypoint" convention).
    """
    jdm, idm = plon.shape
    xyz_all = _to_xyz(plon.ravel(), plat.ravel()).reshape(jdm, idm, 3)

    ftype_l: list[int] = []
    fj_l: list[int] = []
    fi_l: list[int] = []
    fsign_l: list[float] = []
    ft1j_l: list[int] = []
    ft1i_l: list[int] = []
    ft2j_l: list[int] = []
    ft2i_l: list[int] = []
    fwidth_l: list[float] = []
    fdist_l: list[float] = []

    cumdist = 0.0
    for seg in range(len(wp_lons) - 1):
        lon1, lat1 = float(wp_lons[seg]), float(wp_lats[seg])
        lon2, lat2 = float(wp_lons[seg + 1]), float(wp_lats[seg + 1])

        rvec1 = _to_xyz(np.array([lon1]), np.array([lat1]))[0]  # (3,)
        rvec2 = _to_xyz(np.array([lon2]), np.array([lat2]))[0]
        nvec = np.cross(rvec1, rvec2)  # normal to section great-circle plane

        # Pure hemisphere dot products — used later to filter extent-boundary faces.
        # A cell's dot product with nvec < 0 means it's on the right-hand side of
        # the section (the "masked" side). We keep only faces where the unmasked
        # T-cell is genuinely on the other side (dot ≥ 0), not merely excluded by
        # the extent clip.
        dot_hemi: np.ndarray = xyz_all @ nvec  # (jdm, idm), no clipping/zeroing

        # Full mask: hemisphere AND extent AND not a domain boundary row/column.
        mask: np.ndarray = dot_hemi < 0
        cp1 = np.cross(rvec1, xyz_all)  # (jdm, idm, 3)
        cp2 = np.cross(xyz_all, rvec2)
        mask &= np.einsum("...k,...k->...", cp1, cp2) >= -1e-12
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False

        mi = mask.astype(np.int8)

        # Vectorised telescopic sum.
        # flagu[j,i] = mi[j,i] - mi[j,i-1]  (west face of T[j,i], u-velocity)
        # flagv[j,i] = mi[j,i] - mi[j-1,i]  (south face of T[j,i], v-velocity)
        # Interior faces between two masked cells cancel; only boundary faces survive.
        flagu = np.zeros((jdm, idm), dtype=np.int8)
        flagu[:, 1:] = mi[:, 1:] - mi[:, :-1]
        flagu[:, 0] = mi[:, 0]  # left boundary: mi[j, -1] = 0
        flagu[:, -1] = 0  # right domain boundary (mirror MSCPROGS)

        flagv = np.zeros((jdm, idm), dtype=np.int8)
        flagv[1:, :] = mi[1:, :] - mi[:-1, :]
        flagv[0, :] = mi[0, :]  # bottom boundary: mi[-1, i] = 0
        flagv[-1, :] = 0  # top domain boundary (mirror MSCPROGS)

        # Filter: discard extent-boundary faces.  A face is a true section face
        # only when the unmasked T-cell is on the non-masked side of the great
        # circle (dot_hemi ≥ 0).  Extent-boundary faces have the unmasked cell
        # still on the masked side (dot_hemi < 0) — excluded only by the clip.
        # U-face (j,i): T1=(j,i-1), T2=(j,i).
        #   flagu>0 → T2 masked, T1 unmasked; keep if dot_hemi[T1] ≥ 0.
        #   flagu<0 → T1 masked, T2 unmasked; keep if dot_hemi[T2] ≥ 0.
        # V-face (j,i): T1=(j-1,i), T2=(j,i).
        #   flagv>0 → T2 masked, T1 unmasked; keep if dot_hemi[T1] ≥ 0.
        #   flagv<0 → T1 masked, T2 unmasked; keep if dot_hemi[T2] ≥ 0.

        # --- U-faces (flagu ≠ 0): between T(j, i-1) and T(j, i) ---
        uj, ui = np.where(flagu != 0)
        if len(uj):
            ui_m1 = np.maximum(ui.astype(np.intp) - 1, 0)
            unmasked_dot_u = np.where(
                flagu[uj, ui] > 0,
                dot_hemi[uj, ui_m1],  # T1 unmasked
                dot_hemi[uj, ui],  # T2 unmasked
            )
            keep_u = unmasked_dot_u >= 0
            uj, ui, ui_m1 = uj[keep_u], ui[keep_u], ui_m1[keep_u]

        if len(uj):
            flon_u = 0.5 * (plon[uj, ui_m1] + plon[uj, ui])
            flat_u = 0.5 * (plat[uj, ui_m1] + plat[uj, ui])
            fdist_u = cumdist + _haversine_km(lon1, lat1, flon_u, flat_u)
            for k in range(len(uj)):
                ftype_l.append(int(_FACE_U))
                fj_l.append(int(uj[k]))
                fi_l.append(int(ui[k]))
                fsign_l.append(float(flagu[uj[k], ui[k]]))
                ft1j_l.append(int(uj[k]))
                ft1i_l.append(int(ui_m1[k]))
                ft2j_l.append(int(uj[k]))
                ft2i_l.append(int(ui[k]))
                fwidth_l.append(float(scuy[uj[k], ui[k]]))
                fdist_l.append(float(fdist_u[k]))

        # --- V-faces (flagv ≠ 0): between T(j-1, i) and T(j, i) ---
        vj, vi = np.where(flagv != 0)
        if len(vj):
            vj_m1 = np.maximum(vj.astype(np.intp) - 1, 0)
            unmasked_dot_v = np.where(
                flagv[vj, vi] > 0,
                dot_hemi[vj_m1, vi],  # T1 unmasked
                dot_hemi[vj, vi],  # T2 unmasked
            )
            keep_v = unmasked_dot_v >= 0
            vj, vi, vj_m1 = vj[keep_v], vi[keep_v], vj_m1[keep_v]

        if len(vj):
            flon_v = 0.5 * (plon[vj_m1, vi] + plon[vj, vi])
            flat_v = 0.5 * (plat[vj_m1, vi] + plat[vj, vi])
            fdist_v = cumdist + _haversine_km(lon1, lat1, flon_v, flat_v)
            for k in range(len(vj)):
                ftype_l.append(int(_FACE_V))
                fj_l.append(int(vj[k]))
                fi_l.append(int(vi[k]))
                fsign_l.append(float(flagv[vj[k], vi[k]]))
                ft1j_l.append(int(vj_m1[k]))
                ft1i_l.append(int(vi[k]))
                ft2j_l.append(int(vj[k]))
                ft2i_l.append(int(vi[k]))
                fwidth_l.append(float(scvx[vj[k], vi[k]]))
                fdist_l.append(float(fdist_v[k]))

        cumdist += _haversine_km(lon1, lat1, lon2, lat2)

    return (
        ftype_l,
        fj_l,
        fi_l,
        fsign_l,
        ft1j_l,
        ft1i_l,
        ft2j_l,
        ft2i_l,
        fwidth_l,
        fdist_l,
    )


def _break_diagonals(
    j_cells: np.ndarray, i_cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Insert intermediate steps so no consecutive T-cell pair is diagonal.

    For a diagonal step (Δj=±1 and Δi=±1) the i-step is taken first.  Jumps
    larger than 1 in either axis indicate a gap in the model domain and are
    skipped with the destination cell appended regardless (no intermediate
    face is logged).
    """
    j_out: list[int] = [int(j_cells[0])]
    i_out: list[int] = [int(i_cells[0])]
    for k in range(1, len(j_cells)):
        dj = int(j_cells[k]) - j_out[-1]
        di = int(i_cells[k]) - i_out[-1]
        if abs(dj) == 1 and abs(di) == 1:
            # Diagonal: insert i-step first, then j-step
            j_out.append(j_out[-1])
            i_out.append(i_out[-1] + di)
        j_out.append(int(j_cells[k]))
        i_out.append(int(i_cells[k]))
    return np.array(j_out, dtype=np.intp), np.array(i_out, dtype=np.intp)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ResolvedTransect:
    """T-cell path and HYCOM C-grid face data produced by :meth:`Transect.resolve`.

    Attributes
    ----------
    transect:
        The originating :class:`Transect`.
    j, i:
        T-cell indices along the section, shape (N,).
    cell_lon, cell_lat:
        Geographic coordinates of each T-cell, shape (N,).
    distance_km:
        Cumulative great-circle distance from the first T-cell, shape (N,).
    cell_width_km:
        Trapezoidal width of each cell's contribution to integrals, shape (N,).
        Used by T-point-based transport (e.g. future regular-grid support).
    bearing_deg:
        Local section bearing at each cell in degrees clockwise from north,
        shape (N,).  Used for normal-velocity projection in T-point transport.
    face_type:
        Shape (M,) uint8 array: 0 = U-face, 1 = V-face.  Populated by
        :meth:`Transect.resolve`; ``None`` if not yet resolved against a grid.
    face_j, face_i:
        Index of each face in the HYCOM grid (U-point or V-point), shape (M,).
    face_sign:
        ±1.0 per face: +1 when the section traverses the face in the direction
        of the face's positive normal (+i for U, +j for V), -1 otherwise.
    face_t1_j, face_t1_i, face_t2_j, face_t2_i:
        The two T-cells straddling each face (south/west and north/east).
        Thickness and tracers are averaged across these two cells.
    face_width_m:
        Physical face width in metres: ``scuy`` for U-faces, ``scvx`` for V-faces.
    face_dist_km:
        Cumulative distance (km from section start) at the midpoint of each face.
    """

    transect: Transect
    j: np.ndarray
    i: np.ndarray
    cell_lon: np.ndarray
    cell_lat: np.ndarray
    distance_km: np.ndarray
    cell_width_km: np.ndarray
    bearing_deg: np.ndarray
    # Spatial dimension names used to index into the source dataset.
    # Defaults match the HYCOM convention; overridden for generic grids.
    y_dim: str = field(default="y")
    x_dim: str = field(default="x")
    # HYCOM C-grid face data; None when not resolved via resolve()
    face_type: np.ndarray | None = field(default=None)
    face_j: np.ndarray | None = field(default=None)
    face_i: np.ndarray | None = field(default=None)
    face_sign: np.ndarray | None = field(default=None)
    face_t1_j: np.ndarray | None = field(default=None)
    face_t1_i: np.ndarray | None = field(default=None)
    face_t2_j: np.ndarray | None = field(default=None)
    face_t2_i: np.ndarray | None = field(default=None)
    face_width_m: np.ndarray | None = field(default=None)
    face_dist_km: np.ndarray | None = field(default=None)

    @property
    def n_cells(self) -> int:
        """Number of T-cells along the section."""
        return len(self.j)

    @property
    def n_faces(self) -> int:
        """Number of C-grid faces crossed (0 if face data not available)."""
        return 0 if self.face_type is None else len(self.face_type)

    @property
    def has_face_data(self) -> bool:
        """True when HYCOM C-grid face data is available for exact transport."""
        return self.face_type is not None

    # ------------------------------------------------------------------
    # Boundary factory
    # ------------------------------------------------------------------

    @classmethod
    def from_vface_row(
        cls,
        grid: xr.Dataset | str,
        bathy: xr.Dataset | str,
        j: int,
        *,
        sign: float = 1.0,
        i_range: tuple[int, int] | None = None,
        name: str | None = None,
    ) -> ResolvedTransect:
        """Create a ResolvedTransect from V-faces at a domain-boundary j-row.

        Use this instead of ``Transect.from_boundary_row(...).resolve(grid)``
        when you need cross-boundary transport at a domain boundary.
        ``from_boundary_row(...).resolve()`` correctly identifies V-faces, but
        finds the face between row *j* and *j+1* — one row interior to the
        boundary.  This method directly selects V-faces at the specified row
        *j*, i.e. the face between T(j-1, i) and T(j, i), which is the actual
        boundary face for southern (j=1) and northern (j=jdm-1) boundaries.

        The face thickness (used by :func:`~xhycom.transport`) is taken from
        the interior T-cell only, not averaged with the exterior land cell.

        Parameters
        ----------
        grid : Dataset or str
            HYCOM ``regional.grid`` Dataset or path.
        bathy : Dataset or str
            Bathymetry Dataset (opened with ``xhycom.open_dataset``) or path.
            Wet columns are identified by finite depth at the interior T-cell.
        j : int
            V-face row index.  V-face at ``(j, i)`` lies between T ``(j-1, i)``
            and T ``(j, i)``.  Typical values:

            * Southern boundary: ``j=1``
            * Northern boundary: ``j=grid.sizes["y"] - 1``
        sign : float
            Transport sign convention. ``+1`` when positive v-vel is directed
            INTO the domain (use for the southern boundary, where northward
            velocity enters the domain). ``-1`` when positive v-vel is directed
            OUT OF the domain and you want positive transport to mean inflow
            (use for the northern boundary, where northward velocity exits).
        i_range : (int, int), optional
            Column range ``(i_min, i_max)``, both inclusive.  Default: full row.
        name : str, optional
            Human-readable section label.

        Returns
        -------
        ResolvedTransect
            T-cell path at the interior row (for visualization / section_data)
            and V-face data for exact transport via :func:`~xhycom.transport`.

        Examples
        --------
        >>> jdm = grid.sizes["y"]
        >>> southern = xhycom.ResolvedTransect.from_vface_row(
        ...     grid, bathy, j=1, sign=+1, name="tp2_southern"
        ... )
        >>> northern = xhycom.ResolvedTransect.from_vface_row(
        ...     grid, bathy, j=jdm - 1, sign=-1, name="tp2_northern"
        ... )
        >>> tr_south = xhycom.transport(ds, southern)
        >>> tr_north = xhycom.transport(ds, northern)
        """
        grid_ds = _load_grid(grid)
        bathy_ds = _load_grid(bathy)

        plon: np.ndarray = grid_ds["plon"].values  # (jdm, idm)
        plat: np.ndarray = grid_ds["plat"].values
        scvx: np.ndarray = grid_ds["scvx"].values  # V-face east-west width (m)
        _, idm = plon.shape

        # Interior T-cell row: south of the V-face for sign>=0, north for sign<0.
        # V-face at (j, i) is between T(j-1, i) and T(j, i).
        #   Southern (sign≥0): interior ocean row = T(j, i)   → j_int = j
        #   Northern (sign<0): interior ocean row = T(j-1, i) → j_int = j - 1
        j_int: int = j if sign >= 0 else j - 1

        # Wet columns identified by finite depth at the interior T-cell row
        depth_row: np.ndarray = bathy_ds["depth"].values[j_int, :]
        i_lo, i_hi = 0, idm - 1
        if i_range is not None:
            i_lo, i_hi = int(i_range[0]), int(i_range[1])

        wet_idx = np.where(np.isfinite(depth_row[i_lo : i_hi + 1]))[0] + i_lo
        if len(wet_idx) < 2:
            raise ValueError(
                f"Fewer than 2 wet columns at j_int={j_int}"
                + (f", i_range={i_range}" if i_range is not None else "")
                + ". Check j and i_range."
            )

        # T-cell path at the interior row (used for visualization and section_data)
        cell_lons = plon[j_int, wet_idx]
        cell_lats = plat[j_int, wet_idx]
        dist_km = _cumulative_distance_km(cell_lons, cell_lats)
        widths_km = _cell_widths_km(dist_km)
        bearings = _section_bearings(cell_lons, cell_lats)
        n = len(wet_idx)

        # V-face data: one face per wet column.
        # Both t1 and t2 point to j_int (the interior T-cell) so that the
        # thickness / tracer average in transport() uses only interior values
        # and never averages with the exterior land row.
        wet_i = wet_idx.astype(np.intp)

        # Dummy Transect for endpoint labelling in plot()
        dummy = Transect(
            lons=np.array([cell_lons[0], cell_lons[-1]]),
            lats=np.array([cell_lats[0], cell_lats[-1]]),
            name=name,
        )

        return cls(
            transect=dummy,
            j=np.full(n, j_int, dtype=np.intp),
            i=wet_i,
            cell_lon=cell_lons,
            cell_lat=cell_lats,
            distance_km=dist_km,
            cell_width_km=widths_km,
            bearing_deg=bearings,
            face_type=np.full(n, _FACE_V, dtype=np.uint8),
            face_j=np.full(n, j, dtype=np.intp),
            face_i=wet_i.copy(),
            face_sign=np.full(n, float(sign)),
            face_t1_j=np.full(n, j_int, dtype=np.intp),
            face_t1_i=wet_i.copy(),
            face_t2_j=np.full(n, j_int, dtype=np.intp),
            face_t2_i=wet_i.copy(),
            face_width_m=scvx[j, wet_idx],
            face_dist_km=dist_km,
        )

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        *,
        grid: xr.Dataset | str | None = None,
        bathy: xr.Dataset | str | None = None,
        figsize: tuple[float, float] | None = None,
        bathy_levels: int | list[float] = 6,
        pad_deg: float = 2.0,
        i_range: tuple[int, int] | None = None,
        j_range: tuple[int, int] | None = None,
        color: str = "steelblue",
        show_cells: bool = False,
        waypoint_kw: dict | None = None,
        label_endpoints: bool = True,
        **kwargs,
    ) -> None:
        """Plot the resolved T-cell path in longitude–latitude space.

        Parameters
        ----------
        ax:
            Existing axes to draw into.  A new figure is created when ``None``.
            Pass a ``cartopy.crs.PlateCarree()`` axes for a map background.
        grid:
            HYCOM grid Dataset or path to ``regional.grid``.  When provided
            the T-point positions are drawn as a faint scatter clipped to the
            section bounding box, and U/V faces are shown colour-coded by sign.
        bathy:
            Bathymetry Dataset or path (``depth_*`` file, opened with
            ``grid=`` so it carries the coordinates).  When provided a
            ``pcolormesh`` of ocean depth is drawn as the background, with
            land (NaN) shaded in a distinct colour.  Requires *grid*.
        pad_deg:
            Degrees of padding around the section bounding box.  Default 2°.
        color:
            Colour for the T-cell path line and markers.
        show_cells:
            Draw a faint scatter of T-cell centres in the section bounding box
            as a grid reference.  Default ``True``.  Pass ``False`` to hide.
        waypoint_kw:
            Extra keyword arguments for the waypoint scatter call.
        label_endpoints:
            Annotate the first and last waypoints with their coordinates.
        **kwargs:
            Forwarded to :func:`matplotlib.pyplot.plot` for the T-cell path.

        Examples
        --------
        >>> resolved.plot()
        >>> resolved.plot(grid=grid)
        >>> resolved.plot(grid=grid, bathy=bathy)

        With a cartopy map background::

            import cartopy.crs as ccrs
            ax = plt.axes(projection=ccrs.NorthPolarStereo())
            ax.coastlines()
            resolved.plot(ax=ax, grid=grid, bathy=bathy)
        """
        import matplotlib.pyplot as plt

        if bathy is not None and grid is None:
            raise ValueError("grid= is required alongside bathy=.")

        native = bathy is not None  # plot in (i, j) index space when bathy shown

        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (9, 5))
            if native:
                ax.set_xlabel("i (column index)")
                ax.set_ylabel("j (row index)")
            else:
                ax.set_xlabel("Longitude (°E)")
                ax.set_ylabel("Latitude (°N)")

        kwargs.setdefault("lw", 1.5)

        # --- Load grid coordinates once (needed for lon/lat mode and face markers) ---
        plon = plat = None
        ulon_g = ulat_g = vlon_g = vlat_g = None
        if grid is not None:
            grid = _load_grid(grid)
            plon = grid["plon"].values  # (jdm, idm)
            plat = grid["plat"].values
            ulon_g = grid["ulon"].values if "ulon" in grid else plon
            ulat_g = grid["ulat"].values if "ulat" in grid else plat
            vlon_g = grid["vlon"].values if "vlon" in grid else plon
            vlat_g = grid["vlat"].values if "vlat" in grid else plat
            qlon_g = grid["qlon"].values if "qlon" in grid else None
            qlat_g = grid["qlat"].values if "qlat" in grid else None

        # --- Bathymetry / land-mask background — plotted in native (i, j) space ---
        if bathy is not None:
            if isinstance(bathy, str):
                bathy = _load_grid(bathy)
            depth_vals = bathy["depth"].values.astype(float)  # NaN = land

            # Gray land mask — ocean cells left transparent
            land = np.where(np.isnan(depth_vals), 1.0, np.nan)
            land_cmap = plt.cm.Greys.copy()
            land_cmap.set_bad("none")
            ax.pcolormesh(
                land, cmap=land_cmap, vmin=0, vmax=1, shading="nearest", zorder=0
            )

            # Black depth contours (NaN land filled with 0 so contour doesn't choke)
            depth_filled = np.where(np.isnan(depth_vals), 0.0, depth_vals)
            ax.contour(
                depth_filled,
                colors="black",
                linewidths=0.4,
                levels=bathy_levels,
                zorder=1,
            )

        # --- HYCOM T-point scatter (full domain) ---
        if show_cells and plon is not None:
            if native:
                jdm, idm = plon.shape
                ii, jj = np.meshgrid(np.arange(idm), np.arange(jdm))
                ax.scatter(
                    ii.ravel(),
                    jj.ravel(),
                    s=2,
                    color="0.6",
                    linewidths=0,
                    zorder=2,
                    label="HYCOM T-points (cell centres)",
                )
            else:
                ax.scatter(
                    plon.ravel(),
                    plat.ravel(),
                    s=2,
                    color="0.6",
                    linewidths=0,
                    zorder=1,
                    label="HYCOM T-points (cell centres)",
                )

        # --- Face positions, staircase path, and corner markers ---
        # U-face (j,i): between T(j,i-1) and T(j,i);  native position x=i-0.5, y=j
        # V-face (j,i): between T(j-1,i) and T(j,i);  native position x=i,     y=j-0.5
        # Q-point (j,i): NE corner of T(j,i);          native position x=i+0.5, y=j+0.5
        # At each U↔V turn the two faces share a Q-point — the staircase goes through it,
        # creating an L-bend.  Straight U→U or V→V runs have no shared corner.
        fx_all = fy_all = None
        corner_x = corner_y = np.empty(0)
        turn = np.zeros(max(self.n_faces - 1, 0), dtype=bool)
        all_cx = all_cy = np.zeros(len(turn))

        if self.has_face_data:
            is_u = self.face_type == _FACE_U
            fj_all = self.face_j
            fi_all = self.face_i
            if native:
                fx_all = np.where(is_u, fi_all - 0.5, fi_all.astype(float))
                fy_all = np.where(is_u, fj_all.astype(float), fj_all - 0.5)
            elif grid is not None:
                fx_all = np.where(is_u, ulon_g[fj_all, fi_all], vlon_g[fj_all, fi_all])
                fy_all = np.where(is_u, ulat_g[fj_all, fi_all], vlat_g[fj_all, fi_all])

            if fx_all is not None and self.n_faces > 1:
                ft = self.face_type
                is_uv = (ft[:-1] == _FACE_U) & (ft[1:] == _FACE_V)
                is_vu = (ft[:-1] == _FACE_V) & (ft[1:] == _FACE_U)
                # U→V: shared Q = Q(fj[k+1]-1, fi[k]-1)
                # V→U: shared Q = Q(fj[k]-1,   fi[k+1]-1)
                cj_idx = np.where(is_uv, fj_all[1:] - 1, fj_all[:-1] - 1)
                ci_idx = np.where(is_uv, fi_all[:-1] - 1, fi_all[1:] - 1)
                jdm, idm = plon.shape
                in_bounds = (
                    (cj_idx >= 0) & (ci_idx >= 0) & (cj_idx < jdm) & (ci_idx < idm)
                )
                turn = (is_uv | is_vu) & in_bounds
                if native:
                    all_cx = ci_idx + 0.5
                    all_cy = cj_idx + 0.5
                elif qlon_g is not None:
                    sj = np.clip(cj_idx, 0, jdm - 1).astype(np.intp)
                    si = np.clip(ci_idx, 0, idm - 1).astype(np.intp)
                    all_cx = qlon_g[sj, si]
                    all_cy = qlat_g[sj, si]
                else:
                    turn[:] = False
                corner_x = all_cx[turn]
                corner_y = all_cy[turn]

        # Build staircase: start T-cell → [face midpoint → corner? → face midpoint → …] → end T-cell
        if native:
            x0, y0 = float(self.i[0]), float(self.j[0])
            x1, y1 = float(self.i[-1]), float(self.j[-1])
        else:
            x0, y0 = float(self.cell_lon[0]), float(self.cell_lat[0])
            x1, y1 = float(self.cell_lon[-1]), float(self.cell_lat[-1])

        if fx_all is not None:
            pts_x: list[float] = [x0]
            pts_y: list[float] = [y0]
            for k in range(self.n_faces):
                pts_x.append(float(fx_all[k]))
                pts_y.append(float(fy_all[k]))
                if k < self.n_faces - 1 and turn[k]:
                    pts_x.append(float(all_cx[k]))
                    pts_y.append(float(all_cy[k]))
            pts_x.append(x1)
            pts_y.append(y1)
            px = np.array(pts_x)
            py = np.array(pts_y)
        else:
            px = self.i.astype(float) if native else self.cell_lon
            py = self.j.astype(float) if native else self.cell_lat

        ax.plot(px, py, color=color, zorder=3, **kwargs)

        if fx_all is not None:
            ax.scatter(
                np.concatenate([fx_all, corner_x]),
                np.concatenate([fy_all, corner_y]),
                s=12,
                marker="o",
                color=color,
                zorder=4,
                linewidths=0,
            )

        # --- Waypoints / endpoints ---
        wp_kw = {
            "s": 60,
            "color": "tomato",
            "zorder": 5,
            "edgecolors": "white",
            "linewidths": 0.8,
        }
        if waypoint_kw:
            wp_kw.update(waypoint_kw)
        if native:
            # In native space mark the first and last T-cell instead of waypoints
            ax.scatter(self.i[[0, -1]], self.j[[0, -1]], **wp_kw)
        else:
            ax.scatter(self.transect.lons, self.transect.lats, **wp_kw)

        if label_endpoints:
            for ci, cj, lon, lat in [
                (self.i[0], self.j[0], self.transect.lons[0], self.transect.lats[0]),
                (
                    self.i[-1],
                    self.j[-1],
                    self.transect.lons[-1],
                    self.transect.lats[-1],
                ),
            ]:
                ax.annotate(
                    f"({lon:.1f}, {lat:.1f})",
                    xy=(ci if native else lon, cj if native else lat),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="tomato",
                )

        name = self.transect.name or "Section"
        ax.set_title(
            f"{name}  —  {self.n_cells} cells, {self.n_faces} faces, "
            f"{self.distance_km[-1]:.0f} km"
        )

        # Clip view to requested index subset (native mode)
        if native:
            if i_range is not None:
                ax.set_xlim(i_range)
            if j_range is not None:
                ax.set_ylim(j_range)

        # Legend — only when there are labelled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7, loc="best")


# ---------------------------------------------------------------------------
# Transect
# ---------------------------------------------------------------------------


class Transect:
    """An oceanographic section defined by ordered (lon, lat) waypoints.

    The section is the great-circle polyline connecting the waypoints.  It is
    entirely grid-agnostic: call :meth:`resolve` to snap it to a specific HYCOM
    grid and obtain the C-grid faces needed for exact volume and tracer
    transports.

    Sign convention
    ---------------
    Positive transport = flow to the **right** when walking from ``lons[0]``
    to ``lons[-1]``.  Reverse the waypoint order to flip the sign.

    Parameters
    ----------
    lons:
        Waypoint longitudes in degrees east.
    lats:
        Waypoint latitudes in degrees north.
    name:
        Optional human-readable label (e.g. ``"Fram Strait"``).

    Examples
    --------
    >>> t = xhycom.Transect(lons=[-20, 10], lats=[79, 79], name="Fram Strait")
    >>> t = xhycom.Transect.named("fram_strait")
    >>> resolved = t.resolve(grid)
    """

    def __init__(
        self,
        lons: Sequence[float] | ArrayLike,
        lats: Sequence[float] | ArrayLike,
        name: str | None = None,
    ) -> None:
        self.lons: np.ndarray = np.asarray(lons, dtype=float)
        self.lats: np.ndarray = np.asarray(lats, dtype=float)
        if len(self.lons) < 2:
            raise ValueError("Transect requires at least two waypoints.")
        if len(self.lons) != len(self.lats):
            raise ValueError("lons and lats must have the same length.")
        self.name: str | None = name

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def named(cls, name: str) -> Transect:
        """Return a built-in named section.

        Parameters
        ----------
        name:
            Case-insensitive section name.  See :meth:`available_names` for
            the full list.

        Raises
        ------
        ValueError
            If *name* is not recognised.
        """
        key = name.lower().replace(" ", "_")
        if key not in _NAMED_SECTIONS:
            avail = ", ".join(sorted(_NAMED_SECTIONS))
            raise ValueError(f"Unknown section {name!r}. Available: {avail}")
        lons, lats = _NAMED_SECTIONS[key]
        return cls(lons=lons, lats=lats, name=name)

    @classmethod
    def available_names(cls) -> list[str]:
        """Return a sorted list of built-in section names."""
        return sorted(_NAMED_SECTIONS)

    @classmethod
    def from_boundary_row(
        cls,
        grid: xr.Dataset | str,
        j: int,
        bathy: xr.Dataset | str,
        *,
        i_range: tuple[int, int] | None = None,
        name: str | None = None,
    ) -> Transect:
        """Create a Transect from the wet cells along a single grid row.

        .. note::
            For **transport** computation at domain boundaries, use
            :meth:`ResolvedTransect.from_vface_row` (or
            :func:`~xhycom.boundary_transport`) instead.  Resolving this
            Transect via :meth:`resolve` correctly identifies V-faces, but
            locates them at the face between row *j* and row *j+1* — one row
            interior to the boundary.  The actual boundary V-face (between
            the exterior land row and the first wet row) is only accessible
            via :meth:`~ResolvedTransect.from_vface_row`.

        This factory is useful for **visualisation** and for
        :func:`~xhycom.section_data` (hydrographic sections along the boundary
        row), where the T-cell path is the right representation.

        Intended for domain-boundary transects (e.g. TOPAZ open boundaries)
        where the outermost row lies entirely on land.  Pass the first interior
        wet row (``j=1`` for the southern boundary, ``grid.dims['y']-2`` for
        the northern boundary) and this factory extracts the positions of all
        wet cells in that row as waypoints.

        Parameters
        ----------
        grid:
            HYCOM grid Dataset or path to ``regional.grid``.
        j:
            Row index (0-based) to extract wet-cell waypoints from.
        bathy:
            Bathymetry Dataset (opened with ``xhycom.open_dataset``) or path.
            Used to identify wet cells (finite depth) in the row.
        i_range:
            Optional ``(i_min, i_max)`` (inclusive on both ends) to restrict
            the column range.  Useful when a boundary row is only wet over a
            sub-range of columns.
        name:
            Human-readable label for the resulting Transect.

        Returns
        -------
        Transect
            Waypoints at the positions of every wet cell in row *j* (within
            *i_range* if given), ordered west-to-east by column index.

        Raises
        ------
        ValueError
            If fewer than two wet cells are found in the requested row/range.

        Examples
        --------
        >>> topaz_southern = xhycom.Transect.from_boundary_row(
        ...     grid, j=1, bathy=bathy, name="topaz_southern"
        ... )
        >>> topaz_northern = xhycom.Transect.from_boundary_row(
        ...     grid, j=grid.dims["y"] - 2, bathy=bathy, name="topaz_northern"
        ... )
        """
        grid_ds = _load_grid(grid)
        bathy_ds = _load_grid(bathy)

        plon: np.ndarray = grid_ds["plon"].values[j, :]
        plat: np.ndarray = grid_ds["plat"].values[j, :]
        depth_row: np.ndarray = bathy_ds["depth"].values[j, :]

        i_lo = 0
        i_hi = len(plon) - 1
        if i_range is not None:
            i_lo, i_hi = int(i_range[0]), int(i_range[1])

        depth_sub = depth_row[i_lo : i_hi + 1]
        wet_idx = np.where(np.isfinite(depth_sub))[0]

        if len(wet_idx) < 2:
            raise ValueError(
                f"Fewer than 2 wet cells found at j={j}"
                + (f", i_range={i_range}" if i_range is not None else "")
                + ". Check j index and i_range."
            )

        wet_abs = wet_idx + i_lo
        lons = plon[wet_abs].copy()
        # Unwrap so consecutive waypoints never jump more than 180° in longitude,
        # preventing _sample_polyline from interpolating "the long way round"
        # across the date line (e.g. Bering Sea sections near 180°E/W).
        shifts = np.cumsum(
            np.where(
                np.diff(lons) > 180,
                -360.0,
                np.where(np.diff(lons) < -180, 360.0, 0.0),
            )
        )
        lons[1:] += shifts
        return cls(lons=lons, lats=plat[wet_abs], name=name)

    def reverse(self) -> Transect:
        """Return a copy of this transect with the waypoint order reversed.

        Reversing the waypoints flips the sign convention: flow that was
        positive (rightward) becomes negative, and vice versa.  Use this
        to adopt a different positive direction without negating results
        manually::

            fs_north = fs.reverse()           # lons = [15, -20]: positive = northward
            tr_north = xhycom.transport(ds, fs_north, grid=grid)
            # tr_north.volume == -tr.volume

        Returns
        -------
        Transect
            New Transect with waypoints in reversed order and the same name.
        """
        return Transect(lons=self.lons[::-1], lats=self.lats[::-1], name=self.name)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        grid: xr.Dataset | str,
        *,
        lat_var: str | None = None,
        lon_var: str | None = None,
    ) -> ResolvedTransect:
        """Resolve the section against a model grid.

        By default resolves against a HYCOM curvilinear grid using the
        MSCPROGS hemisphere-mask algorithm: each waypoint-pair segment divides
        the grid into two hemispheres; a vectorised telescopic sum of the
        resulting Boolean mask identifies the exact U- and V-faces that cross
        the great-circle boundary.  This correctly handles sections aligned
        with constant grid rows or columns (e.g. boundary sections) as well as
        sections that cross both a U-face and a V-face at the same corner node.

        Pass *lat_var* and *lon_var* to resolve against any rectilinear or
        curvilinear dataset (e.g. GLORYS); in this mode face data is not
        populated (so :func:`~xhycom.section_flux_density` is unavailable,
        but :func:`~xhycom.section_data` works as normal).

        Parameters
        ----------
        grid:
            HYCOM mode: a Dataset from ``xhycom.open_dataset("regional.grid")``
            (must contain ``plon``, ``plat``, ``scuy``, ``scvx``), or a path.
            Generic mode: any Dataset that contains *lat_var* and *lon_var*.
        lat_var:
            Name of the latitude coordinate/variable in *grid*.  When given,
            *lon_var* must also be given and generic mode is used.
        lon_var:
            Name of the longitude coordinate/variable in *grid*.

        Returns
        -------
        ResolvedTransect
            T-cell path, distances, widths, and local bearings.  Face data is
            only populated in HYCOM mode.

        Raises
        ------
        ImportError
            If ``scipy`` is not installed.
        ValueError
            If required variables are absent or fewer than two T-cells are
            found.
        """
        try:
            from scipy.spatial import KDTree
        except ImportError as exc:
            raise ImportError(
                "scipy is required to resolve a Transect against a grid.\n"
                "Install it with: conda install scipy  or  pip install scipy"
            ) from exc

        grid = _load_grid(grid)

        # ------------------------------------------------------------------
        # Generic (non-HYCOM) path
        # ------------------------------------------------------------------
        if lat_var is not None or lon_var is not None:
            if lat_var is None or lon_var is None:
                raise ValueError("lat_var and lon_var must both be supplied together.")
            for var in (lat_var, lon_var):
                if var not in grid and var not in grid.coords:
                    raise ValueError(f"Grid is missing {var!r}. Check lat_var/lon_var.")

            lat_vals = grid[lat_var].values
            lon_vals = grid[lon_var].values

            if lat_vals.ndim == 1 and lon_vals.ndim == 1:
                lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
                y_dim, x_dim = lat_var, lon_var
            elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
                lat2d, lon2d = lat_vals, lon_vals
                y_dim = grid[lat_var].dims[0]
                x_dim = grid[lat_var].dims[1]
            else:
                raise ValueError(
                    f"{lat_var!r} and {lon_var!r} must both be 1-D or both 2-D."
                )

            ny, nx = lon2d.shape
            tree = KDTree(_to_xyz(lon2d.ravel(), lat2d.ravel()))
            slons, slats = _sample_polyline(self.lons, self.lats)
            _, flat_idx = tree.query(_to_xyz(slons, slats))
            j_samp = (flat_idx // nx).astype(np.intp)
            i_samp = (flat_idx % nx).astype(np.intp)

            pairs = np.column_stack([j_samp, i_samp])
            keep = np.concatenate([[True], np.any(pairs[1:] != pairs[:-1], axis=1)])
            j_cells = j_samp[keep]
            i_cells = i_samp[keep]

            if len(j_cells) < 2:
                raise ValueError(
                    "Transect intersects fewer than 2 T-cells — verify that "
                    "the waypoints lie within the dataset domain."
                )

            j_cells, i_cells = _break_diagonals(j_cells, i_cells)

            cell_lons = lon2d[j_cells, i_cells]
            cell_lats = lat2d[j_cells, i_cells]
            dist_km = _cumulative_distance_km(cell_lons, cell_lats)
            widths_km = _cell_widths_km(dist_km)
            bearings = _section_bearings(cell_lons, cell_lats)

            return ResolvedTransect(
                transect=self,
                j=j_cells,
                i=i_cells,
                cell_lon=cell_lons,
                cell_lat=cell_lats,
                distance_km=dist_km,
                cell_width_km=widths_km,
                bearing_deg=bearings,
                y_dim=y_dim,
                x_dim=x_dim,
                # Face data not available for generic grids
            )

        # ------------------------------------------------------------------
        # HYCOM path
        # ------------------------------------------------------------------
        for var in ("plon", "plat", "scuy", "scvx"):
            if var not in grid:
                raise ValueError(
                    f"grid is missing {var!r}. Pass a Dataset from "
                    "xhycom.open_dataset('regional.grid'), or supply "
                    "lat_var and lon_var for a non-HYCOM grid."
                )

        plon: np.ndarray = grid["plon"].values  # (jdm, idm)
        plat: np.ndarray = grid["plat"].values
        scuy: np.ndarray = grid["scuy"].values  # U-face width (y-direction)
        scvx: np.ndarray = grid["scvx"].values  # V-face width (x-direction)

        # ------------------------------------------------------------------
        # Face identification via hemisphere mask (MSCPROGS method).
        # Faces and T-cell path both come from this single algorithm, so they
        # are guaranteed consistent.  No KDTree or _break_diagonals needed.
        # ------------------------------------------------------------------
        (
            ftype_l,
            fj_l,
            fi_l,
            fsign_l,
            ft1j_l,
            ft1i_l,
            ft2j_l,
            ft2i_l,
            fwidth_l,
            fdist_l,
        ) = _faces_from_waypoints(plon, plat, scuy, scvx, self.lons, self.lats)

        if len(fj_l) < 2:
            raise ValueError(
                "Transect intersects fewer than 2 C-grid faces — verify that "
                "the waypoints lie within the model domain."
            )

        def _arr(lst: list, dtype: type = float) -> np.ndarray:
            return np.array(lst, dtype=dtype)

        # Sort faces by distance along section
        fdist_arr = np.asarray(fdist_l)
        _sort = np.argsort(fdist_arr)

        # T-cell path: unique pivot nodes (MSCPROGS ipiv, jpiv) from the
        # sorted face positions.  A corner cell that produces both a U-face
        # and a V-face appears twice in the face arrays but once in the path.
        seen: dict[tuple[int, int], bool] = {}
        path_j: list[int] = []
        path_i: list[int] = []
        for idx in _sort:
            key = (int(fj_l[idx]), int(fi_l[idx]))
            if key not in seen:
                seen[key] = True
                path_j.append(key[0])
                path_i.append(key[1])

        node_j = np.array(path_j, dtype=np.intp)
        node_i = np.array(path_i, dtype=np.intp)
        cell_lons = plon[node_j, node_i]
        cell_lats = plat[node_j, node_i]
        dist_km = _cumulative_distance_km(cell_lons, cell_lats)
        widths_km = _cell_widths_km(dist_km)
        bearings = _section_bearings(cell_lons, cell_lats)

        return ResolvedTransect(
            transect=self,
            j=node_j,
            i=node_i,
            cell_lon=cell_lons,
            cell_lat=cell_lats,
            distance_km=dist_km,
            cell_width_km=widths_km,
            bearing_deg=bearings,
            face_type=_arr(ftype_l, np.uint8)[_sort],
            face_j=_arr(fj_l, np.intp)[_sort],
            face_i=_arr(fi_l, np.intp)[_sort],
            face_sign=_arr(fsign_l, float)[_sort],
            face_t1_j=_arr(ft1j_l, np.intp)[_sort],
            face_t1_i=_arr(ft1i_l, np.intp)[_sort],
            face_t2_j=_arr(ft2j_l, np.intp)[_sort],
            face_t2_i=_arr(ft2i_l, np.intp)[_sort],
            face_width_m=_arr(fwidth_l, float)[_sort],
            face_dist_km=fdist_arr[_sort],
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        name = f"{self.name!r}" if self.name else "unnamed"
        n = len(self.lons)
        return f"Transect({name}, {n} waypoints)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_grid(grid: xr.Dataset | str) -> xr.Dataset:
    """Accept a path or pre-loaded Dataset; return a Dataset."""
    if isinstance(grid, xr.Dataset):
        return grid
    from . import open_dataset

    return open_dataset(str(grid))
