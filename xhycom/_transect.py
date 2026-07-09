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
from typing import Sequence

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

_R_EARTH_KM: float = 6371.0

# Approximate waypoints for common Arctic / sub-Arctic sections.
# Positive transport is rightward when walking from first to last waypoint,
# so adjust waypoint order if you want a specific sign convention.
_NAMED_SECTIONS: dict[str, tuple[list[float], list[float]]] = {
    "fram_strait":      ([-20.0,  10.0], [79.0, 79.0]),
    "bering_strait":    ([-170.0, -166.0], [65.5, 65.5]),
    "barents_opening":  ([20.0,   19.0],  [71.5, 74.5]),
    "svinoy":           ([-5.0,    5.0],  [62.0, 62.0]),
    "gimsoy":           ([13.0,   17.0],  [68.0, 68.0]),
    "fsc":              ([-7.0,   -1.0],  [62.0, 60.0]),
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


def _forward_bearing(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
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
        b[k] = _forward_bearing(lons[k0], lats[k0], lons[k1], lats[k1]) if k0 != k1 else 0.0
    return b


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

    transect: "Transect"
    j: np.ndarray
    i: np.ndarray
    distance_km: np.ndarray
    cell_width_km: np.ndarray
    bearing_deg: np.ndarray
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
    def named(cls, name: str) -> "Transect":
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

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, grid: xr.Dataset | str) -> ResolvedTransect:
        """Resolve the section against a HYCOM curvilinear grid.

        Finds the ordered T-cells the polyline passes through via a 3-D KDTree
        on the T-point coordinates, then determines the exact U- and V-faces
        crossed between consecutive T-cells.

        Parameters
        ----------
        grid:
            A Dataset as returned by ``xhycom.open_dataset("regional.grid")``
            (must contain ``plon``, ``plat``, ``scuy``, ``scvx``), or a path
            to ``regional.grid``.

        Returns
        -------
        ResolvedTransect
            T-cell path, face data, distances, widths, and local bearings.

        Raises
        ------
        ImportError
            If ``scipy`` is not installed.
        ValueError
            If ``grid`` lacks the required variables, or if the section does
            not intersect at least two T-cells in the domain.
        """
        try:
            from scipy.spatial import KDTree
        except ImportError as exc:
            raise ImportError(
                "scipy is required to resolve a Transect against a grid.\n"
                "Install it with: conda install scipy  or  pip install scipy"
            ) from exc

        grid = _load_grid(grid)
        for var in ("plon", "plat", "scuy", "scvx"):
            if var not in grid:
                raise ValueError(
                    f"grid is missing {var!r}. Pass a Dataset from "
                    "xhycom.open_dataset('regional.grid')."
                )

        plon: np.ndarray = grid["plon"].values  # (jdm, idm)
        plat: np.ndarray = grid["plat"].values
        scuy: np.ndarray = grid["scuy"].values  # U-face width (y-direction)
        scvx: np.ndarray = grid["scvx"].values  # V-face width (x-direction)
        jdm, idm = plon.shape

        # Build KDTree in 3-D Cartesian so polar projections work correctly
        tree = KDTree(_to_xyz(plon.ravel(), plat.ravel()))

        # Densely sample the waypoint polyline and find nearest T-cell
        slons, slats = _sample_polyline(self.lons, self.lats)
        _, flat_idx = tree.query(_to_xyz(slons, slats))
        j_samp = (flat_idx // idm).astype(np.intp)
        i_samp = (flat_idx % idm).astype(np.intp)

        # Deduplicate: keep only the first occurrence of each (j, i) pair
        pairs = np.column_stack([j_samp, i_samp])
        keep = np.concatenate([[True], np.any(pairs[1:] != pairs[:-1], axis=1)])
        j_cells = j_samp[keep]
        i_cells = i_samp[keep]

        if len(j_cells) < 2:
            raise ValueError(
                "Transect intersects fewer than 2 T-cells — verify that the "
                "waypoints lie within the model domain."
            )

        # Eliminate diagonal steps (Δj=±1 AND Δi=±1 simultaneously)
        j_cells, i_cells = _break_diagonals(j_cells, i_cells)

        # T-cell coordinates, cumulative distances, widths, bearings
        cell_lons = plon[j_cells, i_cells]
        cell_lats = plat[j_cells, i_cells]
        dist_km = _cumulative_distance_km(cell_lons, cell_lats)
        widths_km = _cell_widths_km(dist_km)
        bearings = _section_bearings(cell_lons, cell_lats)

        # ------------------------------------------------------------------
        # Face identification
        # Face sign convention: +1 when the section steps in the face's
        # positive-normal direction (+i for U-faces, +j for V-faces), else -1.
        # ------------------------------------------------------------------
        face_type_list:  list[int]   = []
        face_j_list:     list[int]   = []
        face_i_list:     list[int]   = []
        face_sign_list:  list[float] = []
        face_t1j_list:   list[int]   = []
        face_t1i_list:   list[int]   = []
        face_t2j_list:   list[int]   = []
        face_t2i_list:   list[int]   = []
        face_width_list: list[float] = []
        face_dist_list:  list[float] = []

        for k in range(len(j_cells) - 1):
            j1, i1 = int(j_cells[k]), int(i_cells[k])
            j2, i2 = int(j_cells[k + 1]), int(i_cells[k + 1])
            dj, di = j2 - j1, i2 - i1
            mid_dist = (dist_km[k] + dist_km[k + 1]) / 2.0

            if dj == 0 and di == 1:
                # Step +i: U-face at (j1, i2), normal in +i  →  sign = +1
                # Face is the western face of T(j1, i2), between T(j1,i1) and T(j1,i2)
                face_type_list.append(int(_FACE_U))
                face_j_list.append(j1);  face_i_list.append(i2)
                face_sign_list.append(1.0)
                face_t1j_list.append(j1); face_t1i_list.append(i1)
                face_t2j_list.append(j1); face_t2i_list.append(i2)
                face_width_list.append(float(scuy[j1, i2]))

            elif dj == 0 and di == -1:
                # Step -i: U-face at (j1, i1), normal in +i  →  sign = -1
                face_type_list.append(int(_FACE_U))
                face_j_list.append(j1);  face_i_list.append(i1)
                face_sign_list.append(-1.0)
                face_t1j_list.append(j1); face_t1i_list.append(i1)
                face_t2j_list.append(j1); face_t2i_list.append(i2)
                face_width_list.append(float(scuy[j1, i1]))

            elif dj == 1 and di == 0:
                # Step +j: V-face at (j2, i1), normal in +j  →  sign = +1
                # Face is the southern face of T(j2, i1), between T(j1,i1) and T(j2,i1)
                face_type_list.append(int(_FACE_V))
                face_j_list.append(j2);  face_i_list.append(i1)
                face_sign_list.append(1.0)
                face_t1j_list.append(j1); face_t1i_list.append(i1)
                face_t2j_list.append(j2); face_t2i_list.append(i1)
                face_width_list.append(float(scvx[j2, i1]))

            elif dj == -1 and di == 0:
                # Step -j: V-face at (j1, i1), normal in +j  →  sign = -1
                face_type_list.append(int(_FACE_V))
                face_j_list.append(j1);  face_i_list.append(i1)
                face_sign_list.append(-1.0)
                face_t1j_list.append(j2); face_t1i_list.append(i2)
                face_t2j_list.append(j1); face_t2i_list.append(i1)
                face_width_list.append(float(scvx[j1, i1]))

            else:
                # Large jump (gap in domain or dense-sampling edge case) — skip
                continue

            face_dist_list.append(mid_dist)

        def _arr(lst: list, dtype: type = float) -> np.ndarray:
            return np.array(lst, dtype=dtype)

        return ResolvedTransect(
            transect=self,
            j=j_cells,
            i=i_cells,
            distance_km=dist_km,
            cell_width_km=widths_km,
            bearing_deg=bearings,
            face_type=_arr(face_type_list, np.uint8),
            face_j=_arr(face_j_list, np.intp),
            face_i=_arr(face_i_list, np.intp),
            face_sign=_arr(face_sign_list, float),
            face_t1_j=_arr(face_t1j_list, np.intp),
            face_t1_i=_arr(face_t1i_list, np.intp),
            face_t2_j=_arr(face_t2j_list, np.intp),
            face_t2_i=_arr(face_t2i_list, np.intp),
            face_width_m=_arr(face_width_list, float),
            face_dist_km=_arr(face_dist_list, float),
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
