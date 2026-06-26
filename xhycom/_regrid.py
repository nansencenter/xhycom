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
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

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
# Target grids: accept a Dataset / path (e.g. GLORYS) instead of raw arrays
# ---------------------------------------------------------------------------
def _open_target(target: "xr.Dataset | xr.DataArray | str") -> xr.Dataset:
    """Accept an ``xr.Dataset`` or a path to one; return an ``xr.Dataset``."""
    if isinstance(target, xr.Dataset):
        return target
    if isinstance(target, xr.DataArray):
        return target.to_dataset()
    return xr.open_dataset(target)


def _load_grid(grid: "xr.Dataset | str | None") -> "xr.Dataset | None":
    """Accept a ``regional.grid`` path or a pre-loaded Dataset; return a Dataset.

    Mirrors :func:`xhycom.open_dataset`'s ``grid=`` so the regrid functions
    take the same argument.  A path is read with :func:`xhycom.open_dataset`
    (imported lazily to avoid a circular import).
    """
    if grid is None or isinstance(grid, xr.Dataset):
        return grid
    from . import open_dataset
    return open_dataset(grid)


def _target_lonlat(tgt: xr.Dataset) -> "tuple[np.ndarray, np.ndarray]":
    """1-D target longitudes / latitudes from a grid Dataset."""
    lon = tgt["longitude"] if "longitude" in tgt.variables else tgt["lon"]
    lat = tgt["latitude"] if "latitude" in tgt.variables else tgt["lat"]
    return np.asarray(lon.values), np.asarray(lat.values)


def _lonlat_names(tgt: xr.Dataset) -> "tuple[str, str]":
    """Names of the 1-D longitude / latitude coordinates on a target grid."""
    lon = "longitude" if "longitude" in tgt.variables else "lon"
    lat = "latitude" if "latitude" in tgt.variables else "lat"
    return lon, lat


def _subset_target(tgt: xr.Dataset, ds: xr.Dataset,
                   pad: float = 1.0) -> xr.Dataset:
    """Trim a regular target grid to the source's lon/lat extent (plus *pad*).

    A regional HYCOM source (e.g. TOPAZ2) usually covers a small fraction of a
    global target like GLORYS, yet xESMF still generates remap weights for every
    target cell — most of which receive no source data (and become NaN).  Weight
    generation cost scales with the target size, so restricting the target to
    the source bounding box is the single biggest speed-up for regional→global
    remaps, with **no effect** on the result inside the covered region.

    Latitude is always boxed (safe, no wrap).  Longitude is boxed only when the
    source clearly does *not* span all meridians — polar caps (Arctic TOPAZ2)
    have every longitude near the pole, so there a latitude box is the whole
    win.  Convention differences (0–360 vs −180–180) are handled by comparing
    modulo 360; if either grid straddles its own seam we conservatively skip the
    longitude box rather than risk dropping covered cells.
    """
    lon_name, lat_name = _lonlat_names(tgt)
    tlon = np.asarray(tgt[lon_name].values)
    tlat = np.asarray(tgt[lat_name].values)
    slat = np.asarray(ds["lat"].values)
    slon = np.asarray(ds["lon"].values)

    lat_lo, lat_hi = np.nanmin(slat) - pad, np.nanmax(slat) + pad
    lat_keep = (tlat >= lat_lo) & (tlat <= lat_hi)

    lon_keep = np.ones(tlon.shape, dtype=bool)
    near_pole = lat_hi >= 88.0 or lat_lo <= -88.0
    s = np.mod(slon, 360.0)
    if not near_pole and (np.nanmax(slon) - np.nanmin(slon)) < 350.0 \
            and (np.nanmax(s) - np.nanmin(s)) < 350.0:
        t = np.mod(tlon, 360.0)
        lon_keep = (t >= np.nanmin(s) - pad) & (t <= np.nanmax(s) + pad)

    # Guard against an empty selection (degenerate extents): keep all if so.
    if not lat_keep.any():
        lat_keep = np.ones(tlat.shape, dtype=bool)
    if not lon_keep.any():
        lon_keep = np.ones(tlon.shape, dtype=bool)

    return tgt.isel({lat_name: lat_keep, lon_name: lon_keep})


# ---------------------------------------------------------------------------
# Weight caching: build the xESMF remap matrix once per (source, target, method)
# ---------------------------------------------------------------------------
def _extent(a: ArrayLike) -> "tuple[float, float]":
    """(min, max) of an array, rounded — a stable, cheap grid signature."""
    a = np.asarray(a)
    return round(float(np.nanmin(a)), 4), round(float(np.nanmax(a)), 4)


def _weights_signature(src: xr.Dataset, lon: ArrayLike, lat: ArrayLike,
                       method: str, periodic: bool) -> "tuple[str, str]":
    """A (label, signature) pair identifying the remap weights.

    The weights depend only on the two grids' geometry and the *method*, not on
    field values or time.  The source side mirrors the ``regional.grid`` ``.b``
    header — ``idm`` / ``jdm`` and the plon/plat min,max — so the three TOPAZ
    configurations (TP0/TP2/TP5), which differ in dimensions, are told apart
    without touching the ``.a`` arrays.  The target side uses the regular grid's
    shape and lon/lat extent (the part of e.g. GLORYS the weights depend on).

    *label* is a short human-readable stem for the filename; *signature* is the
    full string that gets hashed for collision-safe uniqueness.
    """
    jdm, idm = src.sizes["y"], src.sizes["x"]
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    label = f"{idm}x{jdm}_to_{lat.size}x{lon.size}_{method}"
    signature = (
        f"src:{idm}x{jdm}:lon{_extent(src['lon'].values)}:lat{_extent(src['lat'].values)}|"
        f"tgt:{lat.size}x{lon.size}:lon{_extent(lon)}:lat{_extent(lat)}|"
        f"{method}|periodic={bool(periodic)}"
    )
    return label, signature


def _cache_dir() -> str:
    """Directory for cached weight files: ``$XHYCOM_CACHE_DIR`` or XDG default."""
    base = os.environ.get("XHYCOM_CACHE_DIR")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME",
                             os.path.join(os.path.expanduser("~"), ".cache"))
        base = os.path.join(xdg, "xhycom", "regrid_weights")
    return base


def _record_manifest(cache_dir: str, key: str, label: str, signature: str,
                     tgt: "xr.Dataset | None") -> None:
    """Append a human-readable ``hash -> grids`` entry to the cache manifest.

    Best-effort: the cache works without it, so any I/O error is swallowed.  The
    target product name (e.g. GLORYS's ``source`` / ``title`` attr) is recorded
    when available so the cache directory is auditable.
    """
    manifest = os.path.join(cache_dir, "manifest.json")
    try:
        with open(manifest) as f:
            entries = json.load(f)
    except (OSError, ValueError):
        entries = {}
    if key in entries:
        return
    info = {"label": label, "signature": signature}
    if tgt is not None:
        name = tgt.attrs.get("source") or tgt.attrs.get("title")
        if name:
            info["target"] = name
    entries[key] = info
    try:
        with open(manifest, "w") as f:
            json.dump(entries, f, indent=2, sort_keys=True)
    except OSError:
        pass


def _resolve_weights(weights: "str | os.PathLike | bool | None",
                     src: xr.Dataset, lon: ArrayLike, lat: ArrayLike,
                     method: str, periodic: bool,
                     tgt: "xr.Dataset | None" = None) -> "str | None":
    """Map the *weights* argument to a weights-file path (or ``None``).

    ``None`` / ``False`` disable caching; ``True`` derives a content-keyed file
    in the cache directory (and records a manifest entry); a path/str names an
    explicit file the caller manages.
    """
    if weights is None or weights is False:
        return None
    if weights is True:
        label, signature = _weights_signature(src, lon, lat, method, periodic)
        key = hashlib.sha1(signature.encode()).hexdigest()[:8]
        cache_dir = _cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        _record_manifest(cache_dir, key, label, signature, tgt)
        return os.path.join(cache_dir, f"weights_{label}_{key}.nc")
    path = os.fspath(weights)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _target_depth(tgt: xr.Dataset) -> "np.ndarray | None":
    """1-D target depths (metres) from a grid Dataset, or ``None``."""
    if "depth" in tgt.variables:
        return np.asarray(tgt["depth"].values)
    return None


def _apply_target_mask(out: xr.Dataset, tgt: xr.Dataset,
                       surface_only: bool) -> xr.Dataset:
    """Mask *out* to the target ocean mask (1 = sea), if the grid carries one.

    The mask dims (GLORYS: ``longitude``/``latitude``/``depth``) are renamed to
    xhycom's output names and re-homed onto *out*'s coordinates so alignment is
    exact.  When *surface_only* (horizontal-only output, which still has a layer
    dimension), the surface level of a 3-D mask is used.
    """
    if "mask" not in tgt.variables:
        return out
    mask = tgt["mask"].astype(bool)
    rename = {old: new for old, new in
              (("longitude", "lon"), ("latitude", "lat"), ("depth", "depth"))
              if old in mask.dims}
    mask = mask.rename(rename)
    if surface_only and "depth" in mask.dims:
        mask = mask.isel(depth=0, drop=True)
    # Re-home onto out's coords (identical values, but ensures clean alignment).
    mask = mask.assign_coords(
        {d: out[d] for d in mask.dims if d in out.coords}
    )
    return out.where(mask)


def _nan_pole(out: xr.Dataset) -> xr.Dataset:
    """Blank the exact geographic-pole rows (``|lat| = 90``) to NaN.

    A regular lat/lon grid is singular at the pole — every longitude collapses
    to one physical point — and regular-grid ocean products carry no usable data
    there (GLORYS, for instance, is NaN at 90 N while its land/sea ``mask`` still
    marks the pole as sea, so masking alone won't remove it).  The remapped value
    on that row is therefore meaningless.

    This sets those rows to NaN **without changing the grid**: the pole row is
    kept, so the output stays the same shape as the target and a like-for-like
    difference against it (e.g. ``hycom - glorys``) still aligns — GLORYS is
    itself NaN at 90 N, so the blanked row simply drops out of the comparison.
    Dropping the row instead would shrink the grid and break that alignment.
    """
    if "lat" not in out.dims:
        return out
    keep = np.abs(np.asarray(out["lat"].values)) < 90.0 - 1e-6
    if keep.all():
        return out
    return out.where(xr.DataArray(keep, dims="lat", coords={"lat": out["lat"]}))


# ---------------------------------------------------------------------------
# Velocities: de-stagger to T-points and rotate to east/north
# ---------------------------------------------------------------------------
def _uv_to_east_north(ds: xr.Dataset, pang: xr.DataArray) -> xr.Dataset:
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


def _move_to_tpoint(da: xr.DataArray, like: xr.DataArray,
                    direction: str) -> xr.DataArray:
    """Re-home a de-staggered velocity onto the T-point lon/lat coords."""
    da = da.drop_vars([c for c in ("lon_u", "lat_u", "lon_v", "lat_v") if c in da.coords])
    attrs = dict(like.attrs)
    base = attrs.get("long_name", like.name)
    attrs["long_name"] = f"{direction} component of {base}"
    attrs["standard_name"] = f"{direction}_sea_water_velocity"
    attrs["comment"] = "de-staggered to T-points and rotated to geographic axes"
    da.attrs = attrs
    return da.rename(like.name)


def velocities_east_north(ds: xr.Dataset,
                          grid: "xr.Dataset | str | None" = None) -> xr.Dataset:
    """De-stagger HYCOM C-grid velocities to T-points and rotate to true east/north.

    HYCOM stores velocities on a staggered Arakawa C-grid with components along
    the **model grid axes** (``u-vel.`` along x, ``v-vel.`` along y).  On a
    curvilinear grid those axes are not east/north — they rotate across the
    domain (sharply near the grid's poles).  This averages each (u, v) pair onto
    the cell centre (T-point) and rotates the components onto the geographic axes
    using the grid angle ``pang``::

        east  = u * cos(pang) - v * sin(pang)
        north = u * sin(pang) + v * cos(pang)

    Unlike :func:`regrid_horizontal`, the **native curvilinear grid is kept** —
    only the velocity components are de-staggered and rotated.  This is the piece
    needed to compare model velocities against a regular product brought onto the
    HYCOM grid by :func:`regrid_to_hycom`: that function interpolates the
    product's velocities (e.g. GLORYS ``uo``/``vo``) as scalars, so they stay on
    geographic east/north axes.  Rotating the model side here puts both on the
    same axes on the same ``(y, x)`` grid, so they difference directly.

    Parameters
    ----------
    ds : xr.Dataset
        HYCOM Dataset that may contain one or more (u, v) pairs (``u-vel.`` /
        ``v-vel.``, ``u_btrop`` / ``v_btrop``, ``umix`` / ``vmix``,
        ``si_u`` / ``si_v``).  A Dataset with no velocity pair is returned
        unchanged.
    grid : xr.Dataset or str, optional
        HYCOM grid (``regional.grid`` path or a Dataset from
        :func:`xhycom.open_dataset`) supplying the rotation angle ``pang``.  May
        be omitted if ``ds`` already carries a ``pang`` coordinate.

    Returns
    -------
    xr.Dataset
        Copy of *ds* with each velocity pair de-staggered to the T-points and
        rotated to true eastward / northward, re-homed onto the T-point
        ``lon`` / ``lat`` coordinates and keeping the HYCOM names.  Each
        component's ``standard_name`` becomes
        ``eastward`` / ``northward_sea_water_velocity``.

    Notes
    -----
    The de-stagger averages each edge value with its neighbour, so the last
    column (for ``u``) and last row (for ``v``) — which have no neighbour —
    become NaN boundary cells, exactly as inside :func:`regrid_horizontal`.
    """
    if not any(v in ds for v in (_U_VARS | _V_VARS)):
        return ds
    grid = _load_grid(grid)
    pang = _get_pang(ds, grid)
    return _uv_to_east_north(ds, pang)


# ---------------------------------------------------------------------------
# Horizontal: curvilinear -> regular lon/lat (xESMF)
# ---------------------------------------------------------------------------
def regrid_horizontal(ds: xr.Dataset, lon: "ArrayLike | None" = None,
                      lat: "ArrayLike | None" = None,
                      grid: "xr.Dataset | str | None" = None,
                      target: "xr.Dataset | str | None" = None,
                      method: str = "conservative", periodic: bool = False,
                      mask_var: "str | None" = None,
                      apply_target_mask: bool = True,
                      subset_target: bool = True,
                      weights: "str | os.PathLike | bool | None" = None,
                      nan_pole: bool = True) -> xr.Dataset:
    """Regrid a HYCOM Dataset from its curvilinear grid to a regular lon/lat grid.

    Velocities (if present) are first de-staggered to T-points and rotated to
    true east/north (requires the grid angle ``pang``); everything is then
    interpolated with a single T-grid xESMF regridder.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from :func:`xhycom.open_dataset` / ``open_mfdataset``, opened
        **with** a ``grid=`` so that ``lon`` / ``lat`` 2-D coords are attached.
    lon, lat : array-like, optional
        1-D target longitudes and latitudes (degrees).  Omit when *target* is
        given.
    grid : xr.Dataset or str, optional
        Grid Dataset (from ``open_dataset`` on ``regional.grid``), or a path to
        ``regional.grid`` (same as ``open_dataset``'s ``grid=``).  Required to
        rotate velocities — it supplies ``pang`` — and to build source cell
        bounds for conservative regridding (it supplies ``qlon`` / ``qlat``).
        If ``ds`` already carries a ``pang`` coordinate, this may be omitted.
    target : xr.Dataset or str, optional
        A regular target grid (e.g. GLORYS), or a path to one, providing
        ``longitude`` / ``latitude`` (and, when used via :func:`regrid`,
        ``depth``).  Supplied instead of *lon* / *lat*.  If it carries a
        ``mask`` variable (1 = sea), land points are set to NaN in the output
        unless *apply_target_mask* is ``False``.
    method : str
        xESMF interpolation method (``"conservative"``, ``"bilinear"``,
        ``"patch"``, ...).  Default ``"conservative"``, which requires cell
        bounds (source bounds come from the grid's ``qlon`` / ``qlat`` — so
        ``grid=`` must be passed — and target bounds are built from the regular
        target spacing) and thickness-weights layered fields so that the layer
        volume content ``field * thickness`` is conserved.
    periodic : bool
        Whether the source grid is periodic in longitude.  Default ``False``.
    mask_var : str, optional
        Name of the variable used to derive the source land/sea mask.  By
        default the first available of ``temp`` / ``thknss`` is used
        (finite = ocean).
    apply_target_mask : bool
        If ``True`` (default) and *target* carries a ``mask``, apply it to the
        output.
    subset_target : bool
        If ``True`` (default) and the target lon/lat are derived from *target*,
        trim the target to the source's bounding box (plus a small pad) before
        building the regridder.  A regional source over a global target (e.g.
        TOPAZ2 → GLORYS) otherwise pays to remap every global cell, almost all
        of which receive no data.  No effect on the result inside the covered
        region; ignored when explicit *lon* / *lat* are passed.
    weights : str, path-like, or bool, optional
        Cache for the xESMF remap weights, which are the slow part of a remap
        and depend only on the two grids and *method* — not on the field or
        time.  ``True`` keys an auto-named file by source/target geometry (the
        grid ``idm``/``jdm`` + lon/lat extent, the GLORYS shape/extent) under
        ``$XHYCOM_CACHE_DIR`` (default ``~/.cache/xhycom/regrid_weights``), so
        TP0/TP2/TP5 × target × method each get their own and are reused across
        files.  A path names an explicit file (reused if it exists, else
        created).  ``None`` (default) disables caching.
    nan_pole : bool
        If ``True`` (default), set the exact geographic-pole rows
        (``|lat| = 90``) to NaN.  A regular lat/lon grid is singular there — a
        remap deposits a single, meaningless value — and products like GLORYS
        carry no data at 90 N (yet still mark it sea in their mask, so masking
        alone won't remove it).  The row is kept (not dropped), so the grid is
        unchanged and stays aligned with the target for a like-for-like
        difference.  Set ``False`` to keep the raw remapped pole value.

    Accepts a field either on hybrid layers (with ``thknss``) or already on
    fixed depth levels (a ``depth`` dimension, no ``thknss`` — e.g. the output
    of :func:`regrid_vertical`).  In the latter case the static 2-D land mask is
    skipped and NaN source cells are dropped per level, so depth-varying
    bathymetry is honoured during the lateral remap.

    Returns
    -------
    xr.Dataset
        Dataset with 1-D ``lon`` / ``lat`` dimension coordinates, on dims
        ``(time, k, lat, lon)`` for hybrid-layer input (``thknss`` retained for
        a subsequent vertical step) or ``(time, depth, lat, lon)`` for
        depth-level input.
    """
    try:
        import xesmf as xe
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "regrid_horizontal requires xESMF. Install with the 'regrid' extra "
            "via conda-forge (xESMF needs ESMF/esmpy):\n"
            "    conda env create -f ci/environment-regrid.yml"
        ) from exc

    grid = _load_grid(grid)

    tgt = None
    if target is not None:
        tgt = _open_target(target)
        if lon is None and lat is None:
            if subset_target:
                tgt = _subset_target(tgt, ds)
            lon, lat = _target_lonlat(tgt)
    if lon is None or lat is None:
        raise ValueError(
            "Provide target lon/lat either as lon=/lat= arrays or via "
            "target=<grid Dataset/path>."
        )

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
    # A static 2-D land mask is correct for hybrid layers — a HYCOM column is
    # wet at every layer or none — but wrong once the field is on fixed depth
    # levels, where wet/dry varies with depth (below-bottom cells are NaN).  In
    # that case we drop the static mask and let the apply skip NaN source cells
    # per level instead, so each depth level honours the bathymetry locally.
    on_depth_levels = "thknss" not in ds and "depth" in ds.dims
    src = ds
    mask2d = None if on_depth_levels else _ocean_mask(ds, mask_var)
    if mask2d is not None:
        src = ds.assign_coords(mask=mask2d)

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    target_ds = xr.Dataset({"lat": (["lat"], lat), "lon": (["lon"], lon)})

    conservative = method.startswith("conservative")

    # Conservative remapping needs cell corner bounds on both grids.
    if conservative:
        src = _add_source_bounds(src, grid)
        # Latitude edges must stay within [-90, 90]: midpoint extrapolation of a
        # target row sitting on the pole (e.g. GLORYS' top row at exactly 90 N)
        # otherwise lands a cell corner past the pole — an invalid spherical
        # coordinate.  Clamping caps that cell at the pole instead.
        target_ds = target_ds.assign(
            lon_b=("lon_b", _edges_1d(lon)),
            lat_b=("lat_b", np.clip(_edges_1d(lat), -90.0, 90.0)),
        )

    # Thickness-weight layered fields for conservative remapping so the
    # volume-integrated content (field * layer thickness) is conserved: remap
    # field*h and h, then divide.  Layer thickness varies horizontally, so a
    # plain area-conservative remap of an intensive field would not conserve
    # the layer content.  2-D fields are remapped as-is.  Tracers and
    # velocities are treated identically.
    remap_src = src
    layered, layer_attrs = [], {}
    if conservative and "thknss" in src:
        layer_dim = _layer_dim(src["thknss"])
        if layer_dim is not None:
            layered = [v for v in src.data_vars
                       if layer_dim in src[v].dims and v != "thknss"]
        if layered:
            remap_src = src.copy()
            for v in layered:
                layer_attrs[v] = dict(src[v].attrs)
                remap_src[v] = src[v] * src["thknss"]

    weights_path = _resolve_weights(weights, remap_src, lon, lat, method,
                                    periodic, tgt=tgt)
    reuse = weights_path is not None and os.path.exists(weights_path)
    regridder = xe.Regridder(
        remap_src, target_ds, method=method, periodic=periodic,
        ignore_degenerate=True, unmapped_to_nan=True,
        weights=weights_path if reuse else None,
    )
    if weights_path is not None and not reuse:
        regridder.to_netcdf(weights_path)
    out = regridder(remap_src, keep_attrs=True, skipna=on_depth_levels)

    if layered:
        denom = out["thknss"].where(out["thknss"] > 0)
        for v in layered:
            out[v] = out[v] / denom
            out[v].attrs = layer_attrs[v]

    out["lon"].attrs.setdefault("standard_name", "longitude")
    out["lon"].attrs.setdefault("units", "degrees_east")
    out["lat"].attrs.setdefault("standard_name", "latitude")
    out["lat"].attrs.setdefault("units", "degrees_north")

    if tgt is not None and apply_target_mask:
        out = _apply_target_mask(out, tgt, surface_only=True)
    if nan_pole:
        out = _nan_pole(out)
    return out


def _layer_dim(thknss: xr.DataArray) -> "str | None":
    """The vertical (layer) dimension of a thickness field, or ``None``.

    The layer dim is the one that is neither horizontal nor time.
    """
    for d in thknss.dims:
        if d not in ("y", "x", "lat", "lon", "time"):
            return d
    return None


def _edges_1d(centres: ArrayLike) -> np.ndarray:
    """Cell edges (n+1) for a 1-D monotonic centre array, by midpoints."""
    centres = np.asarray(centres, dtype="float64")
    mid = 0.5 * (centres[:-1] + centres[1:])
    first = centres[0] - (mid[0] - centres[0])
    last = centres[-1] + (centres[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _add_source_bounds(src: xr.Dataset,
                       grid: "xr.Dataset | None") -> xr.Dataset:
    """Attach 2-D corner bounds (lon_b/lat_b) to the curvilinear source grid.

    HYCOM ``qlon`` / ``qlat`` are the vorticity points sitting at the SW corner
    of each p-cell, so they provide the cell corners directly — we only need to
    extend by one row/column to close the (ny+1, nx+1) corner mesh, which we do
    by linear extrapolation of the final interval.
    """
    if grid is None or "qlon" not in grid or "qlat" not in grid:
        raise ValueError(
            "conservative regridding needs source cell corners — pass "
            "grid=<grid Dataset> so 'qlon'/'qlat' are available."
        )
    lon_b = _q_to_corners(np.asarray(grid["qlon"].values))
    lat_b = _q_to_corners(np.asarray(grid["qlat"].values))
    return src.assign_coords(
        lon_b=(("y_b", "x_b"), lon_b),
        lat_b=(("y_b", "x_b"), lat_b),
    )


def _q_to_corners(q: np.ndarray) -> np.ndarray:
    """(ny, nx) SW-corner array -> (ny+1, nx+1) full corner mesh by extrapolation."""
    ny, nx = q.shape
    out = np.empty((ny + 1, nx + 1), dtype="float64")
    out[:ny, :nx] = q
    out[:ny, nx] = q[:, -1] + (q[:, -1] - q[:, -2])     # extra east column
    out[ny, :nx] = q[-1, :] + (q[-1, :] - q[-2, :])     # extra north row
    out[ny, nx] = out[ny - 1, nx] + (out[ny - 1, nx] - out[ny - 2, nx])
    return out


def _get_pang(ds: xr.Dataset, grid: "xr.Dataset | None") -> xr.DataArray:
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


def _ocean_mask(ds: xr.Dataset,
                mask_var: "str | None") -> "xr.DataArray | None":
    """2-D (y, x) ocean mask (1 ocean, 0 land) from a representative field."""
    if mask_var is None:
        for cand in ("temp", "thknss"):
            if cand in ds:
                mask_var = cand
                break
    if mask_var is None or mask_var not in ds:
        return None
    da = ds[mask_var]
    # The HYCOM land mask is static in time, so a single step is enough.
    # Reducing over `time` would force reading the *entire* variable (every
    # step — tens of GB for a year) just to build a 2-D mask, which xESMF then
    # pulls in eagerly when the regridder is constructed — swamping any
    # weight-cache saving.  Take the first step instead.
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)   # drop=True: no scalar 'time' coord left
    reduce_dims = [d for d in da.dims if d not in ("y", "x")]
    finite = np.isfinite(da)
    if reduce_dims:
        finite = finite.any(reduce_dims)
    return finite.astype("int8")


# ---------------------------------------------------------------------------
# Reverse: a regular lon/lat product (e.g. GLORYS) -> HYCOM curvilinear grid
# ---------------------------------------------------------------------------
def _resolve_weights_to_hycom(weights: "str | os.PathLike | bool | None",
                              src: xr.Dataset, grid: xr.Dataset,
                              method: str, periodic: bool) -> "str | None":
    """Weights-file path for the reverse (product -> HYCOM) remap, or ``None``.

    Mirrors :func:`_resolve_weights` but keyed the other way round: the regular
    *product* is the source and the HYCOM ``plon``/``plat`` grid is the target.
    """
    if weights is None or weights is False:
        return None
    if weights is True:
        jdm, idm = np.asarray(grid["plat"].values).shape
        nlat, nlon = int(src["lat"].size), int(src["lon"].size)
        signature = (
            f"prod:{nlat}x{nlon}:lon{_extent(src['lon'].values)}:"
            f"lat{_extent(src['lat'].values)}|"
            f"hycom:{idm}x{jdm}:lon{_extent(grid['plon'].values)}:"
            f"lat{_extent(grid['plat'].values)}|{method}|periodic={bool(periodic)}"
        )
        key = hashlib.sha1(signature.encode()).hexdigest()[:8]
        cache_dir = _cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        label = f"{nlat}x{nlon}_to_{idm}x{jdm}_{method}"
        return os.path.join(cache_dir, f"weights_to_hycom_{label}_{key}.nc")
    path = os.fspath(weights)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def regrid_to_hycom(product: "xr.Dataset | xr.DataArray | str",
                    grid: "xr.Dataset | str", *,
                    method: str = "bilinear", periodic: bool = False,
                    like: "xr.Dataset | None" = None,
                    weights: "str | os.PathLike | bool | None" = None,
                    ) -> xr.Dataset:
    """Regrid a regular lon/lat product onto the HYCOM curvilinear ``(y, x)`` grid.

    The lateral inverse of :func:`regrid_horizontal`: a regular product such as
    GLORYS is interpolated onto HYCOM's native curvilinear grid, so it can be
    compared with the model *in the model's own space*.  This is the natural
    direction when the model grid is coarser than the product (regridding HYCOM
    up onto a finer product mostly interpolates, adding no information).

    Only the **horizontal** grid is changed: fields keep their own vertical
    coordinate (``depth``).  All fields are treated as **scalars** — vector
    components (e.g. GLORYS ``uo``/``vo``) are interpolated as-is and stay on
    geographic (east/north) axes; they are *not* rotated onto the model axes or
    re-staggered to the C-grid.

    Parameters
    ----------
    product : xr.Dataset, xr.DataArray, or str
        Regular lon/lat[/depth] source (``longitude``/``latitude`` or
        ``lon``/``lat``), or a path to one.
    grid : xr.Dataset or str
        HYCOM grid (``regional.grid`` path or a Dataset from
        :func:`xhycom.open_dataset`).  Supplies the target points
        ``plon``/``plat`` and, for conservative remapping, the cell corners
        ``qlon``/``qlat``.
    method : str
        xESMF method.  Default ``"bilinear"`` (point interpolation of a coarser
        product onto a finer grid).  ``"conservative"`` additionally needs the
        grid corners ``qlon``/``qlat``.
    periodic : bool
        Whether the *product* is periodic in longitude (e.g. a global grid).
        Default ``False``.
    like : xr.Dataset, optional
        A HYCOM field on the same ``(y, x)`` grid; its land/sea mask
        (finite = ocean, via the first of ``temp``/``thknss``) is applied to
        the output so product values are not carried onto HYCOM land.
    weights : str, path-like, or bool, optional
        Cache for the remap weights, as in :func:`regrid_horizontal`.  ``True``
        keys an auto-named file by product/HYCOM geometry under
        ``$XHYCOM_CACHE_DIR``; a path names an explicit file; ``None`` (default)
        disables caching.

    Returns
    -------
    xr.Dataset
        Product fields on HYCOM dims ``(..., y, x)`` with 2-D ``lon``/``lat``
        coordinates, lined up with a HYCOM Dataset for a like-for-like
        difference.
    """
    try:
        import xesmf as xe
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "regrid_to_hycom requires xESMF. Install with the 'regrid' extra "
            "via conda-forge (xESMF needs ESMF/esmpy):\n"
            "    conda env create -f ci/environment-regrid.yml"
        ) from exc

    grid = _load_grid(grid)
    if grid is None or "plon" not in grid or "plat" not in grid:
        raise ValueError(
            "regrid_to_hycom needs a HYCOM grid carrying 'plon'/'plat' — pass "
            "grid=<regional.grid path or Dataset>."
        )

    src = _open_target(product)
    # Standardise the product's horizontal coordinate names to lon/lat.
    rename = {old: new for old, new in (("longitude", "lon"), ("latitude", "lat"))
              if old in src.variables}
    src = src.rename(rename)
    if "lon" not in src.variables or "lat" not in src.variables:
        raise ValueError(
            "product needs 1-D longitude/latitude (or lon/lat) coordinates."
        )

    # Target = HYCOM p-points (2-D lon/lat on (y, x)).
    plon = np.asarray(grid["plon"].values)
    plat = np.asarray(grid["plat"].values)
    target_ds = xr.Dataset({"lat": (("y", "x"), plat), "lon": (("y", "x"), plon)})

    conservative = method.startswith("conservative")
    if conservative:
        if "qlon" not in grid or "qlat" not in grid:
            raise ValueError(
                "conservative regrid_to_hycom needs target cell corners — the "
                "grid must carry 'qlon'/'qlat'."
            )
        src = src.assign_coords(
            lon_b=("lon_b", _edges_1d(src["lon"].values)),
            lat_b=("lat_b", np.clip(_edges_1d(src["lat"].values), -90.0, 90.0)),
        )
        target_ds = target_ds.assign(
            lon_b=(("y_b", "x_b"), _q_to_corners(np.asarray(grid["qlon"].values))),
            lat_b=(("y_b", "x_b"), _q_to_corners(np.asarray(grid["qlat"].values))),
        )

    weights_path = _resolve_weights_to_hycom(weights, src, grid, method, periodic)
    reuse = weights_path is not None and os.path.exists(weights_path)
    regridder = xe.Regridder(
        src, target_ds, method=method, periodic=periodic,
        ignore_degenerate=True, unmapped_to_nan=True,
        weights=weights_path if reuse else None,
    )
    if weights_path is not None and not reuse:
        regridder.to_netcdf(weights_path)

    # Regrid only the fields that actually carry the horizontal dims; pass the
    # rest (1-D depth helpers, scalars) through untouched.
    spatial = [v for v in src.data_vars
               if "lon" in src[v].dims and "lat" in src[v].dims]
    out = regridder(src[spatial], keep_attrs=True, skipna=True)
    out = out.assign_coords(lon=(("y", "x"), plon), lat=(("y", "x"), plat))

    if like is not None:
        mask = _ocean_mask(like, None)
        if mask is not None:
            out = out.where(xr.DataArray(np.asarray(mask.values).astype(bool),
                                         dims=("y", "x")))
    return out


# ---------------------------------------------------------------------------
# Vertical: hybrid layers -> fixed depth levels (xgcm)
# ---------------------------------------------------------------------------
def regrid_vertical(ds: xr.Dataset, depth: ArrayLike,
                    method: str = "conservative", mask_edges: bool = True,
                    layer_dim: str = "k",
                    variables: "list[str] | None" = None) -> xr.Dataset:
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
        xgcm transform method.  ``"conservative"`` (default) conserves the
        depth-integral of each field: it builds depth bins centred on *depth*
        and returns the thickness-weighted layer mean in each.  ``"linear"``
        instead interpolates each field onto *depth* from the layer centres.
        Either way the output lands on the *depth* levels.
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

    depth = np.asarray(depth, dtype="float64")
    conservative = method.startswith("conservative")

    thknss_m = (ds["thknss"] if ds["thknss"].attrs.get("units") == "m"
                else ds["thknss"] / _ONEM)

    if conservative:
        # Conservative transform needs the layer *interfaces* (an "outer"
        # coordinate, N+1) as target_data and the target *bin edges* as the
        # target.  xgcm conserves the sum of cell values (extensive content),
        # so to conserve the depth-integral of an intensive field we transform
        # the thickness-weighted content and the thickness separately, then
        # divide — which also yields the correct mass-weighted mean in partly
        # filled bins.  Output lands on the bin centres.
        iface_dim = "z_i"
        z_target = layer_interface_depth(ds["thknss"], layer_dim=layer_dim,
                                         interface_dim=iface_dim)
        n_iface = z_target.sizes[iface_dim]
        ds_g = ds.assign_coords({iface_dim: np.arange(n_iface)})
        grid = xgcm.Grid(
            ds_g, coords={"Z": {"center": layer_dim, "outer": iface_dim}},
            periodic=False, autoparse_metadata=False,
        )
        # *depth* are the output levels (centres); build the surrounding bin
        # edges so the result lands on exactly those levels (consistent with
        # the linear method and with target grids that supply depth centres).
        target_edges = _edges_1d(depth)
        depth_out = depth
        # overlapping source thickness per target bin (shared denominator).
        thk_bin = grid.transform(
            thknss_m, "Z", target=target_edges, target_data=z_target,
            method=method, mask_edges=mask_edges,
        )
    else:
        z_target = layer_centre_depth(ds["thknss"], layer_dim=layer_dim)
        grid = xgcm.Grid(
            ds, coords={"Z": {"center": layer_dim}}, periodic=False,
            autoparse_metadata=False,
        )
        depth_out = depth

    if variables is None:
        variables = [
            name for name, da in ds.data_vars.items()
            if layer_dim in da.dims and name != "thknss"
        ]

    out_vars = {}
    for name in variables:
        da = ds[name]
        if conservative:
            content = grid.transform(
                da * thknss_m, "Z", target=target_edges, target_data=z_target,
                method=method, mask_edges=mask_edges,
            )
            transformed = content / thk_bin.where(thk_bin > 0)
        else:
            transformed = grid.transform(
                da, "Z", target=depth, target_data=z_target,
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
        "depth", depth_out,
        {"long_name": "depth", "units": "m", "positive": "down", "axis": "Z"},
    ))
    return out


def layer_centre_depth(thknss: xr.DataArray,
                       layer_dim: str = "k") -> xr.DataArray:
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


def layer_interface_depth(thknss: xr.DataArray, layer_dim: str = "k",
                          interface_dim: str = "z_i") -> xr.DataArray:
    """Layer-interface depths (metres, positive down) from HYCOM ``thknss``.

    Returns an ``N+1`` "outer" coordinate (surface at 0, then the cumulative
    interface below each layer) on *interface_dim*, suitable as ``target_data``
    for an xgcm conservative vertical transform.  Unit-aware like
    :func:`layer_centre_depth`, and carries the same tiny strictly-increasing
    ramp so massless layers don't collapse interfaces onto each other.

    Stays lazy: built with xarray ``cumsum`` / ``pad`` so a dask-backed
    ``thknss`` is never materialized.  Forcing it into a numpy array here would
    eagerly load every time step at once (a regridded year of HYCOM is tens of
    GB), which blows up the kernel during a :func:`regrid` call.
    """
    thknss_m = thknss if thknss.attrs.get("units") == "m" else thknss / _ONEM

    n = thknss_m.sizes[layer_dim]
    # interfaces: 0 at the surface, then cumulative layer bottoms.  Drop the
    # layer coordinate first so pad doesn't introduce a NaN-valued one.
    iface = thknss_m.cumsum(layer_dim)
    if layer_dim in iface.coords:
        iface = iface.drop_vars(layer_dim)
    iface = iface.pad({layer_dim: (1, 0)}, constant_values=0.0)
    iface = iface.rename({layer_dim: interface_dim})
    # xgcm's conservative transform needs the vertical (interface) dim in a
    # single chunk; pad would otherwise leave the prepended surface in its own
    # chunk.  Cheap to coalesce (~tens of levels) and keeps the array lazy.
    if iface.chunks is not None:
        iface = iface.chunk({interface_dim: -1})

    ramp = xr.DataArray(np.arange(n + 1) * 1e-4, dims=[interface_dim])
    iface = (iface + ramp).rename("depth")
    iface.attrs = {"long_name": "layer interface depth", "units": "m",
                   "positive": "down"}
    return iface


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def regrid(ds: xr.Dataset, lon: "ArrayLike | None" = None,
           lat: "ArrayLike | None" = None, depth: "ArrayLike | None" = None,
           grid: "xr.Dataset | str | None" = None,
           target: "xr.Dataset | str | None" = None,
           method: str = "conservative", z_method: str = "conservative",
           periodic: bool = False, mask_edges: bool = True,
           apply_target_mask: bool = True, subset_target: bool = True,
           weights: "str | os.PathLike | bool | None" = None,
           nan_pole: bool = True, order: str = "horizontal_first",
           variables: "list[str] | None" = None) -> xr.Dataset:
    """Regrid HYCOM output to a regular lon/lat/depth grid (lateral + vertical).

    Chains :func:`regrid_horizontal` and :func:`regrid_vertical`; *order* picks
    which runs first, and the two are **not** equivalent near topography and
    fronts:

    * ``"horizontal_first"`` (default) blends neighbouring cells *within each
      hybrid layer* before collapsing to depth levels.  In HYCOM's stratified
      interior the layers are isopycnals, so this mixes water of the same
      density — it is **along-isopycnal** and preserves water masses and the
      T–S relationship, the way tracers actually mix.
    * ``"vertical_first"`` puts every native column on the depth levels first —
      honouring its own bathymetry and giving better coverage on shelves and
      slopes — then blends horizontally at constant depth.  Where isopycnals
      tilt this mixes *across* density surfaces, which can smear water masses,
      so prefer it when geometric/bathymetric fidelity to a z-level product
      matters more than water-mass integrity.

    Both orders conserve the global integral (each step is conservative); the
    difference is local fidelity.

    Parameters
    ----------
    ds : xr.Dataset
        HYCOM Dataset opened with a ``grid=`` (so ``lon`` / ``lat`` exist).
    lon, lat : array-like, optional
        Target 1-D longitudes / latitudes (degrees).  Omit when *target* is
        given.
    depth : array-like, optional
        Target 1-D depths (metres, positive down).  Omit when *target* supplies
        ``depth``.
    grid : xr.Dataset or str, optional
        Grid Dataset, or a path to ``regional.grid`` (same as
        ``open_dataset``'s ``grid=``), needed to rotate velocities (supplies
        ``pang``) and to build source cell bounds for conservative regridding
        (``qlon``/``qlat``).
    target : xr.Dataset or str, optional
        A regular target grid (e.g. GLORYS), or a path to one, providing
        ``longitude`` / ``latitude`` / ``depth`` — supplied instead of
        *lon* / *lat* / *depth*.  Its ``mask`` (1 = sea), if present, is applied
        to the final 3-D output unless *apply_target_mask* is ``False``.
    method : str
        Horizontal interpolation method (xESMF). Default ``"conservative"``
        (requires ``grid=`` for source cell corners).
    z_method : str
        Vertical interpolation method (xgcm). Default ``"conservative"``.
    periodic : bool
        Source grid periodic in longitude. Default ``False``.
    mask_edges : bool
        Mask target depths outside the source column range. Default ``True``.
    apply_target_mask : bool
        If ``True`` (default) and *target* carries a ``mask``, apply it to the
        output.
    subset_target : bool
        If ``True`` (default) and lon/lat are taken from *target*, trim the
        target grid to the source's bounding box (plus a small pad) before
        regridding — the main speed-up for a regional source over a global
        target (e.g. TOPAZ2 → GLORYS).  Ignored when explicit *lon* / *lat*
        are passed.
    weights : str, path-like, or bool, optional
        Cache the xESMF remap weights so they are built once per
        (source grid, target grid, method) and reused across files.  ``True``
        auto-keys a file by grid geometry under ``$XHYCOM_CACHE_DIR``; a path
        names an explicit file; ``None`` (default) disables caching.  See
        :func:`regrid_horizontal`.
    nan_pole : bool
        If ``True`` (default), set the exact geographic-pole rows
        (``|lat| = 90``) to NaN — singular on a regular lat/lon grid and unused
        by products like GLORYS.  The row is kept, so the grid is unchanged and
        stays aligned with the target for differencing.  Set ``False`` to keep
        the raw remapped pole value.
    order : {"horizontal_first", "vertical_first"}
        Which step runs first (see above).  Default ``"horizontal_first"``
        (along-isopycnal, water-mass preserving).
    variables : list of str, optional
        Restrict the vertical step to these layered variables.

    Returns
    -------
    xr.Dataset
        Dataset on dims ``(time, depth, lat, lon)``.
    """
    tgt = None
    if target is not None:
        tgt = _open_target(target)
        if lon is None and lat is None:
            if subset_target:
                tgt = _subset_target(tgt, ds)
            lon, lat = _target_lonlat(tgt)
        if depth is None:
            depth = _target_depth(tgt)
    if depth is None:
        raise ValueError(
            "Provide target depth either as depth= or via a target= grid that "
            "carries a 'depth' coordinate."
        )

    # The target mask and pole-blanking are applied at the end on the full 3-D
    # output, so each intermediate step runs without them.
    if order == "horizontal_first":
        # Blend along the native (isopycnal) layers, then collapse to depth:
        # preserves water masses / the T-S relationship in the interior.
        ds = regrid_horizontal(ds, lon, lat, grid=grid, method=method,
                               periodic=periodic, weights=weights, nan_pole=False)
        ds = regrid_vertical(ds, depth, method=z_method, mask_edges=mask_edges,
                             variables=variables)
    elif order == "vertical_first":
        # Place each column on the depth levels first (honouring its own
        # bathymetry), then blend horizontally at constant depth.
        ds = regrid_vertical(ds, depth, method=z_method, mask_edges=mask_edges,
                             variables=variables)
        ds = regrid_horizontal(ds, lon, lat, grid=grid, method=method,
                               periodic=periodic, weights=weights, nan_pole=False)
    else:
        raise ValueError(
            f"order must be 'horizontal_first' or 'vertical_first', got {order!r}."
        )
    if tgt is not None and apply_target_mask:
        ds = _apply_target_mask(ds, tgt, surface_only=False)
    if nan_pole:
        ds = _nan_pole(ds)
    return ds
