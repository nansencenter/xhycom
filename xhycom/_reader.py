"""Readers for all HYCOM .ab file types, plus file type detection."""
import re
from collections import defaultdict

import numpy as np
import xarray as xr

from ._abfile import (
    ABFileBathy,
    ABFileGrid,
    ABFileArchv,
    grid_ordered_fieldnames,
)
from ._time import model_day_to_datetime


def _fill(arr):
    """Masked array → float64 ndarray, masked values replaced by NaN."""
    return np.ma.filled(arr.astype(np.float64), np.nan)


# ---------------------------------------------------------------------------
# Staggered C-grid helpers
# ---------------------------------------------------------------------------

# Standard HYCOM field names that live on U-points and V-points.
_U_VARS = frozenset({"u-vel.", "u_btrop", "umix"})
_V_VARS = frozenset({"v-vel.", "v_btrop", "vmix"})

# T-point variables used as the preferred source for the dens coordinate.
_TPOINT_DENS_PRIORITY = ("thknss", "temp", "salin", "density")


def _h_coords(fname, grid_ds):
    """Horizontal lon/lat coordinates for fname's staggering point.

    T-point → lon/lat  (plon/plat)
    U-point → lon_u/lat_u  (ulon/ulat)
    V-point → lon_v/lat_v  (vlon/vlat)
    """
    if grid_ds is None:
        return {}
    if fname in _U_VARS:
        return {
            "lon_u": (["y", "x"], grid_ds["ulon"].values),
            "lat_u": (["y", "x"], grid_ds["ulat"].values),
        }
    if fname in _V_VARS:
        return {
            "lon_v": (["y", "x"], grid_ds["vlon"].values),
            "lat_v": (["y", "x"], grid_ds["vlat"].values),
        }
    return {
        "lon": (["y", "x"], grid_ds["plon"].values),
        "lat": (["y", "x"], grid_ds["plat"].values),
    }


def _v_dim(levels):
    """Vertical dimension name for a multi-level variable.

    Layer centres (k = 1..N)       → 'k'
    Layer interfaces (k = 0..N)    → 'ki'
    """
    return "ki" if 0 in levels else "k"


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

def detect_filetype(basename):
    """Detect the type of a HYCOM ``.ab`` file pair from the ``.b`` header.

    Parameters
    ----------
    basename : str
        Path without the ``.a`` / ``.b`` extension.

    Returns
    -------
    str
        One of ``"archv"``, ``"grid"``, ``"bathy"``, or ``"forcing"``.

    Raises
    ------
    ValueError
        If the file type cannot be determined.
    """
    with open(basename + ".b") as f:
        header = f.read(512)

    if re.search(r"'iversn'", header):
        return "archv"
    if re.search(r"'mapflg'", header):
        return "grid"
    if re.search(r"dtime1,range", header):
        return "forcing"
    if re.search(r"min,max\s+depth", header):
        return "bathy"

    raise ValueError(
        f"Cannot determine file type of {basename!r}.  "
        "Expected a HYCOM archv, grid, bathy, or forcing .ab file."
    )


# ---------------------------------------------------------------------------
# Lazy-loading helper (module-level so Dask can serialise it)
# ---------------------------------------------------------------------------

def _read_field_lazy(basename, fname, level, endian):
    """Open the archive and read one 2-D field slice.

    This function is called by Dask tasks; it must live at module level so
    that Dask can serialise it across workers.
    """
    af = ABFileArchv(basename, "r", endian=endian)
    raw = af.read_field(fname, level)
    af.close()
    return _fill(raw)


# ---------------------------------------------------------------------------
# Per-type readers (internal)
# ---------------------------------------------------------------------------

def read_archv(basename, grid_ds=None, endian="big", chunks=None):
    """Read a HYCOM archive ``.ab`` file pair into an ``xr.Dataset``.

    Parameters
    ----------
    chunks : int, dict, "auto", or None
        If not ``None``, field data are read lazily via Dask — the ``.a``
        file is not touched until the returned Dataset is computed.
        The value is forwarded to ``ds.chunk()`` to set chunk boundaries
        (e.g. ``{"k": 1}`` for one layer per chunk).
    """
    af = ABFileArchv(basename, "r", endian=endian)

    field_kdens = defaultdict(dict)
    for rec in af.fields.values():
        field_kdens[rec["field"]][rec["k"]] = rec["dens"]

    jdm, idm = af.jdm, af.idm
    yrflag = af.yrflag
    first_rec = next(iter(af.fields.values())) if af.fields else {}
    model_day = first_rec.get("day")
    global_attrs = {"iversn": af.iversn, "iexpt": af.iexpt, "yrflag": yrflag}

    # k→dens for layer-centre variables only (interfaces sit on 'ki', not 'k').
    # Pass 1: union from all centre vars. Pass 2: prefer T-point values.
    global_kdens = {}
    for fname, kdens in field_kdens.items():
        if len(kdens) > 1 and 0 not in kdens:
            global_kdens.update(kdens)
    for fname in _TPOINT_DENS_PRIORITY:
        if fname in field_kdens and len(field_kdens[fname]) > 1 and 0 not in field_kdens[fname]:
            global_kdens.update(field_kdens[fname])
            break

    if chunks is not None:
        # Lazy path: .b header is parsed above (cheap text read); close the
        # file now.  Each 2-D slab is wrapped in a Dask delayed so the .a
        # binary data is only read when the array is computed.
        af.close()
        try:
            import dask
            import dask.array as da
        except ImportError:
            raise ImportError(
                "Dask is required for lazy/chunked loading. "
                "Install it with: pip install dask"
            )

        def _get_slab(fname, k):
            return da.from_delayed(
                dask.delayed(_read_field_lazy)(basename, fname, k, endian),
                shape=(jdm, idm),
                dtype=np.float64,
            )

        def _stack(slabs):
            return da.stack(slabs, axis=0)

    else:
        # Eager path: file is open, read all data directly.
        def _get_slab(fname, k):
            return _fill(af.read_field(fname, k))

        def _stack(slabs):
            return np.stack(slabs)

    data_vars = {}
    for fname, kdens in field_kdens.items():
        levels = sorted(kdens)
        h_coords = _h_coords(fname, grid_ds)
        if len(levels) == 1:
            data_vars[fname] = xr.DataArray(
                _get_slab(fname, levels[0]), dims=["y", "x"],
                coords=h_coords, name=fname,
            )
        else:
            vdim = _v_dim(levels)
            arr = _stack([_get_slab(fname, k) for k in levels])
            coords = dict(h_coords)
            coords[vdim] = (vdim, levels)
            data_vars[fname] = xr.DataArray(
                arr, dims=[vdim, "y", "x"],
                coords=coords, name=fname,
            )

    if chunks is None:
        af.close()

    ds = xr.Dataset(data_vars, attrs=global_attrs)

    if global_kdens:
        k_vals = sorted(global_kdens)
        ds = ds.assign_coords(dens=("k", [global_kdens[k] for k in k_vals]))

    if model_day is not None and yrflag is not None:
        t = model_day_to_datetime(model_day, yrflag)
        ds = ds.expand_dims({"time": [t]})

    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


def read_grid(basename, endian="big"):
    """Read a HYCOM ``regional.grid`` ``.ab`` file pair into an ``xr.Dataset``."""
    gf = ABFileGrid(basename, "r", endian=endian)
    data_vars = {}
    for fname in grid_ordered_fieldnames:
        raw = gf.read_field(fname)
        if raw is not None:
            data_vars[fname] = xr.DataArray(
                _fill(raw), dims=["y", "x"], name=fname,
            )
    gf.close()
    return xr.Dataset(data_vars)


def read_bathy(basename, grid_ds, endian="big"):
    """Read a HYCOM bathymetry ``.ab`` file pair into an ``xr.Dataset``.

    *grid_ds* is required to supply ``idm`` / ``jdm`` (not stored in the
    bathymetry file itself) and to attach ``lon`` / ``lat`` coordinates.
    """
    jdm, idm = grid_ds["plon"].shape
    bf = ABFileBathy(basename, "r", idm=idm, jdm=jdm, endian=endian)
    raw = bf.read_field("depth")
    bf.close()

    da = xr.DataArray(
        _fill(raw),
        dims=["y", "x"],
        coords={
            "lon": (["y", "x"], grid_ds["plon"].values),
            "lat": (["y", "x"], grid_ds["plat"].values),
        },
        attrs={"units": "m", "long_name": "sea floor depth"},
        name="depth",
    )
    return xr.Dataset({"depth": da})
