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
# Per-type readers (internal)
# ---------------------------------------------------------------------------

def read_archv(basename, grid_ds=None, endian="big"):
    """Read a HYCOM archive ``.ab`` file pair into an ``xr.Dataset``."""
    af = ABFileArchv(basename, "r", endian=endian)

    field_kdens = defaultdict(dict)
    for rec in af.fields.values():
        field_kdens[rec["field"]][rec["k"]] = rec["dens"]

    yrflag = af.yrflag
    first_rec = next(iter(af.fields.values())) if af.fields else {}
    model_day = first_rec.get("day")
    global_attrs = {"iversn": af.iversn, "iexpt": af.iexpt, "yrflag": yrflag}

    base_coords = {}
    if grid_ds is not None:
        base_coords["lon"] = (["y", "x"], grid_ds["plon"].values)
        base_coords["lat"] = (["y", "x"], grid_ds["plat"].values)

    # T-point variables in priority order for choosing the dens coordinate.
    # dens should reflect the layer's nominal target density, which is defined
    # at the tracer point. U/V-point values are spatial averages of neighbouring
    # T-point cells and will differ slightly on a staggered C-grid.
    _TPOINT_VARS = ("thknss", "temp", "salin", "density")

    data_vars = {}
    global_kdens = {}
    _dens_from_tpoint = False

    for fname, kdens in field_kdens.items():
        levels = sorted(kdens)
        if len(levels) == 1:
            raw = af.read_field(fname, levels[0])
            data_vars[fname] = xr.DataArray(
                _fill(raw), dims=["y", "x"],
                coords=dict(base_coords), name=fname,
            )
        else:
            stack = np.stack([_fill(af.read_field(fname, k)) for k in levels])
            coords = dict(base_coords)
            coords["k"] = ("k", levels)
            data_vars[fname] = xr.DataArray(
                stack, dims=["k", "y", "x"],
                coords=coords, name=fname,
            )
            is_tpoint = fname in _TPOINT_VARS
            if is_tpoint or not _dens_from_tpoint:
                global_kdens.update(kdens)
                _dens_from_tpoint = _dens_from_tpoint or is_tpoint

    af.close()
    ds = xr.Dataset(data_vars, attrs=global_attrs)

    if global_kdens:
        k_vals = sorted(global_kdens)
        ds = ds.assign_coords(dens=("k", [global_kdens[k] for k in k_vals]))

    if model_day is not None and yrflag is not None:
        t = model_day_to_datetime(model_day, yrflag)
        ds = ds.expand_dims({"time": [t]})

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
