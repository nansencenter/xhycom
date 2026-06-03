"""xhycom — xarray interface for HYCOM a.b binary output files.

Public API
----------
open_dataset(path, ...)       Open any HYCOM .ab file pair (archv, grid, bathy).
open_mfdataset(paths, ...)    Open a time series of archive snapshots.
"""
import warnings

import xarray as xr

from ._abfile import ABFile
from ._discovery import find_archv_files
from ._reader import detect_filetype, read_archv, read_bathy, read_grid

__version__ = "0.1.0"
__all__ = ["open_dataset", "open_mfdataset"]


def _load_grid(grid, endian):
    """Accept a path string or pre-loaded Dataset; return a Dataset."""
    if grid is None:
        return None
    if isinstance(grid, xr.Dataset):
        return grid
    return open_dataset(grid, endian=endian)


def open_dataset(path, grid=None, endian="big", chunks=None):
    """Open a HYCOM ``.ab`` file pair as an ``xr.Dataset``.

    Automatically detects the file type (archive, grid, or bathymetry) from
    the ``.b`` header, so the same function works for all HYCOM output files.

    If *path* is a glob pattern or directory, it is forwarded to
    :func:`open_mfdataset` automatically.

    Parameters
    ----------
    path : str
        Path to the file.  The ``.a`` / ``.b`` extension is optional.
        Glob patterns (``*``, ``?``, ``[``) and directory paths are forwarded
        to :func:`open_mfdataset`.
    grid : str or xr.Dataset, optional
        Path to ``regional.grid`` (without extension), or a Dataset already
        returned by a previous ``open_dataset`` call on a grid file.

        * For **archive** files: attaches ``lon`` / ``lat`` as non-dimension
          coordinates on every variable.
        * For **bathymetry** files: required (grid dimensions and coordinates
          are not stored in the bathymetry file itself).
        * For **grid** files: ignored.

    endian : str
        Byte order: ``"big"`` (default), ``"little"``, or ``"native"``.
    chunks : int, dict, or "auto", optional
        If provided, the returned Dataset is chunked with Dask.  Passed
        directly to ``ds.chunk()``.  Example: ``chunks={"k": 1}`` to chunk
        one layer at a time.

    Returns
    -------
    xr.Dataset
        Contents depend on file type:

        **Archive** (``archv.YYYY_DDD_HH``)
            * ``time`` dimension of size 1.
            * 2-D fields on ``(time, y, x)``.
            * Layered fields on ``(time, k, y, x)`` with ``k`` (layer index,
              1-based) and ``dens`` (target sigma-2 density) coordinates.
            * Global attributes ``iversn``, ``iexpt``, ``yrflag``.

        **Grid** (``regional.grid``)
            * All 19 grid variables on ``(y, x)``: ``plon``, ``plat``,
              ``ulon``, ``ulat``, ``vlon``, ``vlat``, ``qlon``, ``qlat``,
              ``pang``, ``scpx``, ``scpy``, ``scqx``, ``scqy``, ``scux``,
              ``scuy``, ``scvx``, ``scvy``, ``cori``, ``pasp``.

        **Bathymetry** (``depth_*``)
            * Single ``depth`` variable (metres) on ``(y, x)``.

        ``lon`` / ``lat`` non-dimension coordinates are attached to every
        variable when *grid* is supplied (archive and bathymetry files).

    Raises
    ------
    ValueError
        If the file type cannot be detected, or if *grid* is not provided for
        a bathymetry file.

    Examples
    --------
    Open the grid:

    >>> grid = xhycom.open_dataset("topo/regional.grid")

    Open the bathymetry (grid required for dimensions and coordinates):

    >>> bathy = xhycom.open_dataset("topo/depth_TP2a0.10_04",
    ...                             grid="topo/regional.grid")

    Open a single archive snapshot with grid coordinates:

    >>> ds = xhycom.open_dataset("data/archv.2020_001_00",
    ...                          grid="topo/regional.grid")

    Re-use a pre-loaded grid Dataset to avoid reading the file twice:

    >>> grid = xhycom.open_dataset("topo/regional.grid")
    >>> bathy = xhycom.open_dataset("topo/depth_TP2a0.10_04", grid=grid)
    >>> ds    = xhycom.open_dataset("data/archv.2020_001_00", grid=grid)
    """
    path = str(path)

    # Forward globs and directories to open_mfdataset.
    import os as _os
    if any(c in path for c in "*?[") or _os.path.isdir(path):
        return open_mfdataset(path, grid=grid, endian=endian, chunks=chunks)

    basename = ABFile.strip_ab_ending(path)
    filetype = detect_filetype(basename)

    grid_ds = _load_grid(grid, endian)

    if filetype == "grid":
        ds = read_grid(basename, endian=endian)
    elif filetype == "archv":
        # chunks is handled inside read_archv: data is never loaded eagerly
        # when chunks is set — Dask tasks are created instead.
        return read_archv(basename, grid_ds=grid_ds, endian=endian, chunks=chunks)
    elif filetype == "bathy":
        if grid_ds is None:
            raise ValueError(
                "grid= is required to open a bathymetry file — it provides "
                "the grid dimensions (idm, jdm) and lon/lat coordinates.\n"
                "Example: open_dataset('depth_...', grid='regional.grid')"
            )
        ds = read_bathy(basename, grid_ds=grid_ds, endian=endian)
    else:
        raise ValueError(f"Unsupported file type {filetype!r} for open_dataset.")

    return ds.chunk(chunks) if chunks is not None else ds


def open_mfdataset(paths, grid=None, endian="big", skip_errors=False, chunks=None):
    """Open multiple HYCOM archive ``.ab`` file pairs as a single ``xr.Dataset``.

    Snapshots are concatenated along a ``time`` dimension in chronological
    order.

    Parameters
    ----------
    paths : str or list of str
        One of:

        * A directory path — all ``archv.`` / ``archm.YYYY_DDD_HH.[ab]``
          pairs found inside are used.
        * A glob pattern such as ``"data/archm.1993_*.a"``.
        * An explicit list of archive basenames or filenames.

    grid : str or xr.Dataset, optional
        Grid file path or pre-loaded Dataset.  Loaded once and shared across
        all files.
    endian : str
        Byte order.
    skip_errors : bool
        If ``True``, files that fail to open are skipped with a warning
        rather than raising an exception.  Default ``False``.
    chunks : int, dict, or "auto", optional
        If provided, the returned Dataset is chunked with Dask.  Passed
        directly to ``ds.chunk()``.  Example: ``chunks={"time": 1}``.

    Returns
    -------
    xr.Dataset
        Combined Dataset with a ``time`` dimension spanning all snapshots.

    Examples
    --------
    Open all snapshots in a directory:

    >>> ds = xhycom.open_mfdataset("data/", grid="topo/regional.grid")

    Open a subset using a glob:

    >>> ds = xhycom.open_mfdataset("data/archv.2020_*.a",
    ...                            grid="topo/regional.grid")

    Compute time-mean surface salinity:

    >>> ds["saln"].isel(k=0).mean("time").plot(x="lon", y="lat")
    """
    if isinstance(paths, str):
        basenames = find_archv_files(paths)
    else:
        basenames = [ABFile.strip_ab_ending(str(p)) for p in paths]

    grid_ds = _load_grid(grid, endian)

    datasets = []
    for basename in basenames:
        try:
            datasets.append(read_archv(basename, grid_ds=grid_ds, endian=endian, chunks=chunks))
        except Exception as exc:
            if skip_errors:
                warnings.warn(f"Skipping {basename!r}: {exc}", stacklevel=2)
            else:
                raise

    if not datasets:
        raise RuntimeError("No files were successfully opened.")

    ds = xr.concat(datasets, dim="time", data_vars="minimal", compat="override")
    return ds.chunk(chunks) if chunks is not None else ds
