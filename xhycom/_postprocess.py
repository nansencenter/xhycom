"""Post-processing for HYCOM datasets: native-unit conversions + derived fields.

``open_dataset`` / ``open_mfdataset`` read HYCOM ``.ab`` files verbatim — fields
keep their native units (e.g. ``srfhgt`` as geopotential, layer thicknesses as
pressure in Pa) and only the variables physically stored on disk are present.

:func:`postprocess` turns that raw output into something analysis-ready:

* **Unit conversions** — sea-surface height and the pressure-thickness
  diagnostics are converted to metres (see ``_UNIT_CONVERSIONS``).  Two
  different constants are involved and easy to confuse:

  - ``srfhgt`` is geopotential ``g*eta`` (m^2 s^-2) -> divide by ``g = 9.806``.
  - ``thknss`` / ``mix_dpth`` / ... are pressure (Pa) -> divide by
    ``onem = 9806`` (= rho0 * g).

* **Derived grid fields** — ``area = scpx * scpy`` on grid files.
* **Land/sea mask** — ``landmask`` (1 ocean / 0 land) from the bathymetry.

It is exposed both via ``open_dataset(..., postprocess=True)`` and as the
public :func:`xhycom.postprocess` so it can be applied to an existing Dataset.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

# Gravity (m s^-2) for geopotential -> height; "onem" (Pa per metre) for
# pressure -> thickness.  Keep these named so the two are never conflated.
_G = 9.806
_ONEM = 9806.0

# name -> (factor, new units, new long_name or None)
_UNIT_CONVERSIONS: dict[str, tuple[float, str, str | None]] = {
    "srfhgt":   (1.0 / _G,    "m", "sea surface height"),
    "thknss":   (1.0 / _ONEM, "m", "layer thickness"),
    "mix_dpth": (1.0 / _ONEM, "m", "mixed layer depth"),
    "bl_dpth":  (1.0 / _ONEM, "m", "boundary layer depth"),
    "thmix":    (1.0 / _ONEM, "m", "mixed layer thickness"),
}


def postprocess(ds: xr.Dataset) -> xr.Dataset:
    """Return a copy of *ds* with native units converted and derived fields added.

    Idempotent-ish: a field already carrying ``units='m'`` is not re-scaled, and
    derived fields are not recomputed if already present.

    Parameters
    ----------
    ds : xr.Dataset
        A Dataset from :func:`xhycom.open_dataset` / ``open_mfdataset`` (archive,
        grid, or bathymetry).

    Returns
    -------
    xr.Dataset
        New Dataset; lazy/Dask-backed inputs stay lazy.
    """
    ds = ds.copy()

    for name, (factor, units, long_name) in _UNIT_CONVERSIONS.items():
        if name in ds.data_vars and ds[name].attrs.get("units") != units:
            ds[name] = _scale(ds[name], factor, units, long_name)

    if "scpx" in ds and "scpy" in ds and "area" not in ds:
        ds["area"] = _grid_area(ds)

    if "depth" in ds and "landmask" not in ds:
        ds["landmask"] = _landmask(ds["depth"])

    return ds


def _scale(da: xr.DataArray, factor: float, units: str,
           long_name: str | None = None) -> xr.DataArray:
    """Scale a DataArray, replacing its units/long_name and recording the source."""
    native = da.attrs.get("units", "native")
    out = da * factor                       # lazy for Dask; drops attrs
    attrs = dict(da.attrs)
    attrs["units"] = units
    if long_name is not None:
        attrs["long_name"] = long_name
    attrs["comment"] = f"converted from {native} (factor {factor:.6g})"
    out.attrs = attrs
    return out.rename(da.name)


def _grid_area(ds: xr.Dataset) -> xr.DataArray:
    area = (ds["scpx"] * ds["scpy"]).rename("area")
    area.attrs = {"long_name": "grid cell area", "units": "m2"}
    return area


def _landmask(depth: xr.DataArray) -> xr.DataArray:
    mask = depth.notnull().astype("int8").rename("landmask")
    mask.attrs = {
        "long_name": "land-sea mask",
        "units": "1",
        "flag_values": np.array([0, 1], dtype="int8"),
        "flag_meanings": "land ocean",
        "comment": "1 = ocean, 0 = land (derived from bathymetry)",
    }
    return mask
