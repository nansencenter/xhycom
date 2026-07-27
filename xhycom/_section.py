"""Section-plot utilities for visualising HYCOM cross-section data.

:func:`section_plot` takes the output of :func:`~xhycom.section_data` or
:func:`~xhycom.section_flux_density` and produces a filled-colour cross-
section with distance along the section on the x-axis and depth on the y-axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import matplotlib.axes


def section_plot(
    section_ds: xr.Dataset,
    var: str,
    *,
    ax: matplotlib.axes.Axes | None = None,
    depth_var: str = "depth_m",
    distance_coord: str = "distance_km",
    itime: int | None = None,
    depth_max: float | None = None,
    center_zero: bool = False,
    cmap: str | None = None,
    add_colorbar: bool = True,
    colorbar_label: str | None = None,
    title: str | None = None,
    flip_x: bool = False,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Filled-colour cross-section plot (distance × depth).

    Works with the output of :func:`~xhycom.section_data` (hydrographic
    fields on T-cell positions) and :func:`~xhycom.section_flux_density`
    (transport density on face positions).

    Parameters
    ----------
    section_ds:
        Dataset from :func:`~xhycom.section_data` or
        :func:`~xhycom.section_flux_density`.
    var:
        Variable in *section_ds* to plot (e.g. ``"temp"``,
        ``"flux_density"``).
    ax:
        Existing matplotlib axes to draw into.  A new figure and axes are
        created when ``None``.
    depth_var:
        Name of the depth variable in *section_ds*.  Default ``"depth_m"``
        (produced by both extraction functions).
    distance_coord:
        Name of the distance coordinate on the section dimension.
        Default ``"distance_km"``.
    itime:
        Time index to select when *section_ds* has a ``time`` dimension.
        Defaults to ``0``.
    depth_max:
        Clip the y-axis at this depth [m].
    center_zero:
        If ``True``, use a diverging colormap centred at zero and set
        symmetric colour limits.  Useful for signed transport density.
    cmap:
        Colormap name.  Defaults to ``"RdBu_r"`` when *center_zero* is
        ``True``, otherwise ``"viridis"``.
    add_colorbar:
        Add a colorbar.  Default ``True``.
    colorbar_label:
        Colorbar label.  Falls back to the variable's ``long_name`` attribute.
    title:
        Axes title.  Defaults to the variable name.
    flip_x:
        If ``True``, invert the x-axis so that distance increases from right
        to left.  Useful when the section is defined east → west and you
        want the conventional map orientation (west on the left).
    **kwargs:
        Forwarded to :func:`matplotlib.pyplot.pcolormesh`.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> sec = xhycom.section_data(ds, resolved)
    >>> xhycom.section_plot(sec, "temp", depth_max=500, cmap="thermal")

    >>> sec_flux = xhycom.section_flux_density(ds, resolved)
    >>> xhycom.section_plot(sec_flux, "flux_density", center_zero=True)
    """
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # Extract the data array and handle the time dimension
    # ------------------------------------------------------------------
    da: xr.DataArray = section_ds[var]

    if "time" in da.dims:
        t = itime if itime is not None else 0
        da = da.isel(time=t)
        if depth_var in section_ds and "time" in section_ds[depth_var].dims:
            depth_da = section_ds[depth_var].isel(time=t)
        elif depth_var in section_ds:
            depth_da = section_ds[depth_var]
        else:
            depth_da = None
    else:
        depth_da = section_ds[depth_var] if depth_var in section_ds else None

    # da now has dims (k, section) or (k, face)
    values = da.values.astype(float)  # (k, n)

    # ------------------------------------------------------------------
    # Distance axis
    # ------------------------------------------------------------------
    if distance_coord in da.coords:
        dist = da.coords[distance_coord].values.astype(float)
    elif depth_da is not None and distance_coord in depth_da.coords:
        dist = depth_da.coords[distance_coord].values.astype(float)
    else:
        dist = np.arange(values.shape[-1], dtype=float)

    n_k, n_s = values.shape

    # ------------------------------------------------------------------
    # Depth axis (may be 1-D for z-level or 2-D for isopycnal/hybrid)
    # ------------------------------------------------------------------
    # Use a 1-D representative depth profile so that pcolormesh receives
    # monotonic coordinate arrays — avoids the "not monotonically
    # increasing" warning and produces correct cell-edge computation.
    if depth_da is not None:
        depth_raw = depth_da.values.astype(float)
        if depth_raw.ndim == 1:
            depth_1d = depth_raw
        else:
            # Hybrid/isopycnal: mean depth of each layer across the section,
            # ignoring sub-bathymetry NaN cells.
            depth_1d = np.nanmean(depth_raw, axis=1)
            all_nan = ~np.isfinite(depth_1d)
            if np.any(all_nan):
                depth_1d[all_nan] = np.arange(n_k, dtype=float)[all_nan]
    else:
        depth_1d = np.arange(n_k, dtype=float)

    # ------------------------------------------------------------------
    # Colormap and normalisation
    # ------------------------------------------------------------------
    if center_zero:
        if cmap is None:
            cmap = "RdBu_r"
        if "vmin" not in kwargs and "vmax" not in kwargs:
            vmax = (
                float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 98))
                if np.any(np.isfinite(values))
                else 1.0
            )
            kwargs.setdefault("vmin", -vmax)
            kwargs.setdefault("vmax", vmax)
    else:
        if cmap is None:
            cmap = "viridis"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    pc = ax.pcolormesh(
        dist,
        depth_1d,
        np.ma.masked_invalid(values),
        cmap=cmap,
        shading="nearest",
        **kwargs,
    )

    ax.set_xlabel("Distance along section (km)")
    ax.set_ylabel("Depth (m)")
    if depth_max is not None:
        ax.set_ylim(0, depth_max)
    ax.invert_yaxis()
    if flip_x:
        ax.invert_xaxis()

    if add_colorbar:
        cb = plt.colorbar(pc, ax=ax, pad=0.02)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)
        elif hasattr(da, "attrs"):
            parts = []
            if da.attrs.get("long_name"):
                parts.append(da.attrs["long_name"])
            if da.attrs.get("units"):
                parts.append(f"[{da.attrs['units']}]")
            if parts:
                cb.set_label(" ".join(parts))

    ax.set_title(title if title is not None else var)
    return ax
