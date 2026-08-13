# xhycom

**xhycom** is a Python package for working with HYCOM model output:

- Reads HYCOM `.a/.b` output directly into a labelled [xarray](why-xarray) Dataset, with coordinates, units, and a decoded time axis attached automatically
- Regrids between HYCOM's native curvilinear grid and regular lon/lat/depth grids (and back), for comparison with reanalyses like GLORYS
- Computes volume, heat, salt, and freshwater transports across pre-defined or customized transects, and plots cross-sections
- More HYCOM diagnostics coming

## Why xhycom?

### Reading HYCOM output

xhycom reads HYCOM `.a/.b` output directly into a labelled xarray `Dataset` (names, coordinates, units, a decoded time axis, and lazy out-of-memory access) with no intermediate files:

```python
import xhycom

ds = xhycom.open_dataset("archv.2020_001_00", grid="regional.grid")
ds["temp"].isel(time=0, k=0).plot()        # lon/lat/time already attached
```

The key difference from other workflows is that xhycom can open **decades of output without loading any field data into memory**:

```python
# ~1 TB on disk; ~100 MB RAM.
ds = xhycom.open_mfdataset("data/archm.199*-202*", grid="regional.grid",
                           chunks={"time": 1})
ds["temp"].isel(k=0).mean("time").compute().plot(x="lon", y="lat")
```

The table below compares xhycom with the two most common existing workflows: reading directly with [`abfile`](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/pythonlibs/abfile), and converting to NetCDF first with [`m2nc`](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/hycom/MSCPROGS/src/ExtractNC2D).

|                       | `abfile` + NumPy             | `m2nc` → NetCDF              | xhycom                          |
| --------------------- | ---------------------------- | ---------------------------- | ------------------------------- |
| **Output**            | one masked array per field   | NetCDF file                  | labelled `xr.Dataset` (write to NetCDF with `.to_netcdf()` if needed) |
| **`lon` / `lat`**     | carried separately           | in file                      | attached automatically          |
| **Time axis**         | not decoded                  | one record per file          | calendar-aware datetime         |
| **Layer / density**   | manual                       | in file                      | `k` / `dens` coordinates        |
| **Lazy / out-of-memory** | no (eager into RAM)       | no (must convert first)      | yes, via Dask `chunks=`         |
| **Extra step**        | none                         | compile Fortran, convert     | none                            |
| **Best when**         | low-level field access       | NetCDF needed (NCO/CDO/…)    | interactive / larger-than-RAM   |

### Regridding

`xhycom.regrid` maps HYCOM's curvilinear, hybrid-coordinate output onto any regular lon/lat/depth grid, **conservatively by default** (area-conservative horizontally, thickness-weighted vertically), in a single Python call:

```python
glorys = xr.open_dataset("GLO-MFC_001_030_mask_bathy.nc")
ds_glorys = xhycom.regrid(ds, target=glorys, grid="regional.grid")
```

For NERSC-HYCOM-CICE users, this replaces [`hyc2proj`](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/hycom/MSCPROGS/src/Hyc2proj): no input files to edit, no binary to compile, and the result is a lazy Dask-backed Dataset rather than a static NetCDF file.

|                | [`hyc2proj` (MSCPROGS)](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/hycom/MSCPROGS/src/Hyc2proj) | `xhycom.regrid`                                            |
| -------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| **Horizontal**     | bilinear                                       | conservative (default), bilinear, patch                    |
| **Vertical**       | spline / linear / staircase                    | conservative (default, thickness-weighted) or linear       |
| **Conservative?**  | no                                             | yes                                                        |
| **Target grid**    | native / polar-stereographic / mercator        | any regular grid, incl. a GLORYS Dataset via `target=`     |
| **Interface**      | edit text input files, run a Fortran binary    | one Python call, returns an `xr.Dataset`                   |
| **Output**         | static NetCDF file                             | lazy / Dask Dataset (write NetCDF if you want)             |
| **Velocities**     | rotated to east/north                          | de-staggered to T-points **and** rotated to east/north     |

The inverse direction is also supported: `xhycom.regrid_to_hycom` interpolates a regular lon/lat product (such as GLORYS) onto HYCOM's native curvilinear grid, for direct comparison in the model's own space.

### Computing transports and plotting cross-sections

xhycom can compute volume, heat, salt, and freshwater transport through any section defined by a list of waypoints, and produce filled-colour cross-section plots, all from a Jupyter notebook without leaving Python:

```python
# Define a section by waypoints and resolve it on the HYCOM C-grid
transect = xhycom.Transect(lons=[-20, 10], lats=[65, 65], name="nordic_seas")
sec = transect.resolve(ds, grid)

# Volume, heat, salt and freshwater transport in one call
tr = xhycom.transport(ds, sec)   # tr["volume"] in Sv, tr["heat"] in TW, …

# Filled-colour cross-section plot (distance × depth)
sec_data = xhycom.section_data(ds, sec, "temp")
xhycom.section_plot(sec_data, "temp", depth_max=1000, cmap="thermal")
```

The same workflow applies to a GLORYS reanalysis on its regular grid, and to the open boundaries of the model domain — useful for verifying that HYCOM and GLORYS transports balance across the same boundaries.

The volume transport calculation is a Python re-implementation of the approach used by [`m2transport` (MSCPROGS)](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/hycom/MSCPROGS/src/Section), integrating velocity × layer thickness directly at C-grid cell faces with no interpolation. MSCPROGS can also compute heat and salt transports, but requires recompiling with a `SCALAR_TRANS` flag and a `scalartransport.in` file where you manually supply the reference temperature, reference salinity, and `cp × ρ`. Freshwater transport and GLORYS / open-boundary transports have no MSCPROGS equivalent.

|                                    | [`m2transport` (MSCPROGS)](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/hycom/MSCPROGS/src/Section) | xhycom                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Volume transport (HYCOM C-grid)    | yes                                                                                                                 | yes                                  |
| Heat transport                     | yes (compile flag; manual reference T and `cp × ρ`)                                                                | yes (built-in)    |
| Salt transport                     | yes (compile flag; `scalartransport.in`)                                                                            | yes                                  |
| Freshwater transport               | via salinity offset (manual)                                                                                        | yes (built-in)                 |
| Section plots                      | yes (Fortran → NetCDF → MATLAB)                                                                                     | yes (end-to-end Python)             |
| GLORYS transport                   | no                                                                                                                  | yes                                  |
| Open-boundary transport            | no                                                                                                                  | yes (`tp2_sections`, `tp5_sections`) |
| Interface                          | compile binary, edit text input files                                                                               | Python / Jupyter                     |

## Getting started

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

{octicon}`desktop-download;2em;sd-text-primary`

Install xhycom and set up your environment.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

{octicon}`rocket;2em;sd-text-primary`

Open your first `.a/.b` file and make a plot in minutes.
:::

::::

## Tutorials

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Lazy loading & chunking
:link: lazy-loading
:link-type: doc

{octicon}`database;2em;sd-text-primary`

Work with larger-than-RAM datasets using Dask.
:::

:::{grid-item-card} Analysis
:link: analysis
:link-type: doc

{octicon}`graph;2em;sd-text-primary`

Slice, select, and visualize HYCOM fields with xarray.
:::

:::{grid-item-card} Regridding
:link: regridding
:link-type: doc

{octicon}`globe;2em;sd-text-primary`

Remap onto a regular lon/lat/depth grid for reanalysis comparisons.
:::

:::{grid-item-card} Transects and transport
:link: transects_transports
:link-type: doc

{octicon}`milestone;2em;sd-text-primary`

Define sections, resolve them on the HYCOM C-grid, and compute volume, heat, salt, and freshwater transports.
:::

:::{grid-item-card} Comparing with GLORYS
:link: comparison_transects_transports
:link-type: doc

{octicon}`git-compare;2em;sd-text-primary`

Compare HYCOM transports and hydrographic sections against GLORYS reanalysis using three regridding strategies.
:::

:::{grid-item-card} Boundary transport verification
:link: boundaries
:link-type: doc

{octicon}`sign-in;2em;sd-text-primary`

Verify HYCOM and GLORYS volume transports through the open boundaries of the Arctic domain on their respective native grids.
:::

:::{grid-item-card} Time averaging
:link: time-averaging
:link-type: doc

{octicon}`clock;2em;sd-text-primary`

Compute monthly and seasonal means over long time series.
:::

:::{grid-item-card} Big computations
:link: big-computations
:link-type: doc

{octicon}`server;2em;sd-text-primary`

Scale out to HPC clusters with Dask distributed.
:::

::::

## Reference

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Why xarray?
:link: why-xarray
:link-type: doc

{octicon}`question;2em;sd-text-primary`

A short introduction to xarray for HYCOM users.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc

{octicon}`code;2em;sd-text-primary`

Full documentation of all public functions and classes.
:::

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc

{octicon}`people;2em;sd-text-primary`

How to report issues and contribute to xhycom.
:::

:::{grid-item-card} Changelog
:link: releases
:link-type: doc

{octicon}`tag;2em;sd-text-primary`

Release history and what changed in each version.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

installation
quickstart.ipynb
lazy-loading.ipynb
analysis.ipynb
regridding.ipynb
transects_transports.ipynb
comparison_transects_transports.ipynb
boundaries.ipynb
time-averaging.ipynb
big-computations.ipynb
why-xarray
api
contributing
releases
```
