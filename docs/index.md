# xhycom

**xhycom** is a Python package for working with HYCOM model output:

- Reads HYCOM `.a/.b` output directly into a labelled [xarray](why-xarray) Dataset, with coordinates, units, and a decoded time axis attached automatically
- Regrid between HYCOM's native curvilinear grid and regular lon/lat/depth grids (and back), for comparison with reanalyses like GLORYS
- More HYCOM diagnostics coming

## Why xhycom?

### Reading HYCOM output

xhycom reads HYCOM `.a/.b` output directly into a labelled `Dataset` (names, coordinates, units, a decoded time axis, and lazy out-of-memory access) with no intermediate files:

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

::::

```{toctree}
:hidden:
:maxdepth: 1

installation
quickstart.ipynb
lazy-loading.ipynb
analysis.ipynb
regridding.ipynb
time-averaging.ipynb
big-computations.ipynb
why-xarray
api
contributing
```
