# Why xhycom?

xhycom reads HYCOM `.ab` output directly into a labelled [xarray](why-xarray.ipynb) `Dataset`, names, coordinates, units, a decoded time axis, and lazy out-of-memory access, with no intermediate files.

## Reading HYCOM output

There are three common workflows.

**1. [`abfile`](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/pythonlibs/abfile) + NumPy**: the standard low-level reader; returns one masked array per field:

```python
import abfile.abfile as abf

ab = abf.ABFileArchv("archv.2020_001_00", "r")
temp_sfc = ab.read_field("temp", 1)        # (jdm, idm) masked array, layer 1
```

**2. `m2nc` → xarray**: a Fortran tool ([`hycom/MSCPROGS/src/ExtractNC2D`](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/hycom/MSCPROGS/src/ExtractNC2D)) that converts `.ab` to NetCDF, which you then open with xarray:

```bash
m2nc archv.2020_001_00.a ...               # writes tmp1.nc
```

**3. xhycom**: straight to a labelled, lazy `Dataset`:

```python
import xhycom

ds = xhycom.open_dataset("archv.2020_001_00", grid="regional.grid")
ds["temp"].isel(time=0, k=0).plot()        # lon/lat/time already attached
```

|                       | `abfile` + NumPy             | `m2nc` → xarray              | xhycom                          |
| --------------------- | ---------------------------- | ---------------------------- | ------------------------------- |
| Output                | one masked array per field   | NetCDF file                  | labelled `xr.Dataset`           |
| `lon` / `lat`         | carried separately           | in file                      | attached automatically          |
| Time axis             | not decoded                  | one record per file          | calendar-aware datetime         |
| Layer / density       | manual                       | in file                      | `k` / `dens` coordinates        |
| Lazy / out-of-memory  | no, eager into RAM          | no, must convert first      | yes, Dask via `chunks=`        |
| Extra step            |,                            | compile Fortran, convert     | none                            |
| Best when             | low-level field access       | NetCDF needed (NCO/CDO/…)    | interactive / larger-than-RAM   |

### Out-of-memory analysis with `chunks`

The key difference: xhycom can open **decades of output without loading any field data into memory**. With `chunks={"time": 1}` the Dataset is [Dask](https://docs.dask.org)-backed and each step is read only when computed:

```python
# ~1 TB on disk; returns in seconds, ~100 MB RAM.
ds = xhycom.open_mfdataset("data/archm.199*-202*", grid="regional.grid",
                           chunks={"time": 1})
ds["temp"].isel(k=0).mean("time").compute().plot(x="lon", y="lat")
```

`abfile` always loads eagerly; `m2nc` must process every file before analysis can begin. Directory globs are discovered and concatenated along `time` automatically.

### Conservative regridding to a regular grid

HYCOM's curvilinear, hybrid-coordinate output usually has to be mapped onto a regular lon/lat/depth grid before it can be compared with reanalyses such as [GLORYS](https://doi.org/10.48670/moi-00021). The established tool is **[`hyc2proj`](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/hycom/MSCPROGS/src/Hyc2proj)** (Fortran, `hycom/MSCPROGS/src/Hyc2proj`): you edit `proj.in`, `depthlevels.in` and `extract.archv`, run the compiled binary, and get a NetCDF file. Its horizontal step is bilinear and its vertical step is spline / linear / staircase, none of them conservative.

`xhycom.regrid` does it in one in-process call and **conservatively by default**: area-conservative horizontally, depth-integral-conserving (thickness-weighted) vertically, onto any regular grid, including a GLORYS grid opened straight from its NetCDF file:

```python
glorys = xr.open_dataset("GLO-MFC_001_030_mask_bathy.nc")   # regular lon/lat/depth + mask
ds_glorys = xhycom.regrid(ds, target=glorys, grid="regional.grid")
```

|                | [`hyc2proj` (MSCPROGS)](https://github.com/nansencenter/NERSC-HYCOM-CICE/tree/develop/hycom/MSCPROGS/src/Hyc2proj) | `xhycom.regrid`                                            |
| -------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| Horizontal     | bilinear                                       | conservative (default), bilinear, patch                    |
| Vertical       | spline / linear / staircase                    | conservative (default, thickness-weighted) or linear       |
| Conservative?  | no                                             | yes                                                        |
| Target grid    | native / polar-stereographic / mercator        | any regular grid, incl. a GLORYS Dataset via `target=`     |
| Interface      | edit text input files, run a Fortran binary    | one Python call, returns an `xr.Dataset`                   |
| Output         | static NetCDF file                             | lazy / Dask Dataset (write NetCDF if you want)             |
| Velocities     | rotated to east/north                          | de-staggered to T-points **and** rotated to east/north     |

Regridding many time steps of large output is still expensive, as with the Fortran tools, xhycom doesn't change that, and a batch job (e.g. a Slurm script) is often the right way to run it. For a handful of time steps or a short time horizon, it works fine interactively.

See the [regridding tutorial](regridding.ipynb) for worked examples.

### Exporting to NetCDF

If you do need NetCDF for downstream tools, there is no separate conversion step:

```python
xhycom.open_dataset("archv.2020_001_00", grid="regional.grid").to_netcdf("out.nc")
```
