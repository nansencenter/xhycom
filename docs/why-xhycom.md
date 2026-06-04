# Why xhycom?

There are three common ways to work with HYCOM output.  xhycom is designed to make the third one as easy as the first — without the overhead.

---

## 1. abfile + NumPy

The standard workflow in the HYCOM community uses the
[`abfile`](https://github.com/nansencenter/NERSC-HYCOM-CICE) package to open
`.ab` files and returns individual fields as masked NumPy arrays:

```python
import abfile.abfile as abf
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Open grid and bathymetry
ab_grid = abf.ABFileGrid("regional.grid", "r")
ab_bathy = abf.ABFileBathy("regional.depth", "r",
                           idm=ab_grid.idm, jdm=ab_grid.jdm)

plon = ab_grid.read_field("plon")   # (jdm, idm) masked array
plat = ab_grid.read_field("plat")

# Open an archive snapshot and read one field at a time
ab_archv = abf.ABFileArchv("archv.2020_001_00", "r")
temp_sfc = ab_archv.read_field("temp", 1)   # layer 1

# Plot manually with cartopy
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.pcolormesh(plon, plat, temp_sfc, transform=ccrs.PlateCarree())
```

**Works well when** you are already inside the NERSC-HYCOM-CICE ecosystem,
need low-level access to individual fields, or want to avoid the xarray
dependency.

**Pain points:**
- Each field must be read individually; there is no dataset-level view.
- `lon`/`lat` arrays must be carried around separately and passed explicitly to every plot call.
- No time coordinate: the model day in the `.b` header is not decoded automatically.
- Masked NumPy arrays don't compose as naturally with the broader scientific Python stack (e.g. Dask, hvPlot, Zarr).
- **Every array is loaded eagerly into RAM.** Analysing a full time series (e.g. 10 years of daily output across 83 variables and 40 layers) means looping over thousands of files and stacking the results yourself — the full dataset will not fit in memory on most machines.

---

## 2. Convert to NetCDF with `m2nc`, then use xarray

`m2nc` is a Fortran program in the
[NERSC-HYCOM-CICE toolbox](https://github.com/nansencenter/NERSC-HYCOM-CICE)
(`hycom/MSCPROGS/src/ExtractNC2D`) that converts `.ab` archive files to NetCDF.
Once compiled, it is run from the command line:

```bash
# Convert one or more archive snapshots to tmp1.nc
m2nc archv.2020_001_00.a archv.2020_002_00.a ...
```

The fields to extract are controlled by a configuration file (e.g.
`extract.daily`).  The output is a NetCDF file (`tmp1.nc`) with one time
record per input file, which can then be opened with xarray:

```python
import xarray as xr

ds = xr.open_dataset("tmp1.nc")
ds["temp"].isel(k=0).plot(x="lon", y="lat")
```

**Works well when** you need NetCDF files for tools that cannot read the
native `.ab` format (NCO, CDO, Ferret, external collaborators).
xhycom (approach 3) can produce the same result with less setup — see below.

**Pain points:**
- Requires compiling Fortran and setting up the MSCPROGS build environment.
- Fields to extract must be specified upfront in the configuration file.
- Doubles your storage and adds a mandatory conversion step before analysis.
- **The conversion itself must read every file eagerly.** Running `m2nc` on a multi-year archive is a long blocking job; you cannot start analysis until it finishes. If the resulting NetCDF is opened with `xr.open_dataset` (no `chunks`), the entire file is loaded into memory on first access.

---

## 3. xhycom

xhycom reads `.ab` pairs directly into a labelled `xr.Dataset` — no intermediate files, no boilerplate.

```python
import xhycom

ds = xhycom.open_dataset("archv.2020_001_00", grid="regional.grid")
ds["temp"].isel(time=0, k=0).plot()
```

Everything that approaches 1 and 2 require you to assemble by hand is handled automatically:

- `lon` / `lat` attached as 2-D curvilinear coordinates from `regional.grid`
- `k` (layer index) and `dens` (target sigma-2 density) on 3-D fields
- `time` decoded to a calendar-aware datetime using `yrflag` from the `.b` header
- All xarray operations (`sel`, `isel`, `mean`, `.plot`, …) work immediately

### Out-of-memory analysis with `chunks`

The most important difference from approaches 1 and 2: xhycom can open
**decades of model output without loading a single byte of field data into
memory**.  Pass `chunks={"time": 1}` and the returned Dataset is backed by
[Dask](https://docs.dask.org) — each time step becomes a lazy task that reads
from disk only when you explicitly compute it:

```python
# Open 30 years of monthly output (~1 TB on disk, 83 variables, 40 layers).
# This returns in seconds and uses ~100 MB of RAM — not gigabytes.
ds = xhycom.open_mfdataset("data/archm.199*-202*", grid="regional.grid",
                            chunks={"time": 1})

print(ds)
# <xarray.Dataset>
# Dimensions:  (time: 360, k: 40, y: 880, x: 800)
# Data variables:
#     temp     (time, k, y, x) float64 dask.array<chunksize=(1, 40, 880, 800)>
#     salin    (time, k, y, x) float64 dask.array<chunksize=(1, 40, 880, 800)>
#     ...

# Compute the 30-year mean SST — Dask reads only the data it needs:
sst_mean = ds["temp"].isel(k=0).mean("time").compute()
sst_mean.plot(x="lon", y="lat")
```

Approaches 1 and 2 offer no equivalent: `abfile` always loads eagerly into
RAM, and `m2nc` must process every file upfront before any analysis can begin.
With xhycom, the working memory footprint stays proportional to the chunk size
you request — not to the size of the archive.

### File discovery and concatenation

For a full time series, file discovery is automatic:

```python
ds = xhycom.open_mfdataset("data/", grid="regional.grid",
                            chunks={"time": 1})
# → time dimension spans every archv.YYYY_DDD_HH pair in data/
ds["temp"].isel(k=0).mean("time").plot()
```

### Exporting to NetCDF

If you do need NetCDF for downstream tools, xhycom makes that easy too —
no separate conversion step or full in-memory load required:

```python
ds = xhycom.open_dataset("archv.2020_001_00", grid="regional.grid")
ds.to_netcdf("archv.2020_001_00.nc")
```

**Best choice when** you want to work interactively in a notebook, analyse
datasets that are larger than available RAM, avoid writing conversion glue
code, or integrate HYCOM output into a larger xarray/Dask workflow.
