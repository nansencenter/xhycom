# Why xarray?

xhycom returns [xarray](https://docs.xarray.dev) Datasets.  If you have not
used xarray before, this page explains why that is a good thing.

---

## From raw indices to labelled dimensions

A NumPy array is a grid of numbers.  The only way to refer to a location is
by its integer position:

```python
arr[0, 5, 100, 200]   # what does this mean?
```

An **xarray Dataset** wraps the same numbers but gives every axis a name, attaches
coordinate values, and carries metadata.  Here is what a real HYCOM archive looks
like when opened with xhycom:

```
>>> import xhycom
>>> ds = xhycom.open_dataset("archm.2020_001_12", grid="regional.grid")
>>> ds
<xarray.Dataset> Size: 3GB
Dimensions:  (time: 1, y: 380, x: 400, k: 50, ki: 51)
Coordinates:
  * time     (time)  object   2020-01-01 00:00:00
    lon      (y, x)  float64  -94.75 -94.58 … 98.95 98.82
    lat      (y, x)  float64   39.06  39.16 … 56.29 56.20
    lon_u    (y, x)  float64  -94.83 -94.67 … 99.02 98.89
    lat_u    (y, x)  float64   39.01  39.11 … 56.34 56.25
    lon_v    (y, x)  float64  -94.68 -94.52 … 98.87 98.74
    lat_v    (y, x)  float64   38.99  39.09 … 56.33 56.24
  * k        (k)     int64     1 2 3 … 48 49 50
    dens     (k)     float64   0.1 0.2 … 28.11 28.12
  * ki       (ki)    int64     0 1 2 … 49 50
Data variables: (12/83)
    montg1   (time, y, x)    float64  …
    srfhgt   (time, y, x)    float64  …
    temp     (time, k, y, x) float64  …
    salin    (time, k, y, x) float64  …
    u-vel.   (time, k, y, x) float64  …
    v-vel.   (time, k, y, x) float64  …
    …
Attributes:
    iversn:  23    iexpt:  28    yrflag:  3
```

Every axis has a name (`time`, `k`, `y`, `x`).  The curvilinear grid coordinates
(`lon`, `lat`) and the layer density axis (`dens`) are attached directly.  You
select by *what* rather than *where*, and the code reads like the science:

```python
arr[0, 5, 100, 200]                          # NumPy: what is this?
ds["temp"].isel(time=0, k=5)                 # xarray: surface layer, first snapshot
ds["temp"].isel(time=0).sel(dens=27.0,       # select by density instead of index
                            method="nearest")
```

---

## Coordinates travel with the data

Operations on an xarray object preserve coordinates automatically.  Slice a
region, take a time mean, compute an anomaly — the result always knows where
it is:

```
ds["temp"]                        ds["temp"].isel(k=0).mean("time")
──────────────────────────────    ──────────────────────────────────
dims: (time, k, y, x)             dims: (y, x)
coords:                           coords:
  time  1993-01 … 2022-12           lon  (y, x)  float64 …
  k     1 … 40                      lat  (y, x)  float64 …
  dens  1026.5 …                 attrs: units: degC
  lon   (y, x)
  lat   (y, x)
```

With raw NumPy you would have to carry `lon`, `lat`, and the time axis
bookkeeping yourself and reattach them after every operation.

---

## Plotting just works

Because coordinates are embedded in the data, xarray's `.plot()` method
automatically labels axes, titles, and colourbars:

```python
ds["temp"].isel(time=0, k=0).plot()
```

With NumPy + Matplotlib you would need to pass `plon` and `plat` explicitly,
set axis labels by hand, and add the colourbar yourself.

---

## Larger-than-memory data via Dask

xarray integrates with [Dask](https://docs.dask.org) to represent datasets
that are far larger than available RAM.  Instead of reading data immediately,
xarray builds a *computation graph*: a recipe for what to do when you finally
ask for the result.

```
                          open_mfdataset(..., chunks={"time": 1})
                          ┌─────────────────────────────────────┐
30 years of .ab files ───►│  Dask-backed xr.Dataset             │
(~1 TB on disk)           │  in memory: ~100 MB (graph only)    │
                          └───────────────┬─────────────────────┘
                                          │  .isel(k=0).mean("time")
                                          ▼
                          ┌─────────────────────────────────────┐
                          │  lazy computation graph             │
                          │  (still nothing read from disk)     │
                          └───────────────┬─────────────────────┘
                                          │  .compute()
                                          ▼
                          ┌─────────────────────────────────────┐
                          │  result: (y, x) NumPy array         │
                          │  only the needed chunks were read   │
                          └─────────────────────────────────────┘
```

This is what makes it possible to compute a 30-year mean SST on a laptop
without running out of memory.

---

## The broader ecosystem

An xarray Dataset plugs into a large ecosystem of scientific Python tools
without any glue code:

| Task | Tool |
|---|---|
| Interactive maps | [hvPlot](https://hvplot.holoviz.org) / [GeoViews](https://geoviews.org) |
| Parallel computation | [Dask](https://docs.dask.org) |
| Cloud-optimised storage | [Zarr](https://zarr.dev) |
| Regridding | [xESMF](https://xesmf.readthedocs.io) |
| Statistics | [xskillscore](https://xskillscore.readthedocs.io), [climpred](https://climpred.readthedocs.io) |
| Filtering | [xrft](https://xrft.readthedocs.io), [scipy](https://docs.scipy.org) |

Because xhycom returns standard xarray objects, all of these work on HYCOM
output immediately — no adapters or format conversions required.
