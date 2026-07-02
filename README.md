# xhycom

[![Run Tests](https://github.com/nansencenter/xhycom/actions/workflows/tests.yml/badge.svg)](https://github.com/nansencenter/xhycom/actions/workflows/tests.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/nansencenter/xhycom/graph/badge.svg?token=5S1oNu39xE)](https://codecov.io/gh/nansencenter/xhycom)
[![Documentation Status](https://readthedocs.org/projects/xhycom/badge/?version=latest)](https://xhycom.readthedocs.io/en/latest/?badge=latest)

xhycom integrates HYCOM model output with [xarray](https://docs.xarray.dev) —
giving every field a name, coordinates, units, and lazy out-of-memory access,
directly from the native `.ab` format.

## Installation

```bash
pip install git+https://github.com/nansencenter/xhycom.git
```

**Dependencies:** `numpy`, `xarray`, `cftime`, `dask`, `xgcm` — no Fortran compiler
or external binary readers required. This covers reading, lazy/out-of-memory
loading, and vertical regridding.

Horizontal regridding additionally needs [xESMF](https://xesmf.readthedocs.io),
whose ESMF/esmpy backend is conda-forge only. `ci/environment-regrid.yml` creates
the environment and `pip install`s xhycom itself (in editable mode) into it in one
step:

```bash
conda env create -f ci/environment-regrid.yml
conda activate hycom-analysis-env
```

See [Installation](https://xhycom.readthedocs.io/en/latest/installation.html) for
details, including setup on the Olivia and Betzy HPC clusters.

## Quick example

```python
import xhycom

# Single snapshot — auto-detects file type, attaches lon/lat/time/dens
ds = xhycom.open_dataset("archv.2020_001_00", grid="regional.grid")
ds["temp"].isel(time=0, k=0).plot()

# Multi-year time series — lazy, out-of-memory, no data loaded until .compute()
ds = xhycom.open_mfdataset("data/", grid="regional.grid", chunks={"time": 1})
ds["temp"].isel(k=0).mean("time").compute()
```

## Documentation

Full documentation — why xarray, worked examples, API reference, and a
how-it-works guide — is at **[https://xhycom.readthedocs.io](https://xhycom.readthedocs.io/en/latest/)**.
