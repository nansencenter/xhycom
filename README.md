# xhycom

[![Run Tests](https://github.com/nansencenter/xhycom/actions/workflows/tests.yml/badge.svg)](https://github.com/nansencenter/xhycom/actions/workflows/tests.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/nansencenter/xhycom/graph/badge.svg?token=5S1oNu39xE)](https://codecov.io/gh/nansencenter/xhycom)
[![Documentation Status](https://readthedocs.org/projects/xhycom/badge/?version=latest)](https://xhycom.readthedocs.io/en/latest/?badge=latest)

xhycom reads HYCOM model output in the native `.ab` format directly into [xarray](https://docs.xarray.dev) Datasets, with coordinates, units, and lazy out-of-memory access attached automatically.

## Installation

**conda** (recommended, includes [xESMF](https://xesmf.readthedocs.io) for horizontal regridding):

```bash
conda install -c conda-forge xhycom
```

**pip** (xESMF must come from conda-forge, so create a conda env with it first):

```bash
conda create -n hycom-env -c conda-forge xesmf
conda activate hycom-env
pip install xhycom
```

**From GitHub** (latest unreleased, clone and install in editable mode):

```bash
git clone https://github.com/nansencenter/xhycom.git
cd xhycom
conda env create -f ci/environment-regrid.yml
conda activate hycom-analysis-env
```

See [Installation](https://xhycom.readthedocs.io/en/latest/installation.html) for
details, including setup on the Olivia and Betzy HPC clusters.

## Quick example

```python
import xhycom

# Single snapshot: auto-detects file type, attaches lon/lat/time/dens
ds = xhycom.open_dataset("archv.2020_001_00", grid="regional.grid")
ds["temp"].isel(time=0, k=0).plot()

# Multi-year time series: lazy, out-of-memory, no data loaded until .compute()
ds = xhycom.open_mfdataset("data/", grid="regional.grid", chunks={"time": 1})
ds["temp"].isel(k=0).mean("time").compute()
```

## Documentation

Full documentation (why xarray, worked examples, API reference) is at **[https://xhycom.readthedocs.io](https://xhycom.readthedocs.io/en/latest/)**.
