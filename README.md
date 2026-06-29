# xhycom

xhycom integrates HYCOM model output with [xarray](https://docs.xarray.dev) —
giving every field a name, coordinates, units, and lazy out-of-memory access,
directly from the native `.ab` format.

## Installation

```bash
pip install git+https://github.com/NoraLoose/xhycom.git
```

**Dependencies:** `numpy`, `xarray`, `cftime` — no Fortran compiler or
external binary readers required.

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
