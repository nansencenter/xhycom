# API reference

## Reading data

```{eval-rst}
.. autofunction:: xhycom.open_dataset
.. autofunction:: xhycom.open_mfdataset
```

## Post-processing

Convert native HYCOM units to physical ones and add derived fields.  Applied
automatically with ``open_dataset(..., postprocess=True)``, or called directly
on an existing Dataset.

```{eval-rst}
.. autofunction:: xhycom.postprocess
```

## Regridding

Map HYCOM output onto a regular lon/lat/depth grid (e.g. for comparison with
GLORYS).  See the {doc}`regridding` notebook.  The vertical step needs only
``xgcm`` (pip); the lateral step also needs ``xesmf`` (conda — see
``ci/environment-regrid.yml``).

```{eval-rst}
.. autofunction:: xhycom.regrid
.. autofunction:: xhycom.regrid_horizontal
.. autofunction:: xhycom.regrid_vertical
```

## Internal utilities

These are not part of the public API but are documented here for contributors.

### File discovery

```{eval-rst}
.. autofunction:: xhycom._discovery.find_archv_files
```

### Time conversion

```{eval-rst}
.. autofunction:: xhycom._time.model_day_to_datetime
```
