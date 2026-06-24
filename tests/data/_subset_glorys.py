"""Generate the bundled GLORYS target-grid test fixture.

GLORYS12V1 is a global 1/12-degree regular grid (longitude=4320, latitude=2041,
depth=50); the full mask/bathymetry file is ~0.5 GB.  For tests we only need a
small *regular* target grid covering the model's Arctic window, so this script
spatially subsets and coarsens it (stride in lon/lat) and keeps the land/sea
``mask``, bathymetry ``deptho`` and the ``depth`` levels.  The result is a few
hundred KB and still exercises the full regular-grid + mask + depth-level path.

Run from a machine that can see the source files::

    python tests/data/_subset_glorys.py

Source: /nird/datapeak/NS9481K/MERCATOR_DATA/REGULAR_GRID_COORD (NIRD, NS9481K).
Re-run only if the fixture needs regenerating; the product is committed.
"""
import os

import xarray as xr

SRC = ("/nird/datapeak/NS9481K/MERCATOR_DATA/REGULAR_GRID_COORD/"
       "GLO-MFC_001_030_mask_bathy.nc")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "glorys_grid_subset.nc")

# Window inside the TP0 model domain (Nordic / Arctic sector) and coarsening.
LON_RANGE = (-20.0, 20.0)
LAT_RANGE = (60.0, 80.0)
STRIDE = 6                        # 1/12 deg -> ~0.5 deg


def main():
    g = xr.open_dataset(SRC)
    sub = g.sel(
        longitude=slice(*LON_RANGE),
        latitude=slice(*LAT_RANGE),
    ).isel(
        longitude=slice(None, None, STRIDE),
        latitude=slice(None, None, STRIDE),
    )
    sub = sub[["mask", "deptho", "depth", "longitude", "latitude"]]
    sub["mask"] = sub["mask"].astype("int8")
    encoding = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars}
    sub.to_netcdf(DST, encoding=encoding)
    print(f"wrote {DST} ({os.path.getsize(DST) / 1e3:.0f} KB)")
    print("dims:", dict(sub.sizes))


if __name__ == "__main__":
    main()
