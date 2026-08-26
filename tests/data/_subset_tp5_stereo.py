"""Generate the bundled stereographic (TOPAZ5) target-grid test fixture.

The full TP5 stereographic output is 1137×1185 cells with multiple 3-D fields
(~1 GB).  For tests we need only a small curvilinear target covering the TP0
model domain (Nordic Seas, ~lat 60–80 N, lon -20 to 20 E), spatially coarsened
to a few dozen cells.

Run from a machine that can see the source file::

    python tests/data/_subset_tp5_stereo.py

Source: /cluster/projects/nn2993k/nlo043/TP5a0.06/staged/Hy2.2/archm_1993_01.nc
Re-run only if the fixture needs regenerating; the product is committed.
"""
import os

import numpy as np
import xarray as xr

SRC = ("/cluster/projects/nn2993k/nlo043/TP5a0.06/staged/Hy2.2/archm_1993_01.nc")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "tp5_stereo_subset.nc")

# Region overlapping the TP0 test domain (lat 60–80 N, lon -20 to 20 E) in
# the stereographic projected grid (y/x in degrees of the projection).
# Indices derived by masking on latitude/longitude of the full grid.
Y_SLICE = slice(139, 639)   # ~500 rows covering approx lat 60–80 N
X_SLICE = slice(626, 1126)  # ~500 cols covering approx lon -20 to 20 E
STRIDE = 20                 # → ~25×25 cells
DEPTH_N = 3                 # keep only the shallowest depth levels


def main() -> None:
    ds = xr.open_dataset(SRC)
    sub = ds.isel(y=Y_SLICE, x=X_SLICE).isel(
        y=slice(None, None, STRIDE),
        x=slice(None, None, STRIDE),
        depth=slice(None, DEPTH_N),
        time=0,
    )
    # Keep geographic coords, depth levels, bathymetry, and one tracer for
    # mask derivation (thetao NaN at land and below seafloor).
    keep_vars = ["model_depth", "thetao"]
    sub = sub[keep_vars]
    # Ensure longitude/latitude 2-D coords are included (they are non-index
    # coords on the y/x dims — xarray carries them through isel).
    encoding = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars}
    sub.to_netcdf(DST, encoding=encoding)
    print(f"wrote {DST} ({os.path.getsize(DST) / 1e3:.0f} KB)")
    print("dims:", dict(sub.sizes))
    print("coords:", list(sub.coords))
    # Sanity: lon/lat should span the TP0 domain.
    print(f"lon range: {float(sub.longitude.min()):.1f} .. {float(sub.longitude.max()):.1f}")
    print(f"lat range: {float(sub.latitude.min()):.1f} .. {float(sub.latitude.max()):.1f}")


if __name__ == "__main__":
    main()
