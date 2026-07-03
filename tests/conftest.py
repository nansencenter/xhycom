"""Synthetic HYCOM ``.ab`` fixtures.

These write tiny but format-valid HYCOM ``.a``/``.b`` file pairs to a temp dir so
the readers can be tested without committing binary sample data.  The ``.a``
layout is ``n2drec``-padded big-endian float32 records (see ``AFile``); the
``.b`` headers mirror exactly what ``ABFileGrid`` / ``ABFileBathy`` /
``ABFileArchv`` parse.
"""

import os

import numpy as np
import pytest

IDM, JDM = 5, 4  # tiny grid (x, y)
SPVAL = 2.0**100  # HYCOM land/pad fill value (read as masked)
_N2DREC = ((IDM * JDM + 4095) // 4096) * 4096

# Bundled real-data sample (a subset of the TP0 run; see tests/data/_subset_tp0.py).
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class _TP0:
    """Basenames (no extension) of the bundled real-data fixtures."""

    dir = DATA_DIR
    grid = os.path.join(DATA_DIR, "regional.grid")
    bathy = os.path.join(DATA_DIR, "depth_TP0a1.00_01")
    archive = os.path.join(DATA_DIR, "archm.2006_190_12")
    idm, jdm, nlayers = 100, 110, 28


@pytest.fixture(scope="session")
def tp0():
    """Paths to the bundled real TP0 ``.ab`` fixtures (skips if not present)."""
    if not os.path.exists(_TP0.archive + ".a"):
        pytest.skip("bundled TP0 real-data fixtures are not available")
    return _TP0


GLORYS_GRID = os.path.join(DATA_DIR, "glorys_grid_subset.nc")


@pytest.fixture(scope="session")
def glorys():
    """Coarsened GLORYS regular target grid (skips if not present)."""
    if not os.path.exists(GLORYS_GRID):
        pytest.skip("bundled GLORYS target-grid fixture is not available")
    import xarray as xr

    return xr.open_dataset(GLORYS_GRID)


def _write_a(path_a, arrays):
    """Write 2-D (jdm, idm) arrays as consecutive padded big-endian f4 records."""
    with open(path_a, "wb") as f:
        for arr in arrays:
            flat = np.asarray(arr, dtype=">f4").ravel()
            pad = np.full(_N2DREC - flat.size, SPVAL, dtype=">f4")
            f.write(flat.tobytes())
            f.write(pad.tobytes())


def _ramp(scale=1.0, offset=0.0):
    """A distinctive (jdm, idm) field so round-trips are easy to assert."""
    return (np.arange(JDM * IDM, dtype="f4").reshape(JDM, IDM) * scale) + offset


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@pytest.fixture
def grid_file(tmp_path):
    """Write regional.grid.[ab] with plon/plat/scpx/scpy/pang. Returns (basename, fields)."""
    base = str(tmp_path / "regional.grid")
    fields = {
        "plon": _ramp(0.5, -90.0),
        "plat": _ramp(0.3, 40.0),
        "scpx": np.full((JDM, IDM), 100.0, dtype="f4"),
        "scpy": np.full((JDM, IDM), 200.0, dtype="f4"),
        "pang": np.zeros((JDM, IDM), dtype="f4"),
    }
    lines = [
        f"{IDM:6d}    'idm   ' = longitudinal array size\n",
        f"{JDM:6d}    'jdm   ' = latitudinal array size\n",
        f"{-1:6d}    'mapflg' = map flag\n",
    ]
    for name, arr in fields.items():
        lines.append(f"{name}:  min,max = {arr.min():.6f} {arr.max():.6f}\n")
    with open(base + ".b", "w") as f:
        f.writelines(lines)
    _write_a(base + ".a", list(fields.values()))
    return base, fields


# ---------------------------------------------------------------------------
# Bathymetry
# ---------------------------------------------------------------------------
@pytest.fixture
def bathy_file(tmp_path):
    """Write depth_TEST.[ab] with a land corner (masked). Returns (basename, depth)."""
    base = str(tmp_path / "depth_TEST_01")
    depth = _ramp(50.0, 10.0)
    depth[0, 0] = SPVAL  # one land point
    lines = [
        "Synthetic bathymetry for tests\n",
        "line 2\n",
        "line 3\n",
        "line 4\n",
        "line 5\n",
        f"min,max  depth = {10.0:.4f} {depth[depth < SPVAL].max():.4f}\n",
    ]
    with open(base + ".b", "w") as f:
        f.writelines(lines)
    _write_a(base + ".a", [depth])
    return base, depth


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
def _archive_records():
    """Return (field-line specs, arrays) for a small archive."""
    g = 9.806
    onem = 9806.0
    specs, arrays = [], []

    # 2-D surface field: SSH = 0.5 m stored as geopotential g*0.5
    specs.append(("srfhgt", 12, 40909.5, 0, 0.0))
    arrays.append(np.full((JDM, IDM), g * 0.5, dtype="f4"))

    # layered fields over 3 layers
    densities = {1: 28.0, 2: 29.0, 3: 30.0}
    for name, base_val in (("temp", 10.0), ("salin", 35.0)):
        for k in (1, 2, 3):
            specs.append((name, 12, 40909.5, k, densities[k]))
            arrays.append(np.full((JDM, IDM), base_val - k, dtype="f4"))
    for k in (1, 2, 3):  # thknss = 10 m per layer, in Pa
        specs.append(("thknss", 12, 40909.5, k, densities[k]))
        arrays.append(np.full((JDM, IDM), 10.0 * onem, dtype="f4"))
    return specs, arrays


def _write_archive(base, idm=IDM, jdm=JDM, iexpt=18, day=40909.5):
    specs, arrays = _archive_records()
    # patch the model day so multi-file fixtures get distinct times
    specs = [(n, s, day, k, d) for (n, s, _, k, d) in specs]
    lines = [
        "Synthetic HYCOM archive for tests\n",
        "experiment line\n",
        "comment line\n",
        "comment line\n",
        f"{20:6d}    'iversn' = hycom version number x10\n",
        f"{iexpt:6d}    'iexpt ' = experiment number x10\n",
        f"{3:6d}    'yrflag' = days in year flag\n",
        f"{idm:6d}    'idm   ' = longitudinal array size\n",
        f"{jdm:6d}    'jdm   ' = latitudinal array size\n",
        "field       time step  model day  k  dens        min              max\n",
    ]
    for name, step, d, k, dens in specs:
        arr_min, arr_max = -1.0e3, 1.0e3
        lines.append(
            f"{name:<8s} = {step:7d} {d:11.3f} {k:3d} {dens:7.3f} "
            f"{arr_min:16.7e} {arr_max:16.7e}\n"
        )
    with open(base + ".b", "w") as f:
        f.writelines(lines)
    _write_a(base + ".a", arrays)


@pytest.fixture
def archive_file(tmp_path):
    """Write a single archive snapshot archv.2013_001_12.[ab]. Returns basename."""
    base = str(tmp_path / "archv.2013_001_12")
    _write_archive(base)
    return base


@pytest.fixture
def archive_pair(tmp_path):
    """Write two archive snapshots with distinct times. Returns list of basenames."""
    b1 = str(tmp_path / "archv.2013_001_12")
    b2 = str(tmp_path / "archv.2013_002_12")
    _write_archive(b1, day=40909.5)
    _write_archive(b2, day=40910.5)
    return [b1, b2]


# ---------------------------------------------------------------------------
# AVE (hycave/ensave monthly average)
# ---------------------------------------------------------------------------
def _write_ave(
    base: str,
    idm: int = IDM,
    jdm: int = JDM,
    iexpt: int = 11,
    year: int = 1991,
    month: int = 1,
) -> None:
    """Write a synthetic AVE .ab file pair.

    Contains one 2-D field (ssh, k=0) and one 3-layer field (temp, k=1..3).
    """
    kdm = 3
    # 2-D SSH field at k=0
    ssh_arr = _ramp(0.1, -0.5)
    # 3-D temp fields at k=1..3
    temp_arrs = [_ramp(1.0, 10.0 - k) for k in (1, 2, 3)]
    arrays = [ssh_arr, *temp_arrs]

    lines = [
        f"{'Generated by hycave/ensave':<80s}\n",
        f"{'Generated by hycave/ensave':<80s}\n",
        f"{'Generated by hycave/ensave':<80s}\n",
        f"{'Generated by hycave/ensave':<80s}\n",
        f"{22:6d}    'iversn' = hycom version number x10\n",
        f"{iexpt:6d}    'iexpt ' = experiment number x10\n",
        f"{3:6d}    'yrflag' = days in year flag\n",
        f"{idm:6d}    'idm   ' = longitudinal array size\n",
        f"{jdm:6d}    'jdm   ' = latitudinal  array size\n",
        f"{kdm:6d}    'kdm   ' = Vertical     array size\n",
        f"{9999:6d}    'month ' = Month of this dump\n",
        f"{year % 10:6d}    'year  ' = Year of this dump\n",
        f"{1:6d}    'count ' = Averaging counter\n",
        "field       time step  model day  k  dens        min              max\n",
        f"{'ssh':<8s} = {0:7d} {0.0:11.3f} {0:3d} {0.0:7.3f} {ssh_arr.min():16.7e} {ssh_arr.max():16.7e}\n",
    ]
    for k, arr in enumerate(temp_arrs, start=1):
        lines.append(
            f"{'temp':<8s} = {0:7d} {0.0:11.3f} {k:3d} {0.0:7.3f} {arr.min():16.7e} {arr.max():16.7e}\n"
        )

    with open(base + ".b", "w") as f:
        f.writelines(lines)
    _write_a(base + ".a", arrays)


@pytest.fixture
def ave_file(tmp_path) -> str:
    """Write a single AVE file EXPAVE_1991_01.[ab]. Returns basename."""
    base = str(tmp_path / "EXPAVE_1991_01")
    _write_ave(base)
    return base
