"""Generate the bundled TP0 real-data test fixtures.

The full TP0 sample (a coupled physics + biogeochemistry HYCOM run, idm=100,
jdm=110) ships a 1690-record archive ``.a`` weighing ~83 MB — far too large to
commit.  This script copies the small grid / bathymetry files verbatim and
writes a *variable subset* of the archive containing only the physical fields
that xhycom's reader and regridding code actually exercise.  The output keeps
the exact on-disk HYCOM ``.ab`` layout and the real field values, just with the
biogeochemistry records dropped.

Run from a machine that can see the source files::

    python tests/data/_subset_tp0.py

The source lives at ``/nird/datalake/NS9481K/nlo043/TP0`` (NIRD, NS9481K).
Re-run only if the fixtures need regenerating; the products are committed.
"""
import os
import shutil

import numpy as np

SRC = "/nird/datalake/NS9481K/nlo043/TP0"
DST = os.path.dirname(os.path.abspath(__file__))

IDM, JDM = 100, 110
N2DREC = ((IDM * JDM + 4095) // 4096) * 4096   # floats per padded .a record
RECBYTES = N2DREC * 4                          # big-endian float32

# Physical fields kept in the subset archive.  Everything else (BGC tracers,
# CO2 / ECO_* fields, diffusivities, ...) is dropped.
KEEP = {"montg1", "srfhgt", "temp", "salin", "thknss", "u-vel.", "v-vel."}

ARCHIVE = "archm.2006_190_12"
GRID = "regional.grid"
BATHY = "depth_TP0a1.00_01"


def subset_archive():
    src_b = os.path.join(SRC, ARCHIVE + ".b")
    src_a = os.path.join(SRC, ARCHIVE + ".a")
    dst_b = os.path.join(DST, ARCHIVE + ".b")
    dst_a = os.path.join(DST, ARCHIVE + ".a")

    with open(src_b) as f:
        lines = f.readlines()

    # Header: 4 comment lines + iversn/iexpt/yrflag/idm/jdm + column header.
    header = lines[:10]
    field_lines = lines[10:]

    keep_indices, keep_field_lines = [], []
    for rec_idx, line in enumerate(field_lines):
        if not line.strip():
            continue
        fieldname = line[:8].strip()
        if fieldname in KEEP:
            keep_indices.append(rec_idx)
            keep_field_lines.append(line)

    with open(dst_b, "w") as f:
        f.writelines(header)
        f.writelines(keep_field_lines)

    with open(src_a, "rb") as fin, open(dst_a, "wb") as fout:
        for rec_idx in keep_indices:
            fin.seek(rec_idx * RECBYTES)
            fout.write(fin.read(RECBYTES))

    print(f"{ARCHIVE}: kept {len(keep_indices)}/{len(field_lines)} records "
          f"({os.path.getsize(dst_a) / 1e6:.1f} MB)")


def copy_verbatim(base):
    for ext in (".a", ".b"):
        shutil.copyfile(os.path.join(SRC, base + ext),
                        os.path.join(DST, base + ext))
    print(f"{base}: copied verbatim "
          f"({os.path.getsize(os.path.join(DST, base + '.a')) / 1e6:.2f} MB)")


if __name__ == "__main__":
    copy_verbatim(GRID)
    copy_verbatim(BATHY)
    subset_archive()
