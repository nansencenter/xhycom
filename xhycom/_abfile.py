"""Low-level I/O for HYCOM a.b binary file pairs.

Bundled from the abfile package (https://github.com/NoraLoose/NERSC-HYCOM-CICE),
originally authored by Knut Lisaeter, MIT licence.  Included here so that
xhycom has no external install-time dependencies beyond numpy/xarray/cftime.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Callable

import numpy

logger = logging.getLogger(__name__)

grid_ordered_fieldnames: list[str] = [
    "plon",
    "plat",
    "qlon",
    "qlat",
    "ulon",
    "ulat",
    "vlon",
    "vlat",
    "pang",
    "scpx",
    "scpy",
    "scqx",
    "scqy",
    "scux",
    "scuy",
    "scvx",
    "scvy",
    "cori",
    "pasp",
]


class AFileError(Exception):
    pass


class BFileError(Exception):
    pass


class AFile:
    """Binary I/O for HYCOM .a files."""

    huge = 2.0**100

    def __init__(
        self,
        idm: int,
        jdm: int,
        filename: str,
        action: str,
        mask: bool = False,
        real4: bool = True,
        endian: str = "big",
    ) -> None:
        self._idm = idm
        self._jdm = jdm
        self._filename = filename
        self._action = action
        self._mask = mask
        self._real4 = real4
        self._endian = endian

        if self._action.lower() not in ["r", "w"]:
            raise AFileError("action must be 'r' or 'w'")
        if self._endian.lower() not in ["little", "big", "native"]:
            raise AFileError("endian must be 'native', 'little', or 'big'")
        if self._endian.lower() == "native":
            self._endian = sys.byteorder
        self._endian_structfmt = ">" if self._endian.lower() == "big" else "<"

        self._n2drec = ((self._idm * self._jdm + 4095) // 4096) * 4096
        self._iarec = 0
        self._spval = 2**100.0
        self._filea = open(self._filename, self._action + "b")

    def read_record(self, record: int) -> numpy.ma.MaskedArray:
        self.seekrecord(record)
        struct_fmt = "f" if self._real4 else "d"
        mydtype = numpy.dtype("%s%s" % (self._endian_structfmt, struct_fmt))
        w = numpy.fromfile(self._filea, dtype=mydtype, count=int(self.n2drec))
        w = w[0 : self.idm * self.jdm]
        w.shape = (self.jdm, self.idm)
        return numpy.ma.masked_where(w > self.huge * 0.5, w)

    def seekrecord(self, record: int) -> None:
        nbytes = 4 if self._real4 else 8
        self._filea.seek(int(record * self.n2drec * nbytes))

    def close(self) -> None:
        self._filea.close()

    @property
    def n2drec(self) -> int:
        return self._n2drec

    @property
    def idm(self) -> int:
        return self._idm

    @property
    def jdm(self) -> int:
        return self._jdm


class ABFile:
    """Base class for HYCOM .a/.b file pairs."""

    def __init__(
        self,
        basename: str,
        action: str,
        mask: bool = False,
        real4: bool = True,
        endian: str = "big",
    ) -> None:
        self._basename = ABFile.strip_ab_ending(basename)
        self._action = action
        self._fileb = open(self._basename + ".b", self._action)
        self._filea = None
        self._mask = mask
        self._real4 = real4
        self._endian = endian
        self._firstwrite = True

    def scanitem(
        self, item: str | None = None, conversion: Callable[[str], Any] | None = None
    ) -> tuple:
        line = self._fileb.readline().strip()
        if item is not None:
            m = re.match("^(.*)'(%-6s)'[ =]*" % item, line)
        else:
            m = re.match("^(.*)'(.*)'[ =]*", line)
        if m:
            value = conversion(m.group(1)) if conversion else None
            return m.group(2), value
        return None, None

    def readline(self) -> str:
        return self._fileb.readline()

    def _open_filea_if_necessary(self, field: numpy.ndarray) -> None:
        if self._filea is None:
            self._jdm, self._idm = field.shape
            self._filea = AFile(
                self._idm,
                self._jdm,
                self._basename + ".a",
                self._action,
                mask=self._mask,
                real4=self._real4,
                endian=self._endian,
            )

    def close(self) -> None:
        if self._filea is not None:
            self._filea.close()
        self._fileb.close()

    @property
    def idm(self) -> int:
        return self._idm

    @property
    def jdm(self) -> int:
        return self._jdm

    @property
    def fields(self) -> dict:
        return self._fields

    @property
    def fieldnames(self) -> set:
        return set(elem["field"] for elem in self._fields.values())

    @classmethod
    def strip_ab_ending(cls, fname: str) -> str:
        m = re.match(r"^(.*)(\.[ab]$)", fname)
        return m.group(1) if m else fname


class ABFileGrid(ABFile):
    """HYCOM regional.grid .a/.b file pair."""

    def __init__(
        self,
        basename: str,
        action: str,
        mask: bool = False,
        real4: bool = True,
        endian: str = "big",
        mapflg: int = -1,
    ) -> None:
        super().__init__(basename, action, mask=mask, real4=real4, endian=endian)
        self._mapflg = mapflg
        if action == "r":
            self.read_header()
            self.read_field_info()
            self._open_filea_if_necessary(numpy.zeros((self._jdm, self._idm)))

    def read_header(self) -> None:
        _, self._idm = self.scanitem(item="idm", conversion=int)
        _, self._jdm = self.scanitem(item="jdm", conversion=int)
        _, self._mapflg = self.scanitem(item="mapflg", conversion=int)

    def read_field_info(self) -> None:
        self._fields = {}
        line = self.readline().strip()
        i = 0
        while line:
            fieldname = line[0:4]
            self._fields[i] = {"field": fieldname}
            elems = re.split(r"[ =]+", line)
            self._fields[i]["min"] = float(elems[2])
            self._fields[i]["max"] = float(elems[3])
            i += 1
            line = self.readline().strip()

    def read_field(self, fieldname: str):
        for i, d in self._fields.items():
            if d["field"] == fieldname:
                return self._filea.read_record(i)
        return None


class ABFileBathy(ABFile):
    """HYCOM bathymetry .a/.b file pair."""

    def __init__(
        self,
        basename: str,
        action: str,
        mask: bool = True,
        real4: bool = True,
        endian: str = "big",
        idm: int | None = None,
        jdm: int | None = None,
    ) -> None:
        super().__init__(basename, action, mask=mask, real4=real4, endian=endian)
        if action == "r":
            if idm is None or jdm is None:
                raise BFileError(
                    "ABFileBathy opened as read, but idm and jdm not provided"
                )
            self._idm = idm
            self._jdm = jdm
            self.read_header()
            self.read_field_info()
            self._open_filea_if_necessary(numpy.zeros((self._jdm, self._idm)))

    def read_header(self) -> None:
        self._header = [self.readline() for _ in range(5)]

    def read_field_info(self) -> None:
        self._fields = {}
        line = self.readline().strip()
        i = 0
        while line:
            m = re.match(r"^min,max[ ]+(.*?)[ ]*=(.*)", line)
            if m:
                self._fields[i] = {
                    "field": m.group(1).strip(),
                    **dict(
                        zip(["min", "max"], [float(x) for x in m.group(2).split()[:2]])
                    ),
                }
            i += 1
            line = self.readline().strip()

    def read_field(self, fieldname: str):
        for i, d in self._fields.items():
            if d["field"] == fieldname:
                return self._filea.read_record(i)
        return None


class ABFileArchv(ABFile):
    """HYCOM archive .a/.b file pair."""

    fieldkeys = ["field", "step", "day", "k", "dens", "min", "max"]

    def __init__(
        self,
        basename: str,
        action: str,
        mask: bool = True,
        real4: bool = True,
        endian: str = "big",
        iversn: int | None = None,
        iexpt: int | None = None,
        yrflag: int | None = None,
        idm: int | None = None,
        jdm: int | None = None,
        cline1: str = "",
        cline2: str = "",
        cline3: str = "",
    ) -> None:
        self._iversn = iversn
        self._iexpt = iexpt
        self._yrflag = yrflag
        self._idm = idm
        self._jdm = jdm
        self._cline1 = cline1
        self._cline2 = cline2
        self._cline3 = cline3
        super().__init__(basename, action, mask=mask, real4=real4, endian=endian)
        self._idm = idm
        self._jdm = jdm
        if action == "r":
            self.read_header()
            self.read_field_info()
            # .a file is opened lazily on the first read_record / read_field call.

    def read_header(self) -> None:
        self._header = [self.readline() for _ in range(4)]
        _, self._iversn = self.scanitem(item="iversn", conversion=int)
        _, self._iexpt = self.scanitem(item="iexpt", conversion=int)
        _, self._yrflag = self.scanitem(item="yrflag", conversion=int)
        _, self._idm = self.scanitem(item="idm", conversion=int)
        _, self._jdm = self.scanitem(item="jdm", conversion=int)

    def read_field_info(self) -> None:
        self._fields = {}
        # Column-header line distinguishes a mean archive from an instantaneous
        # one: archv writes "... model day", archm writes "... mean day".  This
        # is the only in-file flag for it (filenames aside), and it matters for
        # velocities (archv stores baroclinic, archm stores total) — see
        # xhycom.postprocess.
        colhdr = self.readline()
        self._is_mean = "mean day" in colhdr
        line = self.readline().strip()
        i = 0
        while line:
            elems = re.split(r"[ =]+", line)
            self._fields[i] = dict(zip(self.fieldkeys, [e.strip() for e in elems]))
            for k in self.fieldkeys:
                if k in ("min", "max", "dens", "day"):
                    self._fields[i][k] = float(self._fields[i][k])
                elif k in ("k", "step"):
                    self._fields[i][k] = int(self._fields[i][k])
            i += 1
            line = self.readline().strip()

    def _ensure_filea(self) -> None:
        if self._filea is None:
            self._filea = AFile(
                self._idm,
                self._jdm,
                self._basename + ".a",
                "r",
                mask=self._mask,
                real4=self._real4,
                endian=self._endian,
            )

    def read_record(self, record_index: int) -> numpy.ma.MaskedArray:
        """Read a 2-D slab from the .a file by absolute record index."""
        self._ensure_filea()
        return self._filea.read_record(record_index)

    def read_field(self, fieldname: str, level: int):
        self._ensure_filea()
        for i, d in self._fields.items():
            if d["field"] == fieldname and level == d["k"]:
                return self._filea.read_record(i)
        logger.warning("Could not find field %s at level %d", fieldname, level)
        return None

    @property
    def fieldlevels(self) -> set:
        return set(elem["k"] for elem in self._fields.values())

    @property
    def is_mean(self) -> bool:
        """True for a mean archive (archm: '... mean day' header), else instantaneous."""
        return getattr(self, "_is_mean", False)

    @property
    def iversn(self) -> int | None:
        return self._iversn

    @property
    def iexpt(self) -> int | None:
        return self._iexpt

    @property
    def yrflag(self) -> int | None:
        return self._yrflag


class ABFileAve(ABFileArchv):
    """HYCOM AVE .a/.b file pair, produced by hycave/ensave (MSCPROGS).

    The header extends the archv format with ``kdm``, ``month``, ``year``,
    and ``count`` entries before the field-column header.
    """

    def read_header(self) -> None:
        self._header = [self.readline() for _ in range(4)]
        _, self._iversn = self.scanitem(item="iversn", conversion=int)
        _, self._iexpt = self.scanitem(item="iexpt", conversion=int)
        _, self._yrflag = self.scanitem(item="yrflag", conversion=int)
        _, self._idm = self.scanitem(item="idm", conversion=int)
        _, self._jdm = self.scanitem(item="jdm", conversion=int)
        _, self._kdm = self.scanitem(item="kdm", conversion=int)
        _, self._month = self.scanitem(item="month", conversion=int)
        _, self._year = self.scanitem(item="year", conversion=int)
        _, self._count = self.scanitem(item="count", conversion=int)

    @property
    def kdm(self) -> int | None:
        return getattr(self, "_kdm", None)

    @property
    def month(self) -> int | None:
        return getattr(self, "_month", None)

    @property
    def year(self) -> int | None:
        return getattr(self, "_year", None)

    @property
    def count(self) -> int | None:
        return getattr(self, "_count", None)
