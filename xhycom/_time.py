"""Time conversion utilities for HYCOM model days.

The ``forday`` function is adapted from modeltools.hycom._timetools
(https://github.com/NoraLoose/NERSC-HYCOM-CICE), originally authored by
Knut Lisaeter, MIT licence.
"""

from __future__ import annotations

import datetime

import cftime

# Map yrflag → CF calendar name
_CALENDAR: dict[int, str] = {
    0: "360_day",  # 360-day year, starts Jan 16
    1: "366_day",  # 366-day year, starts Jan 16
    2: "366_day",  # 366-day year, starts Jan  1
    3: "standard",  # calendar days since 1901-01-01
    4: "365_day",  # 365-day year, starts Jan  1
    5: "365_day",  # 365-day year since 1901-01-01
}


def _leapyear(iyr: int) -> bool:
    return not ((iyr % 4 == 0 and 1901 + iyr % 400 == 0) or iyr % 4 != 0)


def _forday(dtime: float, yrflag: int) -> tuple[int, int, int]:
    """Convert a HYCOM model day to (year, day-of-year, hour).

    Adapted from the HYCOM Fortran routine ``forday`` via the modeltools
    package.
    """
    import numpy as np

    if yrflag == 0:
        iyear = int((dtime + 15.001) / 360.0) + 1
        iday = int(np.mod(dtime + 15.001, 360.0) + 1)
        ihour = int((np.mod(dtime + 15.001, 360.0) + 1.0 - iday) * 24.0)
    elif yrflag == 1:
        iyear = int((dtime + 15.001) / 366.0) + 1
        iday = int(np.mod(dtime + 15.001, 366.0) + 1)
        ihour = int((np.mod(dtime + 15.001, 366.0) + 1.0 - iday) * 24.0)
    elif yrflag == 2:
        iyear = int((dtime + 0.001) / 366.0) + 1
        iday = int(np.mod(dtime + 0.001, 366.0) + 1)
        ihour = int((np.mod(dtime + 0.001, 366.0) + 1.0 - iday) * 24.0)
    elif yrflag == 3:
        iyr = int((dtime - 1.0) / 365.25)
        nleap = int(iyr / 4)
        dtim1 = 365.0 * iyr + nleap + 1.0
        day = dtime - dtim1 + 1.0
        if dtim1 > dtime:
            iyr -= 1
        elif day >= 367.0:
            iyr += 1
        elif day >= 366.0 and iyr % 4 != 3:
            iyr += 1
        nleap = int(iyr / 4)
        dtim1 = 365.0 * iyr + nleap + 1.0
        iyear = 1901 + iyr
        iday = int(dtime - dtim1 + 1.001)
        ihour = int((dtime - dtim1 + 1.001 - iday) * 24.0)
    elif yrflag == 4:
        iyear = int((dtime + 0.001) / 365.0) + 1
        iday = int(np.mod(dtime + 0.001, 365.0) + 1)
        ihour = int((np.mod(dtime + 0.001, 365.0) + 1.0 - iday) * 24.0)
    elif yrflag == 5:
        iyear = int((dtime + 0.001) / 365.0) + 1901
        iday = int(np.mod(dtime + 0.001, 365.0) + 1)
        ihour = int((np.mod(dtime + 0.001, 365.0) + 1.0 - iday) * 24.0)
    else:
        raise ValueError(f"Unsupported yrflag {yrflag}. Must be 0-5.")

    return iyear, iday, ihour


def model_day_to_datetime(model_day: float, yrflag: int) -> cftime.datetime:
    """Convert a HYCOM model day (float) to a ``cftime.datetime`` object.

    HYCOM stores time as a single floating-point "model day" whose meaning
    depends on ``yrflag`` (see HYCOM blkdat documentation).

    Parameters
    ----------
    model_day : float
        HYCOM model day as stored in the archive ``.b`` header.
    yrflag : int
        HYCOM year-flag (0-5).  Read from the ``.b`` header automatically
        when using :func:`xhycom.open_dataset`.

    Returns
    -------
    cftime.datetime
        Absolute date in the calendar implied by ``yrflag``.

    Raises
    ------
    ValueError
        If ``yrflag`` is not one of the supported values (0-5).

    Notes
    -----
    The mapping from ``yrflag`` to CF calendar name is:

    +--------+-----------+----------------------+
    | yrflag | Calendar  | Epoch                |
    +========+===========+======================+
    | 0      | 360_day   | Jan 16, year 1       |
    | 1      | 366_day   | Jan 16, year 1       |
    | 2      | 366_day   | Jan  1, year 1       |
    | 3      | standard  | Jan  1, 1901         |
    | 4      | 365_day   | Jan  1, year 1       |
    | 5      | 365_day   | Jan  1, 1901         |
    +--------+-----------+----------------------+

    Examples
    --------
    >>> model_day_to_datetime(40909.5, yrflag=3)
    cftime.datetime(2013, 1, 1, 12, 0, 0, 0, has_year_zero=False)
    """
    if yrflag not in _CALENDAR:
        raise ValueError(f"Unsupported yrflag {yrflag}. Must be 0-5.")

    iy, id_, ih = _forday(float(model_day), yrflag)
    calendar = _CALENDAR[yrflag]
    origin = cftime.datetime(iy, 1, 1, 0, 0, 0, calendar=calendar)
    return origin + datetime.timedelta(days=int(id_) - 1, hours=int(ih))
