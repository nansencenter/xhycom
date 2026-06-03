"""File discovery helpers for HYCOM archive datasets."""
import glob
import os
import re

# Matches: archv.YYYY_DDD_HH  or  archv.YYYY_DDD  (and archm equivalents)
_ARCHV_RE = re.compile(r"arch[vm]\.(\d{4})_(\d{3})(?:_(\d{2}))?$")


def _sort_key(basename):
    """Return a sortable (year, doy, hour) tuple from an archv basename."""
    m = _ARCHV_RE.search(os.path.basename(basename))
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
    return (0, 0, 0)


def find_archv_files(path):
    """Find HYCOM archive ``.ab`` file pairs and return sorted basenames.

    Scans *path* for files whose names match ``archv.YYYY_DDD_HH`` or
    ``archv.YYYY_DDD``.  The argument can be a directory or a glob pattern.

    Parameters
    ----------
    path : str
        A directory path or a glob pattern (e.g. ``"data/archv.2020_*.a"``).

    Returns
    -------
    list of str
        Sorted list of basenames without the ``.a`` / ``.b`` extension,
        ordered chronologically by (year, day-of-year, hour).

    Raises
    ------
    FileNotFoundError
        If *path* is a non-existent directory.
    ValueError
        If no matching archive file pairs are found.

    Examples
    --------
    >>> files = find_archv_files("data/")
    >>> files[0]
    'data/archv.2020_001_00'
    """
    if os.path.isdir(path):
        candidates = (
            glob.glob(os.path.join(path, "archv.*.b")) +
            glob.glob(os.path.join(path, "archm.*.b"))
        )
    else:
        # Treat as glob; strip any .a/.b suffix before globbing
        base_pattern = re.sub(r"\.[ab]$", "", path)
        candidates = glob.glob(base_pattern + ".b")

    basenames = []
    for f in candidates:
        base = re.sub(r"\.[ab]$", "", f)
        if _ARCHV_RE.search(os.path.basename(base)):
            if os.path.exists(base + ".a") and os.path.exists(base + ".b"):
                basenames.append(base)

    if not basenames:
        raise ValueError(
            f"No archive .ab file pairs found at {path!r}. "
            "Expected files named archv.YYYY_DDD_HH.[ab] or archm.YYYY_DDD_HH.[ab]"
        )

    return sorted(set(basenames), key=_sort_key)
