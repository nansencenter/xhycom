"""Readers for all HYCOM .ab file types, plus file type detection."""
import re
from collections import defaultdict, Counter

import numpy as np
import xarray as xr

from ._abfile import (
    ABFileBathy,
    ABFileGrid,
    ABFileArchv,
    grid_ordered_fieldnames,
)
from ._time import model_day_to_datetime


def _fill(arr):
    """Masked array → float64 ndarray, masked values replaced by NaN."""
    return np.ma.filled(arr.astype(np.float64), np.nan)


# ---------------------------------------------------------------------------
# Variable metadata lookup
# ---------------------------------------------------------------------------
# The .b header only stores field names, not units or descriptions.
# This table supplies CF-style attrs for standard HYCOM archive fields.
# Thickness and depth variables are in Pa (HYCOM stores pressure thickness;
# 1 Pa ≈ 0.1 mm of water at standard density).

_FIELD_ATTRS = {
    # --- layered (3-D) physics variables ---
    "temp":     {"long_name": "sea water potential temperature",        "units": "degC"},
    "salin":    {"long_name": "sea water salinity",                     "units": "PSU"},
    "saln":     {"long_name": "sea water salinity",                     "units": "PSU"},
    "u-vel.":   {"long_name": "sea water x velocity",                   "units": "m s-1"},
    "v-vel.":   {"long_name": "sea water y velocity",                   "units": "m s-1"},
    "thknss":   {"long_name": "layer pressure thickness",               "units": "Pa"},
    "density":  {"long_name": "sea water potential density (sigma-2)",  "units": "kg m-3"},
    "k.e.":     {"long_name": "kinetic energy",                         "units": "m2 s-2"},
    # --- 2-D surface / mixed-layer diagnostics ---
    "montg1":   {"long_name": "Montgomery potential",                   "units": "m2 s-2"},
    "srfhgt":   {"long_name": "sea surface height",                     "units": "Pa"},
    "oneta":    {"long_name": "free surface elevation",                 "units": "m"},
    "surflx":   {"long_name": "net surface heat flux",                  "units": "W m-2"},
    "wtrflx":   {"long_name": "net surface freshwater flux",            "units": "m s-1"},
    "salflx":   {"long_name": "surface salt flux",                      "units": "PSU m s-1"},
    "bl_dpth":  {"long_name": "boundary layer depth",                   "units": "Pa"},
    "mix_dpth": {"long_name": "mixed layer depth",                      "units": "Pa"},
    "tmix":     {"long_name": "mixed layer temperature",                "units": "degC"},
    "smix":     {"long_name": "mixed layer salinity",                   "units": "PSU"},
    "thmix":    {"long_name": "mixed layer thickness",                  "units": "Pa"},
    "umix":     {"long_name": "mixed layer x velocity",                 "units": "m s-1"},
    "vmix":     {"long_name": "mixed layer y velocity",                 "units": "m s-1"},
    "kemix":    {"long_name": "mixed layer kinetic energy",             "units": "m2 s-2"},
    "covice":   {"long_name": "sea ice coverage fraction",              "units": "1"},
    "thkice":   {"long_name": "sea ice thickness",                      "units": "m"},
    "temice":   {"long_name": "sea ice surface temperature",            "units": "degC"},
    "u_btrop":  {"long_name": "barotropic x velocity",                  "units": "m s-1"},
    "v_btrop":  {"long_name": "barotropic y velocity",                  "units": "m s-1"},
    "kebtrop":  {"long_name": "barotropic kinetic energy",              "units": "m2 s-2"},
    "si_u":     {"long_name": "sea ice x velocity",                     "units": "m s-1"},
    "si_v":     {"long_name": "sea ice y velocity",                     "units": "m s-1"},
    # --- biogeochemistry (TOPAZ / ECOSMO) ---
    "ECO_no3":  {"long_name": "nitrate",                                "units": "mmol N m-3"},
    "ECO_nh4":  {"long_name": "ammonium",                               "units": "mmol N m-3"},
    "ECO_pho":  {"long_name": "phosphate",                              "units": "mmol P m-3"},
    "ECO_sil":  {"long_name": "silicate",                               "units": "mmol Si m-3"},
    "ECO_oxy":  {"long_name": "dissolved oxygen",                       "units": "mmol O m-3"},
    "ECO_fla":  {"long_name": "flagellate carbon",                      "units": "mmol C m-3"},
    "ECO_dia":  {"long_name": "diatom carbon",                          "units": "mmol C m-3"},
    "ECO_ccl":  {"long_name": "coccolithophore carbon",                 "units": "mmol C m-3"},
    "ECO_cclc": {"long_name": "coccolithophore calcite carbon",         "units": "mmol C m-3"},
    "ECO_caco": {"long_name": "particulate inorganic carbon (calcite)", "units": "mmol C m-3"},
    "ECO_diac": {"long_name": "diatom calcite carbon",                  "units": "mmol C m-3"},
    "ECO_flac": {"long_name": "flagellate calcite carbon",              "units": "mmol C m-3"},
    "ECO_micr": {"long_name": "microzooplankton carbon",                "units": "mmol C m-3"},
    "ECO_meso": {"long_name": "mesozooplankton carbon",                 "units": "mmol C m-3"},
    "ECO_det":  {"long_name": "detritus carbon",                        "units": "mmol C m-3"},
    "ECO_opa":  {"long_name": "opal (biogenic silica)",                 "units": "mmol Si m-3"},
    "ECO_dom":  {"long_name": "dissolved organic matter carbon",        "units": "mmol C m-3"},
    "ECO_c2ch": {"long_name": "carbon to chlorophyll ratio",            "units": "g C g-1 Chl"},
    "ECO_prim": {"long_name": "primary production",                     "units": "mmol C m-3 s-1"},
    "ECO_secp": {"long_name": "secondary production",                   "units": "mmol C m-3 s-1"},
    "ECO_netp": {"long_name": "net primary production",                 "units": "mmol C m-3 s-1"},
    "ECO_deni": {"long_name": "denitrification",                        "units": "mmol N m-3 s-1"},
    "ECO_snks": {"long_name": "sinking rate",                           "units": "m d-1"},
    "ECO_Nlim": {"long_name": "nitrogen limitation factor",             "units": "1"},
    "ECO_Plim": {"long_name": "phosphorus limitation factor",           "units": "1"},
    "ECO_Slim": {"long_name": "silicate limitation factor",             "units": "1"},
    "ECO_Llim": {"long_name": "light limitation factor",               "units": "1"},
    "ECO_parm": {"long_name": "BGC parameter field",                    "units": "1"},
    "ECO_bots": {"long_name": "bottom sediment flux",                   "units": "1"},
    "ECO_dsnk": {"long_name": "detritus sinking flux",                  "units": "mmol C m-2 s-1"},
    "ECO_sed1": {"long_name": "sediment pool 1",                        "units": "mmol m-2"},
    "ECO_sed2": {"long_name": "sediment pool 2",                        "units": "mmol m-2"},
    "ECO_sed3": {"long_name": "sediment pool 3",                        "units": "mmol m-2"},
    "ECO_sed4": {"long_name": "sediment pool 4",                        "units": "mmol m-2"},
    "CO2_c":    {"long_name": "dissolved inorganic carbon",             "units": "mmol C m-3"},
    "CO2_TA":   {"long_name": "total alkalinity",                       "units": "mmol eq m-3"},
    "CO2_pH":   {"long_name": "seawater pH",                            "units": "1"},
    "CO2_pCO2": {"long_name": "partial pressure of CO2",                "units": "uatm"},
    "CO2_Carb": {"long_name": "carbonate concentration",                "units": "mmol C m-3"},
    "CO2_BiCa": {"long_name": "bicarbonate concentration",              "units": "mmol C m-3"},
    "CO2_Om_c": {"long_name": "calcite saturation state (Omega)",       "units": "1"},
    "CO2_Om_a": {"long_name": "aragonite saturation state (Omega)",     "units": "1"},
    "CO2_fair": {"long_name": "air-sea CO2 flux",                       "units": "mmol C m-2 d-1"},
    "CO2_wind": {"long_name": "wind speed for gas exchange",            "units": "m s-1"},
    "total_ch": {"long_name": "total chlorophyll",                      "units": "mg Chl m-3"},
    "total_ca": {"long_name": "total carbon",                           "units": "mmol C m-3"},
    "light_sw": {"long_name": "shortwave irradiance in water",          "units": "W m-2"},
    "light_pa": {"long_name": "PAR irradiance",                         "units": "W m-2"},
    "attenuat": {"long_name": "light attenuation coefficient",          "units": "m-1"},
}


def _attrs_for(fname):
    """Return metadata attrs for fname, falling back to the base name for
    renamed duplicates (e.g. 'ECO_c2ch_2' → look up 'ECO_c2ch')."""
    if fname in _FIELD_ATTRS:
        return dict(_FIELD_ATTRS[fname])
    base = re.sub(r"_\d+$", "", fname)
    return dict(_FIELD_ATTRS.get(base, {}))


# ---------------------------------------------------------------------------
# Grid variable metadata
# ---------------------------------------------------------------------------

_GRID_ATTRS = {
    "plon": {"long_name": "longitude of T-point",            "units": "degrees_east",  "standard_name": "longitude"},
    "plat": {"long_name": "latitude of T-point",             "units": "degrees_north", "standard_name": "latitude"},
    "qlon": {"long_name": "longitude of Q-point (vorticity)","units": "degrees_east"},
    "qlat": {"long_name": "latitude of Q-point (vorticity)", "units": "degrees_north"},
    "ulon": {"long_name": "longitude of U-point",            "units": "degrees_east"},
    "ulat": {"long_name": "latitude of U-point",             "units": "degrees_north"},
    "vlon": {"long_name": "longitude of V-point",            "units": "degrees_east"},
    "vlat": {"long_name": "latitude of V-point",             "units": "degrees_north"},
    "pang": {"long_name": "local angle of grid x-axis from true east (T-point)", "units": "radians"},
    "scpx": {"long_name": "T-point grid spacing in x",       "units": "m"},
    "scpy": {"long_name": "T-point grid spacing in y",       "units": "m"},
    "scqx": {"long_name": "Q-point grid spacing in x",       "units": "m"},
    "scqy": {"long_name": "Q-point grid spacing in y",       "units": "m"},
    "scux": {"long_name": "U-point grid spacing in x",       "units": "m"},
    "scuy": {"long_name": "U-point grid spacing in y",       "units": "m"},
    "scvx": {"long_name": "V-point grid spacing in x",       "units": "m"},
    "scvy": {"long_name": "V-point grid spacing in y",       "units": "m"},
    "cori": {"long_name": "Coriolis parameter",              "units": "s-1"},
    "pasp": {"long_name": "T-point aspect ratio (scpx / scpy)", "units": "1"},
}


# ---------------------------------------------------------------------------
# Staggered C-grid helpers
# ---------------------------------------------------------------------------

# Standard HYCOM field names that live on U-points and V-points.
_U_VARS = frozenset({"u-vel.", "u_btrop", "umix", "si_u"})
_V_VARS = frozenset({"v-vel.", "v_btrop", "vmix", "si_v"})

# T-point variables used as the preferred source for the dens coordinate.
_TPOINT_DENS_PRIORITY = ("thknss", "temp", "salin", "density")


def _h_coords(fname, grid_ds):
    """Horizontal lon/lat coordinates for fname's staggering point.

    T-point → lon/lat  (plon/plat)
    U-point → lon_u/lat_u  (ulon/ulat)
    V-point → lon_v/lat_v  (vlon/vlat)
    """
    if grid_ds is None:
        return {}
    if fname in _U_VARS:
        return {
            "lon_u": (["y", "x"], grid_ds["ulon"].values,
                      {"long_name": "longitude (U-point)", "units": "degrees_east"}),
            "lat_u": (["y", "x"], grid_ds["ulat"].values,
                      {"long_name": "latitude (U-point)",  "units": "degrees_north"}),
        }
    if fname in _V_VARS:
        return {
            "lon_v": (["y", "x"], grid_ds["vlon"].values,
                      {"long_name": "longitude (V-point)", "units": "degrees_east"}),
            "lat_v": (["y", "x"], grid_ds["vlat"].values,
                      {"long_name": "latitude (V-point)",  "units": "degrees_north"}),
        }
    return {
        "lon": (["y", "x"], grid_ds["plon"].values,
                {"long_name": "longitude (T-point)", "units": "degrees_east",
                 "standard_name": "longitude"}),
        "lat": (["y", "x"], grid_ds["plat"].values,
                {"long_name": "latitude (T-point)",  "units": "degrees_north",
                 "standard_name": "latitude"}),
    }


def _v_dim(levels):
    """Vertical dimension name for a multi-level variable.

    Layer centres (k = 1..N)       → 'k'
    Layer interfaces (k = 0..N)    → 'ki'
    """
    return "ki" if 0 in levels else "k"


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

def detect_filetype(basename):
    """Detect the type of a HYCOM ``.ab`` file pair from the ``.b`` header.

    Parameters
    ----------
    basename : str
        Path without the ``.a`` / ``.b`` extension.

    Returns
    -------
    str
        One of ``"archv"``, ``"grid"``, ``"bathy"``, or ``"forcing"``.

    Raises
    ------
    ValueError
        If the file type cannot be determined.
    """
    with open(basename + ".b") as f:
        header = f.read(512)

    if re.search(r"'iversn'", header):
        return "archv"
    if re.search(r"'mapflg'", header):
        return "grid"
    if re.search(r"dtime1,range", header):
        return "forcing"
    if re.search(r"min,max\s+depth", header):
        return "bathy"

    raise ValueError(
        f"Cannot determine file type of {basename!r}.  "
        "Expected a HYCOM archv, grid, bathy, or forcing .ab file."
    )


# ---------------------------------------------------------------------------
# Lazy-loading helper (module-level so Dask can serialise it)
# ---------------------------------------------------------------------------

def _read_record_lazy(basename, record_idx, endian):
    """Open the archive and read one 2-D slab by record index.

    Module-level so Dask can serialise it across workers.
    """
    af = ABFileArchv(basename, "r", endian=endian)
    raw = af.read_record(record_idx)
    af.close()
    return _fill(raw)


def _read_var_lazy(basename, record_indices, endian):
    """Open the archive and read all vertical levels of one variable.

    *record_indices* is a list of integer record positions, one per level,
    in sorted-level order.  Returns a (nlev, jdm, idm) array for multi-level
    variables and a (jdm, idm) array for 2-D variables.

    One task per (file, variable) instead of per (file, variable, level)
    keeps the Dask graph small for multi-file opens.
    Module-level so Dask can serialise it across workers.
    """
    af = ABFileArchv(basename, "r", endian=endian)
    slabs = [_fill(af.read_record(i)) for i in record_indices]
    af.close()
    return slabs[0] if len(slabs) == 1 else np.stack(slabs, axis=0)


# ---------------------------------------------------------------------------
# Per-type readers (internal)
# ---------------------------------------------------------------------------

def _read_archv_meta(basename, endian="big"):
    """Parse the .b header of an archive file, returning metadata without reading .a data.

    Returns a dict with keys: field_kdens, field_k_record, jdm, idm, yrflag,
    iversn, iexpt, global_kdens, time.
    """
    af = ABFileArchv(basename, "r", endian=endian)

    # Build unique field names, handling duplicate (field, k) pairs.
    pair_count: Counter = Counter()
    for rec in af.fields.values():
        pair_count[(rec["field"], rec["k"])] += 1

    name_running: defaultdict = defaultdict(int)
    field_kdens: defaultdict = defaultdict(dict)
    field_k_record: defaultdict = defaultdict(dict)

    for i, rec in af.fields.items():
        fname = rec["field"]
        k = rec["k"]
        pair = (fname, k)
        name_running[pair] += 1
        uname = f"{fname}_{name_running[pair]}" if pair_count[pair] > 1 else fname
        field_kdens[uname][k] = rec["dens"]
        field_k_record[uname][k] = i

    jdm, idm = af.jdm, af.idm
    yrflag = af.yrflag
    iversn = af.iversn
    iexpt = af.iexpt
    first_rec = next(iter(af.fields.values())) if af.fields else {}
    model_day = first_rec.get("day")

    af.close()  # only closes .b — .a was never opened

    time = None
    if model_day is not None and yrflag is not None:
        time = model_day_to_datetime(model_day, yrflag)

    return {
        "field_kdens": dict(field_kdens),
        "field_k_record": dict(field_k_record),
        "jdm": jdm,
        "idm": idm,
        "yrflag": yrflag,
        "iversn": iversn,
        "iexpt": iexpt,
        "global_kdens": _compute_global_kdens(field_kdens),
        "time": time,
    }


def _compute_global_kdens(field_kdens):
    """Build the k→dens mapping from a (possibly filtered) field_kdens dict.

    Prefers T-point variables as the authoritative source for layer densities.
    """
    global_kdens: dict = {}
    for kdens in field_kdens.values():
        if len(kdens) > 1 and 0 not in kdens:
            global_kdens.update(kdens)
    for fname in _TPOINT_DENS_PRIORITY:
        if fname in field_kdens and len(field_kdens[fname]) > 1 and 0 not in field_kdens[fname]:
            global_kdens.update(field_kdens[fname])
            break
    return global_kdens


def _apply_variables_filter(field_kdens, field_k_record, variables, source):
    """Return filtered copies of field_kdens and field_k_record.

    Emits a warning for any requested variable not present in *source*.
    """
    import warnings
    requested = set(variables)
    available = set(field_kdens)
    missing = sorted(requested - available)
    if missing:
        warnings.warn(
            f"{source}: requested variables not found and skipped: {missing}",
            stacklevel=3,
        )
    keep = requested & available
    return (
        {k: v for k, v in field_kdens.items() if k in keep},
        {k: v for k, v in field_k_record.items() if k in keep},
    )


def _build_mf_lazy(basenames, metas, grid_ds, endian, variables=None):
    """Build a combined lazy Dataset from pre-parsed per-file metadata.

    Constructs Dask arrays directly rather than calling xr.concat, avoiding
    O(N·V) metadata-merging overhead for large file lists.
    """
    import dask
    import dask.array as da

    ref = metas[0]
    times = [m["time"] for m in metas]
    jdm, idm = ref["jdm"], ref["idm"]

    field_kdens = ref["field_kdens"]
    if variables is not None:
        field_kdens, _ = _apply_variables_filter(
            field_kdens, ref["field_k_record"], variables, source="archive",
        )

    global_kdens = _compute_global_kdens(field_kdens)

    data_vars = {}
    for uname, kdens in field_kdens.items():
        levels = sorted(kdens)
        h_coords = _h_coords(uname, grid_ds)
        attrs = _attrs_for(uname)

        # One Dask task per (file, variable): reads all vertical levels at once.
        # This is 40× fewer tasks than one task per slab, which keeps the Dask
        # graph small and avoids GBs of task-graph overhead for many files.
        slab_shape = (jdm, idm) if len(levels) == 1 else (len(levels), jdm, idm)
        file_arrs = []
        for basename, meta in zip(basenames, metas):
            fkr = meta["field_k_record"]
            if uname not in fkr:
                continue
            record_indices = [fkr[uname][k] for k in levels]
            file_arrs.append(
                da.from_delayed(
                    dask.delayed(_read_var_lazy)(basename, record_indices, endian),
                    shape=slab_shape,
                    dtype=np.float64,
                )
            )

        combined = da.stack(file_arrs, axis=0)  # (n_files, [k,] y, x)

        if len(levels) == 1:
            dims = ["time", "y", "x"]
            coords = dict(h_coords)
        else:
            vdim = _v_dim(levels)
            vdim_attrs = (
                {"long_name": "layer index",           "units": "1", "axis": "Z"}
                if vdim == "k" else
                {"long_name": "layer interface index", "units": "1", "axis": "Z"}
            )
            dims = ["time", vdim, "y", "x"]
            coords = dict(h_coords)
            coords[vdim] = (vdim, levels, vdim_attrs)

        data_vars[uname] = xr.DataArray(
            combined, dims=dims, coords=coords, attrs=attrs, name=uname,
        )

    global_attrs = {"iversn": ref["iversn"], "iexpt": ref["iexpt"], "yrflag": ref["yrflag"]}
    ds = xr.Dataset(data_vars, attrs=global_attrs)

    if any(t is not None for t in times):
        ds = ds.assign_coords(time=("time", times))

    if global_kdens:
        k_vals = sorted(global_kdens)
        ds = ds.assign_coords(dens=xr.Variable(
            "k", [global_kdens[k] for k in k_vals],
            {"long_name": "target sigma-2 layer density", "units": "kg m-3"},
        ))

    return ds


def read_archv(basename, grid_ds=None, endian="big", chunks=None, variables=None):
    """Read a HYCOM archive ``.ab`` file pair into an ``xr.Dataset``.

    Parameters
    ----------
    chunks : int, dict, "auto", or None
        If not ``None``, field data are read lazily via Dask — the ``.a``
        file is not touched until the returned Dataset is computed.
        The value is forwarded to ``ds.chunk()`` to set chunk boundaries
        (e.g. ``{"k": 1}`` for one layer per chunk).
    variables : list of str, optional
        If provided, only these variables are included in the returned Dataset.
        Variables not present in the file are silently skipped with a warning.
    """
    meta = _read_archv_meta(basename, endian=endian)

    field_kdens = meta["field_kdens"]
    field_k_record = meta["field_k_record"]

    if variables is not None:
        field_kdens, field_k_record = _apply_variables_filter(
            field_kdens, field_k_record, variables, source=basename,
        )

    jdm, idm = meta["jdm"], meta["idm"]
    global_kdens = _compute_global_kdens(field_kdens)
    global_attrs = {"iversn": meta["iversn"], "iexpt": meta["iexpt"], "yrflag": meta["yrflag"]}

    if chunks is not None:
        try:
            import dask
            import dask.array as da
        except ImportError:
            raise ImportError(
                "Dask is required for lazy/chunked loading. "
                "Install it with: pip install dask"
            )

        def _get_slab(uname, k):
            return da.from_delayed(
                dask.delayed(_read_record_lazy)(
                    basename, field_k_record[uname][k], endian
                ),
                shape=(jdm, idm),
                dtype=np.float64,
            )

        def _stack(slabs):
            return da.stack(slabs, axis=0)

    else:
        af = ABFileArchv(basename, "r", endian=endian)

        def _get_slab(uname, k):
            return _fill(af.read_record(field_k_record[uname][k]))

        def _stack(slabs):
            return np.stack(slabs)

    data_vars = {}
    for uname, kdens in field_kdens.items():
        levels = sorted(kdens)
        h_coords = _h_coords(uname, grid_ds)
        attrs = _attrs_for(uname)
        if len(levels) == 1:
            data_vars[uname] = xr.DataArray(
                _get_slab(uname, levels[0]), dims=["y", "x"],
                coords=h_coords, attrs=attrs, name=uname,
            )
        else:
            vdim = _v_dim(levels)
            vdim_attrs = (
                {"long_name": "layer index",           "units": "1", "axis": "Z"}
                if vdim == "k" else
                {"long_name": "layer interface index", "units": "1", "axis": "Z"}
            )
            arr = _stack([_get_slab(uname, k) for k in levels])
            coords = dict(h_coords)
            coords[vdim] = (vdim, levels, vdim_attrs)
            data_vars[uname] = xr.DataArray(
                arr, dims=[vdim, "y", "x"],
                coords=coords, attrs=attrs, name=uname,
            )

    if chunks is None:
        af.close()

    ds = xr.Dataset(data_vars, attrs=global_attrs)

    if global_kdens:
        k_vals = sorted(global_kdens)
        ds = ds.assign_coords(dens=xr.Variable(
            "k", [global_kdens[k] for k in k_vals],
            {"long_name": "target sigma-2 layer density", "units": "kg m-3"},
        ))

    if meta["time"] is not None:
        ds = ds.expand_dims({"time": [meta["time"]]})

    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


def read_grid(basename, endian="big"):
    """Read a HYCOM ``regional.grid`` ``.ab`` file pair into an ``xr.Dataset``."""
    gf = ABFileGrid(basename, "r", endian=endian)
    data_vars = {}
    for fname in grid_ordered_fieldnames:
        raw = gf.read_field(fname)
        if raw is not None:
            data_vars[fname] = xr.DataArray(
                _fill(raw), dims=["y", "x"], name=fname,
                attrs=_GRID_ATTRS.get(fname, {}),
            )
    gf.close()
    return xr.Dataset(data_vars)


def read_bathy(basename, grid_ds, endian="big"):
    """Read a HYCOM bathymetry ``.ab`` file pair into an ``xr.Dataset``.

    *grid_ds* is required to supply ``idm`` / ``jdm`` (not stored in the
    bathymetry file itself) and to attach ``lon`` / ``lat`` coordinates.
    """
    jdm, idm = grid_ds["plon"].shape
    bf = ABFileBathy(basename, "r", idm=idm, jdm=jdm, endian=endian)
    raw = bf.read_field("depth")
    bf.close()

    da = xr.DataArray(
        _fill(raw),
        dims=["y", "x"],
        coords={
            "lon": (["y", "x"], grid_ds["plon"].values,
                    {"long_name": "longitude (T-point)", "units": "degrees_east",
                     "standard_name": "longitude"}),
            "lat": (["y", "x"], grid_ds["plat"].values,
                    {"long_name": "latitude (T-point)",  "units": "degrees_north",
                     "standard_name": "latitude"}),
        },
        attrs={"units": "m", "long_name": "sea floor depth"},
        name="depth",
    )
    return xr.Dataset({"depth": da})
