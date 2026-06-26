# Installation

## From GitHub

```bash
pip install git+https://github.com/NoraLoose/xhycom.git
```

With lazy / Dask-backed loading:

```bash
pip install "xhycom[lazy] @ git+https://github.com/NoraLoose/xhycom.git"
```

With regridding (vertical regridding and depth interpolation):

```bash
pip install "xhycom[regrid] @ git+https://github.com/NoraLoose/xhycom.git"
```

## Horizontal regridding needs xESMF (conda)

`regrid_vertical` is pure Python and installs from PyPI via the `regrid` extra.
**Horizontal** regridding (`regrid_horizontal` and the full `regrid` wrapper) also
needs [xESMF](https://xesmf.readthedocs.io), whose ESMF/esmpy backend has no PyPI
wheels — it must come from conda-forge. The ready-made environment installs the
whole stack, including xhycom itself in editable mode:

```bash
conda env create -f ci/environment-regrid.yml
conda activate xhycom-regrid
```

## On Olivia (NRIS)

Use a **dedicated, lean conda environment for xhycom** — kept separate from the
NERSC-HYCOM-CICE model environment so the regrid stack stays isolated. The repo ships
the recipe in `ci/environment-regrid.yml` (xESMF/ESMF, xgcm, Dask, and xhycom — no model
libraries).

For interactive work — including running the example notebooks — the recommended entry
point is **Open OnDemand**, the Olivia web portal (see the
[NRIS documentation](https://documentation.sigma2.no/)): launch a JupyterLab or
interactive desktop session and run your setup in its terminal.

### Build the environment (once)

Load Miniforge (Olivia provides it through the module system — check the NRIS docs for
the current module name), then create the environment in project space from your xhycom
clone. Redirect the conda package cache to project space too, since `${HOME}` quota is
small and ESMF is large:

```bash
module load Miniforge3              # name/version per NRIS docs
source ${EBROOTMINIFORGE3}/bin/activate
conda config --append pkgs_dirs /cluster/projects/nn2993k/${USER}/conda/package-cache

cd ${HOME}/xhycom                   # your xhycom clone (holds ci/environment-regrid.yml)
conda env create -f ci/environment-regrid.yml -p /cluster/projects/nn2993k/${USER}/xhycom-env
```

The `-p` prefix puts the environment under project space and overrides the `name:` in
the file; `ci/environment-regrid.yml` installs xhycom from the clone in editable mode.

### Environment setup

Prepend the environment's `bin/` to your `PATH`, and point the regrid-weight cache at
project space. Add these to your `~/.bashrc` (and to any batch job script):

```bash
export PATH="/cluster/projects/nn2993k/${USER}/xhycom-env/bin:${PATH}"
export XHYCOM_CACHE_DIR=/cluster/projects/nn2993k/${USER}/hycom_cache_dir
```

Putting the environment's `bin/` first on `PATH` makes its `python`, `jupyter`, and
`xhycom` resolve to the dedicated install — no `conda activate` needed, which is what you
want inside Open OnDemand sessions. Start JupyterLab from a shell where these exports are
active (e.g. via your `~/.bashrc`) so the notebooks run against this environment and
reuse the cached regrid weights.

## Editable / development install

```bash
git clone https://github.com/NoraLoose/xhycom.git
cd xhycom
pip install -e .            # core only
pip install -e ".[lazy]"    # with Dask
pip install -e ".[regrid]"  # with vertical regridding (xgcm + Dask)
pip install -e ".[dev]"     # with test dependencies
```

## Cache directory for regrid weights

Horizontal regridding builds an xESMF weight matrix that maps the source grid onto
the target grid. Generating it is the slow part of a regrid, and it depends only on
the grids — not on the field values — so xhycom caches it to disk and reuses it on
later calls (see **[Regridding](regridding.ipynb)**).

By default the cache lives under `$XDG_CACHE_HOME/xhycom/regrid_weights`
(i.e. `~/.cache/xhycom/regrid_weights`). Set the `XHYCOM_CACHE_DIR` environment
variable to put it somewhere else; weight files and the manifest are then written
**directly** into that directory. Point it at shared project or scratch space so
the same weights are reused across jobs and by collaborators:

```bash
export XHYCOM_CACHE_DIR=/cluster/projects/nn2993k/${USER}/hycom_cache_dir
```

To make it stick across sessions and batch jobs, add that line to your `~/.bashrc`
(or the job script that launches your runs). On Olivia, use the shared environment and
this cache together — see [On Olivia (NRIS)](#on-olivia-nris).

## Dependencies

### Required

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and binary I/O |
| `xarray` | Dataset construction |
| `cftime` | Calendar-aware datetime objects |

xhycom bundles its own HYCOM binary reader — there are no other required install-time dependencies.

### Optional

| Extra | Package(s) | Purpose |
|-------|------------|---------|
| `lazy` | `dask` | Lazy / out-of-core loading via the `chunks` parameter in `open_dataset` and `open_mfdataset` |
| `regrid` | `xgcm`, `dask` | Vertical regridding (`regrid_vertical`) and depth interpolation. Horizontal regridding additionally needs `xesmf` from conda-forge (see above). |
