# Installation

## 🐍 conda (recommended)

```bash
conda install -c conda-forge xhycom
```

This installs xhycom together with [xESMF](https://xesmf.readthedocs.io), which is required for horizontal regridding and has no PyPI wheels. All features are available.

## 📦 pip

xESMF must come from conda-forge, so first create a conda environment that has it, then pip install xhycom into it:

```bash
conda create -n hycom-env -c conda-forge xesmf
conda activate hycom-env
pip install xhycom
```

## 🔧 From GitHub (latest unreleased)

To get the latest development version before it appears in an official release, clone the repo and create the conda environment from the bundled recipe. This sets up xESMF and installs xhycom in editable mode from the clone in one step:

```bash
git clone https://github.com/nansencenter/xhycom.git
cd xhycom
conda env create -f ci/environment-regrid.yml
conda activate hycom-analysis-env
```

## 🖥️ On Olivia / Betzy (NRIS)

The instructions below follow the **From GitHub** route (editable install from a clone). We can use a dedicated conda environment for HYCOM analysis with `xhycom`. The repo
ships the recipe in `ci/environment-regrid.yml` (xESMF/ESMF, xgcm, Dask, JupyterLab,
and xhycom), named `hycom-analysis-env`. **Olivia and Betzy build and activate this
environment differently**: Betzy uses conda directly; Olivia uses
[HPC-container-wrapper](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html#key-features-of-hpc-container-wrapper),
which builds the environment inside a container instead.

### Build the environment (once)

::::{dropdown} Betzy
:open:

Betzy only provides Miniforge3. First redirect the package cache and environments to
project space, since `${HOME}` quota is limited, this is one-time setup, saved to
`~/.condarc`:

```bash
module load Miniforge3/24.1.2-0
source ${EBROOTMINIFORGE3}/bin/activate
conda config --append pkgs_dirs /cluster/projects/nn2993k/conda/${USER}/package-cache
conda config --append envs_dirs /cluster/projects/nn2993k/conda/${USER}
```

Then create the environment from the clone, this also `pip install`s xhycom itself
(in editable mode, from the clone) as part of `ci/environment-regrid.yml`:

```bash
cd ${HOME}/xhycom                   # your xhycom clone (holds ci/environment-regrid.yml)
conda env create -f ci/environment-regrid.yml
conda activate hycom-analysis-env
```

The `envs_dirs` redirect means the named environment lands in project space
automatically, no `-p` prefix needed.
::::

::::{dropdown} Olivia

Load the container wrapper (session-only, no need to add these to `~/.bashrc`):

```bash
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
module load NRIS/CPU
module load hpc-container-wrapper
```

Build the environment as a container in project space, from the clone, this also
`pip install`s xhycom itself (in editable mode, from the clone) as part of
`ci/environment-regrid.yml`:

```bash
cd ${HOME}/xhycom                   # your xhycom clone (holds ci/environment-regrid.yml)
conda-containerize new --mamba \
    --prefix /cluster/projects/nn2993k/conda/${USER}/hycom-analysis-env \
    ci/environment-regrid.yml
```

Keep `--prefix`'s directory around permanently, it holds the container and
executables. Since the containerised environment can't be modified in place, rebuild
with the same command (after removing the old prefix) if `ci/environment-regrid.yml`
changes.
::::

### Start JupyterLab via Open OnDemand

For interactive work, including running the example notebooks, start a **JupyterLab**
session through **Open OnDemand**:

- Betzy: <https://apps.betzy.sigma2.no/pun/sys/dashboard>
- Olivia: <https://apps.olivia.sigma2.no/pun/sys/dashboard>

Click **JupyterLab**, and paste the appropriate snippet below into the app's
*Environment setup* field before launching, this runs before the Jupyter server
starts and makes the `hycom-analysis-env` environment (`python`, `jupyter`, `xhycom`)
the one the session runs against, reusing the same cached regrid weights across
sessions.

::::{dropdown} Betzy
:open:

Load the module system first, then activate the environment by name:

```bash
module load Miniforge3/24.1.2-0
source ${EBROOTMINIFORGE3}/bin/activate
conda activate hycom-analysis-env
export XHYCOM_CACHE_DIR="/cluster/projects/nn2993k/${USER}/.xhycom-cache-dir"
```
::::

::::{dropdown} Olivia

The containerised environment is activated by prepending its `bin/` to `PATH` (no
`conda activate` needed):

```bash
export PATH="/cluster/projects/nn2993k/conda/${USER}/hycom-analysis-env/bin:${PATH}"
export XHYCOM_CACHE_DIR="/cluster/projects/nn2993k/${USER}/.xhycom-cache-dir"
```
::::

For an editable / development install and how to run the test suite, see the
[Contributor Guide](contributing.md#editable--development-install).

## 💾 Cache directory for regrid weights

Horizontal regridding builds an xESMF weight matrix that maps the source grid onto
the target grid. Generating it is the slow part of a regrid, and it depends only on
the grids, not on the field values, so xhycom caches it to disk and reuses it on
later calls (see **[Regridding](regridding.ipynb)**).

By default the cache lives under `$XDG_CACHE_HOME/xhycom/regrid_weights`
(i.e. `~/.cache/xhycom/regrid_weights`). Set the `XHYCOM_CACHE_DIR` environment
variable to put it somewhere else; weight files and the manifest are then written
**directly** into that directory. Point it at shared project or scratch space so
the same weights are reused across jobs and by collaborators:

```bash
export XHYCOM_CACHE_DIR="/cluster/projects/nn2993k/${USER}/.xhycom-cache-dir"
```

A leading `.` keeps it out of the way of a plain `ls` alongside your other project-space
directories, same as `~/.cache`.

Always add that export to your `~/.bashrc`, even if you also put it in the Open
OnDemand *Environment setup* field (see
[On Olivia / Betzy (NRIS)](#on-olivia-betzy-nris)) or a batch job script.

## 📋 Dependencies

### Required

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and binary I/O |
| `xarray` | Dataset construction |
| `cftime` | Calendar-aware datetime objects |
| `dask` | Lazy / out-of-core loading via the `chunks` parameter in `open_dataset` and `open_mfdataset` |
| `xgcm` | Vertical regridding (`regrid_vertical`) and depth interpolation |

xhycom bundles its own HYCOM binary reader, pip installs everything needed for reading, lazy loading, and vertical regridding.

### Optional

| Package | Purpose |
|---------|---------|
| `xesmf` (conda-forge only) | Horizontal regridding (`regrid_horizontal` and the full `regrid` wrapper). Kept optional since its ESMF/esmpy backend can conflict with other ESMF installs on some platforms (see above). |
