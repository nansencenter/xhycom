# Contributor Guide

## Editable / development install

```bash
git clone https://github.com/nansencenter/xhycom.git
cd xhycom
pip install -e .            # core (includes Dask + xgcm)
pip install -e ".[dev]"     # with test dependencies
```

## Running the tests

```bash
pytest
```

The `dev` extra (`pytest`, `pytest-cov`, `netCDF4`) plus the core dependencies is
enough to run the reader, postprocessing, and vertical-regridding tests — they're
self-contained, with real-data fixtures bundled under `tests/data/`.

The lateral (horizontal) regridding tests additionally need `xesmf`, so they
self-skip (`pytest.importorskip("xesmf")`) unless it's installed. To run the full
suite including those, use the conda environment from
[Horizontal regridding needs xESMF (conda)](installation.md#horizontal-regridding-needs-xesmf-conda):

```bash
conda env create -f ci/environment-regrid.yml
conda activate hycom-analysis-env
pytest
```

## Building the documentation locally

The documentation is built with [Sphinx](https://www.sphinx-doc.org) using
[MyST-NB](https://myst-nb.readthedocs.io), so Markdown files and Jupyter notebooks
render directly.

### Set up the environment

A conda environment with all required packages is provided in `docs/environment.yml`:

```bash
conda env create -f docs/environment.yml
conda activate xhycom-docs
```

If you already have a conda environment for xhycom, you can install the docs
dependencies into it instead:

```bash
conda activate <your-env>
pip install -e ".[docs]"
```

### Build

Run from the `docs/` directory:

```bash
cd docs
make html
```

Then open `docs/_build/html/index.html` in a browser. Other useful targets:

```bash
make clean   # remove the build directory
make help    # list all available targets
```

### Previewing documentation changes in a PR

To see how your changes to the documentation render, you have two options:

1. Build the documentation locally — see [Build](#build) above for instructions.
2. After pushing your changes to the PR, once the Read the Docs build has finished,
   click the yellow link in the PR's checks list (as shown below) to preview the
   rendered docs for this PR:

   <img width="926" height="449" alt="Read the Docs PR check with the Details link highlighted" src="https://github.com/user-attachments/assets/bf73471a-0bee-4dd4-a386-7ce50c019566" />

### Adding or editing pages

- All documentation lives in `docs/` as Markdown files and Jupyter notebooks.
- The table of contents is defined in the `{toctree}` block in `docs/index.md`.
- To add a new page, create a `.md` or `.ipynb` file in `docs/` and add its name
  (without extension) to the `toctree` block in `docs/index.md`.
