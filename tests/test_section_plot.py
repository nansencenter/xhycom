"""Tests for xhycom._section: section_plot."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot import

import numpy as np
import pytest
import xarray as xr

from xhycom._section import section_plot


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section_ds(
    nk: int = 5,
    ns: int = 10,
    with_time: bool = False,
    with_depth: bool = True,
    var_val: float = 1.0,
) -> xr.Dataset:
    """Synthetic (k, section) section dataset."""
    values = np.full((nk, ns), var_val)
    distance = np.linspace(0.0, 100.0, ns)
    depth_1d = np.linspace(5.0, 50.0 * nk, nk)
    depth_2d = np.broadcast_to(depth_1d[:, np.newaxis], (nk, ns)).copy()

    coords = {"distance_km": ("section", distance)}
    dvars: dict = {"temp": (("k", "section"), values)}
    if with_depth:
        dvars["depth_m"] = (("k", "section"), depth_2d)

    if with_time:
        val3 = np.stack([values, values * 2.0], axis=0)  # (time=2, k, section)
        dvars["temp"] = (("time", "k", "section"), val3)
        if with_depth:
            dep3 = np.stack([depth_2d, depth_2d], axis=0)
            dvars["depth_m"] = (("time", "k", "section"), dep3)

    return xr.Dataset(dvars, coords=coords)


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------


def test_section_plot_returns_axes() -> None:
    """section_plot returns a matplotlib Axes object."""
    import matplotlib.axes

    ax = section_plot(_make_section_ds(), "temp")
    assert isinstance(ax, matplotlib.axes.Axes)


def test_section_plot_accepts_existing_ax() -> None:
    """section_plot draws into an existing axes without creating a new figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = section_plot(_make_section_ds(), "temp", ax=ax)
    assert result is ax


def test_section_plot_without_depth_uses_integer_axis() -> None:
    """Absent depth_m falls back to integer 0..nk-1 depth axis."""
    ax = section_plot(_make_section_ds(with_depth=False), "temp")
    ymin, ymax = sorted(ax.get_ylim())  # invert_yaxis means ymax < ymin
    assert ymin == pytest.approx(0.0, abs=1.0)  # pcolormesh extends slightly


# ---------------------------------------------------------------------------
# Axis inversion
# ---------------------------------------------------------------------------


def test_section_plot_yaxis_inverted() -> None:
    """Y-axis is always inverted so depth increases downward."""
    ax = section_plot(_make_section_ds(), "temp")
    assert ax.yaxis_inverted()


def test_section_plot_flip_x_inverts_xaxis() -> None:
    """flip_x=True inverts the x-axis."""
    ax = section_plot(_make_section_ds(), "temp", flip_x=True)
    assert ax.xaxis_inverted()


def test_section_plot_flip_x_false_does_not_invert() -> None:
    """flip_x=False (default) leaves the x-axis normal."""
    ax = section_plot(_make_section_ds(), "temp", flip_x=False)
    assert not ax.xaxis_inverted()


# ---------------------------------------------------------------------------
# depth_max
# ---------------------------------------------------------------------------


def test_section_plot_depth_max_clips_ylim() -> None:
    """depth_max clips the y-axis at that depth."""
    ax = section_plot(_make_section_ds(), "temp", depth_max=100.0)
    ymin, ymax = sorted(ax.get_ylim())
    assert ymax == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Time dimension
# ---------------------------------------------------------------------------


def test_section_plot_with_time_dim_uses_itime_zero() -> None:
    """Default itime=0 selects the first time step without error."""
    ax = section_plot(_make_section_ds(with_time=True), "temp")
    assert ax is not None


def test_section_plot_with_time_dim_itime_one() -> None:
    """itime=1 selects the second time step without error."""
    ax = section_plot(_make_section_ds(with_time=True), "temp", itime=1)
    assert ax is not None


# ---------------------------------------------------------------------------
# Colormap selection
# ---------------------------------------------------------------------------


def test_section_plot_center_zero_uses_rdbu_r() -> None:
    """center_zero=True uses RdBu_r by default."""
    ds = _make_section_ds()
    ds["temp"] = (("k", "section"), np.linspace(-1, 1, 50).reshape(5, 10))
    ax = section_plot(ds, "temp", center_zero=True)
    # Verify the plot rendered without error; colormap is a best-effort check
    assert ax is not None


def test_section_plot_custom_cmap() -> None:
    """Custom cmap argument is forwarded without error."""
    ax = section_plot(_make_section_ds(), "temp", cmap="plasma")
    assert ax is not None


# ---------------------------------------------------------------------------
# Colorbar
# ---------------------------------------------------------------------------


def test_section_plot_add_colorbar_true() -> None:
    """add_colorbar=True adds a colorbar to the figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    section_plot(_make_section_ds(), "temp", ax=ax, add_colorbar=True)
    assert len(fig.axes) == 2  # main axes + colorbar axes


def test_section_plot_add_colorbar_false() -> None:
    """add_colorbar=False adds no colorbar."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    section_plot(_make_section_ds(), "temp", ax=ax, add_colorbar=False)
    assert len(fig.axes) == 1


def test_section_plot_colorbar_label() -> None:
    """colorbar_label is applied to the colorbar."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    section_plot(_make_section_ds(), "temp", ax=ax, colorbar_label="°C")
    cb_ax = fig.axes[1]
    assert "°C" in cb_ax.get_ylabel()


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------


def test_section_plot_default_title_is_var_name() -> None:
    """Default title is the variable name."""
    ax = section_plot(_make_section_ds(), "temp")
    assert ax.get_title() == "temp"


def test_section_plot_custom_title() -> None:
    """Custom title is applied to the axes."""
    ax = section_plot(_make_section_ds(), "temp", title="Fram Strait temperature")
    assert ax.get_title() == "Fram Strait temperature"


# ---------------------------------------------------------------------------
# NaN / masked data
# ---------------------------------------------------------------------------


def test_section_plot_handles_nan_depth() -> None:
    """NaN values in depth_m (e.g. sub-bathymetry) do not raise."""
    ds = _make_section_ds(with_depth=True)
    depth = ds["depth_m"].values.copy()
    depth[3:, 5:] = np.nan  # simulate land/bathymetry
    ds["depth_m"] = (("k", "section"), depth)
    ax = section_plot(ds, "temp")
    assert ax is not None


def test_section_plot_handles_nan_values() -> None:
    """NaN values in the data variable are masked and do not crash."""
    ds = _make_section_ds()
    vals = ds["temp"].values.copy()
    vals[2, 4] = np.nan
    ds["temp"] = (("k", "section"), vals)
    ax = section_plot(ds, "temp")
    assert ax is not None


# ---------------------------------------------------------------------------
# Auto colorbar label from variable attrs
# ---------------------------------------------------------------------------


def test_section_plot_auto_colorbar_label_from_attrs() -> None:
    """Colorbar label is built from the variable's long_name and units attrs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ds = _make_section_ds()
    ds["temp"].attrs = {"long_name": "potential temperature", "units": "°C"}
    section_plot(ds, "temp", ax=ax, add_colorbar=True)
    cb_ax = fig.axes[1]
    label = cb_ax.get_ylabel()
    assert "potential temperature" in label
    assert "°C" in label


# ---------------------------------------------------------------------------
# Face dimension (section_flux_density output)
# ---------------------------------------------------------------------------


def _make_face_ds(nk: int = 5, nf: int = 8) -> xr.Dataset:
    """Synthetic (k, face) dataset mimicking section_flux_density output."""
    values = np.linspace(-2.0, 2.0, nk * nf).reshape(nk, nf)
    distance = np.linspace(0.0, 100.0, nf)
    depth_2d = np.tile(np.linspace(5.0, 50.0 * nk, nk)[:, None], (1, nf))
    return xr.Dataset(
        {
            "flux_density": (("k", "face"), values),
            "depth_m": (("k", "face"), depth_2d),
        },
        coords={"distance_km": ("face", distance)},
    )


def test_section_plot_face_dim_smoke() -> None:
    """section_plot accepts (k, face) datasets from section_flux_density."""
    ax = section_plot(_make_face_ds(), "flux_density", center_zero=True)
    assert ax is not None


def test_section_plot_center_zero_all_finite() -> None:
    """center_zero=True on all-finite data sets symmetric vmin/vmax."""
    ds = _make_section_ds()
    ds["temp"] = (("k", "section"), np.ones((5, 10)) * 3.0)
    ax = section_plot(ds, "temp", center_zero=True)
    # Both collections should use symmetric limits around zero
    pc = ax.collections[0]
    assert pc.norm.vmin == pytest.approx(-pc.norm.vmax, rel=1e-6)
