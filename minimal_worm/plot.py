'''
Created on 13 Jun 2023

@author: amoghasiddhi
'''

# Build-in imports
from typing import List, Tuple, Union

# Thid party imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.colors as colors

# Local imports
# from simple_worm.controls.control_sequence import ControlSequenceNumpy
# from simple_worm.frame.frame_sequence import FrameSequenceNumpy
# from simple_worm.plot3d import MidpointNormalize

# ------------------------------------------------------------------------------
#

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work either side from a prescribed midpoint value
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100)
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def plot_scalar_field(
    ax: Axes,
    M: np.ndarray,
    eps: float = 1e-3,
    v_lim: Tuple[float, float] = None,
    title: List[str] = None,
    T: float = None,
    cmap: str = None,
    cbar_format: str = "%.3f",
):
    """
    Plots a colormap representation of the 2D scalar field M onto the given axes.
    """
    v_min = M.flatten().min()
    v_max = M.flatten().max()

    if np.abs(v_min) < eps:
        np.sign(v_min) * eps
    if np.abs(v_max) < eps:
        np.sign(v_max) * eps

    if v_lim is not None:
        if v_min < v_lim[0]:
            v_min = v_lim[0]
        if v_max > v_lim[1]:
            v_max = v_max[0]

    m = ax.matshow(
        M.T,
        cmap=cmap,
        clim=(v_min, v_max),
        norm=MidpointNormalize(midpoint=0, vmin=v_min, vmax=v_max),
        aspect="auto",
        origin="upper",
    )

    if title is not None:
        ax.set_title(title)

    ax.text(
        -0.02,
        1,
        "H",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontweight="bold",
    )
    ax.text(
        -0.02,
        0,
        "T",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontweight="bold",
    )

    if T is not None:

        ax.text(
            0,
            -0.01,
            "0",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontweight="bold",
        )
        ax.text(
            1,
            -0.01,
            f"{T:.2f}s",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontweight="bold",
        )

    plt.gcf().colorbar(m, ax=ax, format=cbar_format)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    return


def plot_multiple_scalar_fields(
    scalar_fields: list,
    eps: float = 1e-3,
    T: float = None,
    titles: str = None,
    cmaps: bool = None,
    cbar_formats: bool = None,
    fig_size=None,
    grid_layout=None,
):
    """
    Plots a colormap representation for every 2D scalar field in scalar_fields
    in one figure.
    """

    # Number of fields
    M = len(scalar_fields)

    if fig_size is None:
        fig_size = (6 * M, 6)
    if grid_layout is None:
        grid_layout = (1, M)

    fig = plt.figure(figsize=fig_size)
    gs = plt.GridSpec(*grid_layout)
    axes = [plt.subplot(p) for p in gs]

    if titles is None:
        titles = M * [""]
    if cmaps is None:
        cmaps = M * [plt.cm.BrBG]
    if cbar_formats is None:
        cbar_formats = M * ["%.3f"]

    for i, M in enumerate(scalar_fields):

        plot_scalar_field(
            axes[i], M, eps, None, titles[i], T, cmaps[i], cbar_formats[i]
        )

    return fig


def crop_sequence(S, T_min, T_max, times):
    """
    Crop time dimension of given sequence.
    """

    n = len(S)

    idx_arr = np.ones(n, dtype=bool)

    if T_min is not None:
        idx_arr = np.logical_and(idx_arr, times >= T_min)
    if T_max is not None:
        idx_arr = np.logical_and(idx_arr, times <= T_max)

    if not np.all(idx_arr):
        Omega = S.Omega[idx_arr, :, :]
        sigma = S.sigma[idx_arr, :, :]
    else:
        Omega = S.Omega
        sigma = S.sigma

    Omega_list = []
    sigma_list = []

    for i in range(3):
        Omega_list.append(Omega[:, i, :])
        sigma_list.append(sigma[:, i, :])

    return Omega_list + sigma_list


# # ------------------------------------------------------------------------------
# # Wrapper functions to plot chemograms from ControlSequenceNumpy FrameSequenceNumpy objects
#
# DEFAULT_CMAPS = [
#     plt.cm.seismic,
#     plt.cm.seismic,
#     plt.cm.PRGn,
#     plt.cm.BrBG,
#     plt.cm.BrBG,
#     plt.cm.PuOr,
# ]
#
# FS_titles = [
#     r"$\kappa_1$",
#     r"$\kappa_2$",
#     r"$\kappa_3$",
#     r"$\sigma_1$",
#     r"$\sigma_2$",
#     r"$\sigma_3$",
# ]
#
# CS_titles = [
#     r"$\kappa^0_1$",
#     r"$\kappa^0_2$",
#     r"$\kappa^0_3$",
#     r"$\sigma^0_1$",
#     r"$\sigma^0_2$",
#     r"$\sigma^0_3$",
# ]
#
#
# def plot_S(
#     S: Union[FrameSequence, dict],
#     eps: float = 1e-3,
#     T_min: float = None,
#     T_max: float = None,
#     T: float = None,
#     cmaps: List = None,
#     cbar_formats: List[str] = None,
#     dt: float = None,
# ) -> Figure:
#     """
#     Plot strain fields associated with the given sequence.
#     """
#
#     if type(S) == FrameSequenceNumpy:
#         times = S.times
#         titles = FS_titles
#     else:
#         assert dt is not None
#         times = np.array([n * dt for n in range(1, len(S) + 1)])
#         titles = CS_titles
#
#     fields = crop_sequence(S, T_min, T_max, times)
#
#     if cmaps is None:
#         cmaps = DEFAULT_CMAPS
#
#     plot_multiple_scalar_fields(
#         fields,
#         eps=eps,
#         T=T,
#         titles=titles,
#         cmaps=cmaps,
#         cbar_formats=cbar_formats,
#         fig_size=(3 * 6, 3 * 6),
#         grid_layout=(2, 3),
#     )
#
#     return
#
#
# def plot_CS_vs_FS(
#     CS: Union[FrameSequenceNumpy, ControlSequenceNumpy],
#     FS: Union[FrameSequenceNumpy, ControlSequenceNumpy],
#     T_min: float = None,
#     T_max: float = None,
#     T: float = None,
#     eps: float = 1e-3,
#     cmaps: List = None,
#     cbar_formats: bool = None,
# ) -> Figure:
#     """
#     Plots strain fields of given control and frame sequence for comparison
#     """
#     CS_strains = crop_sequence(CS, T_min, T_max, FS.times)
#     FS_strains = crop_sequence(FS, T_min, T_max, FS.times)
#
#     fields = CS_strains + FS_strains
#
#     titles = CS_titles + FS_titles
#
#     if cmaps is None:
#         cmaps = 2 * DEFAULT_CMAPS
#
#     plot_multiple_scalar_fields(
#         fields,
#         eps=eps,
#         T=T,
#         titles=titles,
#         cmaps=cmaps,
#         cbar_formats=cbar_formats,
#         fig_size=(6 * 6, 2 * 6),
#         grid_layout=(2, 6),
#     )
#
#     return

