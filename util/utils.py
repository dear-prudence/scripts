import numpy as np
import matplotlib.pyplot as plt


def add_ticks(ax, extent, del_xTick, del_yTick=None,
              x_label=r'x-coordinate $[$kpc$]$', y_label=r'y-coordinate $[$kpc$]$'):
    if del_yTick is None:
        del_yTick = del_xTick

    font_prop = formatting()

    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect(abs(extent[1] - extent[0]) / abs(extent[3] - extent[2]))

    x_ticksMajor = np.arange(extent[0], extent[1] + 1e-10, del_xTick[0])  # major
    x_ticksMinor = np.arange(extent[0], extent[1], del_xTick[1])  # minor
    y_ticksMajor = np.arange(extent[2], extent[3] + 1e-10, del_yTick[0])
    y_ticksMinor = np.arange(extent[2], extent[3], del_yTick[1])

    x_ticks = x_ticksMajor[(x_ticksMajor >= extent[0]) & (x_ticksMajor <= extent[1])]
    y_ticks = y_ticksMajor[(y_ticksMajor >= extent[2]) & (y_ticksMajor <= extent[3])]
    ax.set_xticks(x_ticks, [str(round(x, 2)).replace("-", '-') for x in x_ticks],
                  fontproperties=font_prop)
    ax.set_xticks(x_ticksMinor[(x_ticksMinor >= extent[0]) & (x_ticksMinor <= extent[1])], minor=True)
    ax.set_yticks(y_ticks, [str(round(y, 2)).replace("-", '-') for y in y_ticks],
                  fontproperties=font_prop)
    ax.set_yticks(y_ticksMinor[(y_ticksMinor >= extent[2]) & (y_ticksMinor <= extent[3])], minor=True)
    ax.set_xlabel(x_label, fontproperties=font_prop)
    ax.set_ylabel(y_label, fontproperties=font_prop)


def add_colorbar(fig, img, label, clip=None, ax_adjust=None):
    font_prop = formatting()
    # [left, bottom, width, height] in figure coords
    cax = fig.add_axes([0.92, 0.11, 0.030, 0.77] if ax_adjust is None else ax_adjust)
    cbar = fig.colorbar(img, cax=cax)
    cbar_values, cbar_labels = [], []
    for t in cbar.ax.get_yticklabels():
        if clip is not None and (t.get_position()[1] < clip[0] or t.get_position()[1] > clip[1]):
            pass
        else:
            cbar_values.append(t.get_position()[1])
            cbar_labels.append(t.get_text())
    cbar.set_ticks([float(t) for t in cbar_values])
    cbar.set_ticklabels(cbar_labels, fontproperties=font_prop)
    cbar.ax.set_ylabel(label, fontproperties=font_prop)


def createFig(dark_mode=False):
    _ = formatting()

    if dark_mode:
        plt.rcParams.update({
            'xtick.color': 'white',
            'ytick.color': 'white',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'text.color': 'white',
        })
    fig, ax = plt.subplots(figsize=(5, 5))

    if dark_mode:
        fig.set_facecolor((33 / 255, 33 / 255, 33 / 255))

    return fig, ax


def formatting():
    import matplotlib.font_manager as fm
    # plt.style.use('dark_background')
    font_prop = fm.FontProperties(fname='/Users/ursa/dear-prudence/scripts/util/fonts/AVHersheySimplexMedium.otf',
                                  size=12)

    # plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams.update({  # "grid.linestyle": "--",  # Dashed grid lines
        'axes.unicode_minus': False,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        'mathtext.fontset': 'cm',
        # 'text.usetex': True,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        'xtick.major.size': 6,
        'xtick.major.width': 0.8,
        'xtick.minor.size': 3,
        'xtick.minor.width': 0.8,
        'ytick.major.size': 7,
        'ytick.major.width': 0.8,
        'ytick.minor.size': 4,
        'ytick.minor.width': 0.8,
        'legend.frameon': False,
    })
    return font_prop
