# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-18 19:38:37
# @Last Modified: 2022-04-12 11:24:06
# ------------------------------------------------------------------------------ #
# Helper functions for dealing with colors
# ------------------------------------------------------------------------------ #

from matplotlib.colors import LinearSegmentedColormap as _ls
from matplotlib.colors import to_hex, to_rgb, to_rgba, Normalize
from matplotlib.patches import Rectangle as _Rectangle
from matplotlib.colorbar import ColorbarBase as _ColorbarBase
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator
import numpy as np

palettes = dict()
# good with edge = False
palettes["cold"] = [
    (0, "0.92"),  # single float as a string for grayscale
    (0.25, "#BEDA9D"),
    (0.45, "#42B3D5"),
    (0.75, "#24295E"),
    (1, "black"),
]
palettes["hot"] = [
    (0, "0.95"),
    (0.3, "#FEEB65"),
    (0.65, "#E4521B"),
    (0.85, "#4D342F"),
    (1, "black"),
]
palettes["pinks"] = [
    (0, "0.95"),
    (0.2, "#E0CB8F"),
    # (0.2, "#FFECB3"),
    (0.45, "#E85285"),
    (0.65, "#6A1B9A"),
    (1, "black"),
]

# good with edge = True
palettes["volcano"] = [
    (0, "#E8E89C"),
    (0.25, "#D29C65"),
    (0.65, "#922C40"),
    (1, "#06102E"),
]

palettes["pastel_1"] = [
    (0, "#E7E7B6"),
    (0.25, "#ffad7e"),
    (0.5, "#cd6772"),
    (0.75, "#195571"),
    (1, "#011A39"),
]

palettes["reds"] = [
    (0, "#d26761"),
    (0.25, "#933a3e"),
    (0.5, "#6b354c"),
    (0.75, "#411c2f"),
    (1, "#050412"),
]
palettes["blues"] = [
    (0, "#bbc2d2"),
    (0.25, "#2865a6"),
    (0.5, "#11395d"),
    (0.75, "#091d35"),
    (1, "#030200"),
]

# hm not super color blind friendly
palettes["bambus"] = [
    (0, "#D9DFD3"),
    (0.25, "#8FA96D"),
    (0.5, "#9C794F"),
    (1, "#3F2301"),
]

palettes["div_red_yellow_blue"] = [
    (0.0, "#f94144"),
    (0.2, "#f3722c"),
    (0.3, "#f8961e"),
    (0.5, "#f9c74f"),
    (0.7, "#90be6d"),
    (0.8, "#43aa8b"),
    (1.0, "#577590"),
]

palettes["div_red_white_blue"] = [
    (0.0, "#233954"),
    (0.25, "#8d99ae"),
    (0.5, "#edf2f4"),
    (0.75, "#ef233c"),
    (1.0, "#d90429"),
]

palettes["div_pastel_1"] = [
    (0, "#C31B2B"),
    (0.25, "#ffad7e"),
    (0.5, "#E7E7B6"),
    (0.85, "#195571"),
    (1, "#011A39"),
]

palettes["div_pastel_2"] = [
    (0.0, "#641002"),
    (0.25, "#D82C0E"),
    (0.5, "#FFD500"),
    (0.75, "#B8E0FF"),
    (1.0, "#F0FBFF"),
]

palettes["grays"] = [
    (0.0, "#999"),
    (1.0, "#000"),
]


# enable the colormaps for matplotlib cmaps and getting discrete values, eg
# cmap["pinks"](0.5)
cmap = dict()
for key in palettes.keys():
    cmap[key] = _ls.from_list(key, palettes[key], N=512)


def create_cmap(start="white", end="black", palette=None, N=512):
    if palette is None:
        palette = [(0.0, start), (1.0, end)]
    return _ls.from_list("custom_colormap", palette, N=N)


def cmap_cycle(palette="hot", N=5, edge=True, format="hex"):
    if palette not in palettes.keys():
        raise KeyError(f"Unrecognized palette '{palette}'")

    assert N >= 1

    if format.lower() == "hex":
        to_format = to_hex
    elif format.lower() == "rgb":
        to_format = to_rgb
    elif format.lower() == "rgba":
        to_format = to_rgba

    res = []
    for idx in range(0, N):
        if N == 1:
            arg = 0.5
        else:
            if edge:
                arg = idx / (N - 1)
            else:
                arg = (idx + 1) / (N + 1)

        this_clr = to_format(cmap[palette](arg))
        if to_format == to_hex:
            this_clr = this_clr.upper()
        res.append(this_clr)

    return res


def demo_cmap(palette="hot", Nmax=7, edge=True):

    dpi = 72
    cell_width = 120
    cell_height = 28
    swatch_width = 117
    swatch_height = 25
    margin = 12
    topmargin = 40
    cbar_width = 50

    ncols = Nmax
    nrows = Nmax
    width = cell_width * ncols + 2 * margin + cbar_width
    height = cell_height * nrows + margin + topmargin

    fig, axes = plt.subplots(
        ncols=2,
        gridspec_kw={"width_ratios": [width - cbar_width, cbar_width]},
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
    )
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - topmargin) / height,
    )
    ax = axes[0]
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(
        f"palette: {palette}", fontsize=24, fontweight="bold", loc="center", pad=10
    )

    print(f"palette: {palette}")
    for N in range(1, Nmax + 1):
        # columns
        colors = cmap_cycle(palette, N, edge, format="rgba")
        clr_desc = cmap_cycle(palette, N, edge, format="hex")
        print(f"N = {N}: {' '.join(clr_desc)}")
        for n in range(1, N + 1):
            # rows

            description = clr_desc[n - 1]
            swatch_clr = colors[n - 1]
            # get a text color that is gray but brightness invert to shown patch
            r, g, b = to_rgb(swatch_clr)
            gray = 0.2989 * (1 - r) + 0.5870 * (1 - g) + 0.1140 * (1 - b)
            if gray < 0.5:
                gray = "black"
            else:
                gray = "white"

            col = N - 1
            row = Nmax / 2 - n + N / 2

            swatch_start_x = col * cell_width
            swatch_start_y = row * cell_height
            text_pos_x = swatch_start_x + cell_width / 2
            text_pos_y = swatch_start_y + cell_height / 2

            if n == N:
                ax.text(
                    text_pos_x,
                    text_pos_y - cell_height,
                    f"N = {N}",
                    fontsize=14,
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            ax.text(
                text_pos_x,
                text_pos_y,
                description,
                fontsize=14,
                color=str(gray),
                horizontalalignment="center",
                verticalalignment="center",
            )

            ax.add_patch(
                _Rectangle(
                    xy=(swatch_start_x, swatch_start_y),
                    width=swatch_width,
                    height=swatch_height,
                    facecolor=swatch_clr,
                )
            )

    # add the full color bar to the right
    cbax = axes[1]
    cbar = _ColorbarBase(
        ax=cbax,
        cmap=cmap[palette],
        norm=Normalize(vmin=0, vmax=1),
        orientation="vertical",
    )
    cbax.axis("off")
    cbar.outline.set_visible(False)

    fig.tight_layout()

    return fig, axes


# this should go elsewhere
def save_all_figures(path, fmt="pdf", save_pickle=False, **kwargs):
    """
    saves all open figures as pdfs and pickle. to load an existing figure:
    ```
    import pickle
    with open('/path/to/fig.pkl','rb') as fid:
        fig = pickle.load(fid)
    ```
    """
    import os

    path = os.path.expanduser(path)
    assert os.path.isdir(path)

    try:
        import pickle
    except ImportError:
        if pickle:
            log.info("Failed to import pickle")
            save_pickle = False

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(*args):
            return iter(*args)

    if "dpi" not in kwargs:
        kwargs["dpi"] = 300
    if "transparent" not in kwargs:
        kwargs["transparent"] = True

    for i in tqdm(plt.get_fignums()):
        fig = plt.figure(i)
        fig.savefig(f"{path}/figure_{i}.{fmt}", **kwargs)
        if save_pickle:
            try:
                os.makedirs(f"{path}/pickle/", exist_ok=True)
                with open(f"{path}/pickle/figure_{i}.pkl", "wb") as fid:
                    pickle.dump(fig, fid)
            except Exception as e:
                print(e)


def load_fig_from_pickle(path):
    import pickle

    with open("/path/to/fig.pkl", "rb") as fid:
        fig = pickle.load(fid)

    return fig


def alpha_to_solid_on_bg(base, alpha, bg="white"):
    """
    Probide a color to start from `base`, and give it opacity `alpha` on
    the background color `bg`
    """

    def rgba_to_rgb(c, bg):
        bg = matplotlib.colors.to_rgb(bg)
        alpha = c[-1]

        res = (
            (1 - alpha) * bg[0] + alpha * c[0],
            (1 - alpha) * bg[1] + alpha * c[1],
            (1 - alpha) * bg[2] + alpha * c[2],
        )
        return res

    new_base = list(matplotlib.colors.to_rgba(base))
    new_base[3] = alpha
    return matplotlib.colors.to_hex(rgba_to_rgb(new_base, bg))


def fade(k, n, start=1, stop=0.4, invert=False):
    """
    helper to get stepwise lower alphas at same color.
    n = total steps
    k = current step, going from 0 to n-1
    start = maximum obtained value
    stop = minimum obtained value
    """

    if n <= 1:
        return 1

    if invert:
        frac = (k) / (n - 1)
    else:
        frac = (n - 1 - k) / (n - 1)
    alpha = stop + (start - stop) * frac
    return alpha


def set_size(ax, w, h=None):
    """
    set the size of an axis, where the size describes the actual area of the plot,
    _excluding_ the axes, ticks, and labels.

    w, h: width, height in cm

    # Example
    ```
        cm = 2.54
        fig, ax = plt.subplots()
        ax.plot(stuff)
        fig.tight_layout()
        _set_size(ax, 3.5*cm, 4.5*cm)
    ```
    """
    # https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w / 2.54) / (r - l)
    if h is None:
        ax.figure.set_figwidth(figw)
    else:
        figh = float(h / 2.54) / (t - b)
        ax.figure.set_size_inches(figw, figh)


def set_size2(ax, w, h):
    # https://newbedev.com/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    from mpl_toolkits.axes_grid1 import Divider, Size

    axew = w / 2.54
    axeh = h / 2.54

    # lets use the tight layout function to get a good padding size for our axes labels.
    # fig = plt.gcf()
    # ax = plt.gca()
    fig = ax.get_figure()
    fig.tight_layout()
    # obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    # work out what the new  ratio values for padding are, and the new fig size.
    neww = axew + oldw * (1 - r + l)
    newh = axeh + oldh * (1 - t + b)
    newr = r * oldw / neww
    newl = l * oldw / neww
    newt = t * oldh / newh
    newb = b * oldh / newh

    # right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
    vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    # we need to resize the figure now, as we have may have made our axes bigger than in.
    fig.set_size_inches(neww, newh)


def set_size3(ax, w, h):
    # https://newbedev.com/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    from mpl_toolkits.axes_grid1 import Divider, Size

    axew = w / 2.54
    axeh = h / 2.54

    # lets use the tight layout function to get a good padding size for our axes labels.
    # fig = plt.gcf()
    # ax = plt.gca()
    fig = ax.get_figure()
    fig.tight_layout()
    # obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    # work out what the new  ratio values for padding are, and the new fig size.
    # ps: adding a bit to neww and newh gives more padding
    # the axis size is set from axew and axeh
    neww = axew + oldw * (1 - r + l) + 0.4
    newh = axeh + oldh * (1 - t + b) + 0.4
    newr = r * oldw / neww - 0.4
    newl = l * oldw / neww + 0.4
    newt = t * oldh / newh - 0.4
    newb = b * oldh / newh + 0.4

    # right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
    vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    # we need to resize the figure now, as we have may have made our axes bigger than in.
    fig.set_size_inches(neww, newh)


def detick(axis, keep_labels=False, keep_ticks=False):
    """
    ```
    detick(ax.xaxis)
    detick([ax.xaxis, ax.yaxis])
    ```
    """
    if not isinstance(axis, list):
        axis = [axis]
    for a in axis:
        if not keep_labels and not keep_ticks:
            a.set_ticks_position("none")
            a.set_ticks([])
        elif not keep_labels and keep_ticks:
            a.set_ticklabels([])
        elif keep_labels and not keep_ticks:
            raise NotImplementedError


def _style_legend(leg):
    """
    a legend style I use frequently

    # Example
    ```
        fig, ax = plt.subplots()
        leg = ax.legend()
        _style_legend(leg)

    ```
    """
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("#e4e5e6")
    leg.get_frame().set_alpha(0.9)


def _legend_into_new_axes(ax):
    """
    draw a legend into a new figure, e.g. for customizing
    """
    cm = 1 / 2.54
    fig, ax_leg = plt.subplots(figsize=(6 * cm, 6 * cm))
    h, l = ax.get_legend_handles_labels()
    ax_leg.axis("off")
    leg = ax_leg.legend(h, l, loc="upper left")

    return leg


def get_shifted_formatter(shift=-60, fmt=".1f"):
    """
    # Example
    ```
    ax.xaxis.set_major_formatter(get_formatter_shifted(shift=-120))
    ```
    """

    def formatter(x, pos):
        return "{{:{}}}".format(fmt).format(x + shift)

    return formatter


def _ticklabels_lin_to_log10(x, pos):
    """
    converts ticks of manually logged data (lin ticks) to log ticks, as follows
     1 -> 10
     0 -> 1
    -1 -> 0.1

    # Example
    ```
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(_ticklabels_lin_to_log10_power)
    )
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(_ticklocator_lin_to_log_minor())
    ```
    """
    prec = int(np.ceil(-np.minimum(x, 0)))
    return "{{:.{:1d}f}}".format(prec).format(np.power(10.0, x))


def _ticklabels_lin_to_log10_power(x, pos, nicer=True, nice_range=[-1, 0, 1]):
    """
    converts ticks of manually logged data (lin ticks) to log ticks, as follows
     1 -> 10^1
     0 -> 10^0
    -1 -> 10^-1
    """
    if x.is_integer():
        # use easy to read formatter if exponents are close to zero
        if nicer and x in nice_range:
            return _ticklabels_lin_to_log10(x, pos)
        else:
            return r"$10^{{{:d}}}$".format(int(x))
    else:
        # return r"$10^{{{:f}}}$".format(x)
        return ""


def _ticklocator_lin_to_log_minor(vmin=-10, vmax=10, nbins=10):
    """
    get minor ticks on when manually converting lin to log
    """
    locs = []
    orders = int(np.ceil(vmax - vmin))
    for o in range(int(np.floor(vmin)), int(np.floor(vmax + 1)), 1):
        locs.extend([o + np.log10(x) for x in range(2, 10)])
    return matplotlib.ticker.FixedLocator(locs, nbins=nbins * orders)


def _fix_log_ticks(ax_el, every=1, hide_label_condition=lambda idx: False):
    """
    this can adapt log ticks to only show every second tick, or so.

    # Parameters
    ax_el: usually `ax.yaxis`
    every: 1 or 2
    hide_label_condition : function e.g. `lambda idx: idx % 2 == 0`
    """
    ax_el.set_major_locator(LogLocator(base=10, numticks=10))
    ax_el.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(0, 1.05, every / 10), numticks=10)
    )
    ax_el.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for idx, lab in enumerate(ax_el.get_ticklabels()):
        # print(idx, lab, hide_label_condition(idx))
        if hide_label_condition(idx):
            lab.set_visible(False)
