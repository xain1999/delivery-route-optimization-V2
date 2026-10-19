"""Figures for routes and instances (matplotlib, print-ready PDF)."""
from __future__ import annotations

import re

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .instances import Instance

# Ink and route colors.  Three validated categorical hues plus a
# neutral ink, routes are also told apart by line style and marker, so
# identity never depends on color alone (and survives gray printing).
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
ROUTE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#3d3c39"]
ROUTE_STYLES = ["-", "--", "-.", ":"]
ROUTE_MARKERS = ["o", "s", "^", "D"]

_PAGELLA = Path("/usr/share/texmf/fonts/opentype/public/tex-gyre")


_PGF_PREAMBLE = "\n".join([
    r"\usepackage{fontspec}",
    r"\setmainfont{TeX Gyre Pagella}",
    r"\usepackage{unicode-math}",
])


def use_book_style() -> None:
    """Match the book's typeface where available, else a generic serif."""
    family = "serif"
    if _PAGELLA.exists():
        for f in _PAGELLA.glob("texgyrepagella-*.otf"):
            font_manager.fontManager.addfont(str(f))
        family = "TeX Gyre Pagella"
    plt.rcParams.update({
        "pgf.texsystem": "xelatex", "pgf.rcfonts": False,
        "pgf.preamble": _PGF_PREAMBLE,
        "font.family": family, "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, path: str | Path) -> None:
    """Save a figure as PDF.

    If XeLaTeX is installed, text is typeset by LaTeX itself (pgf
    backend), so figure labels use exactly the book's fonts.  Otherwise
    matplotlib's own PDF backend is used.
    """
    import shutil
    if shutil.which("xelatex"):
        try:
            fig.savefig(path, backend="pgf")
            return
        except Exception:  # fall back silently, the figure is still usable
            pass
    fig.savefig(path)


def to_miles(inst: Instance) -> tuple[list[float], list[float]]:
    """Project coordinates to miles east/north of the depot."""
    d = inst.sites[0]
    kx = 69.172 * math.cos(math.radians(d.lat))   # miles per degree of longitude
    ky = 68.707                                   # miles per degree of latitude
    xs = [(s.lon - d.lon) * kx for s in inst.sites]
    ys = [(s.lat - d.lat) * ky for s in inst.sites]
    return xs, ys


def _axes(ax) -> None:
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("miles east of depot")
    ax.set_ylabel("miles north of depot")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _depot(ax, x, y) -> None:
    ax.scatter([x], [y], marker="s", s=70, color=INK, zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.annotate("Depot", (x, y), xytext=(6, -10), textcoords="offset points",
                fontsize=8, color=INK, fontweight="bold")


def plot_instance(inst: Instance, path: str | Path, label_ids: bool = True) -> None:
    use_book_style()
    xs, ys = to_miles(inst)
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    sizes = [12 + 4 * q for q in inst.demands[1:]]
    ax.scatter(xs[1:], ys[1:], s=sizes, color=ROUTE_COLORS[0], alpha=0.85,
               edgecolor="white", linewidth=0.8, zorder=4)
    # size key: three reference order sizes
    qs = sorted(set(inst.demands[1:]))
    for q in (qs[0], qs[len(qs) // 2], qs[-1]):
        ax.scatter([], [], s=12 + 4 * q, color=ROUTE_COLORS[0], alpha=0.85,
                   edgecolor="white", label=f"{q} totes")
    ax.legend(title="Order size", loc="center left", bbox_to_anchor=(1.04, 0.5),
              frameon=False, labelspacing=1.2, borderpad=0.8)
    if label_ids:
        for i in range(1, len(xs)):
            ax.annotate(str(i), (xs[i], ys[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=INK_2)
    _depot(ax, xs[0], ys[0])
    _axes(ax)
    save(fig, path)
    plt.close(fig)


def plot_routes(inst: Instance, routes: list[list[int]], path: str | Path,
                title: str | None = None, ax=None, legend: bool = True,
                route_mi: list[float] | None = None, label_ids: bool = False,
                legend_cols: int = 4) -> None:
    own = ax is None
    if own:
        use_book_style()
        fig, ax = plt.subplots(figsize=(5.2, 5.4))
    xs, ys = to_miles(inst)
    used = [r for r in routes if len(r) > 2]
    for k, r in enumerate(used):
        c, ls, mk = ROUTE_COLORS[k % 4], ROUTE_STYLES[k % 4], ROUTE_MARKERS[k % 4]
        label = f"Van {k + 1}"
        if route_mi is not None:
            label += f" ({route_mi[k]:.1f} mi)"
        ax.plot([xs[i] for i in r], [ys[i] for i in r], ls, color=c,
                linewidth=1.4, zorder=3, label=label)
        ax.scatter([xs[i] for i in r[1:-1]], [ys[i] for i in r[1:-1]],
                   marker=mk, s=26, color=c, edgecolor="white",
                   linewidth=0.8, zorder=4)
    if label_ids:
        for r in used:
            for i in r[1:-1]:
                ax.annotate(str(i), (xs[i], ys[i]), xytext=(3, 3),
                            textcoords="offset points", fontsize=6.5,
                            color=INK_2, zorder=6)
    _depot(ax, xs[0], ys[0])
    _axes(ax)
    if title:
        ax.set_title(title, color=INK, loc="left")
    if legend:
        if legend_cols == 1:   # tall city: legend to the right of the map
            ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5),
                      frameon=False, handlelength=2.6, labelspacing=1.0)
        else:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                      ncol=legend_cols, frameon=False, handlelength=2.6,
                      columnspacing=1.2)
    if own:
        save(fig, path)
        plt.close(fig)


_MINUS = re.compile(r"(?<![\w.\-$\\])-(?=\d)")


def write_tex(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write a generated table or results file for the book, typesetting
    negative numbers with a true minus sign instead of a hyphen."""
    Path(path).write_text(_MINUS.sub(r"\\ensuremath{-}", text), encoding=encoding)
