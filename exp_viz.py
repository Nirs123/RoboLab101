from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

Config = Literal["normal", "oriented", "full"]

GRID_COLS = 9
GRID_ROWS = 4
SUBDIVISION_FACTOR = 2  # 2 → 18×8 (4 subcells); 4 → 36×16
SIGMA = 0.75  # logical 9×4 units; do not scale with SUBDIVISION_FACTOR
OUTPUT_DIR = Path("outputs/visualisation")
CONFIGS: tuple[Config, ...] = ("normal", "oriented", "full")
CONFIG_GLOBAL_TITLES: dict[Config, str] = {
    "normal": "Normal scores (in-distribution + out-of-distribution)",
    "oriented": "Oriented scores (in-distribution + out-of-distribution)",
    "full": "Full scores (normal + oriented, in-distribution + out-of-distribution)",
}

HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 1.0
HEATMAP_CMAP_NAME = "viridis"
FIGURE_DPI = 150
COLORBAR_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CMAP = plt.get_cmap(HEATMAP_CMAP_NAME).resampled(256)

data = {
    "7500": {
        "in_distribution": {
            "normal": [1, 0.8, 0.8, 0.7, 0.6, 0.5],
            "oriented": [0.2, 0.7, 0.4, 0.3, 0.7, 0.2],
        },
        "out_of_distribution": {
            "normal": [0.9, 0.7, 0.9, 0.7],
            "oriented": [0.8, 0.5, 1, 0.2],
        },
    },
    "15000": {
        "in_distribution": {
            "normal": [1, 0.6, 0.7, 0.8, 0.6, 0.6],
            "oriented": [0.2, 0.8, 0.6, 0.4, 0.8, 0],
        },
        "out_of_distribution": {
            "normal": [0.9, 0.8, 0.9, 0.8],
            "oriented": [0.7, 0.6, 0.3, 0.2],
        },
    },
    "22500": {
        "in_distribution": {
            "normal": [1, 0.8, 0.7, 0.9, 0.7, 0.7],
            "oriented": [0.4, 0.8, 0.6, 0.4, 0.6, 0.3],
        },
        "out_of_distribution": {
            "normal": [0.9, 0.8, 0.8, 0.9],
            "oriented": [1, 0.5, 0.8, 1],
        },
    },
    "30000": {
        "in_distribution": {
            "normal": [1, 0.8, 0.8, 0.9, 0.9, 0.8],
            "oriented": [0.2, 0.9, 0.5, 0.6, 0.8, 0.2],
        },
        "out_of_distribution": {
            "normal": [0.9, 0.8, 0.9, 0.9],
            "oriented": [0.9, 0.6, 0.7, 1],
        },
    },
    "37500": {
        "in_distribution": {
            "normal": [1, 0.9, 0.8, 0.8, 0.9, 0.8],
            "oriented": [0.3, 0.8, 0.5, 0.4, 0.7, 0.5],
        },
        "out_of_distribution": {
            "normal": [0.9, 0.8, 0.9, 1],
            "oriented": [0.8, 0.5, 0.7, 1],
        },
    },
}

viz_positions = {
    "in_distribution": {
        "normal": [(4.5, 0.5), (2.5, 1.5), (2.5, 2.5), (4.5, 2.5), (6.5, 2.5), (6.5, 1.5)],
        "oriented": [(4.5, 1), (2, 1), (2, 3), (4.5, 3), (6.5, 3), (6.5, 1)],
    },
    "out_of_distribution": {
        "normal": [(3, 1), (3, 3), (6, 3), (6, 1)],
        "oriented": [(3, 1), (3, 3), (6, 3), (6, 1)],
    },
}

_CONFIG_BRANCHES: dict[Config, list[tuple[str, str]]] = {
    "normal": [
        ("in_distribution", "normal"),
        ("out_of_distribution", "normal"),
    ],
    "oriented": [
        ("in_distribution", "oriented"),
        ("out_of_distribution", "oriented"),
    ],
    "full": [
        ("in_distribution", "normal"),
        ("out_of_distribution", "normal"),
        ("in_distribution", "oriented"),
        ("out_of_distribution", "oriented"),
    ],
}


def sample_order() -> list[str]:
    return sorted(data.keys(), key=int)


def fine_grid_shape() -> tuple[int, int]:
    return GRID_ROWS * SUBDIVISION_FACTOR, GRID_COLS * SUBDIVISION_FACTOR


def fine_cell_centers() -> tuple[np.ndarray, np.ndarray]:
    fine_rows, fine_cols = fine_grid_shape()
    cols = (np.arange(fine_cols) + 0.5) / SUBDIVISION_FACTOR
    rows = (np.arange(fine_rows) + 0.5) / SUBDIVISION_FACTOR
    return np.meshgrid(cols, rows)


def collect_scores_and_positions(
    sample_dict: dict,
    config: Config,
) -> tuple[list[float], list[tuple[float, float]]]:
    scores: list[float] = []
    positions: list[tuple[float, float]] = []
    for dist, kind in _CONFIG_BRANCHES[config]:
        scores.extend(sample_dict[dist][kind])
        positions.extend(viz_positions[dist][kind])
    return scores, positions


def build_heatmap(
    scores: list[float],
    positions: list[tuple[float, float]],
) -> np.ndarray:
    col_grid, row_grid = fine_cell_centers()
    fine_rows, fine_cols = fine_grid_shape()

    value_sum = np.zeros((fine_rows, fine_cols), dtype=np.float64)
    weight_sum = np.zeros((fine_rows, fine_cols), dtype=np.float64)

    for score, (x, y) in zip(scores, positions, strict=True):
        dist_sq = (col_grid - x) ** 2 + (row_grid - y) ** 2
        weights = np.exp(-dist_sq / (2 * SIGMA**2))
        value_sum += score * weights
        weight_sum += weights

    heatmap = np.divide(
        value_sum,
        weight_sum,
        out=np.zeros_like(value_sum),
        where=weight_sum > 0,
    )
    return np.clip(heatmap, 0.0, 1.0)


def score_to_rgba(grid: np.ndarray) -> np.ndarray:
    values = np.clip(grid, HEATMAP_VMIN, HEATMAP_VMAX)
    return CMAP(values)


def score_to_rgb_uint8(grid: np.ndarray) -> np.ndarray:
    rgba = score_to_rgba(grid)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _add_logical_grid(ax: plt.Axes) -> None:
    for x in range(1, GRID_COLS):
        ax.axvline(x, color="white", linewidth=0.35, alpha=0.35)
    for y in range(1, GRID_ROWS):
        ax.axhline(y, color="white", linewidth=0.35, alpha=0.35)


def _heatmap_imshow(ax: plt.Axes, grid: np.ndarray):
    return ax.imshow(
        score_to_rgba(grid),
        origin="lower",
        extent=[0, GRID_COLS, 0, GRID_ROWS],
        aspect="auto",
        interpolation="nearest",
    )


def _draw_heatmap_on_ax(ax: plt.Axes, grid: np.ndarray) -> None:
    _heatmap_imshow(ax, grid)
    _add_logical_grid(ax)


def _add_heatmap_colorbar(fig: plt.Figure, ax: plt.Axes) -> None:
    mappable = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX))
    mappable.set_array([])
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        ticks=COLORBAR_TICKS,
    )
    colorbar.set_ticklabels([f"{tick:g}" for tick in COLORBAR_TICKS])
    colorbar.set_label("score")


def _add_shared_colorbar(
    fig: plt.Figure,
    *,
    left: float = 0.91,
    width: float = 0.02,
    bottom: float = 0.06,
    top: float = 0.94,
) -> None:
    """Dedicated colorbar axes on the right, outside all subplots."""
    mappable = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX))
    mappable.set_array([])
    cax = fig.add_axes([left, bottom, width, top - bottom])
    colorbar = fig.colorbar(mappable, cax=cax, ticks=COLORBAR_TICKS)
    colorbar.set_ticklabels([f"{tick:g}" for tick in COLORBAR_TICKS])
    colorbar.set_label("score")


def _build_figure(grid: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    _draw_heatmap_on_ax(ax, grid)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    _add_heatmap_colorbar(fig, ax)
    fig.tight_layout()
    return fig


def _figure_to_rgb(fig: plt.Figure) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=FIGURE_DPI,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _gif_palette_reference() -> Image.Image:
    """P-mode palette: 255 viridis score colors + white for figure margins."""
    ramp = np.linspace(HEATMAP_VMIN, HEATMAP_VMAX, 255, dtype=np.float64)
    score_colors = (CMAP(ramp)[:, :3] * 255).astype(np.uint8)
    white = np.array([[255, 255, 255]], dtype=np.uint8)
    colors = np.vstack([score_colors, white])
    strip = np.repeat(colors[np.newaxis, :, :], 4, axis=0)
    return Image.fromarray(strip, mode="RGB").quantize(colors=256)


def _quantize_frame(frame_rgb: Image.Image, palette_ref: Image.Image) -> Image.Image:
    return frame_rgb.quantize(palette=palette_ref, dither=Image.Dither.NONE)


def _save_gif(frames_rgb: list[Image.Image], path: Path, duration_ms: int = 1000) -> None:
    palette_ref = _gif_palette_reference()
    frames_p = [_quantize_frame(frame, palette_ref) for frame in frames_rgb]
    frames_p[0].save(
        path,
        save_all=True,
        append_images=frames_p[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def plot_heatmap(grid: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = _build_figure(grid, title)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def generate_static_visualisations() -> None:
    for sample_key in sample_order():
        sample_dict = data[sample_key]
        out_dir = OUTPUT_DIR / sample_key
        for config in CONFIGS:
            scores, positions = collect_scores_and_positions(sample_dict, config)
            grid = build_heatmap(scores, positions)
            plot_heatmap(grid, f"{sample_key} — {config}", out_dir / f"{config}.png")


def generate_global_animations() -> None:
    global_dir = OUTPUT_DIR / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_order()

    for config in CONFIGS:
        frames_rgb: list[Image.Image] = []
        for sample_key in samples:
            grid = build_heatmap(*collect_scores_and_positions(data[sample_key], config))
            fig = _build_figure(grid, f"samples = {sample_key} — {config}")
            frames_rgb.append(_figure_to_rgb(fig))
        _save_gif(frames_rgb, global_dir / f"{config}_evolution.gif")


def generate_global_subplots() -> None:
    global_dir = OUTPUT_DIR / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_order()

    for config in CONFIGS:
        fig, axes = plt.subplots(
            len(samples),
            1,
            figsize=(9, 4 * len(samples)),
            squeeze=False,
        )
        fig.patch.set_facecolor("white")
        for ax, sample_key in zip(axes.flat, samples, strict=True):
            grid = build_heatmap(*collect_scores_and_positions(data[sample_key], config))
            _draw_heatmap_on_ax(ax, grid)
            ax.set_title(f"{int(sample_key):,} training samples")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.suptitle(CONFIG_GLOBAL_TITLES[config], fontsize=12, y=0.998)
        fig.tight_layout(rect=[0, 0, 0.86, 0.97])
        _add_shared_colorbar(fig)
        fig.savefig(
            global_dir / f"{config}_all_samples.png",
            dpi=FIGURE_DPI,
            facecolor="white",
        )
        plt.close(fig)


def main() -> None:
    generate_static_visualisations()
    generate_global_animations()
    generate_global_subplots()


if __name__ == "__main__":
    main()
