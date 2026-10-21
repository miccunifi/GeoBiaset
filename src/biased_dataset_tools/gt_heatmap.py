from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from pyproj import Transformer


DEFAULT_INPUT_CSV = Path("/mnt/nniccoli/mp16/Annotations/mp16_places365.csv")
DEFAULT_OUTPUT_DIR = Path("figures/gt_heatmaps")
DEFAULT_BOUNDARY_PATH = Path("data/naturalearth_lowres/naturalearth_lowres.shp")
DEFAULT_LAT_COLUMN = "LAT"
DEFAULT_LON_COLUMN = "LON"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 13
DEFAULT_DPI = 600
DEFAULT_BINS_X = 2400
DEFAULT_BINS_Y = 2400
DEFAULT_SIGMA = 4.0
DEFAULT_LAT_MIN = -85.0
DEFAULT_LAT_MAX = 85.0
DEFAULT_LON_MIN = -180.0
DEFAULT_LON_MAX = 180.0
DEFAULT_BATCH_SIZE = 100_000
DEFAULT_DISPLAY_LAT_MIN = -58.0
DEFAULT_DISPLAY_LAT_MAX = 75.0
DEFAULT_FIGSIZE = (8.6, 5.0)
DEFAULT_THEME = "both"
DEFAULT_COLORBAR_PAD = 0.055
DEFAULT_COLORBAR_ORIENTATION = "horizontal"
DEFAULT_INPUT_FORMAT = "auto"
DEFAULT_LAT_INDEX = -2
DEFAULT_LON_INDEX = -1
DEFAULT_DENSITY_CLIP_PERCENTILE = 99.5
DEFAULT_MASK_FRACTION = 0.02
THEME_CHOICES = ("light", "dark", "both")
COLORBAR_ORIENTATION_CHOICES = ("horizontal", "vertical")
INPUT_FORMAT_CHOICES = ("auto", "csv", "whitespace")
THEME_STYLES = {
    "light": {
        "figure_face": "#FFFFFF",
        "axis_face": "#F7FAFC",
        "land_color": "#F3F0E8",
        "boundary_color": "#9EA3A3",
        "graticule_color": "#D7DEE4",
        "tick_color": "#111827",
        "label_color": "#111827",
        "colorbar_edge": "#9CA3AF",
        "heat_alpha": 0.93,
    },
    "dark": {
        "figure_face": "#05080B",
        "axis_face": "#05080B",
        "land_color": "#101A22",
        "boundary_color": "#536879",
        "graticule_color": "#22303B",
        "tick_color": "#E7EDF2",
        "label_color": "#E7EDF2",
        "colorbar_edge": "#B8C2CC",
        "heat_alpha": 0.97,
    },
}


@dataclass
class CoordinateStats:
    total_rows: int = 0
    valid_rows: int = 0
    plotted_rows: int = 0
    duplicate_rows: int = 0
    missing_rows: int = 0
    non_numeric_rows: int = 0
    nonfinite_rows: int = 0
    out_of_bounds_rows: int = 0


@dataclass(frozen=True)
class RenderedHeatmap:
    png: Path
    pdf: Path


@dataclass(frozen=True)
class HeatmapRunResult:
    metadata_path: Path
    preview: dict[str, RenderedHeatmap]
    full: dict[str, RenderedHeatmap] | None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_boundary_path() -> Path:
    return project_root() / DEFAULT_BOUNDARY_PATH


def mercator_transformer() -> Transformer:
    return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def mercator_extent(
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
) -> tuple[float, float, float, float]:
    transformer = mercator_transformer()
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    return float(x_min), float(x_max), float(y_min), float(y_max)


def validate_lat_bounds(lat_min: float, lat_max: float, label: str) -> None:
    if not (math.isfinite(lat_min) and math.isfinite(lat_max)):
        raise ValueError(f"{label} latitude bounds must be finite")
    if lat_min >= lat_max:
        raise ValueError(f"{label} minimum latitude must be less than maximum latitude")
    if lat_min < DEFAULT_LAT_MIN or lat_max > DEFAULT_LAT_MAX:
        raise ValueError(f"{label} latitude bounds must stay within [{DEFAULT_LAT_MIN}, {DEFAULT_LAT_MAX}]")


def validate_columns(fieldnames: Sequence[str] | None, lat_column: str, lon_column: str, csv_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV has no header: {csv_path}")

    missing = [column for column in (lat_column, lon_column) if column not in fieldnames]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"CSV is missing required coordinate column(s): {joined}")


def resolve_input_format(input_path: Path, input_format: str = DEFAULT_INPUT_FORMAT) -> str:
    if input_format not in INPUT_FORMAT_CHOICES:
        raise ValueError(f"Unsupported input format: {input_format}")
    if input_format != "auto":
        return input_format
    if input_path.suffix.lower() in {".txt", ".tsv", ".lst"}:
        return "whitespace"
    return "csv"


def parse_coordinate_values(
    lat_text: str,
    lon_text: str,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    stats: CoordinateStats | None = None,
) -> tuple[float, float] | None:
    lat_text = lat_text.strip()
    lon_text = lon_text.strip()
    if not lat_text or not lon_text:
        if stats is not None:
            stats.missing_rows += 1
        return None

    try:
        lat = float(lat_text)
        lon = float(lon_text)
    except ValueError:
        if stats is not None:
            stats.non_numeric_rows += 1
        return None

    if not (math.isfinite(lat) and math.isfinite(lon)):
        if stats is not None:
            stats.nonfinite_rows += 1
        return None

    if lat < lat_min or lat > lat_max or lon < lon_min or lon > lon_max:
        if stats is not None:
            stats.out_of_bounds_rows += 1
        return None

    if stats is not None:
        stats.valid_rows += 1
    return lat, lon


def parse_coordinate_row(
    row: dict[str, str],
    lat_column: str = DEFAULT_LAT_COLUMN,
    lon_column: str = DEFAULT_LON_COLUMN,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    stats: CoordinateStats | None = None,
) -> tuple[float, float] | None:
    if stats is not None:
        stats.total_rows += 1

    return parse_coordinate_values(
        row.get(lat_column) or "",
        row.get(lon_column) or "",
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        stats=stats,
    )


def parse_coordinate_line(
    line: str,
    lat_index: int = DEFAULT_LAT_INDEX,
    lon_index: int = DEFAULT_LON_INDEX,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    stats: CoordinateStats | None = None,
) -> tuple[float, float] | None:
    if stats is not None:
        stats.total_rows += 1

    tokens = line.strip().split()
    try:
        lat_text = tokens[lat_index]
        lon_text = tokens[lon_index]
    except IndexError:
        if stats is not None:
            stats.missing_rows += 1
        return None

    return parse_coordinate_values(
        lat_text,
        lon_text,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        stats=stats,
    )


def iter_valid_coordinates(
    csv_path: Path,
    lat_column: str = DEFAULT_LAT_COLUMN,
    lon_column: str = DEFAULT_LON_COLUMN,
    input_format: str = DEFAULT_INPUT_FORMAT,
    lat_index: int = DEFAULT_LAT_INDEX,
    lon_index: int = DEFAULT_LON_INDEX,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    stats: CoordinateStats | None = None,
    unique_coordinates: bool = False,
) -> Iterator[tuple[float, float]]:
    resolved_input_format = resolve_input_format(csv_path, input_format)
    seen_coordinates: set[tuple[float, float]] = set()

    def yield_coordinate(coordinate: tuple[float, float] | None) -> Iterator[tuple[float, float]]:
        if coordinate is None:
            return
        if unique_coordinates:
            if coordinate in seen_coordinates:
                if stats is not None:
                    stats.duplicate_rows += 1
                return
            seen_coordinates.add(coordinate)
        if stats is not None:
            stats.plotted_rows += 1
        yield coordinate

    if resolved_input_format == "whitespace":
        with csv_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                coordinate = parse_coordinate_line(
                    line,
                    lat_index=lat_index,
                    lon_index=lon_index,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    stats=stats,
                )
                yield from yield_coordinate(coordinate)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames, lat_column, lon_column, csv_path)
        for row in reader:
            coordinate = parse_coordinate_row(
                row,
                lat_column=lat_column,
                lon_column=lon_column,
                lat_min=lat_min,
                lat_max=lat_max,
                stats=stats,
            )
            yield from yield_coordinate(coordinate)


def iter_valid_coordinate_batches(
    csv_path: Path,
    lat_column: str = DEFAULT_LAT_COLUMN,
    lon_column: str = DEFAULT_LON_COLUMN,
    input_format: str = DEFAULT_INPUT_FORMAT,
    lat_index: int = DEFAULT_LAT_INDEX,
    lon_index: int = DEFAULT_LON_INDEX,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stats: CoordinateStats | None = None,
    unique_coordinates: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    lats: list[float] = []
    lons: list[float] = []
    for lat, lon in iter_valid_coordinates(
        csv_path,
        lat_column=lat_column,
        lon_column=lon_column,
        input_format=input_format,
        lat_index=lat_index,
        lon_index=lon_index,
        lat_min=lat_min,
        lat_max=lat_max,
        stats=stats,
        unique_coordinates=unique_coordinates,
    ):
        lats.append(lat)
        lons.append(lon)
        if len(lats) >= batch_size:
            yield np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)
            lats.clear()
            lons.clear()

    if lats:
        yield np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)


def sample_coordinates(
    csv_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    lat_column: str = DEFAULT_LAT_COLUMN,
    lon_column: str = DEFAULT_LON_COLUMN,
    input_format: str = DEFAULT_INPUT_FORMAT,
    lat_index: int = DEFAULT_LAT_INDEX,
    lon_index: int = DEFAULT_LON_INDEX,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    unique_coordinates: bool = False,
) -> tuple[list[tuple[float, float]], CoordinateStats]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    rng = random.Random(seed)
    stats = CoordinateStats()
    sample: list[tuple[float, float]] = []
    for coordinate in iter_valid_coordinates(
        csv_path,
        lat_column=lat_column,
        lon_column=lon_column,
        input_format=input_format,
        lat_index=lat_index,
        lon_index=lon_index,
        lat_min=lat_min,
        lat_max=lat_max,
        stats=stats,
        unique_coordinates=unique_coordinates,
    ):
        valid_index = stats.plotted_rows
        if len(sample) < sample_size:
            sample.append(coordinate)
            continue

        replacement_index = rng.randrange(valid_index)
        if replacement_index < sample_size:
            sample[replacement_index] = coordinate

    return sample, stats


def empty_histogram(bins_x: int = DEFAULT_BINS_X, bins_y: int = DEFAULT_BINS_Y) -> np.ndarray:
    if bins_x <= 0 or bins_y <= 0:
        raise ValueError("bins_x and bins_y must be positive")
    return np.zeros((bins_y, bins_x), dtype=np.float64)


def add_projected_points_to_histogram(
    histogram: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    extent: tuple[float, float, float, float],
) -> None:
    if lats.size == 0:
        return

    x_min, x_max, y_min, y_max = extent
    transformer = mercator_transformer()
    xs, ys = transformer.transform(lons.tolist(), lats.tolist())
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    finite_mask = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite_mask):
        return

    xs = xs[finite_mask]
    ys = ys[finite_mask]
    bins_y, bins_x = histogram.shape
    x_indices = np.floor((xs - x_min) / (x_max - x_min) * bins_x).astype(np.int64)
    y_indices = np.floor((ys - y_min) / (y_max - y_min) * bins_y).astype(np.int64)
    inside_mask = (x_indices >= 0) & (x_indices < bins_x) & (y_indices >= 0) & (y_indices < bins_y)

    x_indices = np.clip(x_indices[inside_mask], 0, bins_x - 1)
    y_indices = np.clip(y_indices[inside_mask], 0, bins_y - 1)
    np.add.at(histogram, (y_indices, x_indices), 1.0)


def build_histogram_from_coordinates(
    coordinates: Iterable[tuple[float, float]],
    bins_x: int = DEFAULT_BINS_X,
    bins_y: int = DEFAULT_BINS_Y,
    extent: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    resolved_extent = mercator_extent() if extent is None else extent
    histogram = empty_histogram(bins_x=bins_x, bins_y=bins_y)
    coords = list(coordinates)
    if coords:
        lats = np.asarray([lat for lat, _lon in coords], dtype=np.float64)
        lons = np.asarray([lon for _lat, lon in coords], dtype=np.float64)
        add_projected_points_to_histogram(histogram, lats, lons, resolved_extent)
    return histogram, resolved_extent


def build_histogram_from_csv(
    csv_path: Path,
    bins_x: int = DEFAULT_BINS_X,
    bins_y: int = DEFAULT_BINS_Y,
    lat_column: str = DEFAULT_LAT_COLUMN,
    lon_column: str = DEFAULT_LON_COLUMN,
    input_format: str = DEFAULT_INPUT_FORMAT,
    lat_index: int = DEFAULT_LAT_INDEX,
    lon_index: int = DEFAULT_LON_INDEX,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    batch_size: int = DEFAULT_BATCH_SIZE,
    unique_coordinates: bool = False,
) -> tuple[np.ndarray, CoordinateStats, tuple[float, float, float, float]]:
    stats = CoordinateStats()
    extent = mercator_extent(lat_min=lat_min, lat_max=lat_max)
    histogram = empty_histogram(bins_x=bins_x, bins_y=bins_y)
    for lats, lons in iter_valid_coordinate_batches(
        csv_path,
        lat_column=lat_column,
        lon_column=lon_column,
        input_format=input_format,
        lat_index=lat_index,
        lon_index=lon_index,
        lat_min=lat_min,
        lat_max=lat_max,
        batch_size=batch_size,
        stats=stats,
        unique_coordinates=unique_coordinates,
    ):
        add_projected_points_to_histogram(histogram, lats, lons, extent)
    return histogram, stats, extent


def smoothed_log_density(
    histogram: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    density = histogram.astype(np.float32, copy=False)
    if sigma > 0:
        density = gaussian_filter(density, sigma=sigma, mode="nearest")

    return np.log1p(density)


def compute_density_vmax(
    histogram: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
    clip_percentile: float = DEFAULT_DENSITY_CLIP_PERCENTILE,
) -> float:
    density = smoothed_log_density(histogram, sigma=sigma)
    positive_values = density[density > 0]
    if not positive_values.size:
        return 0.0
    return float(np.percentile(positive_values, clip_percentile))


def prepare_density(
    histogram: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
    clip_percentile: float = DEFAULT_DENSITY_CLIP_PERCENTILE,
    density_vmax: float | None = None,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> np.ma.MaskedArray:
    density = smoothed_log_density(histogram, sigma=sigma)
    if density_vmax is None:
        positive_values = density[density > 0]
        clip_value = float(np.percentile(positive_values, clip_percentile)) if positive_values.size else 0.0
    else:
        clip_value = density_vmax

    if clip_value > 0:
        density = np.clip(density, 0.0, clip_value)
        mask_threshold = max(clip_value * mask_fraction, 0.0)
    else:
        mask_threshold = 0.0

    return np.ma.masked_where(density <= mask_threshold, density)


def heatmap_colormap(theme: str = "light") -> object:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    ramp = np.linspace(0.0, 1.0, 256)
    if theme == "dark":
        base = LinearSegmentedColormap.from_list(
            "paper_dark_truncated_viridis_base",
            [
                "#A78BFA",
                "#8B5CF6",
                "#5B4BC4",
                "#31688E",
                "#26828E",
                "#1F9E89",
                "#35B779",
                "#6DCD59",
                "#B4DE2C",
                "#FDE725",
            ],
            N=256,
        )
        colors = base(ramp)
        colors[:, -1] = 0.52 + 0.46 * (ramp**0.68)
    else:
        base = plt.get_cmap("magma")
        colors = base(np.linspace(0.18, 1.0, 256))
        colors[:, -1] = ramp**0.55 * 0.94
    colormap = ListedColormap(colors, name=f"paper_{theme}_heatmap")
    colormap.set_bad(alpha=0.0)
    return colormap


def draw_boundaries(ax: object, boundary_path: Path | None, theme: str = "light") -> str | None:
    if boundary_path is None:
        return None
    if not boundary_path.exists():
        return f"Boundary file not found: {boundary_path}"

    style = THEME_STYLES[theme]
    try:
        import geopandas as gpd

        world = gpd.read_file(boundary_path).to_crs("EPSG:3857")
        world.plot(
            ax=ax,
            color=style["land_color"],
            edgecolor=style["boundary_color"],
            linewidth=0.28,
            zorder=1,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback for local GIS stack issues.
        return f"Could not draw boundaries from {boundary_path}: {exc}"

    return None


def draw_graticule(
    ax: object,
    lat_min: float,
    lat_max: float,
    theme: str = "light",
    lon_step: int = 30,
    lat_step: int = 20,
) -> None:
    style = THEME_STYLES[theme]
    transformer = mercator_transformer()
    line_style = {
        "color": style["graticule_color"],
        "linewidth": 0.26,
        "alpha": 0.55 if theme == "light" else 0.42,
        "zorder": 2,
    }

    lats = np.linspace(lat_min, lat_max, 240)
    for lon in range(-150, 181, lon_step):
        lons = np.full_like(lats, float(lon))
        xs, ys = transformer.transform(lons.tolist(), lats.tolist())
        ax.plot(xs, ys, **line_style)

    lons = np.linspace(DEFAULT_LON_MIN, DEFAULT_LON_MAX, 360)
    first_parallel = int(math.ceil(lat_min / lat_step) * lat_step)
    last_parallel = int(math.floor(lat_max / lat_step) * lat_step)
    for lat in range(first_parallel, last_parallel + 1, lat_step):
        if lat == 0:
            linewidth = 0.34
            alpha = 0.65 if theme == "light" else 0.5
        else:
            linewidth = line_style["linewidth"]
            alpha = line_style["alpha"]
        lats = np.full_like(lons, float(lat))
        xs, ys = transformer.transform(lons.tolist(), lats.tolist())
        ax.plot(xs, ys, color=style["graticule_color"], linewidth=linewidth, alpha=alpha, zorder=2)


def style_colorbar(colorbar: object, theme: str) -> None:
    from matplotlib.ticker import MaxNLocator

    style = THEME_STYLES[theme]
    colorbar.set_label("log density", fontsize=6.5, color=style["label_color"], labelpad=2)
    colorbar.ax.tick_params(labelsize=5.5, length=1.8, colors=style["tick_color"], width=0.45)
    colorbar.outline.set_edgecolor(style["colorbar_edge"])
    colorbar.outline.set_linewidth(0.45)
    colorbar.locator = MaxNLocator(nbins=4)
    colorbar.update_ticks()


def render_heatmap(
    histogram: np.ndarray,
    extent: tuple[float, float, float, float],
    output_base: Path,
    dpi: int = DEFAULT_DPI,
    sigma: float = DEFAULT_SIGMA,
    boundary_path: Path | None = None,
    include_colorbar: bool = True,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    theme: str = "light",
    colorbar_pad: float = DEFAULT_COLORBAR_PAD,
    colorbar_orientation: str = DEFAULT_COLORBAR_ORIENTATION,
    display_lat_min: float = DEFAULT_DISPLAY_LAT_MIN,
    display_lat_max: float = DEFAULT_DISPLAY_LAT_MAX,
    density_vmax: float | None = None,
    density_clip_percentile: float = DEFAULT_DENSITY_CLIP_PERCENTILE,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> RenderedHeatmap:
    import matplotlib.pyplot as plt

    if theme not in THEME_STYLES:
        raise ValueError(f"Unsupported heatmap theme: {theme}")
    if colorbar_orientation not in COLORBAR_ORIENTATION_CHOICES:
        raise ValueError(f"Unsupported colorbar orientation: {colorbar_orientation}")
    validate_lat_bounds(display_lat_min, display_lat_max, "display")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    x_min, x_max, y_min, y_max = extent
    display_x_min, display_x_max, display_y_min, display_y_max = mercator_extent(
        lat_min=display_lat_min,
        lat_max=display_lat_max,
    )
    density = prepare_density(
        histogram,
        sigma=sigma,
        clip_percentile=density_clip_percentile,
        density_vmax=density_vmax,
        mask_fraction=mask_fraction,
    )
    style = THEME_STYLES[theme]

    fig, ax = plt.subplots(figsize=figsize, facecolor=style["figure_face"], constrained_layout=True)
    ax.set_facecolor(style["axis_face"])
    boundary_warning = draw_boundaries(ax, boundary_path, theme=theme)
    draw_graticule(ax, display_lat_min, display_lat_max, theme=theme)

    image = ax.imshow(
        density,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=heatmap_colormap(theme),
        interpolation="bilinear",
        alpha=style["heat_alpha"],
        zorder=3,
        vmin=0.0,
        vmax=density_vmax,
    )

    if include_colorbar:
        if colorbar_orientation == "horizontal":
            colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.022, pad=colorbar_pad, aspect=72)
        else:
            colorbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.026, pad=colorbar_pad, shrink=0.72)
        style_colorbar(colorbar, theme)

    ax.set_xlim(display_x_min, display_x_max)
    ax.set_ylim(display_y_min, display_y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    if boundary_warning:
        print(boundary_warning)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    return RenderedHeatmap(png=png_path.resolve(), pdf=pdf_path.resolve())


def output_paths(
    input_csv: Path,
    output_dir: Path,
    sample_size: int,
    unique_coordinates: bool = False,
) -> tuple[Path, Path, Path]:
    stem = input_csv.stem
    unique_suffix = "_unique" if unique_coordinates else ""
    preview_base = output_dir / f"{stem}{unique_suffix}_gt_heatmap_preview{sample_size}"
    full_base = output_dir / f"{stem}{unique_suffix}_gt_heatmap_full"
    metadata_path = output_dir / f"{stem}{unique_suffix}_gt_heatmap_metadata.json"
    return preview_base, full_base, metadata_path


def output_base_for_theme(output_base: Path, theme: str) -> Path:
    if theme == "light":
        return output_base
    return output_base.with_name(f"{output_base.name}_{theme}")


def resolve_themes(theme: str) -> list[str]:
    if theme == "both":
        return ["light", "dark"]
    if theme not in THEME_STYLES:
        raise ValueError(f"Unsupported heatmap theme: {theme}")
    return [theme]


def metadata_for_render(rendered: RenderedHeatmap) -> dict[str, object]:
    return {
        "png": str(rendered.png),
        "pdf": str(rendered.pdf),
    }


def metadata_for_renders(renders: dict[str, RenderedHeatmap], stats: CoordinateStats) -> dict[str, object]:
    return {
        "stats": asdict(stats),
        "themes": {theme: metadata_for_render(rendered) for theme, rendered in renders.items()},
    }


def render_theme_set(
    histogram: np.ndarray,
    extent: tuple[float, float, float, float],
    output_base: Path,
    themes: Sequence[str],
    dpi: int = DEFAULT_DPI,
    sigma: float = DEFAULT_SIGMA,
    boundary_path: Path | None = None,
    include_colorbar: bool = True,
    colorbar_pad: float = DEFAULT_COLORBAR_PAD,
    colorbar_orientation: str = DEFAULT_COLORBAR_ORIENTATION,
    display_lat_min: float = DEFAULT_DISPLAY_LAT_MIN,
    display_lat_max: float = DEFAULT_DISPLAY_LAT_MAX,
    density_vmax: float | None = None,
    density_clip_percentile: float = DEFAULT_DENSITY_CLIP_PERCENTILE,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> dict[str, RenderedHeatmap]:
    rendered: dict[str, RenderedHeatmap] = {}
    for theme in themes:
        rendered[theme] = render_heatmap(
            histogram,
            extent,
            output_base_for_theme(output_base, theme),
            dpi=dpi,
            sigma=sigma,
            boundary_path=boundary_path,
            include_colorbar=include_colorbar,
            theme=theme,
            colorbar_pad=colorbar_pad,
            colorbar_orientation=colorbar_orientation,
            display_lat_min=display_lat_min,
            display_lat_max=display_lat_max,
            density_vmax=density_vmax,
            density_clip_percentile=density_clip_percentile,
            mask_fraction=mask_fraction,
        )
    return rendered


def run(args: argparse.Namespace) -> HeatmapRunResult:
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input coordinate file does not exist: {input_csv}")
    if args.density_vmax is not None and args.density_vmax <= 0:
        raise ValueError("--density-vmax must be positive when provided")
    if not (0 < args.density_clip_percentile <= 100):
        raise ValueError("--density-clip-percentile must be in the interval (0, 100]")
    if args.mask_fraction < 0:
        raise ValueError("--mask-fraction must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    resolved_input_format = resolve_input_format(input_csv, args.input_format)
    preview_base, full_base, metadata_path = output_paths(
        input_csv,
        output_dir,
        args.sample_size,
        unique_coordinates=args.unique_coordinates,
    )
    boundary_path = args.boundary_data.expanduser().resolve() if args.boundary_data is not None else None
    themes = resolve_themes(args.theme)
    validate_lat_bounds(args.display_lat_min, args.display_lat_max, "display")

    print(f"Sampling {args.sample_size} valid coordinates from {input_csv}...")
    sample, preview_stats = sample_coordinates(
        input_csv,
        sample_size=args.sample_size,
        seed=args.seed,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
        input_format=resolved_input_format,
        lat_index=args.lat_index,
        lon_index=args.lon_index,
        unique_coordinates=args.unique_coordinates,
    )
    preview_histogram, extent = build_histogram_from_coordinates(
        sample,
        bins_x=args.bins_x,
        bins_y=args.bins_y,
    )
    preview_renders = render_theme_set(
        preview_histogram,
        extent,
        preview_base,
        themes,
        dpi=args.dpi,
        sigma=args.sigma,
        boundary_path=boundary_path,
        include_colorbar=not args.no_colorbar,
        colorbar_pad=args.colorbar_pad,
        colorbar_orientation=args.colorbar_orientation,
        display_lat_min=args.display_lat_min,
        display_lat_max=args.display_lat_max,
        density_vmax=args.density_vmax,
        density_clip_percentile=args.density_clip_percentile,
        mask_fraction=args.mask_fraction,
    )

    metadata: dict[str, object] = {
        "input_csv": str(input_csv),
        "input_format": resolved_input_format,
        "lat_column": args.lat_column,
        "lon_column": args.lon_column,
        "lat_index": args.lat_index if resolved_input_format == "whitespace" else None,
        "lon_index": args.lon_index if resolved_input_format == "whitespace" else None,
        "unique_coordinates": args.unique_coordinates,
        "projection": "EPSG:3857 / Web Mercator",
        "latitude_clip_degrees": [DEFAULT_LAT_MIN, DEFAULT_LAT_MAX],
        "display_latitude_degrees": [args.display_lat_min, args.display_lat_max],
        "longitude_range_degrees": [DEFAULT_LON_MIN, DEFAULT_LON_MAX],
        "bins": {"x": args.bins_x, "y": args.bins_y},
        "sigma": args.sigma,
        "dpi": args.dpi,
        "seed": args.seed,
        "theme": args.theme,
        "themes_rendered": themes,
        "colorbar_pad": args.colorbar_pad,
        "colorbar_orientation": args.colorbar_orientation,
        "density_scale": {
            "mode": "fixed" if args.density_vmax is not None else "auto_per_render",
            "vmin": 0.0,
            "vmax": args.density_vmax,
            "clip_percentile": args.density_clip_percentile,
            "mask_fraction": args.mask_fraction,
        },
        "sample_size_requested": args.sample_size,
        "sample_size_rendered": len(sample),
        "boundary_data": str(boundary_path) if boundary_path is not None else None,
        "preview": metadata_for_renders(preview_renders, preview_stats),
        "full": None,
    }

    full_renders: dict[str, RenderedHeatmap] | None = None
    if not args.preview_only:
        print("Building full streaming histogram...")
        full_histogram, full_stats, extent = build_histogram_from_csv(
            input_csv,
            bins_x=args.bins_x,
            bins_y=args.bins_y,
            lat_column=args.lat_column,
            lon_column=args.lon_column,
            input_format=resolved_input_format,
            lat_index=args.lat_index,
            lon_index=args.lon_index,
            unique_coordinates=args.unique_coordinates,
        )
        full_renders = render_theme_set(
            full_histogram,
            extent,
            full_base,
            themes,
            dpi=args.dpi,
            sigma=args.sigma,
            boundary_path=boundary_path,
            include_colorbar=not args.no_colorbar,
            colorbar_pad=args.colorbar_pad,
            colorbar_orientation=args.colorbar_orientation,
            display_lat_min=args.display_lat_min,
            display_lat_max=args.display_lat_max,
            density_vmax=args.density_vmax,
            density_clip_percentile=args.density_clip_percentile,
            mask_fraction=args.mask_fraction,
        )
        metadata["full"] = metadata_for_renders(full_renders, full_stats)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return HeatmapRunResult(metadata_path=metadata_path.resolve(), preview=preview_renders, full=full_renders)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render paper-ready heatmaps from ground-truth latitude/longitude files.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="CSV or whitespace text file containing ground-truth coordinates.")
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMAT_CHOICES,
        default=DEFAULT_INPUT_FORMAT,
        help="Input coordinate format. In auto mode, .txt/.tsv/.lst files are parsed as whitespace rows.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PNG/PDF/metadata outputs.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of valid coordinates in the preview.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic reservoir-sampling seed.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG output DPI.")
    parser.add_argument("--bins-x", type=int, default=DEFAULT_BINS_X, help="Horizontal density-bin count.")
    parser.add_argument("--bins-y", type=int, default=DEFAULT_BINS_Y, help="Vertical density-bin count.")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Gaussian smoothing sigma in density-bin pixels.")
    parser.add_argument("--lat-column", default=DEFAULT_LAT_COLUMN, help="Latitude column name.")
    parser.add_argument("--lon-column", default=DEFAULT_LON_COLUMN, help="Longitude column name.")
    parser.add_argument("--lat-index", type=int, default=DEFAULT_LAT_INDEX, help="Latitude token index for whitespace input.")
    parser.add_argument("--lon-index", type=int, default=DEFAULT_LON_INDEX, help="Longitude token index for whitespace input.")
    parser.add_argument("--unique-coordinates", action="store_true", help="Deduplicate valid latitude/longitude pairs before sampling and histogramming.")
    parser.add_argument("--boundary-data", type=Path, default=default_boundary_path(), help="Country boundary file for map context.")
    parser.add_argument("--theme", choices=THEME_CHOICES, default=DEFAULT_THEME, help="Render the light theme, dark theme, or both.")
    parser.add_argument("--colorbar-pad", type=float, default=DEFAULT_COLORBAR_PAD, help="Space between map and density colorbar.")
    parser.add_argument(
        "--colorbar-orientation",
        choices=COLORBAR_ORIENTATION_CHOICES,
        default=DEFAULT_COLORBAR_ORIENTATION,
        help="Density colorbar orientation.",
    )
    parser.add_argument(
        "--display-lat-min",
        type=float,
        default=DEFAULT_DISPLAY_LAT_MIN,
        help="Minimum latitude shown in the rendered map; coordinate validation still uses the full Mercator clip.",
    )
    parser.add_argument(
        "--display-lat-max",
        type=float,
        default=DEFAULT_DISPLAY_LAT_MAX,
        help="Maximum latitude shown in the rendered map; coordinate validation still uses the full Mercator clip.",
    )
    parser.add_argument(
        "--density-vmax",
        type=float,
        default=None,
        help="Fixed post-log density maximum for a shared color scale across multiple heatmaps.",
    )
    parser.add_argument(
        "--density-clip-percentile",
        type=float,
        default=DEFAULT_DENSITY_CLIP_PERCENTILE,
        help="Auto color-scale clipping percentile when --density-vmax is not provided.",
    )
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=DEFAULT_MASK_FRACTION,
        help="Hide density below this fraction of the active color-scale maximum.",
    )
    parser.add_argument("--preview-only", action="store_true", help="Only write the deterministic preview heatmap and metadata.")
    parser.add_argument("--no-colorbar", action="store_true", help="Render without the density colorbar.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    for theme, rendered in result.preview.items():
        print(f"Preview {theme} PNG: {rendered.png}")
        print(f"Preview {theme} PDF: {rendered.pdf}")
    if result.full is not None:
        for theme, rendered in result.full.items():
            print(f"Full {theme} PNG: {rendered.png}")
            print(f"Full {theme} PDF: {rendered.pdf}")
    print(f"Metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
