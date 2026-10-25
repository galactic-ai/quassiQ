#!/usr/bin/env python3
"""Fresh raw-coadd bin counts using nsigma6.py, with no old N_sigma pipeline."""
from __future__ import annotations
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_NSIGMA = Path('/work/10579/prisha/ls6/desi_project/nsigma6.py')
PROJECT = Path('/work/10579/prisha/ls6/desi_project')
DEFAULT_CATALOG = PROJECT/'final_subset.csv'
DEFAULT_COADDS = PROJECT/'output_coadds'
SETTING_DEFAULTS = dict(bin_width=2., smooth_width=32., half_window_kms=5000.,
                        hbeta_window='paper', min_bin_coverage=.8, min_smooth_coverage=.8)


def import_method(path):
    spec = importlib.util.spec_from_file_location('current_nsigma6', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ['load_catalog', 'initialize_worker', 'prepare', 'line_bounds',
                'line_hits', 'obs_id', 'smoothing_offsets', 'LINES']
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError(f'{path} is not the supplied nsigma6 interface; missing {missing}')
    return module


def resolve_configuration(args):
    path = args.run_info_json or args.candidates_csv.parent/'run_info.json'
    if args.run_info_json and not path.is_file():
        raise FileNotFoundError(path)
    info = json.loads(path.read_text()) if path.is_file() else {}
    saved = info.get('arguments', {})
    settings = {name: getattr(args, name, None) if getattr(args, name, None) is not None
                else saved.get(name, default) for name, default in SETTING_DEFAULTS.items()}
    def source_path(explicit, saved_value, fallback):
        if explicit is not None:
            return Path(explicit)
        if saved_value:
            path = Path(saved_value)
            # nsigma6 defaults were relative to Prisha's project directory.
            return path if path.is_absolute() else PROJECT/path
        return fallback
    settings.update(input_csv=str(source_path(args.catalog_csv, saved.get('input_csv'), DEFAULT_CATALOG)),
                    coadd_dir=str(source_path(args.coadd_dir, saved.get('coadd_dir'), DEFAULT_COADDS)),
                    threshold=args.nsigma, processing_frame='rest')
    for name in ['bin_width', 'smooth_width', 'half_window_kms', 'min_bin_coverage', 'min_smooth_coverage']:
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f'Invalid {name}: {settings[name]}')
    if settings['half_window_kms'] >= 299792.458 or settings['min_bin_coverage'] > 1 or settings['min_smooth_coverage'] > 1:
        raise ValueError('Invalid velocity window or coverage setting')
    if info and info.get('processing_frame', saved.get('processing_frame', 'rest')) != 'rest':
        raise ValueError('Expected a rest-frame nsigma6 candidate run')
    return SimpleNamespace(**settings), str(path) if info else None


def load_pipeline(path, args):
    module = import_method(path)
    config, info_path = resolve_configuration(args)
    module.smoothing_offsets(config.bin_width, config.smooth_width)
    lines = args.lines or list(module.LINES)
    missing = set(lines)-set(module.LINES)
    if missing:
        raise ValueError(f'Unknown nsigma6 lines: {sorted(missing)}')
    catalog = module.load_catalog(config.input_csv)
    observations = {str(tid): group.to_dict('records') for tid, group in catalog.groupby('TARGETID', sort=False)}
    module.initialize_worker()
    metadata = dict(method_script=str(path), run_info_json=info_path,
                    settings=vars(config), selected_lines=lines,
                    windows_rest_A={name: list(module.line_bounds(name, module.LINES[name], config)) for name in lines},
                    formula='abs(f2-f1)/sqrt(var1+var2)',
                    counting_unit='Unique 2 A rest-frame bins after 32 A median smoothing, or the recorded run widths',
                    overlapping_windows='Union: count each bin once per epoch pair',
                    legacy_columns='*_pixels_* names store processed-bin counts, not native detector pixels')
    print('Pixel method: '+json.dumps(metadata), flush=True)
    if not info_path:
        print('No run_info.json found; using nsigma6 defaults plus explicit options. Candidate consistency is checked before donuts.', flush=True)
    return SimpleNamespace(module=module, config=config, lines=lines, observations=observations, metadata=metadata)


def count_pair(one, two, pipeline):
    """Mirror nsigma6.line_hits() on the union of the chosen line windows."""
    module, config = pipeline.module, pipeline.config
    ids, i, j = np.intersect1d(one[0], two[0], assume_unique=True, return_indices=True)
    f1, v1, f2, v2 = one[1][i], one[2][i], two[1][j], two[2][j]
    good = np.isfinite(f1) & np.isfinite(f2) & np.isfinite(v1) & np.isfinite(v2) & (v1 > 0) & (v2 > 0)
    wave = (ids+.5)*config.bin_width
    window = np.zeros(len(ids), dtype=bool)
    for name in pipeline.lines:
        low, high = module.line_bounds(name, module.LINES[name], config)
        window |= (wave >= low) & (wave <= high)
    selected = good & window
    score = np.full(len(ids), np.nan)
    score[good] = np.abs((f2[good]-f1[good])/np.sqrt(v1[good]+v2[good]))
    significant = selected & (score > config.threshold)
    if selected.any():
        k = np.flatnonzero(selected)[np.argmax(score[selected])]
        peak, peak_wave = float(score[k]), float(wave[k])
    else:
        peak, peak_wave = np.nan, np.nan
    return dict(pixels_nsigma_gt_3=int(significant.sum()), finite_pixels_compared=int(selected.sum()),
                max_nsigma_in_pair=peak, wavelength_at_max_nsigma=peak_wave)


def analyze_target(pipeline, target_id, category):
    module, config = pipeline.module, pipeline.config
    rows = pipeline.observations.get(target_id, [])
    if not rows:
        raise ValueError(f'TARGETID {target_id} absent from input observation catalog {config.input_csv}')
    z = float(np.median([row['Z'] for row in rows]))
    loaded, errors = {}, []
    for row in rows:
        try:
            loaded[module.obs_id(row)] = module.prepare(row, z, config)
        except Exception as exc:
            errors.append(dict(TARGETID=target_id, OBS=module.obs_id(row), ERROR=f'{type(exc).__name__}: {exc}'))
    pairs = []
    for one, two in itertools.combinations(rows, 2):
        first, second = loaded.get(module.obs_id(one)), loaded.get(module.obs_id(two))
        if first is None or second is None:
            continue
        count = count_pair(first, second, pipeline)
        pairs.append(dict(category=category, TARGETID=target_id, first_date=one['NIGHT'], second_date=two['NIGHT'],
                          OBS1=module.obs_id(one), OBS2=module.obs_id(two), **count))
    usable = [p for p in pairs if p['finite_pixels_compared'] > 0]
    best = max(usable, key=lambda p: p['pixels_nsigma_gt_3']) if usable else {}
    peak = max(usable, key=lambda p: p['max_nsigma_in_pair']) if usable else {}
    result = dict(number_of_epochs=len(loaded), number_of_pairs=len(usable),
                  number_of_catalog_observations=len(rows), number_of_attempted_pairs=len(pairs), Z_USED=z,
                  max_pixels_nsigma_gt_3=max((p['pixels_nsigma_gt_3'] for p in usable), default=0),
                  max_nsigma_all_pairs=peak.get('max_nsigma_in_pair', np.nan),
                  wavelength_at_max_nsigma=peak.get('wavelength_at_max_nsigma', np.nan),
                  best_pair_first_date=best.get('first_date',''), best_pair_second_date=best.get('second_date',''),
                  observation_errors_json=json.dumps(errors))
    return result, pairs


def make_category_correlation_plot(
    target_results: pd.DataFrame,
    category: str | None,
    y_column: str,
    y_label: str,
    title: str,
    path: Path,
    *,
    threshold_line: float | None = None,
) -> None:
    """Scatter Average score against a target-level N_sigma summary."""
    if category is None:
        data = target_results.copy()
    else:
        data = target_results[target_results["category"] == category].copy()
    data["Average"] = pd.to_numeric(data["Average"], errors="coerce")
    data[y_column] = pd.to_numeric(data[y_column], errors="coerce")
    data = data.dropna(subset=["Average", y_column])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if not data.empty:
        categories = sorted(data["category"].dropna().unique())
        colors = plt.get_cmap("tab10")
        for index, plot_category in enumerate(categories):
            group = data[data["category"] == plot_category]
            ax.scatter(
                group["Average"], group[y_column],
                s=34, alpha=0.75, color=colors(index % 10),
                label=f"{plot_category} (n={len(group)})",
            )

    if len(data) >= 2 and data["Average"].nunique() > 1 and data[y_column].nunique() > 1:
        pearson_r = float(data["Average"].corr(data[y_column], method="pearson"))
        slope, intercept = np.polyfit(data["Average"], data[y_column], 1)
        x_line = np.linspace(data["Average"].min(), data["Average"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.3,
                label=f"Linear fit (Pearson r={pearson_r:.2f}, n={len(data)})")
    else:
        ax.text(0.03, 0.97, f"Not enough variation to calculate correlation (n={len(data)})",
                transform=ax.transAxes, va="top")

    if threshold_line is not None:
        ax.axhline(threshold_line, color="tab:red", linestyle="--", linewidth=1,
                   label=f"N_sigma = {threshold_line:g}")
    ax.set_xlabel("Average visual-inspection score")
    ax.set_ylabel(y_label)
    ax.set_title(f"{category}: {title}" if category else f"All categories: {title}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def category_filename(category: str) -> str:
    """Make a readable, filesystem-safe category suffix."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", category).strip("_").lower()
