#!/usr/bin/env python3
"""Redraw existing N_sigma results with coadds and reconstructions in raw flux units.

python refresh_nsigma_plots.py --nsigma-py /work/11161/kanyuni/ls6/quassiQ_project/quassiQ/src/pipeline/N_sigma.py --csv-dir /work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/csv

CSV results and N_sigma measurements are never rewritten. Existing PNGs are
replaced in place; --output-dir writes copies elsewhere instead.
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location("nsigma_pipeline", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw(module, row, redshift_map, destination):
    line = str(row.line_name).lower()
    config = dict(module.EMISSION_LINES[line])
    config["window"] = float(row.search_half_width)
    target = str(row.TARGETID)
    high = module.load_epoch(Path(row.high_path), target)
    low = module.load_epoch(Path(row.low_path), target)
    curve = module.calculate_n_sigma(high, low)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    z = redshift_map.get(target)
    for epoch, normalized, label, color in (
        (high, curve["high_flux"], "High", "tab:blue"),
        (low, curve["low_flux"], "Low", "tab:orange"),
    ):
        norm = epoch["normalization"]
        if z is not None and np.isfinite(z):
            wave, flux = module.load_original_coadd(target, epoch["date"], z)
            if wave.size:
                axes[0].plot(wave, flux, color=color, lw=0.4, alpha=0.08)
                coarse_wave, coarse_flux = module.coarse_bin(wave, flux)
                axes[0].plot(coarse_wave, coarse_flux, color=color, lw=1.5,
                             ls="--", alpha=0.7,
                             label=f"{label} coadd: {epoch['date']} "
                                   f"(S/N={epoch['coadd_snr']:.2f})")
        # RECON_FLUX is stored divided by NORM in the FITS file.
        axes[0].plot(curve["wave"], normalized * norm, color=color,
                     lw=2.2, alpha=0.7,
                     label=f"{label} reconstruction: {epoch['date']} "
                           f"(chi2/pixel={epoch['chi2_per_pixel']:.3f})")

    center = config["wavelength"]
    window = config["window"]
    for ax in axes:
        ax.axvline(center, color="tab:green", ls=":", lw=1.2)
        ax.axvspan(center - window, center + window,
                   color="tab:green", alpha=0.08)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Flux (original units)")
    axes[0].set_title(f"TARGETID {target} | {config['label']} "
                      f"({center:.2f} Angstrom) | peak N_sigma="
                      f"{float(row.peak_n_sigma):.2f} at "
                      f"{float(row.peak_wavelength):.2f} Angstrom")
    axes[0].legend(fontsize="small", loc="upper right")
    axes[1].plot(curve["wave"], curve["n_sigma"], color="black", lw=1.5)
    axes[1].plot(float(row.peak_wavelength), float(row.peak_n_sigma),
                 marker="*", markersize=14, color="tab:red", ls="none",
                 label=f"peak={float(row.peak_n_sigma):.2f}")
    axes[1].axhline(module.SIGNIFICANCE_THRESHOLD,
                    color="tab:red", ls="--", lw=1)
    half_width = config["display_half_width"]
    axes[1].set_xlim(center - half_width, center + half_width)
    axes[1].set_xlabel("Rest-frame wavelength [Angstrom]")
    axes[1].set_ylabel("N_sigma (normalized spectra)")
    axes[1].legend(fontsize="small", loc="upper right")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsigma-py", type=Path, required=True,
                        help="Path to the existing N_sigma Python script")
    parser.add_argument("--csv-dir", type=Path,
                        help="Directory with *_results.csv and single-target CSVs")
    parser.add_argument("--output-dir", type=Path,
                        help="Write PNGs here instead of replacing existing plots")
    args = parser.parse_args()
    module = load_pipeline(args.nsigma_py)
    csv_dir = args.csv_dir or module.CSV_OUTPUT_DIR
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        parser.error(f"No CSV files found in {csv_dir}")
    _, redshift_map = module.load_catalog()
    updated, skipped, errors = 0, 0, 0
    seen = set()
    for csv_file in csv_files:
        table = pd.read_csv(csv_file, dtype={"TARGETID": str})
        required = {"TARGETID", "line_name", "search_half_width", "status",
                    "high_path", "low_path", "peak_n_sigma", "peak_wavelength"}
        if not required.issubset(table.columns):
            print(f"Skipping unrelated CSV: {csv_file}")
            continue
        for row in table.itertuples(index=False):
            if row.status != "success" or pd.isna(row.high_path) or pd.isna(row.low_path):
                skipped += 1
                continue
            # A target can occur in multiple result CSVs; process each plot once.
            key = (row.line_name, row.TARGETID, row.high_path, row.low_path)
            if key in seen:
                continue
            seen.add(key)
            run_type = "single_targets" if csv_file.name == f"{row.line_name}_{row.TARGETID}.csv" else "batch"
            significant = float(row.peak_n_sigma) > module.SIGNIFICANCE_THRESHOLD
            plot_dir = (module.line_output_dir(row.line_name, run_type) if significant
                        else module.OUTPUT_BASE / "non_CLQ" / row.line_name / run_type)
            old_path = getattr(row, "plot_path", None)
            if isinstance(old_path, str) and old_path.strip():
                destination = Path(old_path)
            else:
                destination = plot_dir / (f"{row.line_name}_{row.TARGETID}_"
                                          f"{row.low_date}_{row.high_date}.png")
            if args.output_dir:
                destination = args.output_dir / destination.relative_to(module.OUTPUT_BASE)
            try:
                draw(module, row, redshift_map, destination)
                updated += 1
                print(f"Updated {destination}")
            except Exception as exc:
                errors += 1
                print(f"ERROR {csv_file.name} TARGETID={row.TARGETID}: {exc}")
    print(f"Updated: {updated}; skipped failed rows: {skipped}; errors: {errors}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
