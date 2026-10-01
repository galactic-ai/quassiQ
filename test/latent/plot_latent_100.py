#!/usr/bin/env python3
"""Plot one full-spectrum comparison per randomly sampled latent-exclusive target."""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_target(pipeline, target, high, low, redshift, reference, output):
    calc = pipeline.calculate_n_sigma(high, low)
    wave = calc['wave']
    finite_wave = wave[np.isfinite(wave)]
    if finite_wave.size == 0:
        raise ValueError('No finite rest-frame wavelengths')
    xmin, xmax = finite_wave.min(), finite_wave.max()
    fig, axes = plt.subplots(2, 1, figsize=(17, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    try:
        for epoch, key, label, color in [(high, 'high_flux', 'High', 'tab:blue'),
                                         (low, 'low_flux', 'Low', 'tab:orange')]:
            if redshift is not None and np.isfinite(redshift):
                ow, of = pipeline.load_original_coadd(target, epoch['date'], redshift)
                if ow.size:
                    bw, bf = pipeline.coarse_bin(ow, of / epoch['normalization'])
                    axes[0].plot(bw, bf, color=color, alpha=0.3, lw=0.7,
                                 label=f"{label} original (binned)")
            axes[0].plot(wave, calc[key], color=color, lw=1.4,
                         label=f"{label}: {epoch['date']} (S/N={epoch['coadd_snr']:.2f}, "
                               f"χ²/pixel={epoch['chi2_per_pixel']:.2f})")
        axes[1].plot(wave, calc['n_sigma'], color='black', lw=0.8)
        axes[1].axhline(pipeline.SIGNIFICANCE_THRESHOLD, color='tab:red', ls='--',
                        label=f"{pipeline.SIGNIFICANCE_THRESHOLD:g}σ")
        for j, config in enumerate(pipeline.EMISSION_LINES.values()):
            center = config['wavelength']
            if xmin <= center <= xmax:
                for ax in axes:
                    ax.axvline(center, color='tab:green', ls=':', alpha=0.55, lw=0.8)
                axes[0].text(center, 0.98 if j % 2 == 0 else 0.78, config['label'],
                             transform=axes[0].get_xaxis_transform(), rotation=90,
                             va='top', ha='right', fontsize=8)
        axes[0].set_title(f'TARGETID {target} | Latent-exclusive sample | '
                          f'High/low epochs selected at {reference}')
        axes[0].set_ylabel('Normalized flux')
        axes[1].set_ylabel('Nσ')
        axes[1].set_xlabel('Rest-frame wavelength [Å]')
        axes[1].set_xlim(xmin, xmax)
        axes[0].legend(loc='upper right', fontsize=8)
        axes[1].legend(loc='upper right')
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output, dpi=180, bbox_inches='tight')
    finally:
        plt.close(fig)
    return calc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, default=Path(
        '/work/10579/prisha/ls6/desi_project/latent_variance_exclusive.csv'))
    parser.add_argument('--nsigma-script', type=Path, default=Path(
        '/work/11161/kanyuni/ls6/quassiQ_project/quassiQ/src/pipeline/N_sigma.py'))
    parser.add_argument('--number', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--line-name', default='lya',
                        help='Reference line used to select one high/low epoch pair (default: lya).')
    parser.add_argument('--out-dir', type=Path, default=Path(
        '/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/latent/exclusive_plot'))
    args = parser.parse_args()
    if args.number < 1:
        parser.error('--number must be positive')
    if not args.nsigma_script.is_file():
        parser.error(f'Cannot find {args.nsigma_script}')
    spec = importlib.util.spec_from_file_location('nsigma_pipeline', args.nsigma_script)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    if args.line_name not in pipeline.EMISSION_LINES:
        parser.error(f'Choose from {list(pipeline.EMISSION_LINES)}')
    df = pd.read_csv(args.input_csv, dtype='string')
    df['TARGETID'] = df.iloc[:, 0].str.strip().replace('', pd.NA)
    ids = df['TARGETID'].dropna().drop_duplicates()
    if len(ids) < args.number:
        parser.error(f'Only {len(ids)} unique IDs; requested {args.number}')
    selected = ids.sample(n=args.number, random_state=args.seed).tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'TARGETID': selected}).to_csv(args.out_dir / 'selected_targetids.csv', index=False)
    redshifts = {}
    if pipeline.CSV_PATH.is_file():
        _, redshifts = pipeline.load_catalog()
    if 'Z' in df.columns:
        z = pd.to_numeric(df['Z'], errors='coerce')
        good = df['TARGETID'].notna() & np.isfinite(z) & (z > -1)
        zz = df.loc[good, ['TARGETID']].assign(Z=z[good])
        redshifts.update(zz.drop_duplicates('TARGETID').set_index('TARGETID')['Z'].to_dict())
    plot_dir = args.out_dir / 'full_spectrum'
    plot_dir.mkdir(exist_ok=True)
    reference = pipeline.EMISSION_LINES[args.line_name]
    results = []
    for i, target in enumerate(selected, 1):
        result = {'TARGETID': target, 'reference_line': args.line_name,
                  'status': 'failed', 'failure_reason': '', 'plot_path': None}
        try:
            epochs = pipeline.load_candidate_epochs(target)
            high, low = pipeline.select_high_low_epochs(epochs, reference['wavelength'])
            if high is None:
                raise ValueError('Fewer than two epochs pass the original S/N, chi-square, '
                                 'and reference-line coverage cuts')
            result.update(high_date=high['date'], low_date=low['date'])
            # Check that a full-spectrum comparison can actually be made.
            calc = pipeline.calculate_n_sigma(high, low)
            if not np.any(np.isfinite(calc['n_sigma'])):
                raise ValueError('No valid N_sigma pixels for the selected epochs')
            output = plot_dir / f"{target}_{low['date']}_{high['date']}.png"
            plot_target(pipeline, target, high, low, redshifts.get(target),
                        reference['label'], output)
            # These values use ONE shared pair, unlike separate per-line epoch selections.
            for name, config in pipeline.EMISSION_LINES.items():
                peak, _ = pipeline.calculate_peak_n_sigma(
                    calc['wave'], calc['n_sigma'], config['wavelength'], config['window'])
                result[f'{name}_pair_peak_n_sigma'] = peak
            result.update(status='success', plot_path=str(output))
        except Exception as exc:
            result['failure_reason'] = f'{type(exc).__name__}: {exc}'
        results.append(result)
        print(f"[{i}/{len(selected)}] {target}: {result['status']} {result['failure_reason']}", flush=True)
    table = pd.DataFrame(results)
    table.to_csv(args.out_dir / 'full_spectrum_results.csv', index=False)
    print(f"Saved {(table['status'] == 'success').sum()} full-spectrum plots to {plot_dir}")


if __name__ == '__main__':
    main()
