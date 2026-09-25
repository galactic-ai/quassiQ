#!/usr/bin/env python3
"""Sample each of seven Venn regions and plot full-spectrum comparisons.
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_target(pipeline, target, high, low, redshift, reference, output, category):
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
        axes[0].set_title(f'TARGETID {target} | {category} | '
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



def read_ids(path, latent=False):
    frame = pd.read_csv(path, dtype='string')
    ids = frame.iloc[:, 0].str.strip().replace('', pd.NA)
    for column in ('TARGETID', 'TARGET ID'):
        if column in frame:
            ids = frame[column].str.strip().replace('', pd.NA)
            break
    if latent and 'n_latents_exceed_p95' in frame:
        ids = ids[pd.to_numeric(frame['n_latents_exceed_p95'], errors='coerce') > 0]
    return set(ids.dropna())


def main():
    root = Path('/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-dir', type=Path, default=root / 'csv')
    parser.add_argument('--cluster-csv', type=Path, default = Path("/work/10579/prisha/ls6/desi_project/post_clustering.csv"),
                        help='Full cluster-jumper list, not an exclusive subset.')
    parser.add_argument('--latent-csv',  type=Path, default = Path("/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/latent/latent_targets_with_nonzero_p95_counts.csv"),
                        help='Full latent list with n_latents_exceed_p95, or prefiltered outlier list. Not the exclusive CSV.')
    parser.add_argument('--nsigma-script', type=Path, default=Path(
        '/work/11161/kanyuni/ls6/quassiQ_project/quassiQ/src/pipeline/N_sigma.py'))
    parser.add_argument('--out-dir', type=Path, default=root / 'category_plots')
    parser.add_argument('--number', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--line-name', default='lya', help='Reference for choosing high/low epochs.')
    args = parser.parse_args()
    if args.number < 1:
        parser.error('--number must be positive')
    spec = importlib.util.spec_from_file_location('nsigma_pipeline', args.nsigma_script)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    if args.line_name not in pipeline.EMISSION_LINES:
        parser.error(f'Choose from {list(pipeline.EMISSION_LINES)}')

    emission = set()
    # Use exactly the configured emission lines to avoid unrelated/stale CSVs.
    for line in pipeline.EMISSION_LINES:
        path = args.csv_dir / f'{line}_all_results.csv'
        frame = pd.read_csv(path, dtype={'TARGETID': 'string'})
        good = pd.to_numeric(frame['peak_n_sigma'], errors='coerce') > 3
        emission.update(frame.loc[good, 'TARGETID'].str.strip().replace('', pd.NA).dropna())
    cluster = read_ids(args.cluster_csv)
    latent = read_ids(args.latent_csv, latent=True)
    regions = {
        'emission_only': emission - cluster - latent,
        'cluster_only': cluster - emission - latent,
        'latent_only': latent - emission - cluster,
        'emission_cluster_only': (emission & cluster) - latent,
        'emission_latent_only': (emission & latent) - cluster,
        'cluster_latent_only': (cluster & latent) - emission,
        'all_three': emission & cluster & latent,
    }
    redshifts = {}
    if pipeline.CSV_PATH.is_file():
        _, redshifts = pipeline.load_catalog()
    # Supplement redshifts from the category source lists if present.
    for path in (args.cluster_csv, args.latent_csv):
        frame = pd.read_csv(path, dtype='string')
        if 'Z' in frame:
            column = next((c for c in ('TARGETID', 'TARGET ID') if c in frame), frame.columns[0])
            z = pd.to_numeric(frame['Z'], errors='coerce')
            valid = np.isfinite(z) & (z > -1)
            for target, value in zip(frame.loc[valid, column], z[valid]):
                if pd.notna(target):
                    redshifts[str(target).strip()] = float(value)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    reference = pipeline.EMISSION_LINES[args.line_name]
    for category, members in regions.items():
        folder = args.out_dir / category
        plots = folder / 'full_spectrum'
        plots.mkdir(parents=True, exist_ok=True)
        selected = pd.Series(sorted(members), dtype='string').sample(
            n=min(args.number, len(members)), random_state=args.seed).tolist()
        pd.DataFrame({'TARGETID': sorted(members)}).to_csv(folder / 'all_category_targetids.csv', index=False)
        pd.DataFrame({'TARGETID': selected}).to_csv(folder / 'selected_targetids.csv', index=False)
        print(f'\n{category}: {len(members)} total; {len(selected)} sampled', flush=True)
        results = []
        for i, target in enumerate(selected, 1):
            result = {'TARGETID': target, 'category': category, 'reference_line': args.line_name,
                      'status': 'failed', 'failure_reason': '', 'plot_path': None}
            try:
                epochs = pipeline.load_candidate_epochs(target)
                high, low = pipeline.select_high_low_epochs(epochs, reference['wavelength'])
                if high is None:
                    raise ValueError('Fewer than two epochs pass quality and reference-line coverage cuts')
                calc = pipeline.calculate_n_sigma(high, low)
                if not np.any(np.isfinite(calc['n_sigma'])):
                    raise ValueError('No finite N_sigma pixels')
                output = plots / f"{target}_{low['date']}_{high['date']}.png"
                plot_target(pipeline, target, high, low, redshifts.get(target),
                            reference['label'], output, category)
                result.update(status='success', plot_path=str(output),
                              high_date=high['date'], low_date=low['date'])
                for line, config in pipeline.EMISSION_LINES.items():
                    peak, _ = pipeline.calculate_peak_n_sigma(calc['wave'], calc['n_sigma'],
                                                              config['wavelength'], config['window'])
                    result[f'{line}_pair_peak_n_sigma'] = peak
            except Exception as exc:
                result['failure_reason'] = f'{type(exc).__name__}: {exc}'
            results.append(result)
            print(f"[{category} {i}/{len(selected)}] {target}: {result['status']} "
                  f"{result['failure_reason']}", flush=True)
        table = pd.DataFrame(results) if results else pd.DataFrame(
            columns=['TARGETID', 'category', 'status', 'failure_reason', 'plot_path'])
        table.to_csv(folder / 'full_spectrum_results.csv', index=False)
        successful = sum(row['status'] == 'success' for row in results)
        summary.append({'category': category, 'total_targets': len(members),
                        'sampled_targets': len(selected), 'plots_saved': successful,
                        'failed_targets': len(selected) - successful})
        pd.DataFrame(summary).to_csv(args.out_dir / 'category_summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'Outputs: {args.out_dir}')


if __name__ == '__main__':
    main()
