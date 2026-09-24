#!/usr/bin/env python3
"""Merge emission-line CSVs and compare unique TARGETIDs with the latent list."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2


def main():
    root = Path('/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-dir', type=Path, default=root / 'csv')
    parser.add_argument('--latent-csv', type=Path,
                        default=root / 'latent/latent_targets_with_nonzero_p95_counts.csv')
    parser.add_argument('--out-dir', type=Path, default=root / 'overlap')
    parser.add_argument('--nsigma', type=float, default=None,
                        help='Optional: require peak_n_sigma > this value in at least one line.')
    args = parser.parse_args()
    files = sorted(args.csv_dir.glob('*_all_results.csv'))
    if not files:
        raise FileNotFoundError(f'No *_all_results.csv files in {args.csv_dir}')

    frames = []
    for path in files:
        # Read IDs as strings so large DESI identifiers never pass through floats.
        df = pd.read_csv(path, dtype={'TARGETID': 'string'})
        df['TARGETID'] = df['TARGETID'].str.strip()
        df['source_file'] = path.name
        frames.append(df)
        print(f'{path.name}: {df["TARGETID"].nunique():,} unique targets')
    merged = pd.concat(frames, ignore_index=True)
    selected = merged
    label = 'Emission lines\nAll targets'
    tag = 'all'
    if args.nsigma is not None:
        if 'peak_n_sigma' not in merged.columns:
            raise ValueError('Missing peak_n_sigma column')
        selected = merged.loc[pd.to_numeric(merged['peak_n_sigma'], errors='coerce') > args.nsigma]
        label = f'Emission lines\nN_sigma > {args.nsigma:g} in any line'
        tag = f'nsigma_gt_{args.nsigma:g}'

    latent = pd.read_csv(args.latent_csv, dtype={'TARGETID': 'string'})
    latent['TARGETID'] = latent['TARGETID'].str.strip()
    line_ids = set(selected['TARGETID'].dropna()) - {''}
    # latent_ids = set(latent['TARGETID'].dropna()) - {''}
    counts = pd.to_numeric(latent['n_latents_exceed_p95'], errors='coerce')
    latent_ids = set(latent.loc[counts > 3, 'TARGETID'].dropna()) - {''}


    both = line_ids & latent_ids
    line_only = line_ids - latent_ids
    latent_only = latent_ids - line_ids

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / 'merged_emission_line_results_latent_over4.csv', index=False)
    for name, ids in [('emission_targets', line_ids), ('latent_targets', latent_ids),
                      ('both', both), ('emission_only', line_only), ('latent_only', latent_only)]:
        pd.DataFrame({'TARGETID': sorted(ids)}).to_csv(args.out_dir / f'{name}_{tag}.csv', index=False)
    print(f'\nEmission-line targets: {len(line_ids):,}\nLatent targets: {len(latent_ids):,}'
          f'\nBoth: {len(both):,}\nEmission only: {len(line_only):,}\nLatent only: {len(latent_only):,}')

    fig, ax = plt.subplots(figsize=(8, 6))
    if line_ids or latent_ids:
        venn2(subsets=(len(line_only), len(latent_only), len(both)),
              set_labels=(label, 'Latent >3  p95 counts'),
              set_colors=('#2f3061', '#ffe66d'), alpha=0.65, ax=ax)

    else:
        ax.text(0.5, 0.5, 'Both target sets are empty', ha='center', transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title('Unique TARGETID overlap')
    fig.tight_layout()
    destination = args.out_dir / f'emission_vs_latent_venn_{tag}_latent_over4.png'
    fig.savefig(destination, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {destination}')


if __name__ == '__main__':
    main()
