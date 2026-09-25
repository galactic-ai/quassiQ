#!/usr/bin/env python3
"""Merge emission-line CSVs and compare unique TARGETIDs across emission lines, clusters, and latents."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib_venn import venn3, venn3_circles


def main():
    root = Path('/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-dir', type=Path, default=root / 'csv')
    parser.add_argument('--latent-csv', type=Path,
                        default=root / 'latent/latent_targets_with_nonzero_p95_counts.csv')
    parser.add_argument('--cluster-csv', type=Path,
                        default=Path('/work/10579/prisha/ls6/desi_project/post_clustering.csv'))
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
    counts = pd.to_numeric(latent['n_latents_exceed_p95'], errors='coerce')
    latent_ids = set(latent.loc[counts > 0, 'TARGETID'].dropna()) - {''}

    cluster = pd.read_csv(args.cluster_csv, dtype={'TARGETID': 'string'})
    cluster['TARGETID'] = cluster['TARGETID'].str.strip()
    cluster_ids = set(cluster['TARGETID'].dropna()) - {''}

    # Exclusive regions, in matplotlib-venn's order: 100, 010, 110,
    # 001, 101, 011, 111 (emission, cluster, latent).
    regions = {
        'emission_only': line_ids - cluster_ids - latent_ids,
        'cluster_only': cluster_ids - line_ids - latent_ids,
        'emission_cluster_only': (line_ids & cluster_ids) - latent_ids,
        'latent_only': latent_ids - line_ids - cluster_ids,
        'emission_latent_only': (line_ids & latent_ids) - cluster_ids,
        'cluster_latent_only': (cluster_ids & latent_ids) - line_ids,
        'all_three': line_ids & cluster_ids & latent_ids,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / 'merged_emission_line_results_latent_over0.csv', index=False)
    groups = {'emission_targets': line_ids, 'cluster_targets': cluster_ids,
              'latent_targets': latent_ids}
    for name, ids in {**groups, **regions}.items():
        destination = args.out_dir / f'{name}_three_groups_{tag}_latent_ge0.csv'
        pd.DataFrame({'TARGETID': sorted(ids)}).to_csv(destination, index=False)
        print(f'{name}: {len(ids):,}')

    fig, ax = plt.subplots(figsize=(10, 8))
    if line_ids or cluster_ids or latent_ids:
        subsets = tuple(len(ids) for ids in regions.values())
        diagram = venn3(subsets=subsets,
              set_labels=(
                            label,
                            'Cluster Jumpers',
                            'Latent Outliers\nn_latents_exceed_p95 > 0'
                        ),
              set_colors=('#2f3061', '#4ecdc4', '#ffe66d'), alpha=0.65, ax=ax)
        venn3_circles(subsets=subsets, linestyle='solid',
                      linewidth=1.5, color='black', ax=ax)
        for count_label in diagram.subset_labels:
            if count_label is not None:
                count_label.set_fontsize(11)
                count_label.set_bbox(dict(facecolor='white', edgecolor='none',
                                          alpha=0.75, pad=1.0))
        # Position each label just outside its circle.
        positions = [
            (-1.05, 0.65, 'right', 'center'),  # Emission: upper left
            (1.05, 0.65, 'left', 'center'),    # Cluster: upper right
            (0.0, -1.15, 'center', 'top'),     # Latent: below
        ]

        for i, (dx, dy, ha, va) in enumerate(positions):
            text = diagram.get_label_by_id('ABC'[i])
            if text is not None:
                center = diagram.get_circle_center(i)
                radius = diagram.get_circle_radius(i)
                text.set_position((
                    center.x + dx * radius,
                    center.y + dy * radius,
                ))
                text.set_ha(ha)
                text.set_va(va)
                text.set_fontsize(10)
    else:
        ax.text(0.5, 0.5, 'All three target sets are empty',
                ha='center', transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title('Unique TARGETID overlap')
    fig.tight_layout()
    destination = args.out_dir / f'emission_cluster_latent_venn_{tag}.png'
    fig.savefig(destination, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {destination}')


if __name__ == '__main__':
    main()
