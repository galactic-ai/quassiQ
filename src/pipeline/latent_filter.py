#!/usr/bin/env python3
"""Count normalized latent standard deviations strictly above a percentile.

Matches pipeline.py: std(ddof=0)/abs(mean), per-dimension quantiles across
candidate targets, strict > comparison, and selection when count > 0.
Missing values do not contribute to thresholds or exceedance counts.
"""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

OUT = Path('/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/latent')


def compute_target_latent_counts(df, percentile=95.0):
    if not np.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError('Percentile must be strictly between 0 and 100.')
    if 'TARGETID' not in df or df['TARGETID'].isna().any():
        raise ValueError('Input must have nonmissing TARGETIDs.')
    if df['TARGETID'].duplicated().any():
        raise ValueError('Input must have exactly one row per TARGETID.')
    cols = sorted(
        [c for c in df if re.fullmatch(r'Latent\d+_std_normalized', c)],
        key=lambda c: int(c.split('_')[0][6:]),
    )
    if not cols:
        raise ValueError('No LatentN_std_normalized columns found.')
    values = df[cols].apply(pd.to_numeric, errors='coerce').replace(
        [np.inf, -np.inf], np.nan
    ).abs()
    thresholds = values.quantile(percentile / 100, interpolation='linear')
    tag = f'p{percentile:g}'
    count_col = f'n_latents_exceed_{tag}'
    counts = df[['TARGETID']].copy()
    counts[count_col] = values.gt(thresholds, axis='columns').sum(axis=1)
    counts[cols] = values
    counts = counts.sort_values(
        [count_col, 'TARGETID'], ascending=[False, True]
    ).reset_index(drop=True)
    selected = counts.loc[counts[count_col] > 0].copy()
    threshold_table = pd.DataFrame({
        'latent_std_normalized_col': cols,
        f'{tag}_abs_std_normalized': thresholds.reindex(cols).to_numpy(),
    })
    return counts, selected, threshold_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=OUT / 'latent_variance_by_target.csv')
    p.add_argument('--percentile', type=float, default=95.0,
                   help='Percentile on a 0-100 scale, e.g. 90, 95, or 97.5.')
    p.add_argument('--output-dir', type=Path, default=OUT)
    args = p.parse_args()
    if not np.isfinite(args.percentile) or not 0 < args.percentile < 100:
        p.error('--percentile must be strictly between 0 and 100.')
    df = pd.read_csv(args.input, dtype={'TARGETID': 'string'})
    counts, selected, thresholds = compute_target_latent_counts(df, args.percentile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f'p{args.percentile:g}'
    outputs = [
        (counts, f'latent_std_normalized_{tag}_counts_by_target.csv'),
        (thresholds, f'latent_std_normalized_{tag}_by_latent.csv'),
        (selected, f'latent_targets_with_nonzero_{tag}_counts.csv'),
    ]
    for table, name in outputs:
        path = args.output_dir / name
        table.to_csv(path, index=False)
        print(f'Saved: {path}')
    norm_cols = [c for c in counts if c.endswith('_std_normalized')]
    print(f'Targets evaluated: {len(counts)}')
    print(f'Targets with no finite normalized values: '
          f'{counts[norm_cols].isna().all(axis=1).sum()}')
    print(f'Targets passing n_latents_exceed_{tag} > 0: {len(selected)}')


if __name__ == '__main__':
    main()
