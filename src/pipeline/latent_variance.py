#!/usr/bin/env python3
"""Compute population latent statistics for candidate TARGETIDs.

Uses every latent row belonging to a candidate TARGETID, not an epoch-level
join to the catalog. Input must contain per-observation Latent1, Latent2, ...
values already computed by SpenderQ. This script does not run the model.
"""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path('/work/11161/kanyuni/ls6/quassiQ_project')
OUT = ROOT / 'pipeline_output/latent'


def read_ids(path):
    df = pd.read_csv(path, dtype={'TARGETID': 'string'})
    if 'TARGETID' not in df:
        raise ValueError(f'{path}: missing TARGETID')
    df['TARGETID'] = df['TARGETID'].str.strip()
    return df.loc[df['TARGETID'].notna() & df['TARGETID'].ne('')].copy()


def compute_statistics(catalog, latent):
    ids = pd.Index(sorted(catalog['TARGETID'].unique()), name='TARGETID')
    cols = sorted(
        [c for c in latent if re.fullmatch(r'Latent\d+', c)],
        key=lambda c: int(c[6:]),
    )
    if not cols:
        raise ValueError('Expected per-observation columns Latent1, Latent2, ...')
    latent = latent.loc[latent['TARGETID'].isin(ids)].copy()
    for c in cols:
        latent[c] = pd.to_numeric(latent[c], errors='coerce').replace(
            [np.inf, -np.inf], np.nan
        )
    grouped = latent.groupby('TARGETID', sort=True)
    out = pd.DataFrame(index=ids)
    out['NumObservations'] = grouped.size().reindex(ids, fill_value=0)
    means = grouped[cols].mean().reindex(ids)
    variances = grouped[cols].var(ddof=0).reindex(ids)
    stds = grouped[cols].std(ddof=0).reindex(ids)
    counts = grouped[cols].count().reindex(ids, fill_value=0)
    for c in cols:
        out[f'{c}_n_valid'] = counts[c]
        out[f'{c}_mu'] = means[c]
        out[f'{c}_var'] = variances[c]
        out[f'{c}_std'] = stds[c]
        out[f'{c}_std_normalized'] = (
            stds[c] / means[c].abs().replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
    return out.reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path,
                   default=ROOT / 'quality_cut/CLQ_candidates_7days_merged.csv')
    p.add_argument('--latent-csv', type=Path,
                   default=ROOT / 'latent/latent_all_targets.csv')
    p.add_argument('--output-dir', type=Path, default=OUT)
    args = p.parse_args()
    result = compute_statistics(read_ids(args.catalog), read_ids(args.latent_csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'latent_variance_by_target.csv'
    result.to_csv(path, index=False)
    print(f'Candidate targets: {len(result)}')
    print(f'Targets with latent rows: {(result.NumObservations > 0).sum()}')
    print(f'Targets missing latent rows: {(result.NumObservations == 0).sum()}')
    print(f'Targets with >=2 latent rows: {(result.NumObservations >= 2).sum()}')
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
