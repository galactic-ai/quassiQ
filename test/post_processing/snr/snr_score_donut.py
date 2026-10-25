#!/usr/bin/env python3
"""Plot updated visual-score distributions split by target median SNR > 5.

Uses existing MEDIAN_SNR values and updated_scored_memberships.csv; no spectra,
latent-variance calculations, or correlation plots. Requires pandas, numpy,
and matplotlib. Each target counts once. The target's SNR is the median of
finite MEDIAN_SNR values across its distinct catalog observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/post_processing')
LEVELS = ['1', '(1, 2)', '[2, 3)', '[3, 4)', '[4, 5)', '5']
COLORS = ['#f28e8e', '#ffbf78', '#ead16b', '#a8d99b', '#67c5b4', '#49ab69']


def score_band(score):
    if not np.isfinite(score) or not 1 <= score <= 5:
        return None
    if score == 1:
        return '1'
    if score == 5:
        return '5'
    return '(1, 2)' if score < 2 else f'[{int(score)}, {int(score)+1})'


def exact_ids(values, source):
    ids = values.astype('string').str.strip()
    if ids.isna().any() or not ids.str.fullmatch(r'\d+').all():
        raise ValueError(f'{source}: TARGETIDs must be exact full integer strings.')
    return ids


def read_target_snr(path):
    required = ['TARGETID', 'MEDIAN_SNR', 'LASTNIGHT', 'TILEID', 'PETAL_LOC']
    frame = pd.read_csv(path, usecols=required, dtype='string')
    frame['TARGETID'] = exact_ids(frame.TARGETID, path)
    frame['MEDIAN_SNR'] = pd.to_numeric(frame.MEDIAN_SNR, errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Normalize observation keys before deduplication. The source has one row
    # per coadd observation, not one row per target.
    nights = frame.LASTNIGHT.str.strip().str.replace(r'\.0+$', '', regex=True)
    compact = nights.str.fullmatch(r'\d{8}', na=False)
    normalized = pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns]')
    normalized.loc[compact] = pd.to_datetime(nights.loc[compact], format='%Y%m%d', errors='raise')
    normalized.loc[~compact] = pd.to_datetime(nights.loc[~compact], errors='raise')
    if normalized.isna().any():
        raise ValueError('Observation catalog contains missing LASTNIGHT.')
    frame['LASTNIGHT'] = normalized.dt.strftime('%Y%m%d')
    for column in ['TILEID', 'PETAL_LOC']:
        values = pd.to_numeric(frame[column], errors='raise')
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f'Invalid observation key {column}')
        frame[column] = values.astype('int64')
    keys = ['TARGETID', 'LASTNIGHT', 'TILEID', 'PETAL_LOC']
    conflict = frame.groupby(keys, dropna=False).MEDIAN_SNR.nunique(dropna=False)
    if (conflict > 1).any():
        raise ValueError('Same observation has conflicting MEDIAN_SNR values; fix catalog duplicates first.')
    n_duplicates = int(frame.duplicated(keys).sum())
    frame = frame.drop_duplicates(keys)
    target = frame.groupby('TARGETID', as_index=False).agg(
        target_median_snr=('MEDIAN_SNR', 'median'),
        n_observations=('MEDIAN_SNR', 'size'),
        n_finite_snr_observations=('MEDIAN_SNR', 'count'),
        min_observation_median_snr=('MEDIAN_SNR', 'min'),
        max_observation_median_snr=('MEDIAN_SNR', 'max'))
    target['n_missing_snr_observations'] = target.n_observations-target.n_finite_snr_observations
    target['target_median_snr'] = target.target_median_snr.astype('float64')
    return target, n_duplicates


def summarize(frame):
    rows = []
    for level in LEVELS:
        group = frame.loc[frame.score_level == level]
        n = len(group)
        retained = int(group.passes_snr_cut.sum())
        rows.append(dict(score_level=level, n_total=n, n_retained=retained,
                         n_eliminated=n-retained,
                         retained_percent=100*retained/n if n else np.nan,
                         eliminated_percent=100*(n-retained)/n if n else np.nan))
    return pd.DataFrame(rows)


def make_donut(summary, path, group_name, cut, omitted):
    inner, outer, labels, outer_colors = [], [], [], []
    for row, color in zip(summary.itertuples(), COLORS):
        inner.append(row.n_total)
        outer.extend([row.n_eliminated, row.n_retained])
        outer_colors.extend([color+'88', color])
        labels.extend([f'{row.score_level} — Eliminated (SNR ≤{cut:g})',
                       f'{row.score_level} — Retained (SNR >{cut:g})'])
    total = sum(inner)
    fig, ax = plt.subplots(figsize=(12, 8))
    if total:
        wedges, _, _ = ax.pie(outer, radius=1., colors=outer_colors,
            startangle=90, counterclock=False,
            wedgeprops={'width': .30, 'edgecolor': 'white', 'linewidth': 1.2},
            autopct=lambda pct: str(int(round(pct*total/100))) if pct >= 4 else '',
            pctdistance=.85, textprops={'fontsize': 9})
        ax.pie(inner, radius=.69, colors=COLORS, startangle=90, counterclock=False,
            labels=[f'{level}\nn={count}' if count/total >= .035 else ''
                    for level, count in zip(LEVELS, inner)],
            wedgeprops={'width': .30, 'edgecolor': 'white', 'linewidth': 1.2},
            labeldistance=.76, textprops={'fontsize': 8, 'ha': 'center'})
        ax.legend(wedges, labels, title='Outer ring: target median-SNR cut',
                  loc='center left', bbox_to_anchor=(1., .5), fontsize=9)
        ax.text(0, 0, f'{group_name}\n{total} targets', ha='center', va='center', fontsize=11)
    else:
        ax.text(0, 0, 'No assessable\nscored targets', ha='center', va='center')
    ax.set_title(f'{group_name}: visual-score distribution and SNR cut\n'
                 f'Retain if median of observation MEDIAN_SNR values > {cut:g}')
    ax.set_aspect('equal')
    fig.text(.02, .015, f'Omitted for missing/invalid score or no finite SNR: {omitted}. '
             'Exact counts are in the summary CSV.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-csv', type=Path,
        default=Path('/work/10579/prisha/ls6/desi_project/final_subset.csv'))
    parser.add_argument('--memberships-csv', type=Path,
        default=ROOT/'updated_nsigma6/updated_scored_memberships.csv',
        help='Updated target categories and visual scores from update_analysis.py')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'updated_nsigma5/snr')
    parser.add_argument('--snr-cut', type=float, default=5.)
    args = parser.parse_args()
    if not np.isfinite(args.snr_cut):
        parser.error('--snr-cut must be finite')
    cohort = pd.read_csv(args.memberships_csv, dtype={'TARGETID': 'string', 'new_code': 'string'})
    required = {'TARGETID', 'Average', 'new_category', 'new_code'}
    if not required.issubset(cohort):
        raise ValueError(f'Memberships missing {sorted(required-set(cohort))}')
    cohort['TARGETID'] = exact_ids(cohort.TARGETID, args.memberships_csv)
    if cohort.TARGETID.duplicated().any():
        raise ValueError('Membership table must have exactly one row per TARGETID.')
    if not cohort.new_code.str.fullmatch(r'[01]{3}', na=False).all():
        raise ValueError('Invalid new_code; expected three membership bits.')
    if cohort.new_category.isna().any():
        raise ValueError('Missing updated category.')
    cohort['category'] = cohort.new_category
    cohort['Average'] = pd.to_numeric(cohort.Average, errors='coerce')
    snr, duplicates = read_target_snr(args.catalog_csv)
    overlap = set(snr.columns)-{'TARGETID'}
    cohort = cohort.drop(columns=list(overlap & set(cohort)))
    data = cohort.merge(snr, on='TARGETID', how='left', validate='one_to_one')
    data['score_level'] = data.Average.map(score_band)
    finite_snr = np.isfinite(data.target_median_snr.to_numpy(dtype=float, na_value=np.nan))
    data['omission_reason'] = np.select(
        [data.score_level.isna(), ~finite_snr],
        ['missing/invalid visual score', 'no finite catalog MEDIAN_SNR'], default='')
    # Unassessable measurements keep a blank result, rather than False.
    data['passes_snr_cut'] = pd.Series(pd.NA, index=data.index, dtype='boolean')
    data.loc[finite_snr, 'passes_snr_cut'] = data.loc[finite_snr, 'target_median_snr'] > args.snr_cut
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out/'target_median_snr_results.csv', index=False)
    usable = data.loc[data.omission_reason == ''].copy()
    usable.to_csv(out/'assessed_targets.csv', index=False)
    data.loc[data.omission_reason != ''].to_csv(out/'unassessable_targets.csv', index=False)
    groups = [('all_categories', data)]
    groups.extend((name, data.loc[data.category == name]) for name in sorted(data.category.unique()))
    groups.extend([(name, data.loc[mask]) for name, mask in [
        ('latent_involved', data.new_code.str[2] == '1'),
        ('non_latent', data.new_code.str[2] == '0'),
        ('emission_involved', data.new_code.str[0] == '1'),
        ('non_emission', data.new_code.str[0] == '0')]])
    summaries = []
    for name, group in groups:
        assessed = usable.loc[usable.TARGETID.isin(group.TARGETID)]
        summary = summarize(assessed)
        summaries.append(summary.assign(group=name))
        suffix = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_').lower()
        summary.to_csv(out/f'snr_cut_by_score_band_{suffix}.csv', index=False)
        make_donut(summary, out/f'snr_cut_donut_{suffix}.png', name.replace('_', ' '),
                   args.snr_cut, len(group)-len(assessed))
    pd.concat(summaries, ignore_index=True).to_csv(out/'snr_cut_by_category_and_score_band.csv', index=False)
    metadata = dict(catalog_csv=str(args.catalog_csv), memberships_csv=str(args.memberships_csv),
        aggregation='Median of finite MEDIAN_SNR values across distinct observations for each TARGETID',
        observation_keys=['TARGETID', 'LASTNIGHT', 'TILEID', 'PETAL_LOC'],
        identical_duplicate_observations_removed=duplicates, retained_if_strictly_greater_than=args.snr_cut,
        n_cohort=len(data), n_assessed=len(usable), n_omitted=len(data)-len(usable),
        cohort='Existing updated visual-inspection cohort; catalog alone has no visual scores',
        missing_snr='Finite observation values used; no finite values => unassessable')
    (out/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Unique cohort targets: {len(data)}; assessed: {len(usable)}; omitted: {len(data)-len(usable)}')
    print(f'Retained: {int(usable.passes_snr_cut.sum())}; eliminated: {int((~usable.passes_snr_cut).sum())}')
    print(f'Cut: target median SNR > {args.snr_cut:g}. Outputs: {out}')


if __name__ == '__main__':
    main()
