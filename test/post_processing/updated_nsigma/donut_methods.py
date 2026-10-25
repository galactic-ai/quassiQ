#!/usr/bin/env python3
"""Pool existing per-target pixel results and measure score-1/score-5 losses.

Default pass rule: at least 4 pixels with N_sigma > 3 in at least one epoch
pair (max_pixels_nsigma_gt_3 >= 4). Counts are taken from the pixel script's
CSV: the existing line windows, signed N_sigma definition, masks and epoch
pairs are preserved. No spectra or latent variances are recalculated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/post_processing/pixel')
LEVELS = ['1', '(1, 2)', '[2, 3)', '[3, 4)', '[4, 5)', '5']
COLORS = ['#f28e8e', '#ffbf78', '#ead16b', '#a8d99b', '#67c5b4', '#49ab69']


def score_level(score):
    if not np.isfinite(score) or not 1 <= score <= 5:
        return None
    if score == 1:
        return '1'
    if score < 2:
        return '(1, 2)'
    if score < 3:
        return '[2, 3)'
    if score < 4:
        return '[3, 4)'
    if score < 5:
        return '[4, 5)'
    return '5'


def category_key(value):
    key = re.sub(r'[^a-z0-9]', '', str(value).lower())
    return key[:-4] if key.endswith('only') else key


def load_targets(paths, exclude_csv, out):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype={'TARGETID':'string'})
        required = {'TARGETID','Average','max_pixels_nsigma_gt_3'}
        if not required.issubset(frame):
            raise ValueError(f'{path}: missing {sorted(required-set(frame))}')
        frame['TARGETID'] = frame.TARGETID.str.strip()
        if frame.TARGETID.isna().any() or not frame.TARGETID.str.fullmatch(r'\d+').all():
            raise ValueError(f'{path}: use exact full TARGETIDs stored as text')
        frame['source_csv'] = str(path.resolve())
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    excluded = pd.Series(False, index=frame.index)
    if exclude_csv:
        exclusions = pd.read_csv(exclude_csv, dtype={'TARGETID':'string'})
        if 'TARGETID' not in exclusions:
            raise ValueError('Exclusion CSV needs TARGETID')
        exclusions['TARGETID'] = exclusions.TARGETID.str.strip()
        if 'category' in exclusions:
            if 'category' not in frame:
                raise ValueError('Category-specific exclusions require category in input')
            pairs = set(zip(exclusions.category.map(category_key),exclusions.TARGETID))
            excluded = pd.Series([(category_key(c),t) in pairs for c,t in zip(frame.category,frame.TARGETID)],index=frame.index)
        else:
            excluded = frame.TARGETID.isin(exclusions.TARGETID)
    frame.loc[excluded].to_csv(out/'explicitly_excluded_targets.csv',index=False)
    frame = frame.loc[~excluded].copy()
    # A pooled target must appear once. Do not average potentially mismatched
    # visual scores or silently count the same target in two categories.
    repeated = frame.TARGETID.duplicated(False)
    if repeated.any():
        frame.loc[repeated].to_csv(out/'duplicate_targets_to_fix.csv',index=False)
        raise ValueError('Repeated TARGETIDs found. Fix duplicate_targets_to_fix.csv in your source data, or supply category-specific --exclude-csv. Do not supply both a combined CSV and its category subsets.')
    frame['Average'] = pd.to_numeric(frame.Average,errors='coerce')
    frame['max_pixels_nsigma_gt_3'] = pd.to_numeric(frame.max_pixels_nsigma_gt_3,errors='coerce')
    counts = frame.max_pixels_nsigma_gt_3
    finite_counts = np.isfinite(counts)
    if (finite_counts & ((counts<0) | (counts%1!=0))).any():
        raise ValueError('Pixel counts must be nonnegative integers')
    frame['score_level'] = frame.Average.map(score_level)
    valid = finite_counts & frame.score_level.notna()
    reason = pd.Series('',index=frame.index)
    reason.loc[~finite_counts] = 'missing/nonfinite pixel count'
    reason.loc[frame.score_level.isna()] = 'missing/nonfinite score or score outside [1,5]'
    # The supplied producer can write zero counts even when no pair is usable.
    # Preserve such cases as unassessable, rather than calling them eliminated.
    if 'number_of_pairs' in frame:
        pairs = pd.to_numeric(frame.number_of_pairs,errors='coerce')
        no_pairs = ~np.isfinite(pairs) | (pairs<=0)
        valid &= ~no_pairs
        reason.loc[no_pairs] = 'no evaluated epoch pairs'
    if 'max_nsigma_all_pairs' in frame:
        maximum = pd.to_numeric(frame.max_nsigma_all_pairs,errors='coerce')
        no_finite_pixels = ~np.isfinite(maximum)
        valid &= ~no_finite_pixels
        reason.loc[no_finite_pixels] = 'no finite N-sigma measurement in chosen windows'
    frame.loc[~valid].assign(omission_reason=reason.loc[~valid]).to_csv(out/'unassessable_targets.csv',index=False)
    return frame.loc[valid].copy(), frame, int(excluded.sum())


def make_summary(usable):
    rows = []
    for level in LEVELS:
        group = usable.loc[usable.score_level==level]
        total = len(group)
        passed = int(group.passes_pixel_cut.sum())
        failed = total-passed
        rows.append({'score_level':level,'n_total':total,'n_retained':passed,
                     'n_eliminated':failed,
                     'retained_percent':100*passed/total if total else np.nan,
                     'eliminated_percent':100*failed/total if total else np.nan})
    return pd.DataFrame(rows)


def make_donut(summary, path, min_pixels, n_omitted, group_name="All categories", nsigma_threshold=3., variance_cut=None, variance_metric_label="Latent variance"):
    inner_counts, outer_counts, outer_colors, outer_labels = [], [], [], []
    for (_, row), color in zip(summary.iterrows(),COLORS):
        inner_counts.append(int(row.n_total))
        outer_counts.extend([int(row.n_eliminated),int(row.n_retained)])
        outer_colors.extend([color+'88',color])
        if variance_cut is None:
            outer_labels.extend([f'{row.score_level} — Eliminated (<{min_pixels} bins)',
                                 f'{row.score_level} — Retained (≥{min_pixels} bins)'])
        else:
            outer_labels.extend([f'{row.score_level} — Eliminated (variance <{variance_cut:g})',
                                 f'{row.score_level} — Retained (variance ≥{variance_cut:g})'])
    total = sum(inner_counts)
    fig,ax = plt.subplots(figsize=(12,8))
    if total:
        wedges,_,_ = ax.pie(outer_counts,radius=1,colors=outer_colors,startangle=90,
            counterclock=False,wedgeprops={'width':.30,'edgecolor':'white','linewidth':1.2},
            autopct=lambda pct: str(int(round(pct*total/100))) if pct>=4 else '',
            pctdistance=.85,textprops={'fontsize':9})
        ax.pie(inner_counts,radius=.69,
            labels=[f'{level}\nn={count}' if count else '' for level,count in zip(LEVELS,inner_counts)],
            colors=COLORS,startangle=90,counterclock=False,
            wedgeprops={'width':.30,'edgecolor':'white','linewidth':1.2},
            labeldistance=.76,textprops={'fontsize':8,'ha':'center'})
        ax.legend(wedges,outer_labels,title='Outer ring: '+('bin-count' if variance_cut is None else 'variance')+' cut within each score band',
                  loc='center left',bbox_to_anchor=(1,.5),fontsize=9)
        ax.text(0,0,f'{group_name}\n{total} targets',ha='center',va='center',fontsize=11)
    else:
        ax.text(0,0,'No assessable\nscored targets',ha='center',va='center')
    if variance_cut is None:
        ax.set_title(f'{group_name}: visual-score distribution and nsigma6 bin cut\nRetain if ≥{min_pixels} unique bins have |N-sigma| > {nsigma_threshold:g} in any one epoch pair')
    else:
        ax.set_title(f'{group_name}: visual-score distribution and latent variance cut\nRetain if {variance_metric_label.lower()} ≥ {variance_cut:g}')
    ax.set_aspect('equal')
    fig.text(.02,.015,f'Unassessable targets omitted: {n_omitted}. Exact counts and percentages are in the summary CSV.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(path,dpi=200,bbox_inches='tight')
    plt.close(fig)


def make_endpoint_plot(summary,path):
    endpoints = summary[summary.score_level.isin(['1','5'])]
    fig,ax = plt.subplots(figsize=(7,5))
    values = endpoints.eliminated_percent.to_numpy()
    bars = ax.bar(['Exactly 1','Exactly 5'],np.nan_to_num(values),color=[COLORS[0],COLORS[-1]],width=.55)
    labels = [f'{int(r.n_eliminated)}/{int(r.n_total)}\n({r.eliminated_percent:.1f}%)' if r.n_total else 'No targets' for _,r in endpoints.iterrows()]
    ax.bar_label(bars,labels=labels,padding=5)
    ax.set(ylabel='Percent eliminated within exact-score group',ylim=(0,115),
           title='Losses at the two visual-score endpoints')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(path,dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv',type=Path,nargs='+',default=[ROOT/'target_nsigma_pixel_results.csv'],
                        help='Combined per-target output CSV, or disjoint category CSVs; not the summary/pair CSV')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'combined_score_elimination')
    parser.add_argument('--min-pixels',type=int,default=4,help='Default 4 means strictly more than 3 pixels')
    parser.add_argument('--exclude-csv',type=Path,help='Optional TARGETID exclusions; category column restricts to that category')
    args=parser.parse_args()
    if args.min_pixels<1:
        parser.error('--min-pixels must be at least 1')
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    usable, all_rows, n_excluded = load_targets(args.input_csv,args.exclude_csv,out)
    usable['passes_pixel_cut'] = usable.max_pixels_nsigma_gt_3 >= args.min_pixels
    usable.to_csv(out/'combined_assessed_targets.csv',index=False)
    summary=make_summary(usable)
    summary.to_csv(out/'elimination_by_score_band.csv',index=False)
    endpoints=summary[summary.score_level.isin(['1','5'])]
    endpoints.to_csv(out/'exact_score_1_and_5_elimination.csv',index=False)
    for score in (1,5):
        for passes,label in [(False,'eliminated'),(True,'retained')]:
            usable[(usable.Average==score)&(usable.passes_pixel_cut==passes)].to_csv(
                out/f'exact_score_{score}_{label}_targets.csv',index=False)
    make_donut(summary,out/'combined_score_pixel_cut_donut.png',args.min_pixels,len(all_rows)-len(usable))
    make_endpoint_plot(summary,out/'exact_score_1_and_5_eliminated.png')
    print(f'Explicit exclusions: {n_excluded}; unassessable: {len(all_rows)-len(usable)}; assessed unique targets: {len(usable)}')
    print(f'Retain if at least {args.min_pixels} pixels have N-sigma >3 in any one epoch pair.')
    print(summary.to_string(index=False))
    print('\nExact-score 1 and 5 losses:\n'+endpoints.to_string(index=False))
    print(f'Outputs: {out}')


if __name__=='__main__':
    main()
