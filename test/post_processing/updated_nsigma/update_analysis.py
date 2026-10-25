#!/usr/bin/env python3
"""Update selected-cohort categories, latent correlations, and pixel-cut plots.

Run this file; the adjacent *_methods.py files provide the original methods.
Candidate peaks determine emission membership, never pixel counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import donut_methods as donut
import latent_methods as latent
import pixel_methods as pixel
import venn_methods as venn

ROOT = Path('/work/11161/kanyuni/ls6/quassiQ_project')
DEFAULT_CANDIDATES = Path('/work/10579/prisha/ls6/desi_project/nsigma_results5/candidates.csv')
METRICS = [('latent_variance', 'mean'), ('median_latent_variance', 'median'),
           ('max_latent_variance', 'max')]
MEASUREMENTS = ['number_of_epochs', 'number_of_pairs', 'max_pixels_nsigma_gt_3',
                'max_nsigma_all_pairs', 'wavelength_at_max_nsigma',
                'best_pair_first_date', 'best_pair_second_date']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates-csv', type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument('--selected-dir', type=Path, default=ROOT/'pipeline_output/category_plots')
    parser.add_argument('--selected-file', action='append', help='CATEGORY=PATH, repeated for all seven categories')
    parser.add_argument('--membership-csv', type=Path, help='Optional selected-scope target_membership_comparison.csv from the Venn script; must agree with the current candidate rules')
    parser.add_argument('--visual-xlsx', type=Path, default=latent.DEFAULT_VISUAL_XLSX)
    parser.add_argument('--variance-csv', type=Path, default=latent.DEFAULT_VARIANCE_CSV)
    parser.add_argument('--nsigma-script', type=Path, default=pixel.DEFAULT_NSIGMA)
    parser.add_argument('--latent-output-dir', type=Path,
        default=ROOT/'pipeline_output/post_processing/updated_nsigma5/latent')
    parser.add_argument('--latent-cut', type=float, default=0.5,
        help='Retain when the plotted per-target latent-variance metric is >= this value')
    parser.add_argument('--catalog-csv', type=Path, help='Observation catalog used by nsigma6.py; default from run_info.json or Prisha final_subset.csv')
    parser.add_argument('--coadd-dir', type=Path, help='Raw coadd root; default from run_info.json or Prisha output_coadds')
    parser.add_argument('--run-info-json', type=Path, help='Default: run_info.json beside candidates.csv, if present')
    for setting in ['bin-width', 'smooth-width', 'half-window-kms', 'min-bin-coverage', 'min-smooth-coverage']:
        parser.add_argument('--'+setting, type=float, default=None)
    parser.add_argument('--hbeta-window', choices=['paper', 'velocity'], default=None)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'pipeline_output/post_processing/updated_nsigma6')
    parser.add_argument('--nsigma', type=float, default=3.)
    parser.add_argument('--lines', nargs='+', help='Candidate line prefixes; omit for all lines, exactly as in the supplied Venn command')
    parser.add_argument('--min-pixels', type=int, nargs='+', default=[10, 4, 3, 2])
    parser.add_argument('--only', choices=['all', 'latent', 'pixel'], default='all')
    args = parser.parse_args()
    if not np.isfinite(args.nsigma) or args.nsigma < 0:
        parser.error('--nsigma must be finite and nonnegative')
    if not np.isfinite(args.latent_cut) or args.latent_cut < 0:
        parser.error('--latent-cut must be finite and nonnegative')
    if any(t < 1 for t in args.min_pixels):
        parser.error('--min-pixels values must be positive')
    return args


def build_cohort(args):
    """Use the same original selected lists and exclusions as the Venn plot."""
    passed, maxima, evidence, ids, columns, nrows = venn.new_emission_membership(
        args.candidates_csv, args.nsigma, args.lines)
    if args.membership_csv:
        saved = pd.read_csv(args.membership_csv, dtype={'TARGETID': 'string', 'old_code': 'string', 'new_code': 'string'})
        required = {'TARGETID', 'original_category', 'old_code', 'new_category', 'new_code'}
        if not required.issubset(saved):
            raise ValueError(f'Membership CSV missing {sorted(required-set(saved))}')
        original = saved[['TARGETID', 'original_category', 'old_code']].copy()
        original['TARGETID'] = venn.read_ids(original, args.membership_csv)
        if not original.old_code.isin(venn.ORDER[:7]).all():
            raise ValueError('Use the selected-scope membership CSV, with seven original regions.')
        for row in original.itertuples():
            if venn.category_bits(row.original_category) != row.old_code:
                raise ValueError(f'Original membership mismatch for {row.TARGETID}')
        sources = {'membership_csv': str(args.membership_csv.resolve())}
    else:
        original, sources = venn.read_original(args.selected_dir, args.selected_file)
    if original.TARGETID.duplicated().any():
        raise ValueError('Original selected cohort contains duplicate TARGETIDs; fix the selected lists.')
    table = venn.attach_new(original, passed, maxima, evidence, ids)
    if table.empty:
        raise ValueError('Empty selected cohort after anomaly exclusions.')
    if args.membership_csv:
        check = table.merge(saved[['TARGETID', 'new_code', 'new_category']], on='TARGETID', suffixes=('', '_saved'), validate='one_to_one')
        if ((check.new_code != check.new_code_saved) | (check.new_category != check.new_category_saved)).any():
            raise ValueError('Saved memberships disagree with current candidates/threshold/lines. Use matching inputs or omit --membership-csv.')
    # Match the workbook sheet to the ORIGINAL category, before migration.
    scores = pd.DataFrame(latent.read_workbook_targets(args.visual_xlsx))
    if scores.empty:
        raise ValueError('No workbook score rows.')
    scores['old_code'] = scores.category.map(venn.category_bits)
    # Never average scores across different original category sheets.
    score_keys = ['TARGETID', 'old_code']
    if scores.duplicated(score_keys).any():
        raise ValueError('Multiple workbook sheets represent the same original category and TARGETID.')
    keys = table[score_keys]
    extra = scores.merge(keys, on=score_keys, how='left', indicator=True)
    extra.loc[extra._merge == 'left_only'].drop(columns='_merge').to_csv(args.output_dir/'workbook_rows_outside_original_selection.csv', index=False)
    table = table.merge(scores[['TARGETID', 'old_code', 'Average', 'workbook_rows']], on=score_keys, how='left', validate='one_to_one')
    table['category'] = table.new_category
    table['newly_emission_involved'] = (table.old_code.str[0] == '0') & (table.new_code.str[0] == '1')
    table.to_csv(args.output_dir/'updated_scored_memberships.csv', index=False)
    valid_scores = np.isfinite(table.Average) & table.Average.between(1, 5)
    table.loc[~valid_scores].to_csv(args.output_dir/'missing_or_invalid_visual_scores.csv', index=False)
    summary = pd.DataFrame([{'category': venn.NAMES[c], 'old_count': int((table.old_code == c).sum()),
                             'new_count': int((table.new_code == c).sum())} for c in venn.ORDER])
    summary.to_csv(args.output_dir/'category_counts.csv', index=False)
    metadata = {'candidates_csv': str(args.candidates_csv), 'candidate_line_columns': columns,
                'candidate_rows': nrows, 'candidate_unique_targets': len(ids),
                'candidate_threshold_strictly_greater_than': args.nsigma,
                'original_selection_sources': sources, 'visual_xlsx': str(args.visual_xlsx),
                'anomalies_excluded': sorted(venn.ANOMALIES), 'selected_cohort_size': len(table),
                'newly_emission_involved': int(table.newly_emission_involved.sum()),
                'scope': 'Original selected cohort, including targets now outside all sets; no unscored population entrants.',
                'candidate_coverage_assumption': 'Same as Venn script: absence means not selected, assuming complete catalog coverage.'}
    print(summary.to_string(index=False), flush=True)
    return table, metadata


def groups(frame):
    yield 'all_categories', frame
    for category in sorted(frame.category.unique()):
        yield category, frame.loc[frame.category == category]
    for name, mask in [('emission_involved', frame.new_code.str[0] == '1'),
                       ('non_emission', frame.new_code.str[0] == '0'),
                       ('latent_involved', frame.new_code.str[2] == '1'),
                       ('non_latent', frame.new_code.str[2] == '0')]:
        yield name, frame.loc[mask]


def run_latent(args, cohort):
    out = getattr(args, 'latent_output_dir', None) or args.output_dir/'latent'
    out.mkdir(parents=True, exist_ok=True)
    lookup = latent.read_variances(args.variance_csv)
    data = pd.DataFrame([{**row, **latent.lookup_variance(row['TARGETID'], lookup)[0]}
                         for row in cohort.to_dict('records')])
    data.to_csv(out/'updated_score_latent_variance.csv', index=False)
    stats, medians_by_score = [], []
    for name, group in groups(data):
        plot_data = group.copy()
        if name in {'emission_involved', 'non_emission', 'latent_involved', 'non_latent'}:
            plot_data['category'] = name
        for column, metric in METRICS:
            valid = np.isfinite(group.Average) & group.Average.between(1, 5) & np.isfinite(group[column])
            usable = group.loc[valid]
            metric_dir = out/metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            per_score = latent.score_medians(group, column)
            medians_by_score.append(per_score.assign(group=name, metric=metric))
            per_score.to_csv(metric_dir/f'median_variance_by_visual_score_{pixel.category_filename(name)}.csv', index=False)
            cut_data = usable.copy()
            cut_data['score_level'] = cut_data.Average.map(donut.score_level)
            # Reuse the score-band summary schema for the variance cut.
            cut_data['passes_pixel_cut'] = cut_data[column] >= args.latent_cut
            cut_data['passes_latent_cut'] = cut_data['passes_pixel_cut']
            summary = donut.make_summary(cut_data)
            suffix = pixel.category_filename(name)
            summary.to_csv(metric_dir/f'latent_cut_by_score_band_{suffix}.csv', index=False)
            cut_data.drop(columns='passes_pixel_cut').to_csv(metric_dir/f'latent_cut_assessed_targets_{suffix}.csv', index=False)
            donut.make_donut(summary, metric_dir/f'latent_cut_donut_{suffix}.png', 0,
                len(group)-len(usable), group_name=name.replace('_', ' '),
                variance_cut=args.latent_cut, variance_metric_label=f'{metric} latent variance')
            if name == 'all_categories':
                donut.make_endpoint_plot(summary, metric_dir/'exact_score_1_and_5_eliminated.png')
            sufficient = len(usable) >= 2 and usable.Average.nunique() > 1 and usable[column].nunique() > 1
            stats.append({'group': name, 'metric': metric, 'n_total': len(group), 'n_plotted': len(usable),
                          'pearson_r': usable.Average.corr(usable[column]) if sufficient else np.nan,
                          'spearman_rho': usable.Average.rank().corr(usable[column].rank()) if sufficient else np.nan})
            plot_data = plot_data.loc[valid]
            latent.make_plot(plot_data, metric_dir/f'score_vs_{metric}_latent_variance_{pixel.category_filename(name)}.png',
                             category=None if name == 'all_categories' else name, value_column=column, variance_cut=args.latent_cut)
    pd.DataFrame(stats).to_csv(out/'correlation_statistics.csv', index=False)
    for metric in ('mean', 'median', 'max'):
        pd.DataFrame(stats).loc[lambda f: f.metric == metric].to_csv(out/metric/'correlation_statistics.csv', index=False)
    pd.concat(medians_by_score, ignore_index=True).to_csv(out/'median_latent_variance_by_visual_score.csv', index=False)
    # Composition uses every valid scored target, regardless of variance coverage.
    composition = data.copy()
    composition['score_level'] = composition.Average.map(donut.score_level)
    for name, mask in [('latent_involved', composition.new_code.str[2] == '1'),
                       ('non_latent', composition.new_code.str[2] == '0'),
                       ('all_categories', pd.Series(True, index=composition.index))]:
        counts = latent.make_score_donut(composition.loc[mask],
            out/f'score_distribution_donut_{name}.png', donut.LEVELS, donut.COLORS,
            name.replace('_', ' '))
        counts.to_csv(out/f'score_distribution_by_category_{name}.csv', index=False)
    missing = ~np.isfinite(data.Average) | ~data.Average.between(1, 5) | ~np.isfinite(data.latent_variance)
    data.loc[missing].to_csv(out/'unplotted_targets.csv', index=False)


def assessable(frame):
    return (np.isfinite(frame.max_pixels_nsigma_gt_3) & (frame.max_pixels_nsigma_gt_3 >= 0)
            & (frame.max_pixels_nsigma_gt_3 % 1 == 0)
            & np.isfinite(frame.max_nsigma_all_pairs) & (frame.number_of_pairs > 0))


def run_pixels(args, cohort):
    out = args.output_dir/'pixel'
    out.mkdir(parents=True, exist_ok=True)
    pipeline = None
    rows, pairs = [], []
    for i, source in enumerate(cohort.to_dict('records'), 1):
        target = source['TARGETID']
        error = ''
        if pipeline is None:
            pipeline = pixel.load_pipeline(args.nsigma_script, args)
            (out/'pixel_method_metadata.json').write_text(json.dumps(pipeline.metadata, indent=2)+'\n')
        try:
            result, target_pairs = pixel.analyze_target(pipeline, target, source['category'])
            status = 'recalculated'
        except Exception as exc:
            result = dict.fromkeys(MEASUREMENTS, np.nan)
            target_pairs, status, error = [], 'error', f'{type(exc).__name__}: {exc}'
        measured_count = result.get('max_pixels_nsigma_gt_3', np.nan)
        measured_peak = result.get('max_nsigma_all_pairs', np.nan)
        has_measurement = np.isfinite(measured_peak) and result.get('number_of_pairs', 0) > 0
        expected = bool(source['new_emission'])
        observed = bool(has_measurement and measured_count > 0)
        consistent = ((expected == observed) and (not expected or np.isclose(
            measured_peak, source['new_max_nsigma'], rtol=1e-6, atol=1e-8))) if has_measurement else not expected
        rows.append({**source, **result, **{k: result.get(k, np.nan) for k in MEASUREMENTS},
                     'pixel_measurement_source': status, 'pixel_error': error,
                     'candidate_selection_consistent': bool(consistent)})
        pairs.extend(target_pairs)
        # Checkpoint each completed target; errors are never classified as eliminated.
        pd.DataFrame(rows).to_csv(out/'target_nsigma_pixel_results.csv', index=False)
        print(f'[{i}/{len(cohort)}] {target} {source["category"]}: {status} '
              f'loaded={result.get("number_of_epochs")} usable_pairs={result.get("number_of_pairs")} '
              f'bins={measured_count} peak_abs={measured_peak} candidate_peak={source["new_max_nsigma"]} {error}', flush=True)
        if not consistent:
            pd.DataFrame(rows).loc[lambda f: ~f.candidate_selection_consistent].to_csv(out/'candidate_consistency_failures.csv', index=False)
            raise RuntimeError(f'{target}: fresh nsigma6 counts/peak disagree with candidates.csv. '
                               'Stopped before donuts. Check pixel/candidate_consistency_failures.csv, observation_errors_json, '
                               'run_info settings, observation catalog, coadds and candidate provenance.')
    data = pd.DataFrame(rows)
    pd.DataFrame(pairs).to_csv(out/'newly_calculated_epoch_pairs.csv', index=False)
    valid_pixels = assessable(data)
    valid_scores = np.isfinite(data.Average) & data.Average.between(1, 5)
    data['omission_reason'] = np.where(~valid_pixels, 'no usable pixel measurement',
                                      np.where(~valid_scores, 'missing/invalid visual score', ''))
    data.loc[~valid_pixels | ~valid_scores].to_csv(out/'unassessable_targets.csv', index=False)
    data.loc[data.newly_emission_involved].to_csv(out/'newly_emission_involved_pixel_results.csv', index=False)
    assessed = data.loc[valid_pixels & valid_scores].copy()
    assessed['score_level'] = assessed.Average.map(donut.score_level)
    for threshold in dict.fromkeys(args.min_pixels):
        threshold_dir = out/str(threshold)
        threshold_dir.mkdir(parents=True, exist_ok=True)
        assessed['passes_pixel_cut'] = assessed.max_pixels_nsigma_gt_3 >= threshold
        assessed.to_csv(threshold_dir/'combined_assessed_targets.csv', index=False)
        summaries = []
        for name, full_group in groups(data):
            group = assessed.loc[assessed.TARGETID.isin(full_group.TARGETID)]
            summary = donut.make_summary(group)
            summaries.append(summary.assign(category=name))
            suffix = pixel.category_filename(name)
            summary.to_csv(threshold_dir/f'elimination_by_score_band_{suffix}.csv', index=False)
            # Label the center/title with the actual category or pooled subset.
            donut.make_donut(summary, threshold_dir/f'score_pixel_cut_donut_{suffix}.png', threshold,
                             len(full_group)-len(group), group_name=name.replace('_', ' '), nsigma_threshold=args.nsigma)
            if name == 'all_categories':
                donut.make_endpoint_plot(summary, threshold_dir/'exact_score_1_and_5_eliminated.png')
                for score in (1, 5):
                    for passed, label in [(True, 'retained'), (False, 'eliminated')]:
                        group.loc[(group.Average == score) & (group.passes_pixel_cut == passed)].to_csv(
                            threshold_dir/f'exact_score_{score}_{label}_targets.csv', index=False)
        pd.concat(summaries, ignore_index=True).to_csv(threshold_dir/'elimination_by_category_and_score_band.csv', index=False)
    pixel.make_category_correlation_plot(assessed, None, 'max_pixels_nsigma_gt_3',
        f'Maximum unique bins with |N_sigma| > {args.nsigma:g} in any epoch pair', 'Visual score vs significant-bin count (nsigma6)',
        out/'score_vs_nsigma_pixel_count.png')
    print(f'Pixel stage: {len(assessed)} assessable scored targets; {len(data)-len(assessed)} omitted.', flush=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort, metadata = build_cohort(args)
    metadata.update({'pixel_method': str(args.nsigma_script),
                     'pixel_nsigma_threshold_strictly_greater_than': args.nsigma,
                     'pixel_thresholds': args.min_pixels,
                     'pixel_recalculation': 'Fresh raw coadds through nsigma6.prepare; no cached counts or old N_sigma pipeline.',
                     'latent_variance_csv': str(args.variance_csv),
                     'latent_output_dir': str(args.latent_output_dir),
                     'latent_cut_retained_if_at_least': args.latent_cut,
                     'latent_metrics': 'Mean, median, maximum across ten dimensions; require ten finite dimension values.',
                     'epoch_pairs': 'All loaded catalog observation pairs, matching nsigma6.process_target.',
                     'pixel_settings_metadata': 'pixel/pixel_method_metadata.json'})
    (args.output_dir/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.only in ('all', 'latent'):
        run_latent(args, cohort)
    if args.only in ('all', 'pixel'):
        run_pixels(args, cohort)
    print(f'Outputs: {args.output_dir}; latent outputs: {args.latent_output_dir}', flush=True)


if __name__ == '__main__':
    main()
