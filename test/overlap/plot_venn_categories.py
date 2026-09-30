#!/usr/bin/env python3
"""Plot every recorded emission pair and cross-cluster pair in sampled Venn targets.

Latent membership remains target-level. Latent-only targets use a reference-line
high/low display pair. Counts and pair flags are saved separately.
"""
import argparse
import importlib.util
import shutil
from itertools import combinations
import json
import re
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
        for epoch, key, label, color in [
            (high, 'high_flux', 'Epoch A', 'tab:blue'),
            (low, 'low_flux', 'Epoch B', 'tab:orange')
        ]:
            label = f"Epoch {epoch['epoch_label']}"
            if 'cluster_label' in epoch:
                label += f" [cluster {epoch['cluster_label']}]"

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
        axes[1].axhline(pipeline.SIGNIFICANCE_THRESHOLD, color='tab:orange', ls='--',
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
                          f'{reference}')
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


def plot_raw_target(pipeline, target, high, low, redshift, reference,
                    output, category, coarse=False):
    """Plot raw coadds (optionally binned) and, for the full view, reconstructions."""
    if redshift is None or not np.isfinite(redshift):
        raise ValueError(f'No valid redshift for raw coadds of {target}')
    fig, ax = plt.subplots(figsize=(17, 6))
    try:
        found = False
        for epoch, label, color in (
            (high, 'Epoch A', 'tab:blue'),
            (low, 'Epoch B', 'tab:orange')
        ):
            label = f"Epoch {epoch['epoch_label']}"
            if 'cluster_label' in epoch:
                label += f" [cluster {epoch['cluster_label']}]"
            wave, flux = pipeline.load_original_coadd(target, epoch['date'], redshift)
            if not wave.size:
                raise ValueError(f'No original coadd for {target} on {epoch["date"]}')
            found = True
            if coarse:
                wave, flux = pipeline.coarse_bin(wave, flux)
                ax.plot(wave, flux, color=color, lw=1.1,
                        label=f'{label} binned coadd: {epoch["date"]}')
            else:
                ax.plot(wave, flux, color=color, lw=0.45, alpha=0.4,
                        label=f'{label} coadd: {epoch["date"]}')
                # N_sigma.py stores RECON_FLUX divided by NORM.
                ax.plot(epoch['rest_wave'], epoch['recon_flux'] * epoch['normalization'],
                        color=color, lw=1.4, label=f'{label} reconstruction: {epoch["date"]}')
        if not found:
            raise ValueError(f'No original coadds for {target}')
        for config in pipeline.EMISSION_LINES.values():
            center = config['wavelength']
            ax.axvline(center, color='tab:green', ls=':', alpha=0.45, lw=0.8)
        ax.set(title=f'TARGETID {target} | {category} | {reference} | '
                     f'{"Binned raw coadds" if coarse else "Raw coadds and reconstructions"}',
               xlabel='Rest-frame wavelength [Å]', ylabel='Flux (original units)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output, dpi=180, bbox_inches='tight')
    finally:
        plt.close(fig)



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


def date_key(value):
    if pd.isna(value):
        raise ValueError('Missing epoch date')
    value = re.sub(r'\.0$', '', str(value).strip())
    return pd.to_datetime(value).strftime('%Y%m%d')


def pair_key(a, b):
    a, b = date_key(a), date_key(b)
    if a == b:
        raise ValueError(f'Epoch pair has the same date twice: {a}')
    return tuple(sorted((a, b)))


def cluster_pairs(frame, date_column=None):
    """All distinct-date pairs with different valid cluster assignments."""
    date_column = date_column or next((c for c in
        ('OBS_DATE', 'NIGHT_CLEAN', 'LASTNIGHT') if c in frame), None)
    if not date_column or 'CLUSTER' not in frame:
        raise ValueError('Cluster CSV needs CLUSTER and an epoch date column')
    result = {}
    frame = frame.copy()
    frame['TARGETID'] = frame.TARGETID.str.strip()
    for target, rows in frame.groupby('TARGETID'):
        assignments = {}
        for _, row in rows.iterrows():
            cluster = row.CLUSTER
            if pd.isna(cluster) or str(cluster).strip() in ('', '-1', '-1.0'):
                continue  # Missing/noise is not a cluster transition.
            cluster = str(cluster).strip()
            if re.fullmatch(r'-?\d+\.0+', cluster):
                cluster = cluster.split('.')[0]
            date = date_key(row[date_column])
            if date in assignments and assignments[date] != cluster:
                raise ValueError(f'{target}: conflicting clusters on {date}; date alone is ambiguous')
            assignments[date] = cluster
        pairs = {}
        ordered = sorted(assignments)
        for i, a in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                b = ordered[j]
                if assignments[a] != assignments[b]:
                    pairs[(a, b)] = [f'clusters={assignments[a]}->{assignments[b]};'
                                     f'adjacent_valid_epochs={j == i + 1}']
        if pairs:
            result[target] = pairs
    return result


def emission_pair_columns(frame, high_col=None, low_col=None):
    if high_col or low_col:
        if not high_col or not low_col or high_col not in frame or low_col not in frame:
            raise ValueError('Provide existing --emission-high-column and --emission-low-column')
        return high_col, low_col
    for high, low in [('high_date', 'low_date'), ('high_epoch', 'low_epoch'),
                      ('high_epoch_date', 'low_epoch_date'), ('date_high', 'date_low')]:
        if high in frame and low in frame:
            return high, low
    raise ValueError('Emission CSV lacks recognized pair dates. Supply '
                     '--emission-high-column and --emission-low-column; '
                     'target IDs alone cannot identify detected epochs.')
def epoch_letter(index):
    label = ''
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label

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
    parser.add_argument('--final-subset-csv', type=Path, default=Path(
        '/work/10579/prisha/ls6/desi_project/final_subset.csv'),
        help='Only sample TARGETIDs present in this catalog.')
    parser.add_argument('--number', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cluster-date-column', default=None)
    parser.add_argument('--emission-high-column', default=None)
    parser.add_argument('--emission-low-column', default=None)
    args = parser.parse_args()
    if args.number < 1:
        parser.error('--number must be positive')
    spec = importlib.util.spec_from_file_location('nsigma_pipeline', args.nsigma_script)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
  

    emission = set()
    emission_evidence = {}
    # Use exactly the configured emission lines to avoid unrelated/stale CSVs.
    for line in pipeline.EMISSION_LINES:
        path = args.csv_dir / f'{line}_all_results.csv'
        frame = pd.read_csv(path, dtype={'TARGETID': 'string'})
        frame['TARGETID'] = frame['TARGETID'].str.strip().replace('', pd.NA)
        if frame['TARGETID'].isna().any():
            raise ValueError(f'{path}: missing TARGETID')
        good = pd.to_numeric(frame['peak_n_sigma'], errors='coerce') > 3
        emission.update(frame.loc[good, 'TARGETID'].str.strip().replace('', pd.NA).dropna())
        if good.any():
            hc, lc = emission_pair_columns(frame, args.emission_high_column, args.emission_low_column)
            for _, row in frame.loc[good].iterrows():
                target = str(row.TARGETID).strip()
                pair = pair_key(row[hc], row[lc])
                emission_evidence.setdefault(target, {}).setdefault(pair, []).append(line)
    cluster_evidence = cluster_pairs(pd.read_csv(args.cluster_csv, dtype='string'),
                                     args.cluster_date_column)
    cluster_frame = pd.read_csv(args.cluster_csv, dtype='string')
    date_column = args.cluster_date_column or next(
        c for c in ('OBS_DATE', 'NIGHT_CLEAN', 'LASTNIGHT')
        if c in cluster_frame
    )
    cluster_labels = {
        (str(row['TARGETID']).strip(), date_key(row[date_column])):
            str(row['CLUSTER']).strip()
        for _, row in cluster_frame.iterrows()
        if pd.notna(row['CLUSTER'])
    }
    cluster = set(cluster_evidence)
    latent = read_ids(args.latent_csv, latent=True)
    final_subset = read_ids(args.final_subset_csv)
    regions = {
        'emission_only': emission - cluster - latent,
        'cluster_only': cluster - emission - latent,
        'latent_only': latent - emission - cluster,
        'emission_cluster_only': (emission & cluster) - latent,
        'emission_latent_only': (emission & latent) - cluster,
        'cluster_latent_only': (cluster & latent) - emission,
        'all_three': emission & cluster & latent,
    }
    # Restrict every Venn region before sampling so rejected IDs are replaced.
    regions = {name: members & final_subset for name, members in regions.items()}
    # Save every category's IDs together, before sampling.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_categories = pd.DataFrame(
        [(target, category)
        for category, members in regions.items()
        for target in sorted(members)],
        columns=['TARGETID', 'category'],
    )

    all_categories.to_csv(
        args.out_dir / 'all_categories_targetids.csv', index=False
    )
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
    for category, members in regions.items():
        folder = args.out_dir / category
        plots = folder / 'full_spectrum'
        # Clear plots from previous runs, including the old flat layout.
        if plots.is_dir():
            shutil.rmtree(plots)
        plots.mkdir(parents=True, exist_ok=True)
        selected = pd.Series(sorted(members), dtype='string').sample(
            n=min(args.number, len(members)), random_state=args.seed).tolist()
        pd.DataFrame({'TARGETID': sorted(members)}).to_csv(folder / 'all_category_targetids.csv', index=False)
        pd.DataFrame({'TARGETID': selected}).to_csv(folder / 'selected_targetids.csv', index=False)
        print(f'\n{category}: {len(members)} total; {len(selected)} sampled', flush=True)
        results = []
        target_results = []
        for i, target in enumerate(selected, 1):
            evidence = {}
            if target in emission:
                evidence['emission'] = emission_evidence[target]
            if target in cluster:
                evidence['cluster'] = cluster_evidence[target]
            pair_sets = [set(pairs) for pairs in evidence.values()]
            differs = len(pair_sets) > 1 and any(x != pair_sets[0] for x in pair_sets[1:])
            common_pair = bool(set.intersection(*pair_sets)) if len(pair_sets) > 1 else False
            # This also preserves pair provenance for unavailable spectra.
            all_pairs = sorted(set.union(*pair_sets)) if pair_sets else []
            load_error = ''
            try:
                by_date = {}
                for epoch in pipeline.load_candidate_epochs(target):
                    date = date_key(epoch['date'])
                    if date in by_date:
                        raise ValueError(f'Multiple loaded spectra on {date}; cannot select unambiguously')
                    by_date[date] = epoch
            except Exception as exc:
                by_date = {}
                load_error = f'{type(exc).__name__}: {exc}'

            # Outside except: include all pairs for latent-selected targets.
            if target in latent and not load_error:
                evidence['latent'] = {
                    pair: ['All epoch pairs; latent selection is target-level']
                    for pair in combinations(sorted(by_date), 2)
                }
                if len(by_date) < 2:
                    load_error = 'Fewer than two available epochs for latent comparison'

            # Merge pairs across methods without duplicates.
            all_pairs = sorted({
                pair
                for method_pairs in evidence.values()
                for pair in method_pairs
            })
           
            target_plots = plots / target
            epoch_labels = {
                date: epoch_letter(i)
                for i, date in enumerate(sorted(by_date))
            }
            target_plots.mkdir(parents=True, exist_ok=True)
            target_records = []

            for first, second in all_pairs:
                pair = (first, second)
                methods = [method for method, pairs in evidence.items() if pair in pairs]
                details = {method: evidence[method][pair] for method in methods}
                method_tag = '+'.join(methods)
                filename_methods = '+'.join(
                    'emission_' + '-'.join(sorted(set(details[m]))) if m == 'emission' else m
                    for m in methods)
                flag = '__different_method_pairs' if differs else ''
                stem = f'{target}__{first}__{second}__{filename_methods}{flag}'
                result = dict(TARGETID=target, category=category, epoch_a=first, epoch_b=second,
                              methods=method_tag, evidence_json=json.dumps(details),
                              latent_target_selected=target in latent,
                              method_pair_sets_differ=differs, has_pair_common_to_all_methods=common_pair,
                              latent_selection_scope='target_variance_not_epoch_selection'
                              if target in latent else '', status='failed', failure_reason='',
                              normalized_plot_path='', coarse_plot_path='', unnormalized_plot_path='')
                try:
                    if load_error:
                        raise ValueError(load_error)
                    missing_dates = [d for d in pair if d not in by_date]
                    if missing_dates:
                        raise ValueError(f'Detected/contributing epochs unavailable after spectral loading: {missing_dates}')
                    epoch_a['epoch_label'] = epoch_labels[first]
                    epoch_b['epoch_label'] = epoch_labels[second]

                    if target in cluster:
                        epoch_a['cluster_label'] = cluster_labels.get(
                            (target, first), 'unknown'
                        )
                        epoch_b['cluster_label'] = cluster_labels.get(
                            (target, second), 'unknown'
                        )
                    calc = pipeline.calculate_n_sigma(epoch_a, epoch_b)
                    if not np.any(np.isfinite(calc['n_sigma'])):
                        raise ValueError('No finite N_sigma pixels')
                    description = (
                        f"Epoch {epoch_labels[first]} ({first}) vs "
                        f"Epoch {epoch_labels[second]} ({second}) | {filename_methods}"
                    )
                    if target in latent:
                        description += ' | latent: target-level selection'
                    if differs:
                        description += ' | DIFFERENT METHOD PAIRS'
                    for kind in ('normalized', 'coarse', 'unnormalized'):
                        output = target_plots / f'{kind}__{stem}.png'
                        try:
                            if kind == 'normalized':
                                plot_target(pipeline, target, epoch_a, epoch_b, redshifts.get(target),
                                            description, output, category)
                            else:
                                plot_raw_target(pipeline, target, epoch_a, epoch_b, redshifts.get(target),
                                                description, output, category, coarse=kind == 'coarse')
                            result[f'{kind}_plot_path'] = str(output)
                        except Exception as exc:
                            result['failure_reason'] += f'{kind}: {type(exc).__name__}: {exc}; '
                    saved = sum(bool(result[f'{kind}_plot_path']) for kind in
                                ('normalized', 'coarse', 'unnormalized'))
                    result['status'] = 'success' if saved == 3 else 'partial' if saved else 'failed'
                    for line, config in pipeline.EMISSION_LINES.items():
                        peak, _ = pipeline.calculate_peak_n_sigma(calc['wave'], calc['n_sigma'],
                                                                  config['wavelength'], config['window'])
                        result[f'{line}_pair_peak_n_sigma'] = peak
                except Exception as exc:
                    result['failure_reason'] = f'{type(exc).__name__}: {exc}'
                results.append(result)
                target_records.append(result)
            if not all_pairs:
                record = dict(TARGETID=target, category=category, status='failed',
                              failure_reason=load_error or 'No usable epoch pair',
                              epoch_a='', epoch_b='', methods='latent')
                results.append(record)
            successes = sum(r['status'] == 'success' for r in target_records)
            target_results.append(dict(TARGETID=target, category=category,
                                       method_pair_sets_differ=differs,
                                       has_pair_common_to_all_methods=common_pair,
                                       requested_pairs=len(all_pairs), successful_pairs=successes,
                                       incomplete_pairs=len(all_pairs) - successes,
                                       failure_reason=load_error,
                                       status='success' if all_pairs and successes == len(all_pairs) else 'incomplete'))
            # Checkpoint after each target.
            pd.DataFrame(results).to_csv(folder / 'full_spectrum_results.csv', index=False)
            pd.DataFrame(target_results).to_csv(folder / 'target_pair_summary.csv', index=False)
            flag_text = (
                f'; different_method_pairs={differs}'
                if target in emission and target in cluster
                else ''
            )
            print(
                f'[{category} {i}/{len(selected)}] {target}: '
                f'{successes}/{len(all_pairs)} pairs fully plotted{flag_text}',
                flush=True,
            )
        if not results:
            pd.DataFrame(columns=['TARGETID', 'epoch_a', 'epoch_b', 'methods', 'status',
                                  'failure_reason']).to_csv(folder / 'full_spectrum_results.csv', index=False)
            pd.DataFrame(columns=['TARGETID', 'requested_pairs', 'successful_pairs']).to_csv(
                folder / 'target_pair_summary.csv', index=False)
        summary.append(dict(category=category, total_targets=len(members), sampled_targets=len(selected),
                            requested_pairs=sum(r['requested_pairs'] for r in target_results),
                            successful_pairs=sum(r['status'] == 'success' for r in results),
                            incomplete_pairs=sum(r['incomplete_pairs'] for r in target_results),
                            incomplete_targets=sum(r['status'] != 'success' for r in target_results),
                            targets_with_different_method_pairs=sum(r['method_pair_sets_differ'] for r in target_results)))
        pd.DataFrame(summary).to_csv(args.out_dir / 'category_summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'Outputs: {args.out_dir}')


if __name__ == '__main__':
    main()
