#!/usr/bin/env python3
"""Top up the latent-only sample to 100 successful full-spectrum plots."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-script', type=Path,
                        default=Path("/work/11161/kanyuni/ls6/quassiQ_project/quassiQ/test/overlap/plot_venn_categories.py"))
    parser.add_argument('--nsigma-script', type=Path, default=Path(
        '/work/11161/kanyuni/ls6/quassiQ_project/quassiQ/src/pipeline/N_sigma.py'))
    parser.add_argument('--folder', type=Path, default=Path(
        '/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/category_plots/latent_only'))
    args = parser.parse_args()
    plotter = load_module('plotter', args.plot_script)
    pipeline = load_module('pipeline', args.nsigma_script)
    folder = args.folder
    original = pd.read_csv(folder / 'full_spectrum_results.csv', dtype={'TARGETID': 'string'})
    log_path = folder / 'replacement_results.csv'
    previous = pd.read_csv(log_path, dtype={'TARGETID': 'string'}) if log_path.exists() else pd.DataFrame()
    combined = pd.concat([original, previous], ignore_index=True)
    successful = combined.loc[combined.status.eq('success'), 'TARGETID'].dropna().drop_duplicates().tolist()
    attempted = set(combined.TARGETID.dropna())
    attempted.update(pd.read_csv(folder / 'selected_targetids.csv', dtype='string').iloc[:, 0].dropna())
    pool = pd.read_csv(folder / 'all_category_targetids.csv', dtype='string').iloc[:, 0].dropna()
    pool = pool[~pool.isin(attempted)].drop_duplicates().sample(frac=1, random_state=43)
    refs = original.reference_line.dropna().unique()
    if len(refs) != 1:
        raise ValueError('Expected one reference line in original results')
    reference = pipeline.EMISSION_LINES[refs[0]]
    redshifts = pipeline.load_catalog()[1] if pipeline.CSV_PATH.is_file() else {}
    records = previous.to_dict('records')
    plots = folder / 'full_spectrum'
    plots.mkdir(exist_ok=True)
    for target in pool:
        if len(successful) >= 100:
            break
        row = dict(TARGETID=target, category='latent_only', reference_line=refs[0],
                   status='failed', failure_reason='', plot_path=None)
        try:
            high, low = pipeline.select_high_low_epochs(
                pipeline.load_candidate_epochs(target), reference['wavelength'])
            if high is None:
                raise ValueError('Fewer than two qualifying epochs')
            calc = pipeline.calculate_n_sigma(high, low)
            if not np.any(np.isfinite(calc['n_sigma'])):
                raise ValueError('No finite N_sigma pixels')
            output = plots / f"{target}_{low['date']}_{high['date']}.png"
            plotter.plot_target(pipeline, target, high, low, redshifts.get(target),
                                reference['label'], output, 'latent_only')
            row.update(status='success', plot_path=str(output),
                       high_date=high['date'], low_date=low['date'])
            for name, config in pipeline.EMISSION_LINES.items():
                peak, _ = pipeline.calculate_peak_n_sigma(calc['wave'], calc['n_sigma'],
                                                          config['wavelength'], config['window'])
                row[f'{name}_pair_peak_n_sigma'] = peak
            successful.append(target)
        except Exception as exc:
            row['failure_reason'] = f'{type(exc).__name__}: {exc}'
        records.append(row)
        pd.DataFrame(records).to_csv(log_path, index=False)
        print(f"{target}: {row['status']} {row['failure_reason']}", flush=True)
    replacements = [r['TARGETID'] for r in records if r['status'] == 'success']
    pd.DataFrame({'TARGETID': replacements}).to_csv(folder / 'replacement_targetids.csv', index=False)
    pd.DataFrame({'TARGETID': successful}).to_csv(folder / 'successful_targetids.csv', index=False)
    print('Replacement IDs:\n' + '\n'.join(replacements))
    print(f'{len(successful)}/100 successful targets; lists saved in {folder}')
    if len(successful) < 100:
        print('Remaining category targets exhausted before reaching 100.')


if __name__ == '__main__':
    main()
