#!/usr/bin/env python3
"""Compare full-population Venn changes and failures among original selected IDs.

Original cluster/latent definitions follow overlap.py. New emission is any
chosen line/pair > --nsigma in a complete candidate catalog. Both scopes
exclude the 10 known anomalies. Requires pandas, numpy and matplotlib.

run 
python /work/11161/kanyuni/ls6/quassiQ_project/quassiQ/test/post_processing/compare_nsigma_venn_diagram.py  --candidates-csv /work/10579/prisha/ls6/desi_project/nsigma_results5/candidates.csv  --old-emission-dir /work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/n_sigma_archive/csv --old-nsigma 3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path('/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output')
# E, C, L bits; 000 is tracked explicitly outside the circles.
ORDER = ['100', '010', '110', '001', '101', '011', '111', '000']
NAMES = dict(zip(ORDER, ['emission_only', 'cluster_only', 'emission_cluster_only',
    'latent_only', 'emission_latent_only', 'cluster_latent_only', 'all_three', 'none']))
POSITIONS = {'100':(-.83,.40), '010':(.83,.40), '110':(0,.58),
             '001':(0,-.94), '101':(-.46,-.38), '011':(.46,-.38),
             '111':(0,-.02), '000':(-1.65,-1.38)}
ANOMALIES = {
    '39627673550455521', '39627752155907480',
    '39627741670147785', '39627823480046613', '39627902941135614',
    '39627927393931075', '39627963779521305', '39628358258002822',
    '39628011250649588', '39628518056791749',
}



def category_bits(name):
    key = re.sub(r'[^a-z0-9]', '', name.lower())
    aliases = {re.sub(r'[^a-z0-9]', '', label): code for code, label in NAMES.items()}
    aliases.update({'emission': '100', 'cluster': '010', 'latent':'001',
                    'emissioncluster':'110', 'emissionlatent':'101',
                    'clusterlatent':'011', 'latentcluster':'011',
                    'latentemission':'101', 'all':'111'})
    if key not in aliases or aliases[key] == '000':
        raise ValueError(f'Unrecognized original category folder: {name}')
    return aliases[key]


def read_ids(frame, source):
    if 'TARGETID' not in frame:
        raise ValueError(f'{source}: missing TARGETID column')
    ids = frame['TARGETID'].astype('string').str.strip()
    if ids.isna().any() or not ids.str.fullmatch(r'\d+').all():
        raise ValueError(f'{source}: invalid TARGETID; use exact full integer strings')
    return ids


def read_original(directory, overrides):
    sources = []
    if overrides:
        for specification in overrides:
            category, separator, path = specification.partition('=')
            if not separator:
                raise ValueError('--selected-file must be CATEGORY=/absolute/path.csv')
            sources.append((category, Path(path)))
    else:
        sources = [(p.parent.name, p) for p in sorted(directory.glob('*/selected_targetids.csv'))]
    rows, paths, seen = [], {}, set()
    for category, path in sources:
        code = category_bits(category)
        if code in seen:
            raise ValueError(f'Multiple source files for {NAMES[code]}')
        seen.add(code)
        df = pd.read_csv(path, dtype='string')
        ids = read_ids(df, path)
        if ids.duplicated().any():
            raise ValueError(f'{path}: duplicate TARGETIDs')
        paths[NAMES[code]] = str(path.resolve())
        rows.extend({'TARGETID':target, 'original_category':NAMES[code],
                     'old_code':code, 'original_row_1based':i+1}
                    for i, target in enumerate(ids))
    if seen != set(ORDER[:7]):
        raise ValueError(f'Need all seven category files. Missing: {[NAMES[c] for c in ORDER[:7] if c not in seen]}')
    original = pd.DataFrame(rows)
    if original['TARGETID'].duplicated().any():
        repeated = original.loc[original['TARGETID'].duplicated(False), 'TARGETID'].tolist()
        raise ValueError(f'TARGETIDs in multiple original regions: {repeated}. Audit the selected lists first.')
    return original, paths


def new_emission_membership(path, threshold, lines):
    df = pd.read_csv(path, dtype='string')
    ids = read_ids(df, path)
    if lines:
        columns = [f'{line}_max_nsigma' for line in lines]
        missing = set(columns) - set(df)
        if missing:
            raise ValueError(f'Unknown line columns: {sorted(missing)}')
    else:
        columns = [c for c in df if c.endswith('_max_nsigma')]
        if not columns and 'peak_n_sigma' in df:
            columns = ['peak_n_sigma']
    if not columns:
        raise ValueError('No *_max_nsigma or peak_n_sigma columns found')
    peaks = df[columns].apply(pd.to_numeric, errors='coerce')
    malformed = df[columns].notna() & peaks.isna()
    if malformed.any().any():
        raise ValueError('Non-numeric, nonblank values in N-sigma columns')
    if np.isinf(peaks.to_numpy(dtype=float)).any():
        raise ValueError('Infinite N-sigma values found')
    # A candidate can have multiple epoch pairs; any qualifying pair is enough.
    row_max = peaks.max(axis=1)
    passed = row_max > threshold
    maxima = pd.DataFrame({'TARGETID':ids, 'new_max_nsigma':row_max}).groupby('TARGETID')['new_max_nsigma'].max()
    evidence = pd.DataFrame({'TARGETID':ids, 'qualifying_pair':passed.astype(int)}).groupby('TARGETID')['qualifying_pair'].sum()
    return set(ids[passed]), maxima, evidence, set(ids), columns, len(df)


def reclassify(code, emission):
    return ('1' if emission else '0') + code[1:]


def draw_base(ax):
    # Identical fixed geometry in every plot. Circle areas do not encode counts.
    for center, color in [((-.5,.18),'#2f3061'), ((.5,.18),'#4ecdc4'), ((0,-.57),'#ffe66d')]:
        ax.add_patch(Circle(center, .96, facecolor=color, edgecolor='#444444', alpha=.23, lw=1.4))
    ax.text(-1.10,1.28,'Emission',ha='center',fontsize=12)
    ax.text(1.10,1.28,'Cluster',ha='center',fontsize=12)
    ax.text(0,-1.68,'Latent',ha='center',fontsize=12)
    ax.set(xlim=(-2.12,1.68),ylim=(-1.86,1.54),aspect='equal')
    ax.axis('off')


def draw_counts(ax, counts):
    draw_base(ax)
    for code, position in POSITIONS.items():
        text = f'{counts[code]}' if code != '000' else f'Outside all sets\n{counts[code]}'
        ax.text(*position, text, ha='center', va='center', fontsize=13,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':3})


def make_figures(old, new, table, transitions, out):
    fig, axes = plt.subplots(1,2,figsize=(14,7))
    for ax, counts, title in zip(axes, [old,new], ['Original selection','New N-sigma emission selection']):
        draw_counts(ax,counts)
        ax.set_title(title,fontsize=14)
    fig.suptitle(f'Same comparison universe, anomalies excluded (n={len(table)})',fontsize=16)
    fig.text(.5,.025,'Fixed geometry; areas are not proportional to counts. Cluster and latent memberships are held fixed.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(out/'venn_before_after.png',dpi=200,bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,9))
    draw_base(ax)
    # Only four old/new region pairs can exchange members when E alone changes.
    for left,right in [('100','000'),('010','110'),('001','101'),('011','111')]:
        forward = int(transitions.loc[left,right])
        reverse = int(transitions.loc[right,left])
        if forward or reverse:
            arrow = FancyArrowPatch(POSITIONS[left],POSITIONS[right],
                connectionstyle='arc3,rad=.24',arrowstyle='<->' if forward and reverse else ('->' if forward else '<-'),
                mutation_scale=15,color='#676767',lw=1.6,shrinkA=34,shrinkB=34,zorder=2)
            ax.add_patch(arrow)
    for code,position in POSITIONS.items():
        delta = new[code]-old[code]
        text = f'{old[code]} → {new[code]}\n({delta:+d})'
        if code=='000':
            text = 'Outside all sets\n'+text
        ax.text(*position,text,ha='center',va='center',fontsize=12,
                color='#16784d' if delta>0 else ('#b14442' if delta<0 else '#333333'),
                bbox={'facecolor':'white','edgecolor':'#dddddd','alpha':.94,'pad':4},zorder=3)
    moved = table[table['changed']]
    ax.set_title(f'Original → new count in each Venn region\n{len(moved)} of {len(table)} targets move',fontsize=16,pad=18)
    fig.savefig(out/'venn_shift_overlay.png',dpi=200,bbox_inches='tight')
    plt.close(fig)

    fig, ax=plt.subplots(figsize=(12,10))
    matrix=transitions.to_numpy()
    im=ax.imshow(matrix,cmap='Blues',vmin=0)
    for i in range(8):
        for j in range(8):
            if matrix[i,j]:
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    labels=[NAMES[c].replace('_',' ') for c in ORDER]
    ax.set(xticks=range(8),yticks=range(8),xticklabels=labels,yticklabels=labels,
           xlabel='New region',ylabel='Original region',title='Target migration between exclusive Venn regions')
    plt.setp(ax.get_xticklabels(),rotation=40,ha='right')
    fig.colorbar(im,ax=ax,label='Number of targets')
    fig.tight_layout()
    fig.savefig(out/'region_transition_heatmap.png',dpi=200,bbox_inches='tight')
    plt.close(fig)



def id_set(path):
    frame = pd.read_csv(path, dtype='string')
    return set(read_ids(frame, path))


def old_emission_set(args):
    if args.old_emission_ids:
        return id_set(args.old_emission_ids), {
            'mode':'saved original emission-ID list',
            'source':str(args.old_emission_ids.resolve()),
            'old_nsigma':'already encoded in saved ID list',
        }
    files = sorted(args.old_emission_dir.glob('*_all_results.csv'))
    if not files:
        raise FileNotFoundError(
            f'No original *_all_results.csv in {args.old_emission_dir}. '
            'Use --old-emission-ids with a saved ORIGINAL emission-target list '
            'or --old-emission-dir pointing to the original results.'
        )
    selected = set()
    for path in files:
        frame = pd.read_csv(path, dtype='string')
        ids = read_ids(frame, path)
        if args.old_nsigma is None:
            selected.update(ids)
        else:
            if 'peak_n_sigma' not in frame:
                raise ValueError(f'{path}: --old-nsigma requires peak_n_sigma')
            selected.update(ids[pd.to_numeric(frame.peak_n_sigma, errors='coerce') > args.old_nsigma])
    return selected, {
        'mode':'original overlap.py emission rule',
        'sources':[str(p.resolve()) for p in files],
        'old_nsigma':args.old_nsigma,
        'note':'No old threshold means all original result IDs, matching overlap.py default.',
    }


def full_population(args, new_ids):
    old_emission, old_meta = old_emission_set(args)
    # Exactly reproduce attached overlap.py: all cluster CSV IDs. Do not
    # silently replace its rule with a different cluster-transition algorithm.
    cluster = id_set(args.cluster_csv)
    frame = pd.read_csv(args.latent_csv, dtype='string')
    ids = read_ids(frame, args.latent_csv)
    if 'n_latents_exceed_p95' not in frame:
        raise ValueError('Latent CSV needs n_latents_exceed_p95 to reproduce overlap.py')
    latent = set(ids[pd.to_numeric(frame.n_latents_exceed_p95, errors='coerce') > 0])
    # The union of old/new selected populations lets us track new entrants
    # and old targets that exit all sets, without dropping either.
    universe = old_emission | new_ids | cluster | latent
    restriction = None
    if args.universe_csv:
        universe &= id_set(args.universe_csv)
        restriction = str(args.universe_csv.resolve())
    rows = []
    for target in sorted(universe):
        code = ''.join('1' if target in group else '0' for group in (old_emission, cluster, latent))
        rows.append({'TARGETID':target, 'original_category':NAMES[code], 'old_code':code})
    original = pd.DataFrame(rows, columns=['TARGETID','original_category','old_code'])
    return original, {
        'old_emission':old_meta,
        'cluster_csv':str(args.cluster_csv.resolve()),
        'cluster_rule':'All unique TARGETIDs, exactly as attached overlap.py.',
        'latent_csv':str(args.latent_csv.resolve()),
        'latent_rule':'n_latents_exceed_p95 > 0',
        'optional_universe_csv':restriction,
        'scope':'Entire old/new population union, optionally restricted to supplied universe.',
    }


def attach_new(original, passed, maxima, evidence, catalog_ids):
    table = original[~original.TARGETID.isin(ANOMALIES)].copy()
    table['present_in_candidates'] = table.TARGETID.isin(catalog_ids)
    table['new_emission'] = table.TARGETID.isin(passed)
    table['new_max_nsigma'] = table.TARGETID.map(maxima)
    table['qualifying_candidate_rows'] = table.TARGETID.map(evidence).fillna(0).astype(int)
    table['new_code'] = [reclassify(c,e) for c,e in zip(table.old_code, table.new_emission)]
    table['new_category'] = table.new_code.map(NAMES)
    table['changed'] = table.old_code != table.new_code
    table['emission_change'] = np.where(table.changed, np.where(table.new_emission,'gained','lost'),'unchanged')
    table['new_emission_status'] = np.select(
        [~table.present_in_candidates, table.new_emission, table.new_max_nsigma.isna()],
        ['absent_from_candidate_catalog', 'passes_new_cut', 'present_but_no_finite_chosen_line'],
        default='present_but_not_above_cut',
    )
    return table


def export_comparison(original, table, args, metadata, out):
    out.mkdir(parents=True, exist_ok=True)
    original[original.TARGETID.isin(ANOMALIES)].to_csv(out/'excluded_anomalies.csv', index=False)
    table.to_csv(out/'target_membership_comparison.csv', index=False)
    table[table.changed].to_csv(out/'moved_targets.csv', index=False)
    old = {c:int((table.old_code==c).sum()) for c in ORDER}
    new = {c:int((table.new_code==c).sum()) for c in ORDER}
    summary = pd.DataFrame([{
        'code':c, 'category':NAMES[c], 'old_count':old[c], 'new_count':new[c],
        'net_change':new[c]-old[c],
        'entered':int(((table.new_code==c)&table.changed).sum()),
        'left':int(((table.old_code==c)&table.changed).sum()),
    } for c in ORDER])
    summary.to_csv(out/'region_count_changes.csv', index=False)
    transitions = pd.crosstab(table.old_code,table.new_code).reindex(index=ORDER,columns=ORDER,fill_value=0)
    transitions.rename(index=NAMES,columns=NAMES).to_csv(out/'region_transition_counts.csv')
    flows = [{'old_category':NAMES[a], 'new_category':NAMES[b], 'n_targets':int(transitions.loc[a,b])}
             for a in ORDER for b in ORDER if a!=b and transitions.loc[a,b]]
    pd.DataFrame(flows,columns=['old_category','new_category','n_targets']).to_csv(out/'migration_flows.csv',index=False)
    for c in ORDER:
        pd.DataFrame({'TARGETID':sorted(table.loc[table.new_code==c,'TARGETID'])}).to_csv(
            out/f'new_{NAMES[c]}_targetids.csv', index=False)
    make_figures(old,new,table,transitions,out)
    metadata.update({
        'candidates_csv':str(args.candidates_csv.resolve()),
        'new_threshold_strictly_greater_than':args.nsigma,
        'rows_before_anomaly_exclusion':len(original),
        'anomalies_removed':len(original)-len(table),
        'n_compared':len(table), 'n_moved':int(table.changed.sum()),
        'old_union_count':int((table.old_code!='000').sum()),
        'new_union_count':int((table.new_code!='000').sum()),
        'absent_from_candidates':int((~table.present_in_candidates).sum()),
        'assumption':'Candidate catalog is complete: absence means not newly selected, not proof of successful spectral processing.',
        'geometry':'Fixed circle geometry; counts are exact, areas not proportional.',
    })
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('\n'+str(out)+'\n'+summary.to_string(index=False))


def selected_failure_report(original, table, threshold, out):
    # These are the four ORIGINAL emission-involved 100-target selections.
    rows = []
    for code in ['100','110','101','111']:
        before = original[original.old_code==code]
        tested = table[table.old_code==code]
        failed = tested[~tested.new_emission]
        rows.append({
            'original_category':NAMES[code],
            'original_selected_count':len(before),
            'anomalies_removed':len(before)-len(tested),
            'n_tested':len(tested),
            'n_pass_new_cut':int(tested.new_emission.sum()),
            'n_fail_new_selection':len(failed),
            'fail_percent':100*len(failed)/len(tested) if len(tested) else np.nan,
            'n_absent_from_candidates':int((~tested.present_in_candidates).sum()),
            'n_present_not_above_cut':int((tested.new_emission_status=='present_but_not_above_cut').sum()),
            'n_present_no_finite_chosen_line':int((tested.new_emission_status=='present_but_no_finite_chosen_line').sum()),
        })
        failed.to_csv(out/f'failed_original_{NAMES[code]}.csv',index=False)
    report = pd.DataFrame(rows)
    report.to_csv(out/'selected_emission_failure_summary.csv',index=False)
    table[(table.old_code.str[0]=='1') & ~table.new_emission].to_csv(
        out/'all_failed_original_emission_targets.csv',index=False)
    fig,ax = plt.subplots(figsize=(11,6))
    x = np.arange(len(report))
    fields = [
        ('n_absent_from_candidates','Absent from candidate catalog','#c47b64'),
        ('n_present_not_above_cut','Present; chosen lines not above cut','#7d8eac'),
        ('n_present_no_finite_chosen_line','Present; no finite chosen-line value','#c8b56f'),
    ]
    bottom = np.zeros(len(report))
    for field,label,color in fields:
        values = report[field].to_numpy()
        ax.bar(x,values,bottom=bottom,label=label,color=color)
        bottom += values
    for i,row in report.iterrows():
        label = f"{row.n_fail_new_selection}/{row.n_tested} ({row.fail_percent:.1f}%)"
        ax.text(i,row.n_fail_new_selection+.15,label,ha='center',fontsize=10)
    ax.set(xticks=x,xticklabels=report.original_category,
           ylabel='Number failing new emission selection',
           title=f'Original emission-involved selections: failures at N-sigma > {threshold:g}',
           ylim=(0,max(5,float(bottom.max())*1.25)))
    ax.legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.15))
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'selected_emission_failures.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    print('\nOriginal emission-involved selections:\n'+report.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates-csv',type=Path,required=True)
    parser.add_argument('--scope',choices=['both','population','selected'],default='both')
    parser.add_argument('--selected-dir',type=Path,default=ROOT/'category_plots')
    parser.add_argument('--selected-file',action='append',help='Optional CATEGORY=PATH; repeat for all seven files')
    parser.add_argument('--old-emission-dir',type=Path,default=ROOT/'csv')
    parser.add_argument('--old-emission-ids',type=Path,help='Saved ORIGINAL full emission-ID list; overrides old-emission-dir')
    parser.add_argument('--old-nsigma',type=float,default=None,
                        help='Old peak_n_sigma cut; omit to reproduce overlap.py default (all result IDs)')
    parser.add_argument('--cluster-csv',type=Path,default=Path('/work/10579/prisha/ls6/desi_project/post_clustering.csv'))
    parser.add_argument('--latent-csv',type=Path,default=ROOT/'latent/latent_targets_with_nonzero_p95_counts.csv')
    parser.add_argument('--universe-csv',type=Path,help='Optional common ID restriction for population only, e.g. final_subset.csv')
    parser.add_argument('--nsigma',type=float,default=3.)
    parser.add_argument('--lines',nargs='+',help='Optional chosen new-line prefixes, e.g. Lyalpha CIV MgII')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'overlap/new_nsigma_full_comparison')
    args = parser.parse_args()
    for name,value in [('nsigma',args.nsigma),('old-nsigma',args.old_nsigma)]:
        if value is not None and (not np.isfinite(value) or value<0):
            parser.error(f'--{name} must be finite and nonnegative')
    passed,maxima,evidence,catalog_ids,columns,nrows = new_emission_membership(args.candidates_csv,args.nsigma,args.lines)
    candidate_meta = {'candidate_line_columns':columns,'candidate_rows':nrows,'candidate_unique_targets':len(catalog_ids)}
    # Resolve all required inputs before producing any figures.
    work = []
    if args.scope in ('both','population'):
        original,meta = full_population(args,passed)
        work.append(('population',original,{**meta,**candidate_meta}))
    if args.scope in ('both','selected'):
        original,paths = read_original(args.selected_dir,args.selected_file)
        work.append(('selected',original,{'scope':'Original selected cohort only','selected_files':paths,**candidate_meta}))
    for scope,original,meta in work:
        table = attach_new(original,passed,maxima,evidence,catalog_ids)
        if table.empty:
            raise ValueError(f'{scope}: empty comparison after exclusions')
        out = args.out_dir/scope
        export_comparison(original,table,args,meta,out)
        if scope=='selected':
            selected_failure_report(original,table,args.nsigma,out)
    print('\nBoth scopes exclude the ten known anomalies; cluster and latent membership are fixed.')
    print('Absent from catalog means not selected, assuming complete catalog coverage.')


if __name__=='__main__':
    main()

