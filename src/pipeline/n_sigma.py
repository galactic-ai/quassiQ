#!/usr/bin/env python3
import os
for _key in ('NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import itertools
import json
import multiprocessing as mp
import re
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

C_KMS = 299792.458
LINES = {'Lyalpha': 1215.67, 'SiIV_OIV_blend': 1400.,
         'CIV':1549.48, 'CIII_1909':1908.73, 'MgII':2799.12}
# Rest-frame Angstroms: line, left continuum, right continuum (Table 1).
PAPER_WINDOWS = {
    'Lyalpha': ((1190.,1250.), (1150.,1190.), (1250.,1290.)),
    'SiIV_OIV_blend': ((1360.,1430.), (1310.,1360.), (1430.,1480.)),
    'CIV': ((1510.,1590.), (1460.,1510.), (1590.,1640.)),
    'CIII_1909': ((1850.,1970.), (1820.,1850.), (1970.,2000.)),
    'MgII': ((2750.,2850.), (2680.,2720.), (2880.,2920.)),
}
METRIC_FLOATS = ['LINE_FLUX1','LINE_FLUX2','LINE_FLUX_ERR1','LINE_FLUX_ERR2',
    'LINE_NOISE_ERR1','LINE_NOISE_ERR2','F_BRIGHT','F_DIM','F_BRIGHT_ERR','F_DIM_ERR',
    'R','SIGMA_R','F_SIGMA_DIM','LINE_COVERAGE']
METRIC_FLOATS += [f'{key}{i}' for i in (1, 2) for key in
    ('CONTINUUM_AT_PIVOT', 'CONTINUUM_SLOPE_PER_A')]
METRIC_FLOATS += ['LINE_DIAG_NOISE_ERR1', 'LINE_DIAG_NOISE_ERR2',
                  'F_BRIGHT_ERR_FOR_R', 'F_DIM_ERR_FOR_R']
METRIC_INTS = ['BRIGHT_EPOCH','DIM_EPOCH','N_INTEGRATION_PIXELS']
METRIC_INTS += [f'N_CONTINUUM_{side}{i}' for i in (1, 2) for side in ('LEFT', 'RIGHT')]
METRIC_COLS = METRIC_FLOATS + METRIC_INTS

META = ['TARGETID', 'OBS1', 'OBS2', 'NIGHT1', 'NIGHT2', 'TILE1', 'TILE2',
        'PETAL1', 'PETAL2', 'Z_USED', 'DELTA_DAYS']
PAIR_COLS = META + ['N_LINES_GT_THRESHOLD', 'LINES_GT_THRESHOLD'] + [
    col for name in LINES for col in
    (f'{name}_max_nsigma', f'{name}_signed_nsigma', f'{name}_peak_rest_A',
     f'{name}_snr1_region', f'{name}_snr2_region',
     f'{name}_n_valid_pixels', f'{name}_n_pixels_gt_threshold', f'{name}_n_pixels_gt_3')]
HIT_COLS = META + ['LINE', 'MAX_NSIGMA', 'SIGNED_NSIGMA', 'REST_WAVE_A',
                   'OBS_WAVE_A', 'FLUX1', 'FLUX2', 'SIGMA1', 'SIGMA2',
                   'N_VALID_PIXELS', 'N_PIXELS_GT_THRESHOLD', 'N_PIXELS_GT_3',
                   'SNR1_REGION', 'SNR2_REGION', 'ERROR_MODEL', 'EBV_GAL', 'RV_GAL']
HIT_COLS += METRIC_COLS
AUDIT_COLS = META + ['LINE','MAX_NSIGMA', 'EBV_GAL', 'RV_GAL'] + METRIC_COLS + [
    'PASS_R','PASS_R_SIGNIFICANCE','PASS_F_SIGMA_DIM','PASS_ALL','REJECTION_REASON']
PAIR_COLS += [f'{name}_{col}' for name in LINES for col in METRIC_COLS]
TARGET_COLS = ['TARGETID', 'N_OBSERVATIONS', 'N_LOADED', 'N_PAIRS_CATALOG',
               'N_PAIRS_COMPARED', 'N_CANDIDATE_PAIRS', 'N_LINE_HITS', 'ELAPSED_SECONDS']
ERROR_COLS = ['TARGETID', 'OBS', 'STAGE', 'ERROR']
# One row per target per line. 
LINE_TARGET_COLS = ['TARGETID', 'LINE', 'Z_USED', 'N_PASSING_PAIRS', 'MAX_NSIGMA'] + [
    'BEST_'+key for key in HIT_COLS if key not in {'TARGETID', 'LINE', 'Z_USED', 'MAX_NSIGMA'}
] + ['PASSING_PAIRS_JSON']


def available_cpus():
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        n = min(n, max(1, int(float(os.environ['SLURM_CPUS_PER_TASK']))))
    except (KeyError, ValueError):
        pass
    return n


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-csv', default='subset_cands.csv')
    p.add_argument('--coadd-dir', default='output_coadds',
                   help='Contains TARGETID/coadd-PETAL-TILE-NIGHT-TARGETID.fits')
    p.add_argument('--rest-bin-width', type=float, default=2.,
                   help='Common rest-frame bin width in Angstrom; default 2')
    p.add_argument('--rest-min', type=float, default=1100., help='First rest-frame bin edge')
    p.add_argument('--rest-max', type=float, default=3000., help='Last rest-frame bin edge')
    p.add_argument('--out-dir', default='nsigma_results')
    p.add_argument('--workers', type=int, default=available_cpus(),
                   help='Worker processes; default: all CPUs available within affinity and SLURM allocation')
    p.add_argument('--threshold', type=float, default=3.)
    p.add_argument('--half-window-kms', type=float, default=5000.,
                   help='Existing n_sigma half-window; R integrations always use fixed paper windows')
    p.add_argument('--progress-every', type=int, default=1)
    p.add_argument('--target-id')
    p.add_argument('--max-targets', type=int)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--export-lines-from', metavar='LINE_HITS_CSV',
                   help='Only export per-line target CSVs from an existing line_hits.csv; no FITS imports or spectra reads')
    p.add_argument('--r-min', type=float, default=2.)
    p.add_argument('--r-sigma-factor', type=float, default=2.)
    p.add_argument('--ebv-column', default='EBV',
                   help='Catalog column containing foreground Milky Way E(B-V) in mag; '
                        'if absent/blank, use --ebv-header from the coadd FIBERMAP or FITS headers')
    p.add_argument('--ebv-header', default='EBV',
                   help='Foreground E(B-V) fallback keyword in coadd FIBERMAP/primary header')
    p.add_argument('--rv', type=float, default=3.1,
                   help='Fitzpatrick 1999 A_V/E(B-V); default 3.1')
    a = p.parse_args()
    if (not np.isfinite([a.rest_min, a.rest_max, a.rest_bin_width]).all()
            or a.rest_min <= 0 or a.rest_max <= a.rest_min or a.rest_bin_width <= 0):
        p.error('Invalid rest-frame grid settings')
    nbins = (a.rest_max - a.rest_min) / a.rest_bin_width
    if not np.isclose(nbins, round(nbins), rtol=0, atol=1e-8) or nbins < 2:
        p.error('Rest-frame range must contain an integer number (>=2) of bins')
    if not np.isfinite(a.rv) or a.rv <= 0:
        p.error('--rv must be positive and finite')
    if not all(np.isfinite(x) and x > 0 for x in [a.r_min,a.r_sigma_factor]):
        p.error('R cut settings must be positive and finite')
    if (a.workers < 1 or a.progress_every < 1 or a.threshold < 0 or
            not np.isfinite(a.threshold) or
            not 0 < a.half_window_kms < C_KMS or
            (a.max_targets is not None and a.max_targets < 1)):
        p.error('Invalid worker, threshold, velocity window, or count setting')
    if not a.export_lines_from and a.workers > available_cpus():
        p.error(f'--workers exceeds the detected allocation of {available_cpus()} CPUs')
    return a


def exact_id(value):
    value = re.sub(r'\.0+$', '', str(value).strip())
    if not re.fullmatch(r'\d+', value):
        raise ValueError(f'Invalid TARGETID: {value!r}; retain IDs as full integer strings')
    return str(int(value))


def night_string(value):
    value = str(value).strip()
    if re.fullmatch(r'\d{8}(?:\.0+)?', value):
        return pd.to_datetime(value[:8], format='%Y%m%d').strftime('%Y%m%d')
    return pd.to_datetime(value).strftime('%Y%m%d')


def load_catalog(path):
    df = pd.read_csv(path, dtype={'TARGETID': 'string', 'LASTNIGHT': 'string'})
    required = {'TARGETID', 'LASTNIGHT', 'TILEID', 'PETAL_LOC', 'Z', 'MEDIAN_SNR'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
    snr = pd.to_numeric(df['MEDIAN_SNR'], errors='coerce')
    df = df.loc[np.isfinite(snr) & (snr > 2.)].copy()
    df['TARGETID'] = df.TARGETID.map(exact_id)
    df['NIGHT'] = df.LASTNIGHT.map(night_string)
    for key in ('TILEID', 'PETAL_LOC'):
        n = pd.to_numeric(df[key], errors='raise')
        if n.isna().any() or (n % 1 != 0).any():
            raise ValueError(f'Invalid integer {key}')
        df[key] = n.astype('int64')
    df['Z'] = pd.to_numeric(df.Z, errors='raise')
    if not (np.isfinite(df.Z) & (df.Z > -1)).all():
        raise ValueError('Invalid redshift')
    keys = ['TARGETID', 'NIGHT', 'TILEID', 'PETAL_LOC']
    df = df.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    nights = pd.to_datetime(df['NIGHT'], format='%Y%m%d')
    cadence = df.assign(_date=nights).groupby('TARGETID')['_date'].agg(['min','max','count'])
    keep = cadence.index[(cadence['count'] >= 2) & ((cadence['max']-cadence['min']).dt.days >= 7)]
    return df.loc[df['TARGETID'].isin(keep)].reset_index(drop=True)


def obs_id(row):
    return f"{row['NIGHT']}_tile{row['TILEID']}_petal{row['PETAL_LOC']}"


def line_bounds(name, center, a):
    half = center * a.half_window_kms / C_KMS
    return center - half, center + half


def initialize_worker():
    global FITS, FITZPATRICK99
    try:
        from astropy.io import fits
        from extinction import fitzpatrick99
    except ImportError as exc:
        raise ImportError('Install dependencies: python -m pip install astropy extinction') from exc
    FITS = fits
    FITZPATRICK99 = fitzpatrick99


def foreground_ebv(row, header, primary_header, a):
    """Use supplied foreground E(B-V)"""
    value = row.get(a.ebv_column)
    source = f'catalog column {a.ebv_column}'
    if value is None or pd.isna(value) or str(value).strip() == '':
        value = header.get(a.ebv_header, primary_header.get(a.ebv_header))
        source = f'FITS keyword {a.ebv_header}'
    if value is None:
        raise ValueError(f'Missing foreground E(B-V): supply catalog column '
                         f'{a.ebv_column!r} or FITS keyword {a.ebv_header!r}')
    try:
        ebv = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid foreground E(B-V) in {source}: {value!r}') from exc
    if not np.isfinite(ebv) or ebv < 0:
        raise ValueError(f'Foreground E(B-V) in {source} must be finite and >= 0')
    return ebv


def galactic_deredden(wave_observed, flux, ivar, ebv, rv):
    """Undo foreground extinction on observed wavelengths (Angstrom)."""
    if not np.isfinite([ebv, rv]).all() or ebv < 0 or rv <= 0:
        raise ValueError('Invalid E(B-V) or R_V')
    if not np.isfinite(wave_observed).all() or np.any(wave_observed <= 0):
        raise ValueError('Invalid observed wavelengths for Galactic correction')
    if np.any((wave_observed < 1150.) | (wave_observed > 60000.)):
        raise ValueError('Observed wavelengths outside F99 interval 1150--60000 Angstrom')
    if ebv == 0:
        return flux.copy(), ivar.copy()
    a_lambda = FITZPATRICK99(np.ascontiguousarray(wave_observed, dtype=float),
                            rv * ebv, r_v=rv, unit='aa')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        factor = 10. ** (0.4 * a_lambda)
        factor_squared = factor**2
    if not np.isfinite(factor_squared).all() or np.any(factor_squared <= 0):
        raise ValueError('Nonfinite/zero Galactic extinction correction factor')
    # sigma_corrected = factor * sigma; IVAR_corrected = IVAR / factor**2.
    corrected_flux = flux * factor
    corrected_ivar = ivar / factor_squared
    if np.any(np.isfinite(flux) & ~np.isfinite(corrected_flux)):
        raise ValueError('Galactic correction overflowed finite input flux')
    return corrected_flux, corrected_ivar


def rest_edges(a):
    n = int(round((a.rest_max-a.rest_min)/a.rest_bin_width))
    return a.rest_min + np.arange(n+1)*a.rest_bin_width


def rebin_band(wave_rest, flux, ivar, mask, edges):
    wave_rest = np.asarray(wave_rest, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    mask = np.asarray(mask)

    if (wave_rest.ndim != 1 or len(wave_rest) < 2 or
            flux.shape != wave_rest.shape or ivar.shape != wave_rest.shape or
            mask.shape != wave_rest.shape or not np.isfinite(wave_rest).all() or
            not (np.diff(wave_rest) > 0).all()):
        raise ValueError('Invalid coadd wavelength grid or array shapes')

    native = cell_edges(wave_rest)
    valid = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (mask == 0)

    rows, cols, values = [], [], []
    nout = len(edges) - 1

    for k in np.flatnonzero(valid):
        left, right = native[k:k+2]
        j0 = max(0, np.searchsorted(edges, left, side='right') - 1)
        j1 = min(nout - 1, np.searchsorted(edges, right, side='left') - 1)

        for j in range(j0, j1 + 1):
            overlap = min(right, edges[j+1]) - max(left, edges[j])
            if overlap > 0:
                rows.append(j)
                cols.append(k)
                values.append(overlap / (edges[j+1] - edges[j]))

    operator = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(nout, len(flux))
    )

    coverage = np.asarray(operator.sum(axis=1)).ravel()
    complete = np.isclose(coverage, 1., rtol=0, atol=1e-8)
    operator = sparse.diags(complete.astype(float)) @ operator

    input_variance = np.zeros(len(flux))
    input_variance[valid] = 1. / ivar[valid]

    rebinned = np.asarray(
        operator @ np.where(valid, flux, 0.)
    ).ravel()

    # Propagate individual-bin variances
    variance = np.asarray(
        operator.power(2) @ input_variance
    ).ravel()

    rebinned[~complete] = np.nan
    variance[~complete] = np.nan

    return rebinned, variance


def combine_rebinned_bands(bands):
    """Combine cameras using per-bin inverse-variance weights."""
    invvars = []

    for flux, variance in bands:
        good = (
            np.isfinite(flux)
            & np.isfinite(variance)
            & (variance > 0)
        )
        weight = np.zeros(len(flux))
        weight[good] = 1. / variance[good]
        invvars.append(weight)

    total = np.sum(invvars, axis=0)
    valid = total > 0
    flux_out = np.zeros(len(total))

    for (flux, variance), weight in zip(bands, invvars):
        alpha = np.zeros(len(total))
        alpha[valid] = weight[valid] / total[valid]

        safe_flux = np.where(weight > 0, flux, 0.)
        flux_out += alpha * safe_flux

    variance_out = np.full(len(total), np.nan)
    variance_out[valid] = 1. / total[valid]
    flux_out[~valid] = np.nan

    return flux_out, variance_out


def prepare(row, z, a):
    tid, night = row['TARGETID'], row['NIGHT']
    path = Path(a.coadd_dir)/tid/f"coadd-{row['PETAL_LOC']}-{row['TILEID']}-{night}-{tid}.fits"
    edges = rest_edges(a)
    with FITS.open(path, memmap=False) as hdul:
        fmap = hdul['FIBERMAP'].data
        names = set(fmap.dtype.names or ())
        if 'TARGETID' not in names:
            raise ValueError('Coadd FIBERMAP has no TARGETID')
        matches = np.flatnonzero(np.asarray(fmap['TARGETID']) == int(tid))
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one coadded spectrum for TARGETID {tid}; found {len(matches)}')
        index = int(matches[0])
        record = {key:fmap[key][index] for key in names}
        for key in ('TILEID', 'PETAL_LOC'):
            value = record.get(key, hdul[0].header.get(key))
            if value is not None and int(value)!=int(row[key]):
                raise ValueError(f'Coadd {key} does not match catalog observation')
        for key in ('NIGHT', 'LASTNIGHT'):
            if key in record and night_string(record[key])!=night:
                raise ValueError(f'Coadd {key} does not match requested night')
        ebv = foreground_ebv(row, record, hdul[0].header, a)
        extensions = {h.name.upper() for h in hdul}
        cameras = (['BRZ'] if 'BRZ_FLUX' in extensions else
                   [b for b in ('B','R','Z') if b+'_FLUX' in extensions])
        if not cameras:
            raise ValueError('No BRZ or B/R/Z flux extensions in coadd')
        bands = []
        for band in cameras:
            required = {band+suffix for suffix in ('_WAVELENGTH','_FLUX','_IVAR','_MASK')}
            if not required.issubset(extensions):
                raise ValueError(f'Missing coadd extensions: {sorted(required-extensions)}')
            wave = np.asarray(hdul[band+'_WAVELENGTH'].data, dtype=float).copy()
            flux = np.asarray(hdul[band+'_FLUX'].data[index], dtype=float).copy()
            ivar = np.asarray(hdul[band+'_IVAR'].data[index], dtype=float).copy()
            mask = np.asarray(hdul[band+'_MASK'].data[index]).copy()
            flux, ivar = galactic_deredden(wave, flux, ivar, ebv, a.rv)
            bands.append(rebin_band(wave/(1+z), flux, ivar, mask, edges))
    flux, variance = combine_rebinned_bands(bands)
    if not np.any(np.isfinite(variance) & (variance>0)):
        raise ValueError('No fully covered valid bins on common rest-frame grid')
    rest = (edges[:-1]+edges[1:])/2.
    return rest, flux, variance, ebv


def metadata(tid, one, two, z):
    days = (pd.to_datetime(two['NIGHT'], format='%Y%m%d')-pd.to_datetime(one['NIGHT'], format='%Y%m%d')).days
    return dict(zip(META, [tid, obs_id(one), obs_id(two), one['NIGHT'], two['NIGHT'],
                    one['TILEID'], two['TILEID'], one['PETAL_LOC'], two['PETAL_LOC'], z, days]))


def cell_edges(wave):
    mid = (wave[1:] + wave[:-1]) / 2.
    return np.r_[wave[0]-(wave[1]-wave[0])/2., mid,
                 wave[-1]+(wave[-1]-wave[-2])/2.]


def integrated_window_measure(rest, flux, variance, good, name):
    (lo, hi), left, right = PAPER_WINDOWS[name]
    edges = cell_edges(rest)
    if edges[0] > lo or edges[-1] < hi:
        raise ValueError('incomplete_wavelength_coverage_of_line_window')
    if edges[0] > left[0] or edges[-1] < right[1]:
        raise ValueError('incomplete_wavelength_coverage_of_continuum_windows')
    valid = (good & np.isfinite(flux) & np.isfinite(variance) & (variance > 0))
    widths = np.maximum(0., np.minimum(edges[1:], hi)-np.maximum(edges[:-1], lo))
    select = valid & (rest >= lo) & (rest <= hi) & (widths > 0)
    if np.count_nonzero(select) < 2:
        raise ValueError('fewer_than_two_pair_valid_line_pixels')
    # Do not let shared endpoints enter both the fit and the line integral.
    left_sel = valid & (rest >= left[0]) & (rest <= left[1]) & (rest < lo)
    right_sel = valid & (rest >= right[0]) & (rest <= right[1]) & (rest > hi)
    if left_sel.sum() < 2 or right_sel.sum() < 2:
        raise ValueError('fewer_than_two_pair_valid_pixels_in_a_continuum_sideband')
    side = left_sel | right_sel
    pivot = (lo + hi) / 2.
    # Centered/scaled coordinates keep the two-parameter fit well conditioned.
    x = (rest - pivot) / 100.
    design = np.column_stack((np.ones(side.sum()), x[side]))
    sigma = np.sqrt(variance[side])
    weighted_design = design / sigma[:, None]
    weighted_flux = flux[side] / sigma
    beta, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_flux, rcond=None)
    if rank < 2:
        raise ValueError('singular_continuum_fit')
    dl = widths[select]
    continuum = beta[0] + beta[1] * x[select]
    integrated = float(np.dot(dl, flux[select] - continuum))
    noise_var = float(np.dot(dl**2, variance[select]))
    if not np.isfinite([integrated, noise_var]).all() or noise_var <= 0:
        raise ValueError('invalid_continuum_subtracted_flux_or_uncertainty')
    error = float(np.sqrt(noise_var))
    return dict(flux=integrated, error=error, noise_error=error,
                diagonal_noise_error=error,
                continuum_at_pivot=float(beta[0]),
                continuum_slope_per_A=float(beta[1] / 100.),
                n_cont_left=int(left_sel.sum()), n_cont_right=int(right_sel.sum()),
                n_line=int(select.sum()), line_coverage=float(dl.sum()/(hi-lo)))


def paper_metrics(rest, f1, f2, v1, v2, good, name, a):
    out = {key:None for key in METRIC_COLS}
    out.update(PASS_R=False,PASS_R_SIGNIFICANCE=False,PASS_F_SIGMA_DIM=False,PASS_ALL=False,REJECTION_REASON='')
    try:
        first = integrated_window_measure(rest,f1,v1,good,name)
        second = integrated_window_measure(rest,f2,v2,good,name)
    except (ValueError,np.linalg.LinAlgError) as exc:
        out['REJECTION_REASON']=str(exc)
        return out
    for i,m in [(1,first),(2,second)]:
        out[f'LINE_FLUX{i}']=m['flux'];out[f'LINE_FLUX_ERR{i}']=m['error']
        out[f'LINE_NOISE_ERR{i}']=m['noise_error']
        out[f'LINE_DIAG_NOISE_ERR{i}']=m['diagonal_noise_error']
        out[f'CONTINUUM_AT_PIVOT{i}']=m['continuum_at_pivot']
        out[f'CONTINUUM_SLOPE_PER_A{i}']=m['continuum_slope_per_A']
        out[f'N_CONTINUUM_LEFT{i}']=m['n_cont_left']
        out[f'N_CONTINUUM_RIGHT{i}']=m['n_cont_right']
    out.update(N_INTEGRATION_PIXELS=first['n_line'],LINE_COVERAGE=first['line_coverage'])
    b,d=(first,second) if first['flux']>=second['flux'] else (second,first)
    bright_epoch=1 if first['flux']>=second['flux'] else 2
    out.update(BRIGHT_EPOCH=bright_epoch,DIM_EPOCH=3-bright_epoch,
               F_BRIGHT=b['flux'],F_DIM=d['flux'],F_BRIGHT_ERR=b['error'],F_DIM_ERR=d['error'],
               F_BRIGHT_ERR_FOR_R=b['diagonal_noise_error'],
               F_DIM_ERR_FOR_R=d['diagonal_noise_error'])
    out['F_SIGMA_DIM'] = float(d['flux'] / d['noise_error'])
    if d['flux']<=0:
        out['REJECTION_REASON']='nonpositive_dim_line_flux_ratio_undefined'
        return out
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        r=b['flux']/d['flux']-1.
        sigma_r=float(np.hypot(b['diagonal_noise_error']/d['flux'],
                               (b['flux']/d['flux'])*(d['diagonal_noise_error']/d['flux'])))
        dim_snr=d['flux']/d['noise_error']
    if not np.isfinite([r,sigma_r]).all():
        out['REJECTION_REASON']='nonfinite_R_or_sigma_R'
        return out
    out.update(R=float(r),SIGMA_R=sigma_r,F_SIGMA_DIM=float(dim_snr) if np.isfinite(dim_snr) else None,
               PASS_R=bool(r>a.r_min),PASS_R_SIGNIFICANCE=bool(r>a.r_sigma_factor*sigma_r),
               PASS_F_SIGMA_DIM=bool(np.isfinite(dim_snr) and dim_snr < 2.*r))
    reasons=[]
    if not out['PASS_R']:reasons.append('R_not_above_minimum')
    if not out['PASS_R_SIGNIFICANCE']:reasons.append('R_not_above_uncertainty_cut')
    if not out['PASS_F_SIGMA_DIM']:reasons.append('F_sigma_dim_not_below_2R')
    out['PASS_ALL']=not reasons;out['REJECTION_REASON']=';'.join(reasons)
    return out


def line_hits(one, two, z, a):
    rest1,f1,v1,ebv1=one;rest2,f2,v2,ebv2=two
    # Foreground dust is the same for repeat observations of one target.
    if not np.isclose(ebv1, ebv2, rtol=0., atol=1e-6):
        raise ValueError('Different foreground E(B-V) for the same target; '
                         'supply a common target EBV in the catalog')
    if rest1.shape!=rest2.shape or not np.allclose(rest1,rest2,rtol=0.,atol=1e-8):
        raise ValueError('Prepared rest-frame bin grids differ')
    good=np.isfinite(f1)&np.isfinite(f2)&np.isfinite(v1)&np.isfinite(v2)&(v1>0)&(v2>0)
    if not np.any(good):return [],False,[]
    rest=rest1
    signed=np.full(len(rest),np.nan)
    signed[good]=(f2[good]-f1[good])/np.sqrt(v1[good]+v2[good])
    hits,audits=[],[]
    for name,center in LINES.items():
        lo,hi=line_bounds(name,center,a)
        select=np.flatnonzero(good&(rest>=lo)&(rest<=hi))
        if not len(select):continue
        k=select[np.argmax(np.abs(signed[select]))]
        score=float(abs(signed[k]))
        if not score>a.threshold:continue
        # After the n_sigma cut: fit/subtract local continua and integrate line excess.
        metrics=paper_metrics(rest,f1,f2,v1,v2,good,name,a)
        audits.append(dict(LINE=name,MAX_NSIGMA=score,EBV_GAL=ebv1,RV_GAL=a.rv,**metrics))
        if not metrics['PASS_ALL']:continue
        # Diagnostic only; no cut on median region SNR.
        snr1=float(np.median(f1[select]/np.sqrt(v1[select])))
        snr2=float(np.median(f2[select]/np.sqrt(v2[select])))
        hit=dict(LINE=name,MAX_NSIGMA=score,SIGNED_NSIGMA=float(signed[k]),
            REST_WAVE_A=float(rest[k]),OBS_WAVE_A=float(rest[k]*(1+z)),
            FLUX1=float(f1[k]),FLUX2=float(f2[k]),SIGMA1=float(np.sqrt(v1[k])),SIGMA2=float(np.sqrt(v2[k])),
            N_VALID_PIXELS=len(select),N_PIXELS_GT_THRESHOLD=int(np.sum(np.abs(signed[select])>a.threshold)),
            N_PIXELS_GT_3=int(np.sum(np.abs(signed[select])>3.)),SNR1_REGION=snr1,SNR2_REGION=snr2,
            EBV_GAL=ebv1,RV_GAL=a.rv,
            ERROR_MODEL='original_coadd_IVAR;overlap_rebin_diagonal_variance;F99_Galactic_dereddened_flux_and_noise;local_linear_continuum_subtracted;fixed_continuum;independent_rebinned_bins')
        hit.update({key:metrics[key] for key in METRIC_COLS})
        hits.append(hit)
    return hits,True,audits


def process_target(task):
    tid, rows, a = task
    start = time.perf_counter()
    z = float(np.median([row['Z'] for row in rows]))
    loaded, errors, candidates, hits_out, audit_out = {}, [], [], [], []
    for row in rows:
        try:
            loaded[obs_id(row)] = prepare(row, z, a)
        except Exception as exc:
            errors.append(dict(TARGETID=tid, OBS=obs_id(row), STAGE='read_or_prepare', ERROR=str(exc)))
    compared = 0
    for one, two in itertools.combinations(rows, 2):
        first, second = loaded.get(obs_id(one)), loaded.get(obs_id(two))
        if first is None or second is None:
            continue
        try:
            hits, usable, audits = line_hits(first, second, z, a)
        except Exception as exc:
            errors.append(dict(TARGETID=tid, OBS=f'{obs_id(one)} vs {obs_id(two)}',
                               STAGE='compare', ERROR=str(exc)))
            continue
        compared += int(usable)
        meta = metadata(tid, one, two, z)
        audit_out.extend(dict(meta, **record) for record in audits)
        if not hits:
            continue
        meta = metadata(tid, one, two, z)
        pair = dict(meta, N_LINES_GT_THRESHOLD=len(hits), LINES_GT_THRESHOLD=';'.join(h['LINE'] for h in hits))
        for hit in hits:
            name = hit['LINE']
            pair[f'{name}_max_nsigma'] = hit['MAX_NSIGMA']
            pair[f'{name}_signed_nsigma'] = hit['SIGNED_NSIGMA']
            pair[f'{name}_peak_rest_A'] = hit['REST_WAVE_A']
            for suffix, key in (('snr1_region', 'SNR1_REGION'),
                                ('snr2_region', 'SNR2_REGION'),
                                ('n_valid_pixels', 'N_VALID_PIXELS'),
                                ('n_pixels_gt_threshold', 'N_PIXELS_GT_THRESHOLD'),
                                ('n_pixels_gt_3', 'N_PIXELS_GT_3')):
                pair[f'{name}_{suffix}'] = hit[key]
            for key in METRIC_COLS:
                pair[f'{name}_{key}'] = hit[key]
            hits_out.append(dict(meta, **hit))
        candidates.append(pair)
    stats = dict(TARGETID=tid, N_OBSERVATIONS=len(rows), N_LOADED=len(loaded),
        N_PAIRS_CATALOG=len(rows)*(len(rows)-1)//2, N_PAIRS_COMPARED=compared,
        N_CANDIDATE_PAIRS=len(candidates), N_LINE_HITS=len(hits_out),
        ELAPSED_SECONDS=round(time.perf_counter()-start, 3))
    return stats, candidates, hits_out, errors, audit_out


def parallel_results(tasks, a):
    if a.workers == 1:
        for task in tasks:
            yield process_target(task)
        return
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker) as pool:
        pending = {}
        def submit_one():
            try:
                task = next(tasks)
            except StopIteration:
                return False
            pending[pool.submit(process_target, task)] = task[0]
            return True
        for _ in range(a.workers*2):
            if not submit_one():
                break
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                yield future.result()
                submit_one()



def line_target_rows(line_rows):
    """Group a completed target's passing pairs by line"""
    grouped = {}
    for hit in line_rows:
        grouped.setdefault((str(hit['TARGETID']), hit['LINE']), []).append(hit)
    result = []
    for (tid, line), records in grouped.items():
        best = max(records, key=lambda record: float(record['MAX_NSIGMA']))
        row = dict(TARGETID=tid, LINE=line, Z_USED=best['Z_USED'],
                   N_PASSING_PAIRS=len(records), MAX_NSIGMA=best['MAX_NSIGMA'])
        for key in HIT_COLS:
            if key not in {'TARGETID', 'LINE', 'Z_USED', 'MAX_NSIGMA'}:
                row['BEST_'+key] = best[key]
        row['PASSING_PAIRS_JSON'] = json.dumps(records, separators=(',', ':'), allow_nan=False)
        result.append(row)
    return result


def export_line_csvs(source, out_dir, overwrite=False):
    out = Path(out_dir)/'per_line'
    paths = [out/f'{name}.csv' for name in LINES]
    if not overwrite and any(path.exists() for path in paths):
        raise SystemExit('Per-line outputs already exist; choose a new --out-dir or explicitly --overwrite.')
    numeric_float = {'Z_USED', 'MAX_NSIGMA', 'SIGNED_NSIGMA', 'REST_WAVE_A',
                     'OBS_WAVE_A', 'FLUX1', 'FLUX2', 'SIGMA1', 'SIGMA2',
                     'SNR1_REGION', 'SNR2_REGION', 'EBV_GAL', 'RV_GAL'}
    numeric_int = {'TILE1', 'TILE2', 'PETAL1', 'PETAL2', 'DELTA_DAYS',
                   'N_VALID_PIXELS', 'N_PIXELS_GT_THRESHOLD', 'N_PIXELS_GT_3'}
    numeric_float.update(METRIC_FLOATS)
    numeric_int.update(METRIC_INTS)
    source = Path(source)
    with source.open(newline='') as fh, ExitStack() as stack:
        reader = csv.DictReader(fh)
        if not set(HIT_COLS).issubset(reader.fieldnames or []):
            raise SystemExit('Input must be a pipeline line_hits.csv containing all required hit columns.')
        out.mkdir(parents=True, exist_ok=True)
        writers = {}
        for name, path in zip(LINES, paths):
            stream = stack.enter_context(path.open('w', newline=''))
            writer = csv.DictWriter(stream, fieldnames=LINE_TARGET_COLS)
            writer.writeheader()
            writers[name] = writer
        done, seen, current, block = 0, set(), None, []
        counts = dict.fromkeys(LINES, 0)
        def emit(records):
            for row in line_target_rows(records):
                writers[row['LINE']].writerow(row)
                counts[row['LINE']] += 1
        for raw in reader:
            tid = exact_id(raw['TARGETID'])
            if current is not None and tid != current:
                emit(block)
                seen.add(current)
                done += 1
                block = []
            if tid != current and tid in seen:
                raise SystemExit('Source target rows must be contiguous. Sort the source by TARGETID before export.')
            current = tid
            record = {key: raw[key] for key in HIT_COLS}
            record['TARGETID'] = tid
            if record['LINE'] not in writers:
                raise SystemExit(f"Unrecognized line in input: {record['LINE']}")
            for key in numeric_float:
                record[key] = float(record[key])
            for key in numeric_int:
                record[key] = int(record[key])
            block.append(record)
        if block:
            emit(block)
            done += 1
    print(f'Exported {done:,} candidate targets to {out.resolve()}', flush=True)
    for line, count in counts.items():
        print(f'  {line}: {count:,} targets', flush=True)


def main():
    a = arguments()
    if a.export_lines_from:
        export_line_csvs(a.export_lines_from, a.out_dir, a.overwrite)
        return
    df = load_catalog(a.input_csv)
    if a.target_id:
        df = df[df.TARGETID == exact_id(a.target_id)]
    if a.max_targets:
        df = df[df.TARGETID.isin(df.TARGETID.drop_duplicates().iloc[:a.max_targets])]
    if df.empty:
        raise SystemExit('No matching observations')
    initialize_worker()
    out = Path(a.out_dir)
    files = ['nsigma_passed_line_metrics.csv', 'candidates.csv', 'line_hits.csv', 'target_summary.csv', 'errors.csv', 'run_info.json']
    files += [f'per_line/{name}.csv' for name in LINES]
    if not a.overwrite and any((out/name).exists() for name in files):
        raise SystemExit('Output already exists; use a new --out-dir or explicitly --overwrite. No resume is performed.')
    out.mkdir(parents=True, exist_ok=True)
    (out/'per_line').mkdir(exist_ok=True)
    line_target_counts = dict.fromkeys(LINES, 0)
    total_targets = int(df.TARGETID.nunique())
    sizes = df.groupby('TARGETID', sort=False).size()
    total_pairs = sum(int(n)*(int(n)-1)//2 for n in sizes)
    tasks = ((tid, group.to_dict('records'), a) for tid, group in df.groupby('TARGETID', sort=False))
    done = pairs = compared = candidate_pairs = candidate_targets = hits = failed = loaded = 0
    start = time.perf_counter()
    print(f'Starting {total_targets:,} targets, {total_pairs:,} pairs, {a.workers} workers | '
          f'original coadd FLUX + pre-SpenderQ IVAR | {a.rest_bin_width:g} A rest-frame bins', flush=True)
    with ExitStack() as stack:
        streams = []
        def writer(name, columns):
            fh = stack.enter_context(open(out/name, 'w', newline=''))
            streams.append(fh)
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
            w.writeheader()
            return w
        pw, hw = writer('candidates.csv', PAIR_COLS), writer('line_hits.csv', HIT_COLS)
        tw, ew = writer('target_summary.csv', TARGET_COLS), writer('errors.csv', ERROR_COLS)
        aw = writer('nsigma_passed_line_metrics.csv', AUDIT_COLS)
        per_line_writers = {name: writer(f'per_line/{name}.csv', LINE_TARGET_COLS) for name in LINES}
        for stats, cand, line_rows, errors, audit_rows in parallel_results(iter(tasks), a):
            pw.writerows(cand); hw.writerows(line_rows); tw.writerow(stats); ew.writerows(errors)
            aw.writerows(audit_rows)
            for record in line_target_rows(line_rows):
                per_line_writers[record['LINE']].writerow(record)
                line_target_counts[record['LINE']] += 1
            done += 1
            pairs += stats['N_PAIRS_CATALOG']; compared += stats['N_PAIRS_COMPARED']
            loaded += stats['N_LOADED']; failed += len(errors)
            candidate_pairs += len(cand); candidate_targets += int(bool(cand)); hits += len(line_rows)
            if done % a.progress_every == 0 or done == total_targets:
                elapsed = time.perf_counter()-start
                eta = elapsed/done*(total_targets-done)
                print(f'[{done:,}/{total_targets:,} targets | {100*done/total_targets:.1f}%] '
                      f'Pairs checked {pairs:,}/{total_pairs:,} ({compared:,} usable) | '
                      f'Candidates {candidate_targets:,} targets / {candidate_pairs:,} pairs / {hits:,} line hits | '
                      f'Read errors {failed:,} | Elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min', flush=True)
                for fh in streams:
                    fh.flush()
    summary = dict(
    arguments=vars(a),
    total_targets=done,
    pairs_compared=compared,
    candidate_targets=candidate_targets,
    candidate_pairs=candidate_pairs,
    line_hits=hits,
    observation_errors=failed,
    sigma_R_model="Independent rebinned-bin variances; fixed fitted continuum; used for all line errors, SIGMA_R and F_SIGMA_DIM",
    f_sigma_dim_cut="F_SIGMA_DIM < 2*R",
)
    (out/'run_info.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Done: {done:,} targets | {candidate_targets:,} candidate targets | {candidate_pairs:,} candidate pairs | '
          f'{hits:,} line hits. Results: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
