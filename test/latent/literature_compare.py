#!/usr/bin/env python3

# DOES NOT WORK WELL
"""Match Chen+2026 CL quasar coordinates to DESI and retain every available latent row.

The Chen+2026 coordinates are embedded below (no download needed).
Requires numpy, pandas, astropy (and scipy for sky matching).
No latent, redshift, or final_subset cuts are applied.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u

CATALOG = '/work/11161/kanyuni/ls6/quassiQ_project/QSO_iron/iron/QSO_cat_iron_cumulative_v0.fits'
LATENTS = '/work/10579/prisha/ls6/desi_project/latent_info.csv'
OUT_DIR = '/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/latent/literature_compare/chen2026'

# Chen+2026 (ApJS 282, 28) table "CL quasar candidates identified in this work"
# (file apjsae23cbt5_mrt.txt). Columns: SDSS name, RAdeg, DEdeg, z, MJD-SDSS, MJD-DESI, Trans
CHEN_TEXT = """
J000908.69+303733.6   2.2862 30.626  2.3471 56566 59475.43985 on
J002245.89+061729.0   5.6912  6.2914 1.4869 58396 59525.21465 on
J002829.91-005055.4   7.1246 -0.8487 2.361  55447 59492.30014 off
J003111.32-001121.2   7.7972 -0.1892 1.5139 58045 59494.29318 on
J003646.67+070100.0   9.1944  7.0167 1.5224 58398 59498.29944 on
J003827.88+314744.2   9.6162 31.7956 1.7597 57364 59531.15362 off
J004108.69+223401.7  10.2862 22.5672 1.0473 56949 59554.14933 on
J004537.42+061652.6  11.4059  6.2813 2.465  55888 59532.54183 on
J004830.19+010622.6  12.1258  1.1063 1.6745 57278 59484.34243 off
J010345.18-000711.7  15.9382 -0.1199 2.3171 55475 59493.08173 on
J010751.35-015349.2  16.9639 -1.897  1.23   56980 59494.33008 on
J010912.80+193423.0  17.3033 19.5731 1.5828 57006 59553.17469 off
J010924.59+183558.1  17.3524 18.5995 3.6349 56987 59553.17469 off
J010958.66+183143.3  17.4944 18.5287 2.172  55894 59537.21584 off
J011031.71-001609.5  17.6321 -0.2693 1.7482 57279 59493.33265 off
J011105.27+070256.1  17.772   7.0489 1.6224 58436 59527.24939 off
J011955.69+030428.7  19.9821  3.0746 2.944  55855 59498.33583 on
J012204.04-014800.8  20.5168 -1.8002 2.3518 55532 59593.09318 on
J012342.48+161544.6  20.927  16.2624 2.2608 58507 59520.28381 on
J012448.56+012727.6  21.2023  1.4577 2.568  56989 59489.3646  on
J012505.74+011353.0  21.2739  1.2314 1.6284 56989 59489.3646  on
J012526.31+012318.3  21.3596  1.3884 1.6658 56989 59489.3646  on
J012542.60+012410.5  21.4275  1.4029 1.8894 56989 59489.3646  on
J013047.73+003548.0  22.6989  0.5967 1.6409 56989 59489.3646  on
J013114.44-004127.2  22.8102 -0.6909 1.3975 57373 59593.09318 on
J013119.23+030319.6  22.8301  3.0554 2.2884 57700 59496.35895 on
J013321.36+145354.0  23.339  14.8983 1.163  58523 59549.21008 off
J013528.21-010540.9  23.8676 -1.0947 1.3593 58125 59495.34671 on
J014414.22+144852.5  26.0592 14.8146 2.3788 55835 59549.2222  on
J014758.26+124403.2  26.9928 12.7342 1.2889 58437 59521.29207 off
J014848.56+145729.3  27.2023 14.9581 2.1323 55592 59549.2222  off
J015029.18+124430.1  27.6216 12.7417 2.1489 55587 59537.7539  off
J015208.92+012301.2  28.0372  1.3837 1.6297 58108 59490.37072 on
J015301.18+115545.1  28.2549 11.9292 1.3877 58437 59521.29207 off
J015403.61+264402.7  28.515  26.7341 1.355  57046 59586.11711 on
J022324.82-002504.6  35.8534 -0.418  2.9657 55179 59535.78706 on
J022655.44+005555.5  36.731   0.9321 2.1787 55241 59535.78706 on
J022719.44-002937.7  36.831  -0.4938 2.1852 56267 59488.40757 off
J022939.15+005123.6  37.4131  0.8565 1.2575 56577 59583.16655 off
J023012.83-001258.3  37.5535 -0.2162 1.1638 56274 59583.16655 off
J023200.68-010859.9  38.0028 -1.15   2.7901 55209 59485.44439 on
J023514.48-001658.7  38.8103 -0.283  1.494  56274 59488.4217  on
J023629.51+001754.9  39.1229  0.2986 2.2572 55244 59490.4148  off
J023814.15+001941.4  39.559   0.3282 1.6039 56576 59488.42171 on
J024056.50+002912.4  40.2354  0.4868 2.1837 55247 59488.42171 off
J024738.75+005709.8  41.9115  0.9527 1.2371 56602 59489.436   on
J025011.14+033215.7  42.5464  3.5377 2.7635 55481 59495.91374 on
J025306.21-012642.8  43.2759 -1.4452 1.5294 56978 59489.436   off
J033154.28-003650.2  52.9762 -0.6139 3.1184 52934 59535.33237 off
J074247.95+184249.8 115.6998 18.7138 1.4337 58515 59253.33393 on
J074643.67+201954.1 116.682  20.3317 1.4015 58508 59497.48543 on
J080031.32+320259.3 120.1305 32.0498 2.0246 54537 59519.44299 on
J080847.76+432621.3 122.199  43.4393 1.8985 56330 59521.44    on
J081734.79+280029.6 124.395  28.0082 1.6674 58133 59553.36479 on
J081946.37+265551.4 124.9432 26.9309 1.4605 58433 59522.44477 on
J082911.67+431933.3 127.2986 43.3259 2.149  57063 59631.16068 on
J082944.65+173105.1 127.4361 17.5181 2.6933 55545 59542.39781 on
J084554.37+403800.1 131.4765 40.6334 2.1851 57401 59631.17885 off
J090908.63-020612.9 137.2859 -2.1036 3.1701 55240 59608.12851 on
J091209.23+392824.7 138.0385 39.4735 1.7405 57447 59557.40256 on
J091238.78+332429.1 138.1616 33.4081 1.4768 54550 59257.19334 on
J091246.40-012317.9 138.1933 -1.3883 0.9545 59292 59622.22202 on
J091329.24+324603.0 138.3718 32.7675 1.0605 54550 59257.19334 on
J092838.70+113446.7 142.1612 11.5796 2.4303 55956 59554.43852 on
J093102.90+154146.4 142.7621 15.6962 2.1723 55983 59586.34428 on
J094354.80+075245.7 145.9783  7.8794 1.8376 55953 59551.45592 off
J095153.27+425805.4 147.972  42.9682 2.1202 57062 59547.47713 off
J095614.67+393107.5 149.0611 39.5187 1.7695 52765 59588.34794 off
J102012.73+475227.3 155.053  47.8743 1.0429 56769 59547.50524 off
J102128.34+281227.9 155.3681 28.2078 2.1722 58456 59548.51217 on
J102702.96+292620.1 156.7623 29.4389 0.9711 58456 59548.51217 off
J111404.06+255006.4 168.5169 25.8351 1.5594 56336 59690.16701 on
J111535.25-010134.4 168.8969 -1.0262 2.3419 55588 59625.74664 off
J113550.49-023741.7 173.9604 -2.6283 1.1445 52282 59648.30356 on
J114601.80+512630.7 176.5075 51.4419 2.3068 56416 59308.26451 on
J115149.48+135219.9 177.9561 13.8722 2.2731 56008 59589.47367 on
J115639.60+000031.4 179.165   0.0087 2.3595 55278 59307.27511 off
J115658.11+023727.7 179.2421  2.6244 1.2164 52051 59611.41702 off
J120242.56+035707.5 180.6773  3.9521 2.5865 55652 59653.64055 on
J120814.35+344859.8 182.0598 34.8166 0.949  58161 59666.27985 off
J122319.81+054814.2 185.8325  5.8039 2.3827 55679 59696.21023 on
J123218.94+584026.5 188.0789 58.674  1.7955 58191 59307.3128  off
J123301.27+570100.9 188.2553 57.0169 1.0951 56426 59307.3128  off
J124416.33+573806.3 191.068  57.6351 2.602  56443 59307.3128  off
J124603.85+451926.8 191.516  45.3241 3.0279 56370 59621.44347 off
J124617.11+544701.2 191.5713 54.7837 1.5539 56420 59309.27937 off
J124931.53+364816.3 192.3814 36.8045 1.4287 57481 59351.17429 on
J125418.09+371116.1 193.5754 37.1878 1.5205 53772 59728.17066 on
J125633.76+453410.2 194.1407 45.5695 2.1564 56401 59621.44347 off
J131532.92+013409.1 198.8872  1.5692 2.3637 55321 59641.66547 off
J132409.56+333251.0 201.0398 33.5475 1.8841 58514 59350.21336 on
J140216.40+385322.1 210.5683 38.8895 2.2353 58526 59672.36665 on
J141226.78+524120.3 213.1116 52.689  2.1919 56747 59280.83758 off
J141235.35+524316.7 213.1473 52.7213 2.3556 56751 59319.14964 off
J141354.84+530024.6 213.4785 53.0068 1.6117 57805 59311.27572 off
J141439.11+345257.0 213.6629 34.8825 1.4463 58220 59656.40607 off
J141634.37+515849.3 214.1432 51.9804 0.9938 56448 59311.5215  off
J141807.62+534204.5 214.5318 53.7012 3.0147 57817 59323.43114 off
J141922.97+534135.5 214.8457 53.6932 1.3201 57817 59332.68328 on
J141959.94+541255.3 214.9997 54.2154 2.3997 57805 59345.28204 off
J142242.15+533252.0 215.6756 53.5478 1.9947 56722 59313.43711 on
J142355.50+383150.5 215.9813 38.5307 1.2079 53084 59698.32479 off
J142405.10+533206.3 216.0212 53.5351 1.7378 57805 59312.60264 on
J142832.66+531203.3 217.1361 53.2009 1.3552 56449 59314.25563 off
J143122.78-010856.0 217.8449 -1.1489 3.2625 52404 59586.96039 on
J144104.51+370755.3 220.2688 37.132  2.2874 56046 59681.88575 off
J144728.21+411854.1 221.8675 41.315  2.1356 58161 59363.22055 on
J145521.14+332155.2 223.8381 33.3653 1.1508 55273 59648.97818 off
J150649.16+365634.5 226.7048 36.9429 1.3532 58501 59691.38598 off
J152005.44+400318.9 230.0227 40.0552 2.7145 56067 59541.30293 on
J152926.71+102312.3 232.3613 10.3868 1.9496 56009 59393.22292 off
J153752.75+164057.5 234.4698 16.6826 2.5041 55335 59374.28446 off
J155647.00+364205.9 239.1958 36.7016 1.4725 58254 59375.29122 off
J155728.04+342846.2 239.3668 34.4795 1.688  58255 59350.37492 off
J155806.32+443136.5 239.5263 44.5268 2.6846 56069 59317.38011 on
J163357.16+442729.3 248.4882 44.4581 1.4826 58287 59729.36139 on
J163626.64+393039.7 249.111  39.511  3.116  58248 59709.4128  on
J164222.63+310058.4 250.5943 31.0162 2.3581 55832 59533.42043 on
J164623.36+315010.7 251.5973 31.8363 2.7185 58257 59587.10035 off
J165156.94+322420.5 252.9872 32.4057 1.598  58280 59700.96982 on
J165913.68+323817.4 254.807  32.6382 1.4954 58397 59374.37291 on
J170835.33+303355.8 257.1472 30.5655 1.4344 58526 59710.46002 on
J214102.81-001946.1 325.2617 -0.3295 2.216  55478 59382.45074 on
J214515.42-002624.7 326.3142 -0.4402 2.36   57665 59488.83244 off
J224321.62+175545.7 340.8401 17.9294 2.1666 55855 59522.13447 off
J224427.53+205751.9 341.1147 20.9644 1.9871 56181 59523.1223  off
J225048.45+011443.4 342.7019  1.2454 1.6452 55471 59490.21678 on
J225318.39-001647.9 343.3266 -0.28   1.4371 57687 59490.21678 on
J232116.92+005157.0 350.3205  0.8658 1.5807 51816 59493.24667 off
J232128.00+273509.5 350.3667 27.586  1.6925 58017 59530.14221 off
J232816.17+013338.3 352.0674  1.5606 1.4377 58049 59569.05912 on
J233736.70+080318.0 354.4029  8.055  2.1721 56208 59548.10231 on
J233944.57+001016.0 354.9357  0.1711 1.4714 58044 59490.26196 off
J234127.27+011114.9 355.3636  1.1875 1.58   52591 59489.27294 on
J235134.68+015543.3 357.8945  1.9287 1.3802 58081 59490.27717 on
"""

SOURCE_LABEL = 'Chen+2026 CL quasar table'


def pick_column(frame, explicit, candidates):
    if explicit:
        if explicit not in frame:
            raise ValueError(f'Column {explicit!r} missing. Available: {list(frame.columns)}')
        return explicit
    for col in candidates:
        if col in frame:
            return col
    raise ValueError(f'Expected one of {candidates}; available: {list(frame.columns)}')


def embedded_chen():
    """Parse the embedded table text into a DataFrame."""
    rows = []
    for line in CHEN_TEXT.strip().splitlines():
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f'Bad embedded row (expected 7 fields): {line!r}')
        rows.append(parts)
    raw = pd.DataFrame(rows, columns=['SDSS', 'RAdeg', 'DEdeg', 'z', 'MJD-SDSS', 'MJD-DESI', 'Trans'])
    return raw


def load_chen(args):
    if args.chen_table:
        # Optional override: your own CSV / FITS / VOTable
        path = Path(args.chen_table)
        if path.suffix.lower() == '.csv':
            raw = pd.read_csv(path, dtype=str)
        else:
            kwargs = {'format': args.table_format} if args.table_format else {}
            raw = Table.read(path, **kwargs).to_pandas()
    else:
        raw = embedded_chen()
    name = pick_column(raw, args.name_col, ['SDSS', 'SDSS_NAME', 'name'])
    ra = pick_column(raw, args.ra_col, ['RAJ2000', 'RAdeg', 'R.A.', 'RA', 'ra'])
    dec = pick_column(raw, args.dec_col, ['DEJ2000', 'DEdeg', 'Dec.', 'DEC', 'dec'])
    chen = pd.DataFrame({
        'name': raw[name].map(lambda v: v.decode().strip() if isinstance(v, bytes) else str(v).strip()),
        'ra': pd.to_numeric(raw[ra], errors='raise'),
        'dec': pd.to_numeric(raw[dec], errors='raise'),
    })
    if chen['name'].nunique() != len(chen):
        raise ValueError('Duplicate object names in the Chen table.')
    if not (np.isfinite(chen[['ra', 'dec']]).all().all()
            and chen.ra.between(0, 360).all() and chen.dec.between(-90, 90).all()):
        raise ValueError('Chen coordinates must be finite decimal degrees.')
    chen['source'] = SOURCE_LABEL
    chen.index.name = 'chen_row'
    return chen.reset_index()


def match_catalog(chen, catalog_path, max_sep):
    table = Table.read(catalog_path)
    cat = pd.DataFrame({
        'TARGETID': np.asarray(table['TARGETID'], dtype=np.int64),
        'desi_ra': np.asarray(table['TARGET_RA'], dtype=float),
        'desi_dec': np.asarray(table['TARGET_DEC'], dtype=float),
    })
    if 'Z' in table.colnames:
        cat['Z_desi'] = np.asarray(table['Z'], dtype=float)
    valid = (np.isfinite(cat.desi_ra) & np.isfinite(cat.desi_dec)
             & cat.desi_ra.between(0, 360) & cat.desi_dec.between(-90, 90))
    print(f'Catalog rows with invalid coordinates excluded: {(~valid).sum()}')
    cat = cat.loc[valid].copy()
    if cat.empty:
        raise ValueError('DESI catalog has no valid coordinates.')
    # Keep coordinate variants until matching, then retain one association per ID.
    cc = SkyCoord(chen.ra.to_numpy() * u.deg, chen.dec.to_numpy() * u.deg)
    dc = SkyCoord(cat.desi_ra.to_numpy() * u.deg, cat.desi_dec.to_numpy() * u.deg)
    lit_i, cat_i, sep, _ = search_around_sky(cc, dc, max_sep * u.arcsec)
    pairs = cat.iloc[cat_i].reset_index(drop=True)
    pairs['chen_row'] = lit_i
    pairs['sep_arcsec'] = sep.arcsec
    pairs = pairs.loc[pairs.sep_arcsec < max_sep].sort_values('sep_arcsec')
    pairs = pairs.drop_duplicates(['chen_row', 'TARGETID'])
    pairs['TARGETID'] = pairs['TARGETID'].astype('Int64')
    report = chen.merge(pairs, on='chen_row', how='left', validate='one_to_many')
    report['in_catalog'] = report['TARGETID'].notna()
    report['n_catalog_matches'] = report.groupby('chen_row')['TARGETID'].transform('count')
    report['ambiguous_match'] = report.n_catalog_matches > 1
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chen-table', help='Optional: your own CSV/FITS/VOTable instead of the embedded table')
    parser.add_argument('--table-format', help='Astropy format, e.g. ascii.cds (requires its ReadMe)')
    parser.add_argument('--ra-col')
    parser.add_argument('--dec-col')
    parser.add_argument('--name-col')
    parser.add_argument('--catalog', default=CATALOG)
    parser.add_argument('--latent', default=LATENTS)
    parser.add_argument('--out-dir', default=OUT_DIR)
    parser.add_argument('--max-sep', type=float, default=1.0, help='Strict separation limit in arcseconds')
    args = parser.parse_args()
    if not np.isfinite(args.max_sep) or args.max_sep <= 0:
        parser.error('--max-sep must be finite and positive')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chen = load_chen(args)
    report = match_catalog(chen, args.catalog, args.max_sep)

    # Read IDs as integers directly: floating-point conversion can corrupt DESI IDs.
    latents = pd.read_csv(args.latent, dtype={'TARGETID': 'int64', 'LASTNIGHT': 'string'})
    required = ['TARGETID', 'LASTNIGHT'] + [f'Latent{i}' for i in range(1, 11)]
    missing = set(required) - set(latents.columns)
    if missing:
        raise ValueError(f'Latent file is missing columns: {sorted(missing)}')
    counts = latents.groupby('TARGETID').size()
    nights = latents.groupby('TARGETID')['LASTNIGHT'].nunique()
    report['in_latent_file'] = report.in_catalog & report.TARGETID.isin(counts.index)
    report['n_latent_rows'] = report.TARGETID.map(counts).fillna(0).astype(int)
    report['n_unique_lastnight'] = report.TARGETID.map(nights).fillna(0).astype(int)
    hits = report.loc[report.in_latent_file]
    # Aggregate names so ambiguous literature associations cannot multiply latent rows.
    names = hits.groupby('TARGETID')['name'].agg(lambda s: ';'.join(sorted(set(s))))
    out = latents.loc[latents.TARGETID.isin(names.index)].copy()
    out['chen_name'] = out.TARGETID.map(names)
    out['literature_source'] = SOURCE_LABEL
    out = out.sort_values(['TARGETID', 'LASTNIGHT'])
    report.to_csv(out_dir / 'chen2026_match_report.csv', index=False)
    out.to_csv(out_dir / 'chen2026_matched_latents.csv', index=False)

    print(f'Literature objects: {len(chen)}')
    print(f'Literature objects matched in this DESI catalog: {report.loc[report.in_catalog, "chen_row"].nunique()}')
    print(f'Literature objects with latents: {hits.chen_row.nunique()}')
    print(f'Unique matched TARGETIDs with latents: {hits.TARGETID.nunique()}')
    print(f'Latent rows retained: {len(out)}')
    print(f'Literature objects with multiple DESI matches (review report): {report.loc[report.ambiguous_match, "chen_row"].nunique()}')
    print(f'Saved {out_dir / "chen2026_match_report.csv"}')
    print(f'Saved {out_dir / "chen2026_matched_latents.csv"}')
    print('Overlap is limited to TARGETIDs represented in the supplied coordinate catalog.')


if __name__ == '__main__':
    main()