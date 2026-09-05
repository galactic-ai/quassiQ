#!/usr/bin/env python
# coding: utf-8

from astropy.io import fits
import numpy as np
import pandas as pd

fits_file = "QSO_cat_iron_cumulative_v0.fits"

# Load FITS table
with fits.open(fits_file) as hdul:
    # prevents potential FITS byte-order/endianness issues in pandas
    data = np.ascontiguousarray(hdul[1].data)

df = pd.DataFrame(data)
print(f"Initial row count: {len(df)}")

# Quality Filtering (ZWARN & COADD_FIBERSTATUS)
df = df[df['ZWARN'].isin([0, 4])]
df = df[df['COADD_FIBERSTATUS'] == 0]
print(f"Rows after quality filtering: {len(df)}")

# Redshift Filtering (2.1 <= Z <= 3.5), keep TARGETIDs that have at least one valid observation in this range
mask_z = (df['Z'] >= 2.1) & (df['Z'] <= 3.5)
valid_z_ids = df.loc[mask_z, 'TARGETID'].unique()
df = df[df['TARGETID'].isin(valid_z_ids)]
print(f"Rows after redshift filtering: {len(df)}")

# Filter for Duplicate Observations, keep TARGETIDs that have more than one valid observation remaining
df = df[df['TARGETID'].duplicated(keep=False)]
print(f"Rows with multiple observations: {len(df)}")

# Datetime Conversion & Duration Calculation
df['LASTNIGHT'] = pd.to_datetime(df['LASTNIGHT'], format='%Y%m%d')

duration_df = df.groupby('TARGETID')['LASTNIGHT'].agg(['min', 'max'])
duration_df['duration_days'] = (duration_df['max'] - duration_df['min']).dt.days

# Merge duration back to main DataFrame
df = df.merge(duration_df['duration_days'], on='TARGETID')

# Filter by Minimum Duration (>= 30 days)
valid_duration_ids = df.loc[df['duration_days'] >= 30, 'TARGETID'].unique()
df = df[df['TARGETID'].isin(valid_duration_ids)]

# Final Sort & Export
df = df.sort_values(['duration_days', 'TARGETID'], ascending=[False, True])

output_csv = "CLQ_candidates.csv"
df.to_csv(output_csv, index=False)

print("\n Summary")
print(f"Saved final candidates to: {output_csv}")
print(f"Final remaining rows: {len(df)}")
print(f"Final unique TARGETIDs: {df['TARGETID'].nunique()}")