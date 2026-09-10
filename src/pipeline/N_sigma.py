"""
Flow
1. CLQ_candidates.py - only use target ID whose recon exist 
2. Take sigma
3. Determine two epochs out of N nights (N>=2) and highest and lowest luminosity at 4750A
"""
from astropy.io import fits
import numpy as np

from pathlib import Path
from astropy.io import fits
import numpy as np


#path
RECON_BASE = Path(
    "/work/10579/prisha/ls6/desi_project/reconstructed_spectra/"
)

CSV_PATH = Path("/work/11161/kanyuni/ls6/quassiQ_project/quassiQ/src/pipeline/CLQ_candidates.csv")


# Read TARGETID in the candidate.csv and match 
candidates = pd.read_csv(
    CSV_PATH,
    dtype={"TARGETID": str},
)

candidate_ids = set(
    candidates["TARGETID"].dropna().str.strip()
)

print(f"Number of candidate IDs: {len(candidate_ids)}")

# ==============================================================

def grab_candidate_weights():
    for target_id in candidate_ids:
        target_dir = RECON_BASE / target_id

        if not target_dir.is_dir():
            print(f"Directory not found: {target_id}")
            continue

        fits_files = list(target_dir.glob("recon_*.fits"))

        if not fits_files:
            print(f"No FITS file found: {target_id}")
            continue

        for fits_path in fits_files:
            with fits.open(fits_path) as hdul:
                data = hdul[1].data

                rest_wave = np.asarray(
                    data["REST_WAVE"]
                ).squeeze()

                orig_flux = np.asarray(
                    data["ORIG_FLUX"]
                ).squeeze()

                recon_flux = np.asarray(
                    data["RECON_FLUX"]
                ).squeeze()

                weights = np.asarray(
                    data["WEIGHTS"]
                ).squeeze()

                final_weights = np.asarray(
                    data["FINAL_WEIGHTS"]
                ).squeeze()

            yield {
                "target_id": target_id,
                "fits_path": fits_path,
                "rest_wave": rest_wave,
                "orig_flux": orig_flux,
                "recon_flux": recon_flux,
                "weights": weights,
                "final_weights": final_weights,
            }

def sigma():
    for result in grab_candidate_weights():
        target_id = result["target_id"]
        weights = result["weights"]

        sigma = np.full_like(weights, np.nan, dtype=float)
        valid = weights > 0
        sigma[valid] = 1.0 / np.sqrt(weights[valid])

def pair_epoch():
from pathlib import Path
from itertools import combinations
import pandas as pd
import re

RECON_BASE = Path(
    "/work/10579/prisha/ls6/desi_project/reconstructed_spectra/"
)

CSV_PATH = Path("CLQ_candidates.csv")


def extract_observation_date(file_path):
    """
    Extract YYYYMMDD from a reconstruction filename.

    Example:
    recon_39627559461192666_coadd-1-7852-20220604-39627559461192666.fits
                                         ^^^^^^^^
    """
    match = re.search(
        r"-(\d{8})-\d+\.(?:fits|png)$",
        file_path.name,
    )

    if match is None:
        return None

    return match.group(1)


def create_night_pairs(
    csv_path=CSV_PATH,
    recon_base=RECON_BASE,
    extension="fits",
):
    candidates = pd.read_csv(
        csv_path,
        dtype={"TARGETID": str},
    )

    candidate_ids = (
        candidates["TARGETID"]
        .dropna()
        .str.strip()
        .unique()
    )

    all_pairs = []

    for target_id in candidate_ids:
        target_dir = recon_base / target_id

        if not target_dir.is_dir():
            continue

        observations = []

        for file_path in sorted(
            target_dir.glob(f"recon_*.{extension}")
        ):
            observation_date = extract_observation_date(file_path)

            if observation_date is None:
                print(f"Could not extract date: {file_path.name}")
                continue

            observations.append({
                "date": observation_date,
                "path": file_path,
            })

        # Construct every pair, excluding files from the same night
        for obs1, obs2 in combinations(observations, 2):
            if obs1["date"] == obs2["date"]:
                continue

            # Make sure the earlier observation is first
            if obs1["date"] > obs2["date"]:
                obs1, obs2 = obs2, obs1

            all_pairs.append({
                "TARGETID": target_id,
                "date_1": obs1["date"],
                "date_2": obs2["date"],
                "path_1": str(obs1["path"]),
                "path_2": str(obs2["path"]),
            })

    pairs = pd.DataFrame(all_pairs)

    if not pairs.empty:
        pairs["date_1"] = pd.to_datetime(
            pairs["date_1"],
            format="%Y%m%d",
        )
        pairs["date_2"] = pd.to_datetime(
            pairs["date_2"],
            format="%Y%m%d",
        )

        pairs["bots_frame_days"] = (
            pairs["date_2"] - pairs["date_1"]
        ).dt.days

        pairs = pairs.sort_values(
            ["TARGETID", "date_1", "date_2"]
        ).reset_index(drop=True)

    return pairs


def N_sigma(wavelength):
    #assuming wavelength is restframe
    recon_flux (high) - recon_flux (low) / (sigma (high)^ 2 + sigma (low)^2)

def determine_highest():
    wavelength_range = (4750, 4940)
    biggest_dif = np.argmax(N_sigma in wavelength_range) # 1 pixel 

def substract
    r = N_sigma(highest_dif) - N_sigma(4850)

if substract>=3
    label as clq
    plot two highest and lowest 