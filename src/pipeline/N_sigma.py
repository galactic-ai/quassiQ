"""
Flow
1. Read CLQ_candidates_7days.csv and only use TARGETIDs with reconstructions.
2. Convert the original inverse-variance weights into sigma.
3. For each TARGETID with at least two observing nights:
   - Select the highest and lowest reconstructed flux at Ly-alpha.
   - Calculate N_sigma between those two epochs.
   - Evaluate N_sigma at 1215.67 Angstrom.
   - Label as a CLQ candidate if N_sigma(1215.67) > 3.
   - Plot the high- and low-state spectra.
"""

from pathlib import Path
import argparse
import re

from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the exact same per-coadd S/N calculation that quality_cut.py uses
# to build CLQ_candidates_7days.csv, so "SNR >= SNR_CUT" means the same thing
# in both places. fits_qc_utils.py must be importable (e.g. this script
# lives alongside it in src/pipeline/, or it's on PYTHONPATH).
from fits_qc_utils import compute_median_snr_from_original_coadd

RECON_BASE = Path(
    "/work/10579/prisha/ls6/desi_project/reconstructions/"
)

#temporarily
# RECON_BASE = Path(
#     "/work/11161/kanyuni/ls6/quassiQ_project/test_reconstructed_spectra/"
# )

COADD_BASE = Path(
    "/work/10579/prisha/ls6/desi_project/output_coadds/"
)



CSV_PATH = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/"
    "quassiQ/src/pipeline/CLQ_candidates_7days.csv"
)

OUTPUT_DIR = CSV_PATH.parent / "clq_nsigma_results"
SINGLE_OUTPUT_DIR = OUTPUT_DIR / "single_targets"
BATCH_OUTPUT_DIR = OUTPUT_DIR / "batch"
LYA_WAVE = 1215.67
LYA_WINDOW = 2.0  # Angstrom; search LYA_WAVE +/- LYA_WINDOW for the peak N_sigma
PLOT_MIN = 1100.0
PLOT_MAX = 1325.0
CLQ_THRESHOLD = 3.0
SNR_CUT = 2.0  # must match quality_cut.py's SNR_CUT so the same definition of
               # "usable coadd" is enforced when picking high/low epochs here

# --- Conservative C IV / C III] changing-look configuration ---
# Shares all loading/plotting helpers below with the Ly-alpha pipeline;
# this used to live in a separate "wrapper" script that reimplemented
# the N_sigma math and called an outdated function signature. It now
# reuses calculate_n_sigma / load_original_coadd / coarse_bin directly.
CIV_CIII_LINE_CENTERS = {
    "CIV": 1549.48,
    "CIII]": 1908.73,
}
CIV_CIII_DEFAULT_TARGET_ID = "39632986693436647"
CIV_CIII_PRIMARY_HALF_WIDTH = 10.0
CIV_CIII_CHECK_HALF_WIDTHS = (5.0, 10.0, 20.0)
CIV_CIII_NSIGMA_THRESHOLD = 3.0
CIV_CIII_MIN_VALID_FRACTION = 0.70
CIV_CIII_OUTPUT_DIR = OUTPUT_DIR / "civ_ciii"
CIV_CIII_SINGLE_OUTPUT_DIR = CIV_CIII_OUTPUT_DIR / "single_targets"
CIV_CIII_BATCH_OUTPUT_DIR = CIV_CIII_OUTPUT_DIR / "batch"

#read candidate csv
candidates = pd.read_csv(
    CSV_PATH,
    dtype={"TARGETID": str},
)

candidate_ids = set(
    candidates["TARGETID"]
    .dropna()
    .str.strip()
)

redshift_map = (
    candidates.drop_duplicates("TARGETID")
    .set_index("TARGETID")["Z"]
    .to_dict()
)

print(f"Number of candidate IDs in CSV: {len(candidate_ids)}")

def get_original_coadd_path(target_id, night):
    """
    Given a TARGETID and observing NIGHT, find the matching original
    DESI coadd file:

        output_coadds/<TARGETID>/coadd-<petal>-<tileid>-<NIGHT>-<TARGETID>.fits

    The petal/tile numbers aren't known ahead of time, so this globs
    for the NIGHT and TARGETID, which together should be unique.
    """
    target_id = str(target_id)
    night = str(night)

    coadd_dir = COADD_BASE / target_id
    pattern = f"coadd-*-*-{night}-{target_id}.fits"

    matches = sorted(coadd_dir.glob(pattern))

    if not matches:
        print(f"Original coadd not found for TARGETID={target_id}, NIGHT={night}")
        return None

    if len(matches) > 1:
        print(
            f"Multiple coadds found for TARGETID={target_id}, NIGHT={night}, "
            f"using first: {matches}"
        )

    return matches[0]


def load_original_coadd(target_id, night, redshift):
    """
    Load the B, R, and Z spectra from the original DESI coadd and
    convert the wavelength to the rest frame.
    """
    coadd_path = get_original_coadd_path(target_id, night)

    if coadd_path is None:
        return np.array([]), np.array([])

    with fits.open(coadd_path) as hdul:
        hdu_names = {hdu.name.upper(): hdu for hdu in hdul}

        # Determine which row belongs to this TARGETID.
        row_index = 0

        if "FIBERMAP" in hdu_names:
            fibermap = hdu_names["FIBERMAP"].data

            if (
                fibermap is not None
                and "TARGETID" in fibermap.names
            ):
                matches = np.where(
                    fibermap["TARGETID"].astype(str)
                    == str(target_id)
                )[0]

                if matches.size == 0:
                    print(
                        f"TARGETID {target_id} not found in "
                        f"{coadd_path}"
                    )
                    return np.array([]), np.array([])

                row_index = matches[0]

        all_wave = []
        all_flux = []

        for band in ("B", "R", "Z"):
            wave_name = f"{band}_WAVELENGTH"
            flux_name = f"{band}_FLUX"

            if (
                wave_name not in hdu_names
                or flux_name not in hdu_names
            ):
                continue

            wave = np.asarray(
                hdu_names[wave_name].data,
                dtype=float,
            ).squeeze()

            flux_data = np.asarray(
                hdu_names[flux_name].data,
                dtype=float,
            )

            if flux_data.ndim == 2:
                flux = flux_data[row_index]
            else:
                flux = flux_data.squeeze()

            wave_rest = wave / (1.0 + redshift)

            good = (
                np.isfinite(wave_rest)
                & np.isfinite(flux)
            )

            all_wave.append(wave_rest[good])
            all_flux.append(flux[good])

    if not all_wave:
        return np.array([]), np.array([])

    wave_rest = np.concatenate(all_wave)
    flux = np.concatenate(all_flux)

    order = np.argsort(wave_rest)

    return wave_rest[order], flux[order]


def coarse_bin(wave, flux, n_points=10):
    """
    Average every n_points neighboring pixels.
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)

    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]

    if wave.size == 0:
        return np.array([]), np.array([])

    number_of_bins = wave.size // n_points

    if number_of_bins == 0:
        return wave, flux

    usable_size = number_of_bins * n_points

    wave_coarse = (
        wave[:usable_size]
        .reshape(number_of_bins, n_points)
        .mean(axis=1)
    )

    flux_coarse = (
        flux[:usable_size]
        .reshape(number_of_bins, n_points)
        .mean(axis=1)
    )

    return wave_coarse, flux_coarse

def extract_observation_date(fits_path):
    """
    Extract YYYYMMDD from a filename such as:

    recon_39627968208701622_20210331.fits

    The date is the last underscore-separated 8-digit token before
    the .fits extension.
    """
    match = re.search(
        r"_(\d{8})\.fits$",
        fits_path.name,
    )

    if match is None:
        return None

    return match.group(1)


def weights_to_sigma(weights):
    """
    Convert inverse-variance weights to 1-sigma flux uncertainty.
    """
    sigma = np.full(weights.shape, np.nan, dtype=float)

    valid = np.isfinite(weights) & (weights > 0)
    sigma[valid] = 1.0 / np.sqrt(weights[valid])

    return sigma


def get_epoch_snr(target_id, night):
    """
    Median per-pixel S/N of the ORIGINAL coadd for one night, using the
    same calculation quality_cut.py/fits_qc_utils.py used to build
    CLQ_candidates.csv in the first place. Returns NaN if the coadd is
    missing or unreadable.
    """
    coadd_path = get_original_coadd_path(target_id, night)

    if coadd_path is None:
        return np.nan

    median_snr, _n_pix = compute_median_snr_from_original_coadd(
        coadd_path,
        target_id,
    )

    return median_snr


def load_epoch(fits_path, target_id):
    """
    Load one reconstructed spectral epoch.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data

        rest_wave = np.asarray(
            data["REST_WAVE"],
            dtype=float,
        ).squeeze()

        orig_flux = np.asarray(
            data["ORIG_FLUX"],
            dtype=float,
        ).squeeze()

        recon_flux = np.asarray(
            data["RECON_FLUX"],
            dtype=float,
        ).squeeze()

        # weights = np.asarray(
        #     data["WEIGHTS"],
        #     dtype=float,
        # ).squeeze()

        updated_weights = np.asarray(
            data["UPDATED_WEIGHTS"],
            dtype=float,
        ).squeeze()

    date = extract_observation_date(fits_path)
    if date is None:
        date = fits_path.parent.name

    coadd_snr = get_epoch_snr(target_id, date)

    return {
        "target_id": target_id,
        "date": date,
        "fits_path": fits_path,
        "rest_wave": rest_wave,
        "orig_flux": orig_flux,
        "recon_flux": recon_flux,
        # "weights": weights,
        "updated_weights": updated_weights,
        "sigma": weights_to_sigma(updated_weights),
        "coadd_snr": coadd_snr,
        "passes_snr_cut": bool(np.isfinite(coadd_snr) and coadd_snr >= SNR_CUT),
    }


def load_candidate_epochs(target_id):
    """
    Load all available observing nights for one TARGETID.

    Directory layout:
        RECON_BASE/<TARGETID>/<NIGHT>/recon_<TARGETID>_<NIGHT>.fits
    """
    target_dir = RECON_BASE / target_id

    if not target_dir.is_dir():
        return []

    epochs_by_date = {}

    # One level down: <TARGETID>/<NIGHT>/recon_*.fits
    for fits_path in sorted(target_dir.glob("*/recon_*.fits")):
        date = extract_observation_date(fits_path)

        if date is None:
            # Fall back to the parent (NIGHT) folder name, which is
            # the source of truth for this layout.
            date = fits_path.parent.name

        # Use one reconstruction per observing night.
        if date not in epochs_by_date:
            epochs_by_date[date] = load_epoch(
                fits_path,
                target_id,
            )

    return list(epochs_by_date.values())


def flux_at_wavelength(epoch, wavelength):
    """
    Interpolate the reconstructed flux at an exact wavelength.
    """
    wave = epoch["rest_wave"]
    flux = epoch["recon_flux"]

    valid = np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(valid) < 2:
        return np.nan

    wave_valid = wave[valid]
    flux_valid = flux[valid]
    order = np.argsort(wave_valid)
    wave_valid = wave_valid[order]
    flux_valid = flux_valid[order]

    wave_min = wave_valid[0]
    wave_max = wave_valid[-1]
    if not wave_min <= wavelength <= wave_max:
        return np.nan

    return np.interp(wavelength, wave_valid, flux_valid)


def filter_epochs_by_snr(epochs, snr_cut=SNR_CUT):
    """
    Keep only epochs whose ORIGINAL coadd has a finite median S/N >=
    snr_cut. This is what actually guarantees the high/low epochs used
    in the N_sigma calculation are good coadds -- quality_cut.py only
    guarantees a TARGETID has >=2 passing coadds SOMEWHERE, not that
    the specific high/low pair chosen here are among them.
    """
    kept = []
    dropped_info = []

    for epoch in epochs:
        snr = epoch.get("coadd_snr", np.nan)
        if np.isfinite(snr) and snr >= snr_cut:
            kept.append(epoch)
        else:
            dropped_info.append(f"{epoch['date']} (S/N={snr})")

    if dropped_info:
        print(
            f"  Dropped {len(dropped_info)} epoch(s) below S/N >= "
            f"{snr_cut}: {dropped_info}"
        )

    return kept


def select_high_low_epochs(epochs, snr_cut=SNR_CUT):
    """
    Select the epochs with the highest and lowest reconstructed
    flux at Ly-alpha (1215.67 Angstrom), restricted to epochs whose
    original coadd passes the S/N cut.
    """
    return select_high_low_epochs_at_wavelength(
        epochs, LYA_WAVE, snr_cut=snr_cut
    )


def select_high_low_epochs_at_wavelength(epochs, wavelength, snr_cut=SNR_CUT):
    """
    Generic version of select_high_low_epochs: select the epochs with
    the highest and lowest reconstructed flux at any rest-frame
    wavelength, restricted to epochs whose original coadd passes the
    S/N cut.
    """
    snr_ok_epochs = filter_epochs_by_snr(epochs, snr_cut=snr_cut)

    valid_epochs = []

    for epoch in snr_ok_epochs:
        flux_value = flux_at_wavelength(
            epoch,
            wavelength,
        )

        if np.isfinite(flux_value):
            epoch["flux_lya"] = flux_value  # kept for backward compatibility
            epoch["flux_at_selection_wave"] = flux_value
            valid_epochs.append(epoch)

    if len(valid_epochs) < 2:
        return None, None

    high_epoch = max(
        valid_epochs,
        key=lambda epoch: epoch["flux_at_selection_wave"],
    )

    low_epoch = min(
        valid_epochs,
        key=lambda epoch: epoch["flux_at_selection_wave"],
    )

    return high_epoch, low_epoch


def interpolate_to_grid(values, old_wave, new_wave):
    """
    Interpolate values onto a reference wavelength grid after removing
    padded/invalid wavelengths and sorting the input grid.
    """
    values = np.asarray(values, dtype=float)
    old_wave = np.asarray(old_wave, dtype=float)
    new_wave = np.asarray(new_wave, dtype=float)

    valid = (
        np.isfinite(old_wave)
        & np.isfinite(values)
        & (old_wave > 0)
    )

    if np.count_nonzero(valid) < 2:
        return np.full(new_wave.shape, np.nan, dtype=float)

    wave_valid = old_wave[valid]
    values_valid = values[valid]
    order = np.argsort(wave_valid)
    wave_valid = wave_valid[order]
    values_valid = values_valid[order]

    # np.interp expects an increasing grid. Remove repeated padded or
    # duplicated wavelengths after sorting.
    wave_valid, unique_indices = np.unique(
        wave_valid,
        return_index=True,
    )
    values_valid = values_valid[unique_indices]

    return np.interp(
        new_wave,
        wave_valid,
        values_valid,
        left=np.nan,
        right=np.nan,
    )


def calculate_n_sigma(high_epoch, low_epoch):
    """
    Calculate:

                      |f_high - f_low|
    N_sigma(lambda) = -----------------
                       sqrt(1/ivar_high
                            + 1/ivar_low)

    Interpolate inverse variance before converting it to uncertainty.
    Interpolating sigma arrays containing NaNs can make an otherwise
    valid N_sigma array entirely non-finite.
    """
    wave = high_epoch["rest_wave"]

    high_flux = high_epoch["recon_flux"]
    high_weights = high_epoch["updated_weights"]

    low_flux = interpolate_to_grid(
        low_epoch["recon_flux"],
        low_epoch["rest_wave"],
        wave,
    )

    #interpolating might be wrong? Might be just select wavelength that had highest N_sigma in the window
    low_weights = interpolate_to_grid(
        low_epoch["updated_weights"],
        low_epoch["rest_wave"],
        wave,
    )

    denominator = np.full(wave.shape, np.nan, dtype=float)

    valid_weights = (
        np.isfinite(high_weights)
        & np.isfinite(low_weights)
        & (high_weights > 0)
        & (low_weights > 0)
    )

    denominator[valid_weights] = np.sqrt(
        1.0 / high_weights[valid_weights]
        + 1.0 / low_weights[valid_weights]
    )

    n_sigma = np.full(wave.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(high_flux)
        & np.isfinite(low_flux)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    n_sigma[valid] = np.abs(
        high_flux[valid] - low_flux[valid]
    ) / denominator[valid]

    return {
        "wave": wave,
        "high_flux": high_flux,
        "low_flux": low_flux,
        "high_weights": high_weights,
        "low_weights": low_weights,
        "denominator": denominator,
        "n_sigma": n_sigma,
    }


def calculate_peak_n_sigma(wave, n_sigma, center, window=2.0):
    """
    Find the peak N_sigma within center +/- window (Angstrom) and use
    that as the representative N_sigma for that line, rather than
    interpolating at the exact line center. Generalizes what used to
    be Ly-alpha-only logic to any rest-frame wavelength.

    Returns (peak_n_sigma, peak_wavelength). Both are NaN if there is
    no finite N_sigma within the window.
    """
    valid = np.isfinite(wave) & np.isfinite(n_sigma)

    if not np.any(valid):
        return np.nan, np.nan

    wave_valid = wave[valid]
    n_sigma_valid = n_sigma[valid]

    in_window = (
        (wave_valid >= center - window)
        & (wave_valid <= center + window)
    )

    if not np.any(in_window):
        return np.nan, np.nan

    windowed_wave = wave_valid[in_window]
    windowed_n_sigma = n_sigma_valid[in_window]

    peak_index = np.argmax(windowed_n_sigma)

    return windowed_n_sigma[peak_index], windowed_wave[peak_index]


def calculate_lya_n_sigma(wave, n_sigma, window=LYA_WINDOW):
    """Ly-alpha-specific convenience wrapper around calculate_peak_n_sigma."""
    return calculate_peak_n_sigma(wave, n_sigma, center=LYA_WAVE, window=window)
    
def plot_candidate(
    target_id,
    high_epoch,
    low_epoch,
    wave,
    high_flux,
    low_flux,
    n_sigma,
    lya_n_sigma,
    lya_n_sigma_wave,
    output_dir,
):
    """
    Plot:

    - Original coadd, lightly
    - Original coarse-binned spectrum, dashed
    - SpenderQ reconstruction, solid
    - N_sigma in the lower panel
    """
    redshift = redshift_map.get(str(target_id))

    if redshift is None or not np.isfinite(redshift):
        print(f"Missing redshift for TARGETID {target_id}")
        return None

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    epoch_settings = [
        {
            "epoch": high_epoch,
            "label": "High",
            "color": "tab:blue",
        },
        {
            "epoch": low_epoch,
            "label": "Low",
            "color": "tab:orange",
        },
    ]

    for setting in epoch_settings:
        epoch = setting["epoch"]
        state_label = setting["label"]
        color = setting["color"]

        # Original coadd
        wave_obs, flux_obs = load_original_coadd(
            target_id=target_id,
            night=epoch["date"],
            redshift=redshift,
        )

        if wave_obs.size > 0:
            axes[0].plot(
                wave_obs,
                flux_obs,
                color=color,
                linewidth=0.4,
                alpha=0.08,
                zorder=1,
            )

            wave_coarse, flux_coarse = coarse_bin(
                wave_obs,
                flux_obs,
                n_points=10,
            )

            axes[0].plot(
                wave_coarse,
                flux_coarse,
                color=color,
                linewidth=1.8,
                linestyle="--",
                alpha=0.7,
                zorder=2,
                label=(
                    f"{state_label} original coarse: "
                    f"{epoch['date']}"
                ),
            )

        # Reconstructed spectrum
        axes[0].plot(
            epoch["rest_wave"],
            epoch["recon_flux"],
            color=color,
            linewidth=2.4,
            alpha=0.65,
            zorder=3,
            label=(
                f"{state_label} reconstruction: "
                f"{epoch['date']}"
            ),
        )

    axes[0].axvline(
        LYA_WAVE,
        color="tab:green",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label=rf"Ly$\alpha$: {LYA_WAVE:.2f} $\AA$",
    )

    axes[0].set_ylabel("Flux")
    axes[0].set_title(
        f"TARGETID {target_id} | "
        f"z={redshift:.4f} | "
        rf"$N_\sigma(\mathrm{{Ly}}\alpha, \pm{LYA_WINDOW:.1f}"
        rf"\,\AA)={lya_n_sigma:.2f}$ "
        rf"at {lya_n_sigma_wave:.2f} $\AA$"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(
        fontsize="small",
        loc="upper right",
    )

    # N_sigma panel
    axes[1].plot(
        wave,
        n_sigma,
        color="black",
        linewidth=1.5,
    )

    axes[1].axvline(
        LYA_WAVE,
        color="tab:green",
        linestyle="--",
        linewidth=1.2,
        label=rf"Ly$\alpha$: {LYA_WAVE:.2f} $\AA$",
    )

    axes[1].axvspan(
        LYA_WAVE - LYA_WINDOW,
        LYA_WAVE + LYA_WINDOW,
        color="tab:green",
        alpha=0.08,
        label=rf"search window ($\pm${LYA_WINDOW:.1f} $\AA$)",
    )

    if np.isfinite(lya_n_sigma_wave):
        axes[1].plot(
            lya_n_sigma_wave,
            lya_n_sigma,
            marker="*",
            markersize=14,
            color="tab:red",
            linestyle="none",
            zorder=5,
            label=rf"peak $N_\sigma$ = {lya_n_sigma:.2f}",
        )

    axes[1].axhline(
        CLQ_THRESHOLD,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    axes[1].set_xlim(PLOT_MIN, PLOT_MAX)
    axes[1].set_xlabel(
        r"Rest-frame wavelength [$\AA$]"
    )
    axes[1].set_ylabel(r"$N_\sigma(\lambda)$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(
        fontsize="small",
        loc="upper right",
    )

    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / (
            f"clq_lya_{target_id}_"
            f"{low_epoch['date']}_"
            f"{high_epoch['date']}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path

def analyze_candidate(
    target_id,
    make_plot=True,
    plot_only_if_clq=True,
    output_dir=BATCH_OUTPUT_DIR,
):
    epochs = load_candidate_epochs(target_id)

    if len(epochs) < 2:
        return None

    high_epoch, low_epoch = select_high_low_epochs(epochs)

    if high_epoch is None or low_epoch is None:
        return None

    n_sigma_calc = calculate_n_sigma(
        high_epoch,
        low_epoch,
    )
    wave = n_sigma_calc["wave"]
    high_flux = n_sigma_calc["high_flux"]
    low_flux = n_sigma_calc["low_flux"]
    n_sigma = n_sigma_calc["n_sigma"]

    lya_n_sigma, lya_n_sigma_wave = calculate_lya_n_sigma(
        wave,
        n_sigma,
    )

    if not np.isfinite(lya_n_sigma):
        return None

    is_clq = lya_n_sigma > CLQ_THRESHOLD
    plot_path = None

    if make_plot and (is_clq or not plot_only_if_clq):
        plot_path = plot_candidate(
            target_id=target_id,
            high_epoch=high_epoch,
            low_epoch=low_epoch,
            wave=wave,
            high_flux=high_flux,
            low_flux=low_flux,
            n_sigma=n_sigma,
            lya_n_sigma=lya_n_sigma,
            lya_n_sigma_wave=lya_n_sigma_wave,
            output_dir=output_dir,
        )

    return {
        "TARGETID": target_id,
        "number_of_nights": len(epochs),
        "high_date": high_epoch["date"],
        "low_date": low_epoch["date"],
        "high_flux_lya": high_epoch["flux_lya"],
        "low_flux_lya": low_epoch["flux_lya"],
        "high_coadd_snr": high_epoch.get("coadd_snr"),
        "low_coadd_snr": low_epoch.get("coadd_snr"),
        "lya_wavelength": LYA_WAVE,
        "lya_n_sigma": lya_n_sigma,
        "lya_n_sigma_wavelength": lya_n_sigma_wave,
        "is_clq": is_clq,
        "high_path": str(high_epoch["fits_path"]),
        "low_path": str(low_epoch["fits_path"]),
        "plot_path": (
            str(plot_path)
            if plot_path is not None
            else None
        ),
    }


def clear_previous_results(output_dir):
    """Remove only files produced by this script."""
    if not output_dir.exists():
        return

    patterns = (
        "clq_*.png",
        "clq_nsigma_results.csv",
        "clq_lya_nsigma_results.csv",
        "single_target_*.csv",
    )

    removed = 0
    for pattern in patterns:
        for output_path in output_dir.glob(pattern):
            if output_path.is_file():
                output_path.unlink()
                removed += 1

    print(f"Cleared {removed} previous result file(s) from {output_dir}")


def diagnose_failed_candidate(target_id):
    """Return diagnostics when the standard analysis returns None."""
    target_id = str(target_id)
    target_dir = RECON_BASE / target_id
    fits_paths = sorted(target_dir.glob("*/recon_*.fits"))
    epochs = load_candidate_epochs(target_id)

    result = {
        "TARGETID": target_id,
        "analysis_status": "failed",
        "failure_reason": "",
        "number_of_reconstruction_files": len(fits_paths),
        "number_of_nights": len(epochs),
        "high_date": None,
        "low_date": None,
        "high_flux_lya": np.nan,
        "low_flux_lya": np.nan,
        "high_coadd_snr": np.nan,
        "low_coadd_snr": np.nan,
        "lya_wavelength": LYA_WAVE,
        "lya_n_sigma": np.nan,
        "lya_n_sigma_wavelength": np.nan,
        "nearest_n_sigma_wavelength": np.nan,
        "nearest_n_sigma": np.nan,
        "is_clq": False,
        "high_path": None,
        "low_path": None,
        "plot_path": None,
    }

    if not target_dir.is_dir():
        result["failure_reason"] = (
            f"Reconstruction directory does not exist or is not accessible: "
            f"{target_dir}"
        )
        return result

    if len(epochs) < 2:
        dates = [epoch["date"] for epoch in epochs]
        result["failure_reason"] = (
            "N_sigma requires two distinct observing nights, but only "
            f"{len(epochs)} usable night(s) were found. Dates: {dates}"
        )
        return result

    snr_ok_epochs = filter_epochs_by_snr(epochs)
    if len(snr_ok_epochs) < 2:
        snr_details = [
            f"{epoch['date']}=S/N:{epoch.get('coadd_snr')}"
            for epoch in epochs
        ]
        result["failure_reason"] = (
            f"Only {len(snr_ok_epochs)}/{len(epochs)} night(s) have an "
            f"original coadd with S/N >= {SNR_CUT}, need at least 2. "
            f"Values: {snr_details}"
        )
        return result

    high_epoch, low_epoch = select_high_low_epochs(epochs)

    if high_epoch is None or low_epoch is None:
        flux_details = []
        for epoch in epochs:
            flux_lya = flux_at_wavelength(epoch, LYA_WAVE)
            flux_details.append(
                f"{epoch['date']}={flux_lya}"
            )

        result["failure_reason"] = (
            "Fewer than two epochs have finite reconstructed flux at "
            f"{LYA_WAVE:.2f} Angstrom. Values: "
            + ", ".join(flux_details)
        )
        return result

    result.update({
        "high_date": high_epoch["date"],
        "low_date": low_epoch["date"],
        "high_flux_lya": high_epoch["flux_lya"],
        "low_flux_lya": low_epoch["flux_lya"],
        "high_coadd_snr": high_epoch.get("coadd_snr"),
        "low_coadd_snr": low_epoch.get("coadd_snr"),
        "high_path": str(high_epoch["fits_path"]),
        "low_path": str(low_epoch["fits_path"]),
    })

    n_sigma_calc = calculate_n_sigma(
        high_epoch,
        low_epoch,
    )
    wave = n_sigma_calc["wave"]
    n_sigma = n_sigma_calc["n_sigma"]
    lya_n_sigma, lya_n_sigma_wave = calculate_lya_n_sigma(wave, n_sigma)
    result["lya_n_sigma"] = lya_n_sigma
    result["lya_n_sigma_wavelength"] = lya_n_sigma_wave

    finite = np.isfinite(wave) & np.isfinite(n_sigma)
    if np.any(finite):
        finite_indices = np.flatnonzero(finite)
        nearest_index = finite_indices[
            np.argmin(np.abs(wave[finite] - LYA_WAVE))
        ]
        result["nearest_n_sigma_wavelength"] = wave[nearest_index]
        result["nearest_n_sigma"] = n_sigma[nearest_index]

    result["failure_reason"] = (
        f"N_sigma could not be evaluated exactly at {LYA_WAVE:.2f} "
        "Angstrom because the interpolated value is not finite. This "
        "usually indicates invalid/zero weights or insufficient common "
        "wavelength coverage."
    )

    return result


def run_one_target(target_id):
    """Analyze one target and always make a diagnostic plot."""
    target_id = str(target_id)

    if target_id not in candidate_ids:
        raise ValueError(
            f"TARGETID {target_id} is not present in {CSV_PATH}"
        )

    result = analyze_candidate(
        target_id,
        make_plot=True,
        plot_only_if_clq=False,
        output_dir=SINGLE_OUTPUT_DIR,
    )

    if result is None:
        result = diagnose_failed_candidate(target_id)

    result_table = pd.DataFrame([result])
    SINGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = SINGLE_OUTPUT_DIR / f"single_target_{target_id}.csv"
    result_table.to_csv(output_csv, index=False)

    print(result_table.to_string(index=False))
    print(
        f"N_sigma(Ly-alpha, {LYA_WAVE:.2f} Angstrom) = "
        f"{result['lya_n_sigma']}"
    )

    if result.get("analysis_status") == "failed":
        print(f"Analysis status: failed")
        print(f"Reason: {result['failure_reason']}")

        nearest_n_sigma = result.get("nearest_n_sigma", np.nan)
        if np.isfinite(nearest_n_sigma):
            print(
                "Nearest finite N_sigma = "
                f"{nearest_n_sigma:.6f} at "
                f"{result['nearest_n_sigma_wavelength']:.2f} Angstrom"
            )

    if result["plot_path"] is not None:
        print(f"Diagnostic plot: {result['plot_path']}")

    print(f"Single-target result: {output_csv}")

    return result_table


def run_all_targets():
    """Analyze all CSV targets that have usable reconstructions."""
    results = []

    for index, target_id in enumerate(
        sorted(candidate_ids),
        start=1,
    ):
        result = analyze_candidate(
            target_id,
            output_dir=BATCH_OUTPUT_DIR,
        )

        if result is None:
            continue

        results.append(result)

        print(
            f"[{index}/{len(candidate_ids)}] "
            f"{target_id}: "
            f"N nights={result['number_of_nights']}, "
            f"N_sigma(Ly-alpha)={result['lya_n_sigma']:.2f}, "
            f"CLQ={result['is_clq']}"
        )

    results_table = pd.DataFrame(results)

    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = BATCH_OUTPUT_DIR / "clq_lya_nsigma_results.csv"
    results_table.to_csv(output_csv, index=False)

    if not results_table.empty:
        number_clq = results_table["is_clq"].sum()

        print(f"\nAnalyzed targets: {len(results_table)}")
        print(f"CLQ candidates: {number_clq}")

    print(f"Results saved to: {output_csv}")

    return results_table


def analyze_custom_line(
    target_id,
    line_name,
    line_wave,
    window=2.0,
    threshold=CLQ_THRESHOLD,
    snr_cut=SNR_CUT,
    make_plot=True,
    output_dir=None,
):
    """
    Generic single-line N_sigma analysis: given ANY emission line name
    and its theoretical rest-frame wavelength, pick the highest/lowest
    reconstructed-flux epochs at that wavelength (restricted to epochs
    whose original coadd passes the S/N cut), run the shared
    calculate_n_sigma, and report the peak N_sigma within +/- window
    Angstrom -- the same style of analysis Ly-alpha already gets, just
    parameterized instead of hardcoded.
    """
    target_id = str(target_id)

    if output_dir is None:
        output_dir = OUTPUT_DIR / "custom_lines" / line_name.replace("]", "").replace("[", "")

    epochs = load_candidate_epochs(target_id)

    result = {
        "TARGETID": target_id,
        "line_name": line_name,
        "line_wave": line_wave,
        "window": window,
        "number_of_nights": len(epochs),
        "high_date": None,
        "low_date": None,
        "high_coadd_snr": np.nan,
        "low_coadd_snr": np.nan,
        "peak_n_sigma": np.nan,
        "peak_wavelength": np.nan,
        "is_significant": False,
        "plot_path": None,
        "status": "failed",
        "failure_reason": "",
    }

    if len(epochs) < 2:
        result["failure_reason"] = (
            f"Only {len(epochs)} usable observing night(s) were found."
        )
        return result

    high_epoch, low_epoch = select_high_low_epochs_at_wavelength(
        epochs, line_wave, snr_cut=snr_cut
    )

    if high_epoch is None or low_epoch is None:
        n_passing = len(filter_epochs_by_snr(epochs, snr_cut=snr_cut))
        result["failure_reason"] = (
            f"Fewer than two S/N-passing epochs ({n_passing} passed S/N "
            f">= {snr_cut}) have finite reconstructed flux at "
            f"{line_wave:.2f} A."
        )
        return result

    n_sigma_calc = calculate_n_sigma(high_epoch, low_epoch)
    peak_n_sigma, peak_wave = calculate_peak_n_sigma(
        n_sigma_calc["wave"], n_sigma_calc["n_sigma"], center=line_wave, window=window
    )

    result.update({
        "high_date": high_epoch["date"],
        "low_date": low_epoch["date"],
        "high_coadd_snr": high_epoch.get("coadd_snr"),
        "low_coadd_snr": low_epoch.get("coadd_snr"),
        "peak_n_sigma": peak_n_sigma,
        "peak_wavelength": peak_wave,
        "is_significant": bool(np.isfinite(peak_n_sigma) and peak_n_sigma > threshold),
        "status": "success" if np.isfinite(peak_n_sigma) else "failed",
    })

    if not np.isfinite(peak_n_sigma):
        result["failure_reason"] = (
            f"N_sigma could not be evaluated within +/-{window:.1f} A of "
            f"{line_wave:.2f} A (no finite N_sigma pixels in that window)."
        )
        return result

    if make_plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_custom_line(
            target_id, line_name, line_wave, window,
            high_epoch, low_epoch, n_sigma_calc, peak_n_sigma, peak_wave,
            output_dir,
        )
        result["plot_path"] = str(plot_path)

    return result


def plot_custom_line(
    target_id, line_name, line_wave, window,
    high_epoch, low_epoch, n_sigma_calc, peak_n_sigma, peak_wave,
    output_dir, display_half_width=40.0,
):
    """Single-line diagnostic plot, same visual style as the Ly-alpha plot."""
    redshift = redshift_map.get(str(target_id))

    figure, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for epoch, recon_flux, state, color in (
        (high_epoch, n_sigma_calc["high_flux"], "High", "tab:blue"),
        (low_epoch, n_sigma_calc["low_flux"], "Low", "tab:orange"),
    ):
        if redshift is not None and np.isfinite(redshift):
            original_wave, original_flux = load_original_coadd(
                target_id=target_id, night=epoch["date"], redshift=redshift,
            )
            if original_wave.size > 0:
                axes[0].plot(original_wave, original_flux, color=color, linewidth=0.4, alpha=0.08)
                coarse_wave, coarse_flux = coarse_bin(original_wave, original_flux, n_points=10)
                axes[0].plot(
                    coarse_wave, coarse_flux, color=color, linewidth=1.5,
                    linestyle="--", alpha=0.7,
                    label=f"{state} original coarse: {epoch['date']} (S/N={epoch.get('coadd_snr'):.2f})",
                )

        axes[0].plot(
            n_sigma_calc["wave"], recon_flux, color=color, linewidth=2.2, alpha=0.7,
            label=f"{state} reconstruction: {epoch['date']}",
        )

    axes[0].axvline(line_wave, color="tab:green", linestyle=":", linewidth=1.2, alpha=0.8)
    axes[0].set_ylabel("Flux")
    axes[0].set_title(
        f"TARGETID {target_id} | {line_name} ({line_wave:.2f} $\\AA$) | "
        f"peak $N_\\sigma$={peak_n_sigma:.2f} at {peak_wave:.2f} $\\AA$"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize="small", loc="upper right")

    axes[1].plot(n_sigma_calc["wave"], n_sigma_calc["n_sigma"], color="black", linewidth=1.5)
    axes[1].axvline(line_wave, color="tab:green", linestyle="--", linewidth=1.2)
    axes[1].axvspan(line_wave - window, line_wave + window, color="tab:green", alpha=0.08)
    axes[1].plot(
        peak_wave, peak_n_sigma, marker="*", markersize=14, color="tab:red",
        linestyle="none", zorder=5, label=f"peak $N_\\sigma$={peak_n_sigma:.2f}",
    )
    axes[1].axhline(CLQ_THRESHOLD, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_xlim(line_wave - display_half_width, line_wave + display_half_width)
    axes[1].set_xlabel(r"Rest-frame wavelength [$\AA$]")
    axes[1].set_ylabel(r"$N_\sigma(\lambda)$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize="small", loc="upper right")

    figure.tight_layout()
    output_path = output_dir / f"{line_name}_{target_id}_{low_epoch['date']}_{high_epoch['date']}.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    return output_path


def median_line_flux(epoch, center, half_width=CIV_CIII_PRIMARY_HALF_WIDTH):
    """Median reconstructed flux within +/- half_width of an emission line."""
    wave = epoch["rest_wave"]
    flux = epoch["recon_flux"]
    inside = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & (wave >= center - half_width)
        & (wave <= center + half_width)
    )
    if np.count_nonzero(inside) == 0:
        return np.nan
    return float(np.nanmedian(flux[inside]))


def choose_high_low_epochs_by_median(epochs, center, half_width=CIV_CIII_PRIMARY_HALF_WIDTH, snr_cut=SNR_CUT):
    """
    Choose high/low epochs for one emission line using the median
    reconstructed flux within the line window, rather than a single
    interpolated point (used for Ly-alpha selection instead).
    Restricted to epochs whose original coadd passes the S/N cut.
    """
    snr_ok_epochs = filter_epochs_by_snr(epochs, snr_cut=snr_cut)

    scored = []

    for epoch in snr_ok_epochs:
        line_flux = median_line_flux(epoch, center, half_width)
        if np.isfinite(line_flux):
            scored.append((line_flux, epoch))

    if len(scored) < 2:
        return None, None, np.nan, np.nan

    scored.sort(key=lambda item: item[0])
    low_flux, low_epoch = scored[0]
    high_flux, high_epoch = scored[-1]

    return high_epoch, low_epoch, high_flux, low_flux


def summarize_line_n_sigma(wave, n_sigma, center, half_width):
    """Median/percentile/max N_sigma within +/- half_width of a line center."""
    inside = (
        np.isfinite(wave)
        & (wave >= center - half_width)
        & (wave <= center + half_width)
    )
    valid = inside & np.isfinite(n_sigma)
    total_pixels = int(np.count_nonzero(inside))
    valid_pixels = int(np.count_nonzero(valid))
    valid_fraction = valid_pixels / total_pixels if total_pixels > 0 else 0.0

    if valid_pixels == 0:
        return {
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "valid_fraction": valid_fraction,
            "median_nsigma": np.nan,
            "p25_nsigma": np.nan,
            "p75_nsigma": np.nan,
            "max_nsigma": np.nan,
        }

    values = n_sigma[valid]
    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "valid_fraction": valid_fraction,
        "median_nsigma": float(np.nanmedian(values)),
        "p25_nsigma": float(np.nanpercentile(values, 25)),
        "p75_nsigma": float(np.nanpercentile(values, 75)),
        "max_nsigma": float(np.nanmax(values)),
    }


def analyze_civ_ciii_lines(
    target_id,
    epochs,
    line_centers=CIV_CIII_LINE_CENTERS,
    primary_half_width=CIV_CIII_PRIMARY_HALF_WIDTH,
    check_half_widths=CIV_CIII_CHECK_HALF_WIDTHS,
    threshold=CIV_CIII_NSIGMA_THRESHOLD,
    min_valid_fraction=CIV_CIII_MIN_VALID_FRACTION,
):
    """
    For each line in line_centers, pick high/low epochs by median flux
    in the line window, run the SHARED calculate_n_sigma, and summarize
    N_sigma at each check window width. Returns (result_row, line_details,
    line_passes) where line_details holds the epochs/calc needed for
    plotting and result_row is flat enough to append straight to a CSV.
    """
    result = {
        "TARGETID": str(target_id),
        "number_of_nights": len(epochs),
    }
    line_details = {}
    line_passes = {}

    for line_name, center in line_centers.items():
        high_epoch, low_epoch, high_flux, low_flux = choose_high_low_epochs_by_median(
            epochs,
            center,
            primary_half_width,
        )

        if high_epoch is None or low_epoch is None:
            result[f"{line_name}_high_date"] = None
            result[f"{line_name}_low_date"] = None
            result[f"{line_name}_high_median_flux"] = np.nan
            result[f"{line_name}_low_median_flux"] = np.nan
            for half_width in check_half_widths:
                label = f"{int(2 * half_width)}A_window"
                for key in (
                    "total_pixels", "valid_pixels", "valid_fraction",
                    "median_nsigma", "p25_nsigma", "p75_nsigma", "max_nsigma",
                ):
                    result[f"{line_name}_{label}_{key}"] = np.nan
            result[f"{line_name}_passes"] = False
            line_passes[line_name] = False
            continue

        n_sigma_calc = calculate_n_sigma(high_epoch, low_epoch)
        wave = n_sigma_calc["wave"]
        n_sigma = n_sigma_calc["n_sigma"]

        result[f"{line_name}_high_date"] = high_epoch["date"]
        result[f"{line_name}_low_date"] = low_epoch["date"]
        result[f"{line_name}_high_median_flux"] = high_flux
        result[f"{line_name}_low_median_flux"] = low_flux
        result[f"{line_name}_high_coadd_snr"] = high_epoch.get("coadd_snr")
        result[f"{line_name}_low_coadd_snr"] = low_epoch.get("coadd_snr")

        for half_width in check_half_widths:
            summary = summarize_line_n_sigma(wave, n_sigma, center, half_width)
            label = f"{int(2 * half_width)}A_window"
            for key, value in summary.items():
                result[f"{line_name}_{label}_{key}"] = value

        primary_summary = summarize_line_n_sigma(
            wave, n_sigma, center, primary_half_width
        )
        passes = bool(
            primary_summary["valid_fraction"] >= min_valid_fraction
            and np.isfinite(primary_summary["median_nsigma"])
            and primary_summary["median_nsigma"] > threshold
        )
        line_passes[line_name] = passes
        result[f"{line_name}_passes"] = passes
        result[f"{line_name}_median_nsigma"] = primary_summary["median_nsigma"]
        result[f"{line_name}_valid_fraction"] = primary_summary["valid_fraction"]

        line_details[line_name] = {
            "high_epoch": high_epoch,
            "low_epoch": low_epoch,
            "wave": wave,
            "n_sigma": n_sigma,
            "high_flux_array": n_sigma_calc["high_flux"],
            "low_flux_array": n_sigma_calc["low_flux"],
            "primary_summary": primary_summary,
        }

    valid_scores = {
        line_name: line_details[line_name]["primary_summary"]["median_nsigma"]
        for line_name in line_details
        if (
            line_details[line_name]["primary_summary"]["valid_fraction"]
            >= min_valid_fraction
            and np.isfinite(
                line_details[line_name]["primary_summary"]["median_nsigma"]
            )
        )
    }

    if valid_scores:
        trigger_line = max(valid_scores, key=valid_scores.get)
        selection_score = float(valid_scores[trigger_line])
    else:
        trigger_line = None
        selection_score = np.nan

    result["trigger_line"] = trigger_line
    result["selection_score"] = selection_score
    result["conservative_cl_both_lines"] = bool(
        line_passes
        and all(line_passes.get(name, False) for name in line_centers)
    )

    return result, line_details, line_passes


def plot_civ_ciii_lines(target_id, line_details, output_dir, display_half_width=40.0):
    """
    2x2 diagnostic plot: one column per emission line, spectrum on top,
    N_sigma below, reusing load_original_coadd/coarse_bin for the
    lightly-drawn original coadd overlay.
    """
    redshift = redshift_map.get(str(target_id))
    line_names = list(line_details.keys())

    figure, axes = plt.subplots(2, len(line_names), figsize=(15, 9))
    if len(line_names) == 1:
        axes = axes.reshape(2, 1)

    for column, line_name in enumerate(line_names):
        details = line_details[line_name]
        center = CIV_CIII_LINE_CENTERS[line_name]
        spectrum_axis = axes[0, column]
        nsigma_axis = axes[1, column]

        for epoch, recon_flux, state, color in (
            (details["high_epoch"], details["high_flux_array"], "High", "tab:blue"),
            (details["low_epoch"], details["low_flux_array"], "Low", "tab:orange"),
        ):
            if redshift is not None and np.isfinite(redshift):
                original_wave, original_flux = load_original_coadd(
                    target_id=target_id,
                    night=epoch["date"],
                    redshift=redshift,
                )
                if original_wave.size > 0:
                    spectrum_axis.plot(
                        original_wave, original_flux,
                        color=color, linewidth=0.4, alpha=0.08,
                    )
                    coarse_wave, coarse_flux = coarse_bin(
                        original_wave, original_flux, n_points=10,
                    )
                    spectrum_axis.plot(
                        coarse_wave, coarse_flux,
                        color=color, linewidth=1.5, linestyle="--", alpha=0.7,
                        label=f"{state} original coarse: {epoch['date']}",
                    )

            spectrum_axis.plot(
                details["wave"], recon_flux,
                color=color, linewidth=2.2, alpha=0.7,
                label=f"{state} reconstruction: {epoch['date']}",
            )

        for axis in (spectrum_axis, nsigma_axis):
            axis.axvspan(
                center - CIV_CIII_PRIMARY_HALF_WIDTH,
                center + CIV_CIII_PRIMARY_HALF_WIDTH,
                color="tab:green", alpha=0.12,
                label="Primary +/-10 A window" if axis is spectrum_axis else None,
            )
            axis.axvline(center, color="tab:green", linestyle=":", linewidth=1)
            axis.set_xlim(center - display_half_width, center + display_half_width)
            axis.grid(alpha=0.25)

        summary = details["primary_summary"]
        spectrum_axis.set_title(
            f"{line_name} | median N_sigma={summary['median_nsigma']:.2f} | "
            f"valid={summary['valid_fraction']:.0%}"
        )
        spectrum_axis.set_ylabel("Reconstructed flux")
        spectrum_axis.legend(fontsize="small")

        nsigma_axis.plot(
            details["wave"], details["n_sigma"], color="black", linewidth=1.3,
        )
        nsigma_axis.axhline(
            CIV_CIII_NSIGMA_THRESHOLD, color="tab:red", linestyle="--", linewidth=1,
        )
        nsigma_axis.set_xlabel(r"Rest-frame wavelength [$\AA$]")
        nsigma_axis.set_ylabel(r"$N_\sigma$")

    figure.suptitle(f"TARGETID {target_id}: C IV and C III] variability")
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"civ_ciii_nsigma_{target_id}.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    return output_path


def run_civ_ciii_target(target_id):
    """Run the conservative C IV / C III] analysis for one TARGETID."""
    target_id = str(target_id)
    epochs = load_candidate_epochs(target_id)

    if len(epochs) < 2:
        raise RuntimeError(
            f"TARGETID {target_id} has only {len(epochs)} usable "
            "observing night(s); need at least 2."
        )

    CIV_CIII_SINGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result, line_details, line_passes = analyze_civ_ciii_lines(target_id, epochs)

    complete_details = {
        name: details
        for name, details in line_details.items()
        if details  # skip lines that failed to produce high/low epochs
    }

    plot_path = None
    if complete_details:
        plot_path = plot_civ_ciii_lines(
            target_id, complete_details, CIV_CIII_SINGLE_OUTPUT_DIR
        )
        result["plot_path"] = str(plot_path)
    else:
        result["plot_path"] = None

    result_path = CIV_CIII_SINGLE_OUTPUT_DIR / f"civ_ciii_nsigma_{target_id}.csv"
    pd.DataFrame([result]).to_csv(result_path, index=False)

    print("\n========== C IV / C III] RESULT ==========")
    print(f"TARGETID = {target_id}")
    for line_name in CIV_CIII_LINE_CENTERS:
        print(
            f"{line_name}: high/low dates = "
            f"{result.get(f'{line_name}_high_date')} / "
            f"{result.get(f'{line_name}_low_date')}, "
            f"median N_sigma = {result.get(f'{line_name}_median_nsigma')}, "
            f"passes = {line_passes.get(line_name)}"
        )
    print(f"Both lines pass = {result['conservative_cl_both_lines']}")
    print(f"Trigger line = {result['trigger_line']}, score = {result['selection_score']}")
    print(f"Result CSV = {result_path}")
    if plot_path is not None:
        print(f"Plot = {plot_path}")
    print("===========================================")

    return result


def run_civ_ciii_batch(number_of_targets=100):
    """
    Scan the first N sorted candidates with reconstructions, and plot
    only the ones whose strongest line clears the threshold (falling
    back to a looser 2-sigma cut if nothing clears 3-sigma).
    """
    available_ids = [
        target_id
        for target_id in sorted(candidate_ids)
        if (RECON_BASE / target_id).is_dir()
    ]
    selected_ids = available_ids[:number_of_targets]

    if len(selected_ids) < number_of_targets:
        print(
            f"Only {len(selected_ids)} accessible reconstruction "
            f"directories were found; requested {number_of_targets}."
        )

    rows = []
    for index, target_id in enumerate(selected_ids, start=1):
        epochs = load_candidate_epochs(target_id)

        if len(epochs) < 2:
            row = {
                "TARGETID": target_id,
                "status": "failed",
                "failure_reason": f"only {len(epochs)} usable night(s)",
                "selection_score": np.nan,
                "trigger_line": None,
                "conservative_cl_both_lines": False,
            }
        else:
            result, _, _ = analyze_civ_ciii_lines(target_id, epochs)
            result["status"] = "success"
            result["failure_reason"] = ""
            row = result

        rows.append(row)
        score = row.get("selection_score", np.nan)
        score_text = f"{score:.3f}" if np.isfinite(score) else "NaN"
        print(
            f"[{index:03d}/{len(selected_ids)}] TARGETID={target_id} "
            f"max_line_median_N_sigma={score_text} "
            f"line={row.get('trigger_line')} status={row['status']}"
        )

    results = pd.DataFrame(rows)
    CIV_CIII_BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results_path = CIV_CIII_BATCH_OUTPUT_DIR / "civ_ciii_all_results.csv"
    results.to_csv(all_results_path, index=False)

    above_three = results[
        (results["status"] == "success") & (results["selection_score"] > 3.0)
    ].copy()

    if len(above_three) > 0:
        threshold = 3.0
        selected = above_three
        fallback_used = False
    else:
        threshold = 2.0
        selected = results[
            (results["status"] == "success") & (results["selection_score"] > 2.0)
        ].copy()
        fallback_used = True

    selected = selected.sort_values("selection_score", ascending=False)

    plot_paths = []
    for _, row in selected.iterrows():
        target_id = str(row["TARGETID"])
        epochs = load_candidate_epochs(target_id)
        _, line_details, _ = analyze_civ_ciii_lines(target_id, epochs)
        complete_details = {
            name: details for name, details in line_details.items() if details
        }
        if complete_details:
            plot_path = plot_civ_ciii_lines(
                target_id, complete_details, CIV_CIII_BATCH_OUTPUT_DIR
            )
            plot_paths.append(str(plot_path))
        else:
            plot_paths.append(None)

    selected["plot_path"] = plot_paths if len(selected) > 0 else pd.Series(dtype=str)

    selected_path = (
        CIV_CIII_BATCH_OUTPUT_DIR
        / f"civ_ciii_candidates_above_{int(threshold)}sigma.csv"
    )
    selected.to_csv(selected_path, index=False)

    print("\n========== C IV / C III] BATCH ==========")
    print(f"Targets attempted: {len(results)}")
    print(f"Targets with any line above 3 sigma: {len(above_three)}")
    print(f"Fallback to 2 sigma used: {fallback_used}")
    print(f"Selection threshold: > {threshold:g} sigma")
    print(f"Selected targets: {len(selected)}")
    print(f"All results: {all_results_path}")
    print(f"Selected candidates: {selected_path}")
    print("==========================================")

    return results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Measure Ly-alpha spectral-change significance."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--target-id",
        help=(
            "Run one TARGETID and always save its diagnostic plot. "
            "Runs the Ly-alpha CLQ pipeline by default, or the C IV / "
            "C III] pipeline if --civ-ciii is also given."
        ),
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run all CSV targets through the Ly-alpha CLQ pipeline.",
    )
    mode.add_argument(
        "--first-n",
        type=int,
        help=(
            "Scan the first N sorted candidates with reconstructions "
            "through the C IV / C III] pipeline (batch mode)."
        ),
    )

    parser.add_argument(
        "--civ-ciii",
        action="store_true",
        help="With --target-id, run the C IV / C III] pipeline instead of Ly-alpha.",
    )

    parser.add_argument(
        "--line-name",
        help=(
            "With --target-id, run a generic single-line N_sigma analysis "
            "for this line name (e.g. MgII). Requires --line-wave."
        ),
    )
    parser.add_argument(
        "--line-wave",
        type=float,
        help="Theoretical rest-frame wavelength (Angstrom) for --line-name.",
    )
    parser.add_argument(
        "--line-window",
        type=float,
        default=2.0,
        help="+/- window (Angstrom) to search for the peak N_sigma. Default: 2.0.",
    )

    parser.add_argument(
        "--clear-results",
        action="store_true",
        help="Clear this script's previous PNG and CSV outputs first.",
    )

    args = parser.parse_args()

    if (args.line_name is None) != (args.line_wave is None):
        parser.error("--line-name and --line-wave must be given together.")

    if args.line_name is not None and args.target_id is None:
        parser.error("--line-name requires --target-id.")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.first_n is not None:
        if args.clear_results:
            clear_previous_results(CIV_CIII_BATCH_OUTPUT_DIR)
        results = run_civ_ciii_batch(args.first_n)

    elif args.target_id is not None and args.line_name is not None:
        result = analyze_custom_line(
            args.target_id, args.line_name, args.line_wave, window=args.line_window,
        )
        print(f"\n========== {args.line_name} RESULT ==========")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print("===========================================")
        results = pd.DataFrame([result])

    elif args.target_id is not None and args.civ_ciii:
        if args.clear_results:
            clear_previous_results(CIV_CIII_SINGLE_OUTPUT_DIR)
        results = run_civ_ciii_target(args.target_id)

    elif args.target_id is not None:
        if args.clear_results:
            clear_previous_results(SINGLE_OUTPUT_DIR)
        results = run_one_target(args.target_id)

    else:
        if args.clear_results:
            clear_previous_results(BATCH_OUTPUT_DIR)
        results = run_all_targets()