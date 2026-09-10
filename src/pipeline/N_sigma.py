"""
Flow
1. Read CLQ_candidates.csv and only use TARGETIDs with reconstructions.
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

RECON_BASE = Path(
    "/work/10579/prisha/ls6/desi_project/reconstructed_spectra/"
)

COADD_BASE = Path(
    "/work/10579/prisha/ls6/desi_project/output_coadds/"
)



CSV_PATH = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/"
    "quassiQ/src/pipeline/CLQ_candidates.csv"
)

OUTPUT_DIR = CSV_PATH.parent / "clq_nsigma_results"
SINGLE_OUTPUT_DIR = OUTPUT_DIR / "single_targets"
BATCH_OUTPUT_DIR = OUTPUT_DIR / "batch"
LYA_WAVE = 1215.67
PLOT_MIN = 1100.0
PLOT_MAX = 1325.0
CLQ_THRESHOLD = 3.0

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

def get_original_coadd_path(recon_path, target_id):
    """
    Convert:

    reconstructed_spectra/TARGETID/
    recon_TARGETID_coadd-1-7852-20220604-TARGETID.fits

    into:

    output_coadds/TARGETID/
    coadd-1-7852-20220604-TARGETID.fits
    """
    recon_path = Path(recon_path)
    target_id = str(target_id)

    prefix = f"recon_{target_id}_"

    if not recon_path.name.startswith(prefix):
        return None

    coadd_filename = recon_path.name[len(prefix):]
    coadd_path = COADD_BASE / target_id / coadd_filename

    if not coadd_path.is_file():
        print(f"Original coadd not found: {coadd_path}")
        return None

    return coadd_path


def load_original_coadd(recon_path, target_id, redshift):
    """
    Load the B, R, and Z spectra from the original DESI coadd and
    convert the wavelength to the rest frame.
    """
    coadd_path = get_original_coadd_path(
        recon_path,
        target_id,
    )

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

    recon_39627559461192666_coadd-1-7852-20220604-
    39627559461192666.fits
    """
    match = re.search(
        r"-(\d{8})-\d+\.fits$",
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

        weights = np.asarray(
            data["WEIGHTS"],
            dtype=float,
        ).squeeze()

        final_weights = np.asarray(
            data["FINAL_WEIGHTS"],
            dtype=float,
        ).squeeze()

    return {
        "target_id": target_id,
        "date": extract_observation_date(fits_path),
        "fits_path": fits_path,
        "rest_wave": rest_wave,
        "orig_flux": orig_flux,
        "recon_flux": recon_flux,
        "weights": weights,
        "final_weights": final_weights,
        "sigma": weights_to_sigma(weights),
    }


def load_candidate_epochs(target_id):
    """
    Load all available observing nights for one TARGETID.
    """
    target_dir = RECON_BASE / target_id

    if not target_dir.is_dir():
        return []

    epochs_by_date = {}

    for fits_path in sorted(target_dir.glob("recon_*.fits")):
        date = extract_observation_date(fits_path)

        if date is None:
            print(f"Could not extract date: {fits_path.name}")
            continue

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


def select_high_low_epochs(epochs):
    """
    Select the epochs with the highest and lowest reconstructed
    flux at Ly-alpha (1215.67 Angstrom).
    """
    valid_epochs = []

    for epoch in epochs:
        flux_lya = flux_at_wavelength(
            epoch,
            LYA_WAVE,
        )

        if np.isfinite(flux_lya):
            epoch["flux_lya"] = flux_lya
            valid_epochs.append(epoch)

    if len(valid_epochs) < 2:
        return None, None

    high_epoch = max(
        valid_epochs,
        key=lambda epoch: epoch["flux_lya"],
    )

    low_epoch = min(
        valid_epochs,
        key=lambda epoch: epoch["flux_lya"],
    )

    return high_epoch, low_epoch


def interpolate_to_grid(values, old_wave, new_wave):
    """
    Interpolate values onto a reference wavelength grid.
    """
    return np.interp(
        new_wave,
        old_wave,
        values,
        left=np.nan,
        right=np.nan,
    )


def calculate_n_sigma(high_epoch, low_epoch):
    """
    Calculate:

                       f_high - f_low
    N_sigma(lambda) = -----------------
                      sqrt(sigma_high^2
                           + sigma_low^2)
    """
    wave = high_epoch["rest_wave"]

    high_flux = high_epoch["recon_flux"]
    high_sigma = high_epoch["sigma"]

    low_flux = interpolate_to_grid(
        low_epoch["recon_flux"],
        low_epoch["rest_wave"],
        wave,
    )

    low_sigma = interpolate_to_grid(
        low_epoch["sigma"],
        low_epoch["rest_wave"],
        wave,
    )

    denominator = np.sqrt(
        high_sigma**2 + low_sigma**2
    )

    n_sigma = np.full(wave.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(high_flux)
        & np.isfinite(low_flux)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    n_sigma[valid] = (
        high_flux[valid] - low_flux[valid]
    ) / denominator[valid]

    return wave, high_flux, low_flux, n_sigma


def calculate_lya_n_sigma(wave, n_sigma):
    """
    Interpolate N_sigma at the Ly-alpha wavelength, 1215.67 A.
    """
    valid = np.isfinite(wave) & np.isfinite(n_sigma)
    if np.count_nonzero(valid) < 2:
        return np.nan

    wave_valid = wave[valid]
    n_sigma_valid = n_sigma[valid]
    order = np.argsort(wave_valid)
    wave_valid = wave_valid[order]
    n_sigma_valid = n_sigma_valid[order]

    if not wave_valid[0] <= LYA_WAVE <= wave_valid[-1]:
        return np.nan

    return np.interp(LYA_WAVE, wave_valid, n_sigma_valid)
def plot_candidate(
    target_id,
    high_epoch,
    low_epoch,
    wave,
    high_flux,
    low_flux,
    n_sigma,
    lya_n_sigma,
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
            recon_path=epoch["fits_path"],
            target_id=target_id,
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
        rf"$N_\sigma(\mathrm{{Ly}}\alpha)"
        rf"={lya_n_sigma:.2f}$"
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

    wave, high_flux, low_flux, n_sigma = calculate_n_sigma(
        high_epoch,
        low_epoch,
    )

    lya_n_sigma = calculate_lya_n_sigma(
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
            output_dir=output_dir,
        )

    return {
        "TARGETID": target_id,
        "number_of_nights": len(epochs),
        "high_date": high_epoch["date"],
        "low_date": low_epoch["date"],
        "high_flux_lya": high_epoch["flux_lya"],
        "low_flux_lya": low_epoch["flux_lya"],
        "lya_wavelength": LYA_WAVE,
        "lya_n_sigma": lya_n_sigma,
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
        print(
            f"TARGETID {target_id} could not be analyzed. "
            "Check that it has at least two reconstruction nights, "
            "valid weights, and wavelength coverage around Ly-alpha."
        )
        return None

    result_table = pd.DataFrame([result])
    SINGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = SINGLE_OUTPUT_DIR / f"single_target_{target_id}.csv"
    result_table.to_csv(output_csv, index=False)

    print(result_table.to_string(index=False))
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Measure Ly-alpha spectral-change significance."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--target-id",
        help="Run one TARGETID and always save its diagnostic plot.",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run all CSV targets that have usable reconstructions.",
    )

    parser.add_argument(
        "--clear-results",
        action="store_true",
        help="Clear this script's previous PNG and CSV outputs first.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear_results:
        selected_output_dir = (
            SINGLE_OUTPUT_DIR
            if args.target_id is not None
            else BATCH_OUTPUT_DIR
        )
        clear_previous_results(selected_output_dir)

    if args.target_id is not None:
        results = run_one_target(args.target_id)
    else:
        results = run_all_targets()
