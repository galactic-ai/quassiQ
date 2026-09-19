"""Measure multi-epoch spectral variability around a requested emission line.

To support another line, add one entry to EMISSION_LINES. All lines use the
same loading, quality filtering, N-sigma calculation, plotting, and batch code.
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
    "/work/10579/prisha/ls6/desi_project/reconstructions"
)
COADD_BASE = Path(
    "/work/10579/prisha/ls6/desi_project/output_coadds"
)
CSV_PATH = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/"
    "quality_cut/CLQ_candidates_7days_merged.csv"
)
OUTPUT_BASE = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output"
)

CSV_OUTPUT_DIR = OUTPUT_BASE / "csv"

EMISSION_LINES = {
    "lya": {
        "label": "Ly-alpha",
        "wavelength": 1215.67,
        "window": 2.0,
        "display_half_width": 112.5,
    },
    "nv": {
        "label": "N V",
        "wavelength": 1240.81,
        "window": 2.0,
        "display_half_width": 40.0,
    },
    "siiv_oiv": {
        "label": "Si IV + O IV]",
        "wavelength": 1399.80,
        "window": 2.0,
        "display_half_width": 40.0,
    },
    "civ": {
        "label": "C IV",
        "wavelength": 1549.48,
        "window": 2.0,
        "display_half_width": 40.0,
    },
    "heii": {
        "label": "He II",
        "wavelength": 1640.42,
        "window": 2.0,
        "display_half_width": 40.0,
    },
    "oiii": {
        "label": "O III]",
        "wavelength": 1663.48,
        "window": 2.0,
        "display_half_width": 40.0,
    },
    "aliii": {
        "label": "Al III",
        "wavelength": 1857.40,
        "window": 2.0,
        "display_half_width": 40.0,
    },
    "ciii": {
        "label": "C III]",
        "wavelength": 1908.73,
        "window": 2.0,
        "display_half_width": 50.0,
    },
    "mgii": {
        "label": "Mg II",
        "wavelength": 2798.75,
        "window": 2.0,
        "display_half_width": 40.0,
    },
}


SIGNIFICANCE_THRESHOLD = 3.0
SNR_CUT = 2.0
CHI2_MAX = 1.4


def load_catalog():
    candidates = pd.read_csv(
        CSV_PATH,
        dtype={"TARGETID": str},
    )
    candidates["TARGETID"] = candidates["TARGETID"].str.strip()

    candidate_ids = set(
        candidates["TARGETID"].dropna()
    )

    redshift_map = (
        candidates.drop_duplicates("TARGETID")
        .set_index("TARGETID")["Z"]
        .to_dict()
    )

    return candidate_ids, redshift_map


def get_original_coadd_path(target_id, night):
    pattern = f"coadd-*-*-{night}-{target_id}.fits"
    matches = sorted(
        (COADD_BASE / str(target_id)).glob(pattern)
    )

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"Warning: multiple coadds for "
            f"TARGETID={target_id}, NIGHT={night}; "
            f"using {matches[0]}"
        )

    return matches[0]


def load_original_coadd(target_id, night, redshift):
    """Load the unnormalized flux from the original DESI coadd."""

    coadd_path = get_original_coadd_path(
        target_id,
        night,
    )

    if coadd_path is None:
        return np.array([]), np.array([])

    with fits.open(coadd_path) as hdul:
        hdu_names = {
            hdu.name.upper(): hdu
            for hdu in hdul
        }
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
                    return np.array([]), np.array([])

                row_index = matches[0]

        wave_parts = []
        flux_parts = []

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

            rest_wave = wave / (1.0 + redshift)

            valid = (
                np.isfinite(rest_wave)
                & np.isfinite(flux)
            )

            wave_parts.append(rest_wave[valid])
            flux_parts.append(flux[valid])

    if not wave_parts:
        return np.array([]), np.array([])

    rest_wave = np.concatenate(wave_parts)
    flux = np.concatenate(flux_parts)

    order = np.argsort(rest_wave)

    return rest_wave[order], flux[order]


def coarse_bin(wave, flux, n_points=10):
    wave = np.asarray(wave)
    flux = np.asarray(flux)

    valid = (
        np.isfinite(wave)
        & np.isfinite(flux)
    )

    wave = wave[valid]
    flux = flux[valid]

    number_of_bins = wave.size // n_points

    if number_of_bins == 0:
        return wave, flux

    usable_size = number_of_bins * n_points

    binned_wave = (
        wave[:usable_size]
        .reshape(number_of_bins, n_points)
        .mean(axis=1)
    )

    binned_flux = (
        flux[:usable_size]
        .reshape(number_of_bins, n_points)
        .mean(axis=1)
    )

    return binned_wave, binned_flux


def extract_observation_date(fits_path):
    match = re.search(
        r"_(\d{8})\.fits$",
        fits_path.name,
    )

    if match:
        return match.group(1)

    return fits_path.parent.name


def compute_median_snr_from_original_coadd(
    coadd_path,
    target_id,
):
    """Return median per-pixel S/N and contributing pixel count.

    S/N is calculated directly from each DESI B/R/Z coadd as
    flux * sqrt(ivar).
    """

    snr_parts = []

    with fits.open(coadd_path) as hdul:
        hdu_names = {
            hdu.name.upper(): hdu
            for hdu in hdul
        }
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
                    return np.nan, 0

                row_index = matches[0]

        for band in ("B", "R", "Z"):
            flux_name = f"{band}_FLUX"
            ivar_name = f"{band}_IVAR"
            mask_name = f"{band}_MASK"

            if (
                flux_name not in hdu_names
                or ivar_name not in hdu_names
            ):
                continue

            flux_data = np.asarray(
                hdu_names[flux_name].data,
                dtype=float,
            )
            ivar_data = np.asarray(
                hdu_names[ivar_name].data,
                dtype=float,
            )

            if flux_data.ndim == 2:
                flux = flux_data[row_index]
            else:
                flux = flux_data.squeeze()

            if ivar_data.ndim == 2:
                ivar = ivar_data[row_index]
            else:
                ivar = ivar_data.squeeze()

            valid = (
                np.isfinite(flux)
                & np.isfinite(ivar)
                & (ivar > 0)
            )

            if mask_name in hdu_names:
                mask_data = np.asarray(
                    hdu_names[mask_name].data
                )

                if mask_data.ndim == 2:
                    mask = mask_data[row_index]
                else:
                    mask = mask_data.squeeze()

                valid &= mask == 0

            if np.any(valid):
                snr_parts.append(
                    flux[valid] * np.sqrt(ivar[valid])
                )

    if not snr_parts:
        return np.nan, 0

    snr = np.concatenate(snr_parts)

    return (
        float(np.median(snr)),
        int(snr.size),
    )


def get_epoch_snr(target_id, night):
    coadd_path = get_original_coadd_path(
        target_id,
        night,
    )

    if coadd_path is None:
        return np.nan

    median_snr, _ = (
        compute_median_snr_from_original_coadd(
            coadd_path,
            target_id,
        )
    )

    return median_snr


def calculate_chi2_per_pixel(
    orig_flux,
    recon_flux,
    weights,
):
    """Calculate chi-square per valid reconstruction pixel.

    ORIG_FLUX and RECON_FLUX in the reconstruction FITS file are
    already divided by NORM. UPDATED_WEIGHTS has already been
    multiplied by NORM**2.

    Therefore no additional normalization should be applied here:

        ((orig / norm) - (recon / norm))**2
        * (ivar * norm**2)

    is equal to:

        (orig - recon)**2 * ivar
    """

    orig_flux = np.asarray(
        orig_flux,
        dtype=float,
    ).squeeze()

    recon_flux = np.asarray(
        recon_flux,
        dtype=float,
    ).squeeze()

    weights = np.asarray(
        weights,
        dtype=float,
    ).squeeze()

    valid = (
        np.isfinite(orig_flux)
        & np.isfinite(recon_flux)
        & np.isfinite(weights)
        & (weights > 0)
    )

    number_of_pixels = np.count_nonzero(valid)

    if number_of_pixels == 0:
        return np.nan, 0

    chi2 = np.sum(
        (
            orig_flux[valid]
            - recon_flux[valid]
        ) ** 2
        * weights[valid]
    )

    chi2_per_pixel = chi2 / number_of_pixels

    return (
        float(chi2_per_pixel),
        int(number_of_pixels),
    )


def load_epoch(fits_path, target_id):
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        header = hdul[1].header

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

        updated_weights = np.asarray(
            data["UPDATED_WEIGHTS"],
            dtype=float,
        ).squeeze()

        normalization = float(
            header.get("NORM", 1.0)
        )

    if (
        not np.isfinite(normalization)
        or normalization <= 0
    ):
        print(
            f"Warning: invalid NORM={normalization} "
            f"in {fits_path}; using NORM=1."
        )
        normalization = 1.0

    chi2_per_pixel, chi2_pixels = (
        calculate_chi2_per_pixel(
            orig_flux,
            recon_flux,
            updated_weights,
        )
    )

    date = extract_observation_date(fits_path)
    coadd_snr = get_epoch_snr(
        target_id,
        date,
    )

    return {
        "target_id": str(target_id),
        "date": date,
        "fits_path": fits_path,
        "rest_wave": rest_wave,
        "orig_flux": orig_flux,
        "recon_flux": recon_flux,
        "updated_weights": updated_weights,
        "normalization": normalization,
        "coadd_snr": coadd_snr,
        "chi2_per_pixel": chi2_per_pixel,
        "chi2_pixels": chi2_pixels,
    }


def load_candidate_epochs(target_id):
    target_dir = RECON_BASE / str(target_id)

    if not target_dir.is_dir():
        return []

    epochs_by_date = {}

    for fits_path in sorted(
        target_dir.glob("*/recon_*.fits")
    ):
        date = extract_observation_date(fits_path)

        if date not in epochs_by_date:
            epochs_by_date[date] = load_epoch(
                fits_path,
                str(target_id),
            )

    return list(epochs_by_date.values())


def flux_at_wavelength(epoch, wavelength):
    wave = epoch["rest_wave"]
    flux = epoch["recon_flux"]

    valid = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & (wave > 0)
    )

    if np.count_nonzero(valid) < 2:
        return np.nan

    wave = wave[valid]
    flux = flux[valid]

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    wave, unique_indices = np.unique(
        wave,
        return_index=True,
    )
    flux = flux[unique_indices]

    if not wave[0] <= wavelength <= wave[-1]:
        return np.nan

    return np.interp(
        wavelength,
        wave,
        flux,
    )


def get_cut_counts(epochs, wavelength):
    """Count epochs surviving each consecutive quality cut."""

    epochs_after_snr = [
        epoch
        for epoch in epochs
        if (
            np.isfinite(
                epoch.get("coadd_snr", np.nan)
            )
            and epoch["coadd_snr"] >= SNR_CUT
        )
    ]

    epochs_after_chi2 = [
        epoch
        for epoch in epochs_after_snr
        if (
            np.isfinite(
                epoch.get("chi2_per_pixel", np.nan)
            )
            and epoch["chi2_per_pixel"] <= CHI2_MAX
        )
    ]

    epochs_covering_line = [
        epoch
        for epoch in epochs_after_chi2
        if np.isfinite(
            flux_at_wavelength(
                epoch,
                wavelength,
            )
        )
    ]

    return (
        len(epochs_after_snr),
        len(epochs_after_chi2),
        len(epochs_covering_line),
    )


def select_high_low_epochs(
    epochs,
    wavelength,
    snr_cut=SNR_CUT,
    chi2_max=CHI2_MAX,
):
    ranked = []

    for epoch in epochs:
        snr = epoch.get(
            "coadd_snr",
            np.nan,
        )

        if (
            not np.isfinite(snr)
            or snr < snr_cut
        ):
            continue

        chi2 = epoch.get(
            "chi2_per_pixel",
            np.nan,
        )

        if (
            not np.isfinite(chi2)
            or chi2 > chi2_max
        ):
            continue

        selection_flux = flux_at_wavelength(
            epoch,
            wavelength,
        )

        if np.isfinite(selection_flux):
            ranked.append(
                (selection_flux, epoch)
            )

    if len(ranked) < 2:
        return None, None

    ranked.sort(
        key=lambda item: item[0]
    )

    low_flux, low_epoch = ranked[0]
    high_flux, high_epoch = ranked[-1]

    low_epoch = dict(
        low_epoch,
        selection_flux=low_flux,
    )
    high_epoch = dict(
        high_epoch,
        selection_flux=high_flux,
    )

    return high_epoch, low_epoch


def interpolate_to_grid(
    values,
    old_wave,
    new_wave,
):
    values = np.asarray(
        values,
        dtype=float,
    )
    old_wave = np.asarray(
        old_wave,
        dtype=float,
    )
    new_wave = np.asarray(
        new_wave,
        dtype=float,
    )

    valid = (
        np.isfinite(old_wave)
        & np.isfinite(values)
        & (old_wave > 0)
    )

    if np.count_nonzero(valid) < 2:
        return np.full(
            new_wave.shape,
            np.nan,
        )

    wave = old_wave[valid]
    values = values[valid]

    order = np.argsort(wave)
    wave = wave[order]
    values = values[order]

    wave, unique_indices = np.unique(
        wave,
        return_index=True,
    )
    values = values[unique_indices]

    return np.interp(
        new_wave,
        wave,
        values,
        left=np.nan,
        right=np.nan,
    )


def calculate_n_sigma(
    high_epoch,
    low_epoch,
):
    wave = high_epoch["rest_wave"]
    high_flux = high_epoch["recon_flux"]
    high_weights = high_epoch["updated_weights"]

    low_flux = interpolate_to_grid(
        low_epoch["recon_flux"],
        low_epoch["rest_wave"],
        wave,
    )

    low_weights = interpolate_to_grid(
        low_epoch["updated_weights"],
        low_epoch["rest_wave"],
        wave,
    )

    denominator = np.full(
        wave.shape,
        np.nan,
    )

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

    n_sigma = np.full(
        wave.shape,
        np.nan,
    )

    valid = (
        np.isfinite(high_flux)
        & np.isfinite(low_flux)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    n_sigma[valid] = (
        np.abs(
            high_flux[valid]
            - low_flux[valid]
        )
        / denominator[valid]
    )

    return {
        "wave": wave,
        "high_flux": high_flux,
        "low_flux": low_flux,
        "n_sigma": n_sigma,
    }


def calculate_peak_n_sigma(
    wave,
    n_sigma,
    center,
    window,
):
    valid = (
        np.isfinite(wave)
        & np.isfinite(n_sigma)
    )

    in_window = (
        valid
        & (wave >= center - window)
        & (wave <= center + window)
    )

    if not np.any(in_window):
        return np.nan, np.nan

    indices = np.flatnonzero(in_window)

    peak_index = indices[
        np.argmax(n_sigma[indices])
    ]

    return (
        n_sigma[peak_index],
        wave[peak_index],
    )


def line_output_dir(line_name, run_type):
    return (
        OUTPUT_BASE
        / line_name
        / run_type
    )


def plot_line_result(
    target_id,
    line_name,
    line_config,
    high_epoch,
    low_epoch,
    calculation,
    peak_n_sigma,
    peak_wave,
    redshift_map,
    output_dir,
):
    center = line_config["wavelength"]
    window = line_config["window"]
    half_width = line_config[
        "display_half_width"
    ]
    label = line_config["label"]

    redshift = redshift_map.get(
        str(target_id)
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={
            "height_ratios": [2, 1]
        },
    )

    plot_data = (
        (
            high_epoch,
            calculation["high_flux"],
            "High",
            "tab:blue",
        ),
        (
            low_epoch,
            calculation["low_flux"],
            "Low",
            "tab:orange",
        ),
    )

    for epoch, flux, state, color in plot_data:
        if (
            redshift is not None
            and np.isfinite(redshift)
        ):
            original_wave, original_flux = (
                load_original_coadd(
                    target_id,
                    epoch["date"],
                    redshift,
                )
            )

            if original_wave.size:
                # Reconstruction flux was divided by NORM
                # before being saved. Divide the raw coadd
                # by the same factor for a valid comparison.
                original_flux = (
                    original_flux
                    / epoch["normalization"]
                )

                axes[0].plot(
                    original_wave,
                    original_flux,
                    color=color,
                    linewidth=0.4,
                    alpha=0.08,
                )

                coarse_wave, coarse_flux = (
                    coarse_bin(
                        original_wave,
                        original_flux,
                    )
                )

                axes[0].plot(
                    coarse_wave,
                    coarse_flux,
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                    label=(
                        f"{state} original: "
                        f"{epoch['date']} "
                        f"(S/N={epoch['coadd_snr']:.2f}, "
                        f"NORM={epoch['normalization']:.3f})"
                    ),
                )

        axes[0].plot(
            calculation["wave"],
            flux,
            color=color,
            linewidth=2.2,
            alpha=0.7,
            label=(
                f"{state} reconstruction: "
                f"{epoch['date']} "
                f"(chi2/pixel="
                f"{epoch['chi2_per_pixel']:.3f})"
            ),
        )

    for axis in axes:
        axis.axvline(
            center,
            color="tab:green",
            linestyle=":",
            linewidth=1.2,
        )
        axis.axvspan(
            center - window,
            center + window,
            color="tab:green",
            alpha=0.08,
        )
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Normalized flux")
    axes[0].set_title(
        f"TARGETID {target_id} | "
        f"{label} ({center:.2f} Angstrom) | "
        f"peak N_sigma={peak_n_sigma:.2f} "
        f"at {peak_wave:.2f} Angstrom"
    )
    axes[0].legend(
        fontsize="small",
        loc="upper right",
    )

    axes[1].plot(
        calculation["wave"],
        calculation["n_sigma"],
        color="black",
        linewidth=1.5,
    )

    axes[1].plot(
        peak_wave,
        peak_n_sigma,
        marker="*",
        markersize=14,
        color="tab:red",
        linestyle="none",
        label=f"peak={peak_n_sigma:.2f}",
    )

    axes[1].axhline(
        SIGNIFICANCE_THRESHOLD,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_xlim(
        center - half_width,
        center + half_width,
    )
    axes[1].set_xlabel(
        "Rest-frame wavelength [Angstrom]"
    )
    axes[1].set_ylabel("N_sigma")
    axes[1].legend(
        fontsize="small",
        loc="upper right",
    )

    figure.tight_layout()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / (
        f"{line_name}_{target_id}_"
        f"{low_epoch['date']}_"
        f"{high_epoch['date']}.png"
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def analyze_target(
    target_id,
    line_name,
    line_config,
    redshift_map,
    output_dir,
    make_plot,
    plot_only_if_significant,
):
    target_id = str(target_id)
    center = line_config["wavelength"]
    window = line_config["window"]

    epochs = load_candidate_epochs(
        target_id
    )

    (
        number_after_snr,
        number_after_chi2,
        number_covering_line,
    ) = get_cut_counts(
        epochs,
        center,
    )

    result = {
        "TARGETID": target_id,
        "line_name": line_name,
        "line_label": line_config["label"],
        "line_wavelength": center,
        "search_half_width": window,
        "number_of_nights": len(epochs),
        "number_of_epochs_after_snr_cut":
            number_after_snr,
        "number_of_epochs_after_chi2_cut":
            number_after_chi2,
        "number_of_epochs_covering_line":
            number_covering_line,
        "passes_chi2_target_cut":
            number_after_chi2 >= 2,
        "high_date": None,
        "low_date": None,
        "high_flux_at_line": np.nan,
        "low_flux_at_line": np.nan,
        "high_coadd_snr": np.nan,
        "low_coadd_snr": np.nan,
        "high_normalization": np.nan,
        "low_normalization": np.nan,
        "high_chi2_per_pixel": np.nan,
        "low_chi2_per_pixel": np.nan,
        "high_chi2_pixels": 0,
        "low_chi2_pixels": 0,
        "peak_n_sigma": np.nan,
        "peak_wavelength": np.nan,
        "is_significant": False,
        "status": "failed",
        "failure_reason": "",
        "high_path": None,
        "low_path": None,
        "plot_path": None,
    }

    if len(epochs) < 2:
        result["failure_reason"] = (
            "Fewer than two reconstructed "
            "observing nights."
        )
        return result

    high_epoch, low_epoch = (
        select_high_low_epochs(
            epochs,
            center,
        )
    )

    if high_epoch is None:
        result["failure_reason"] = (
            f"Only {number_covering_line} epoch(s) "
            f"pass S/N >= {SNR_CUT}, "
            f"chi2/pixel <= {CHI2_MAX}, "
            f"and cover {center:.2f} Angstrom; "
            f"at least two are required."
        )
        return result

    result.update({
        "high_date":
            high_epoch["date"],
        "low_date":
            low_epoch["date"],
        "high_flux_at_line":
            high_epoch["selection_flux"],
        "low_flux_at_line":
            low_epoch["selection_flux"],
        "high_coadd_snr":
            high_epoch["coadd_snr"],
        "low_coadd_snr":
            low_epoch["coadd_snr"],
        "high_normalization":
            high_epoch["normalization"],
        "low_normalization":
            low_epoch["normalization"],
        "high_chi2_per_pixel":
            high_epoch["chi2_per_pixel"],
        "low_chi2_per_pixel":
            low_epoch["chi2_per_pixel"],
        "high_chi2_pixels":
            high_epoch["chi2_pixels"],
        "low_chi2_pixels":
            low_epoch["chi2_pixels"],
        "high_path":
            str(high_epoch["fits_path"]),
        "low_path":
            str(low_epoch["fits_path"]),
    })

    calculation = calculate_n_sigma(
        high_epoch,
        low_epoch,
    )

    peak_n_sigma, peak_wave = (
        calculate_peak_n_sigma(
            calculation["wave"],
            calculation["n_sigma"],
            center,
            window,
        )
    )

    if not np.isfinite(peak_n_sigma):
        result["failure_reason"] = (
            f"No finite N_sigma pixels within "
            f"+/-{window:.2f} Angstrom of "
            f"{center:.2f} Angstrom."
        )
        return result

    is_significant = (
        peak_n_sigma
        > SIGNIFICANCE_THRESHOLD
    )

    result.update({
        "peak_n_sigma":
            peak_n_sigma,
        "peak_wavelength":
            peak_wave,
        "is_significant":
            is_significant,
        "status":
            "success",
    })

    if (
        make_plot
        and (
            is_significant
            or not plot_only_if_significant
        )
    ):
        result["plot_path"] = str(
            plot_line_result(
                target_id,
                line_name,
                line_config,
                high_epoch,
                low_epoch,
                calculation,
                peak_n_sigma,
                peak_wave,
                redshift_map,
                output_dir,
            )
        )

    return result


def available_candidate_ids(candidate_ids):
    return [
        target_id
        for target_id in sorted(candidate_ids)
        if (
            RECON_BASE / target_id
        ).is_dir()
    ]


def format_value(value, decimals=4):
    if np.isfinite(value):
        return f"{value:.{decimals}f}"

    return "NaN"


def run_pipeline(
    args,
    candidate_ids,
    redshift_map,
):
    line_config = dict(
        EMISSION_LINES[args.line_name]
    )

    if args.line_window is not None:
        line_config["window"] = (
            args.line_window
        )

    if args.target_id is not None:
        if args.target_id not in candidate_ids:
            raise ValueError(
                f"TARGETID {args.target_id} "
                f"is not present in {CSV_PATH}"
            )

        selected_ids = [args.target_id]
        run_type = "single_targets"
        plot_only_if_significant = False

    else:
        selected_ids = (
            available_candidate_ids(
                candidate_ids
            )
        )

        if args.first_n is not None:
            selected_ids = selected_ids[
                :args.first_n
            ]

        run_type = "batch"
        plot_only_if_significant = (
            not args.plot_all
        )

    output_dir = line_output_dir(
        args.line_name,
        run_type,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for index, target_id in enumerate(
        selected_ids,
        start=1,
    ):
        result = analyze_target(
            target_id=target_id,
            line_name=args.line_name,
            line_config=line_config,
            redshift_map=redshift_map,
            output_dir=output_dir,
            make_plot=not args.no_plots,
            plot_only_if_significant=(
                plot_only_if_significant
            ),
        )

        results.append(result)

        peak_text = format_value(
            result["peak_n_sigma"],
            decimals=6,
        )
        high_chi2_text = format_value(
            result["high_chi2_per_pixel"]
        )
        low_chi2_text = format_value(
            result["low_chi2_per_pixel"]
        )

        print(
            f"[{index:04d}/{len(selected_ids)}] "
            f"TARGETID={target_id} "
            f"line={args.line_name} "
            f"peak_N_sigma={peak_text} "
            f"high_chi2/pixel={high_chi2_text} "
            f"low_chi2/pixel={low_chi2_text} "
            f"epochs_after_chi2="
            f"{result['number_of_epochs_after_chi2_cut']} "
            f"status={result['status']}"
        )

        if result["failure_reason"]:
            print(
                f"    Reason: "
                f"{result['failure_reason']}"
            )

    results_table = pd.DataFrame(results)

    CSV_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.target_id is not None:
        output_csv = CSV_OUTPUT_DIR / (
            f"{args.line_name}_"
            f"{args.target_id}.csv"
        )
    else:
        if args.first_n is not None:
            suffix = f"first_{args.first_n}"
        else:
            suffix = "all"

        output_csv = CSV_OUTPUT_DIR / (
            f"{args.line_name}_"
            f"{suffix}_results.csv"
        )

    results_table.to_csv(
        output_csv,
        index=False,
    )

    if results_table.empty:
        targets_after_chi2 = 0
        successful = 0
        significant = 0
    else:
        targets_after_chi2 = int(
            results_table[
                "passes_chi2_target_cut"
            ].sum()
        )

        successful = int(
            (
                results_table["status"]
                == "success"
            ).sum()
        )

        significant = int(
            results_table[
                "is_significant"
            ].sum()
        )

    print("\n========== SUMMARY ==========")
    print(
        f"Emission line: "
        f"{line_config['label']} "
        f"({line_config['wavelength']:.2f} A)"
    )
    print(
        f"Targets attempted: "
        f"{len(results_table)}"
    )
    print(
        f"Targets with >=2 epochs after "
        f"chi-square cut: {targets_after_chi2}"
    )
    print(
        f"Successfully analyzed: "
        f"{successful}"
    )
    print(
        f"Above "
        f"{SIGNIFICANCE_THRESHOLD:g} sigma: "
        f"{significant}"
    )
    print(
        f"Results CSV: {output_csv}"
    )
    print("=============================")

    return results_table


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Measure spectral-change "
            "significance at one emission line."
        )
    )

    scope = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    scope.add_argument(
        "--target-id",
        help=(
            "Analyze one TARGETID "
            "and plot it."
        ),
    )

    scope.add_argument(
        "--all",
        action="store_true",
        help="Analyze all candidates.",
    )

    scope.add_argument(
        "--first-n",
        type=int,
        metavar="N",
        help=(
            "Analyze the first N sorted "
            "candidates with reconstructions."
        ),
    )

    line_scope = parser.add_mutually_exclusive_group(
        required=True
    )

    line_scope.add_argument(
        "--line-name",
        type=str.lower,
        choices=sorted(EMISSION_LINES),
        help="Analyze one emission line.",
    )

    line_scope.add_argument(
        "--all-lines",
        action="store_true",
        help="Analyze every emission line listed in EMISSION_LINES.",
    )

    parser.add_argument(
        "--line-window",
        type=float,
        help=(
            "Override the configured +/- "
            "search window in Angstrom."
        ),
    )

    parser.add_argument(
        "--plot-all",
        action="store_true",
        help=(
            "In batch mode, plot every "
            "successful target instead of "
            "only >3 sigma."
        ),
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help=(
            "Do not create diagnostic plots."
        ),
    )

    args = parser.parse_args()

    if (
        args.first_n is not None
        and args.first_n <= 0
    ):
        parser.error(
            "--first-n must be greater than zero."
        )

    if (
        args.line_window is not None
        and args.line_window <= 0
    ):
        parser.error(
            "--line-window must be greater than zero."
        )

    return args


def main():
    args = parse_arguments()

    candidate_ids, redshift_map = load_catalog()

    print(
        f"Number of candidate IDs in CSV: "
        f"{len(candidate_ids)}"
    )

    if args.all_lines:
        line_names = list(EMISSION_LINES)
    else:
        line_names = [args.line_name]

    for line_number, line_name in enumerate(
        line_names,
        start=1,
    ):
        args.line_name = line_name

        print(
            "\n"
            "========================================"
        )
        print(
            f"Running emission line "
            f"{line_number}/{len(line_names)}: "
            f"{EMISSION_LINES[line_name]['label']} "
            f"({line_name})"
        )
        print(
            "========================================"
        )

        run_pipeline(
            args,
            candidate_ids,
            redshift_map,
        )

if __name__ == "__main__":
    main()