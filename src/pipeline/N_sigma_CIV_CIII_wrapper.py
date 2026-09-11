"""Conservative C IV/C III] changing-look wrapper around N_sigma.py."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import N_sigma as base


DEFAULT_TARGET_ID = "39632986693436647"
LINE_CENTERS = {
    "CIV": 1549.48,
    "CIII]": 1908.73,
}
PRIMARY_HALF_WIDTH = 10.0
CHECK_HALF_WIDTHS = (5.0, 10.0, 20.0)
NSIGMA_THRESHOLD = 3.0
MIN_VALID_FRACTION = 0.70

OUTPUT_DIR = base.OUTPUT_DIR / "civ_ciii_single_targets"


def sorted_finite_interp(new_wave, old_wave, values):
    """Interpolate finite values after explicitly sorting wavelength."""
    good = np.isfinite(old_wave) & np.isfinite(values)
    if np.count_nonzero(good) < 2:
        return np.full(new_wave.shape, np.nan, dtype=float)

    wave_good = old_wave[good]
    values_good = values[good]
    order = np.argsort(wave_good)

    return np.interp(
        new_wave,
        wave_good[order],
        values_good[order],
        left=np.nan,
        right=np.nan,
    )


def median_line_flux(epoch, center, half_width=PRIMARY_HALF_WIDTH):
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


def choose_common_high_low_epochs(epochs):
    """Choose one high/low pair using both line-core windows together."""
    scored = []

    for epoch in epochs:
        civ_flux = median_line_flux(epoch, LINE_CENTERS["CIV"])
        ciii_flux = median_line_flux(epoch, LINE_CENTERS["CIII]"])

        if np.isfinite(civ_flux) and np.isfinite(ciii_flux):
            # Equal weighting of the two line-window median fluxes.
            state_score = 0.5 * (civ_flux + ciii_flux)
            scored.append((state_score, epoch, civ_flux, ciii_flux))

    if len(scored) < 2:
        raise RuntimeError(
            "Fewer than two epochs have finite reconstructed flux in both "
            "the C IV and C III] windows."
        )

    scored.sort(key=lambda item: item[0])
    low = scored[0]
    high = scored[-1]
    return high, low


def calculate_nsigma_ivar_first(high_epoch, low_epoch):
    """Calculate N_sigma after interpolating inverse variance, not sigma."""
    wave = np.asarray(high_epoch["rest_wave"], dtype=float)
    high_flux = np.asarray(high_epoch["recon_flux"], dtype=float)
    high_ivar = np.asarray(high_epoch["weights"], dtype=float)

    low_flux = sorted_finite_interp(
        wave,
        np.asarray(low_epoch["rest_wave"], dtype=float),
        np.asarray(low_epoch["recon_flux"], dtype=float),
    )
    low_ivar = sorted_finite_interp(
        wave,
        np.asarray(low_epoch["rest_wave"], dtype=float),
        np.asarray(low_epoch["weights"], dtype=float),
    )

    denominator = np.full(wave.shape, np.nan, dtype=float)
    valid_ivar = (
        np.isfinite(high_ivar)
        & np.isfinite(low_ivar)
        & (high_ivar > 0)
        & (low_ivar > 0)
    )
    denominator[valid_ivar] = np.sqrt(
        1.0 / high_ivar[valid_ivar]
        + 1.0 / low_ivar[valid_ivar]
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

    return {
        "wave": wave,
        "high_flux": high_flux,
        "low_flux": low_flux,
        "high_ivar": high_ivar,
        "low_ivar": low_ivar,
        "denominator": denominator,
        "n_sigma": n_sigma,
    }


def summarize_line(calculation, center, half_width):
    wave = calculation["wave"]
    n_sigma = calculation["n_sigma"]
    inside = (
        np.isfinite(wave)
        & (wave >= center - half_width)
        & (wave <= center + half_width)
    )
    valid = inside & np.isfinite(n_sigma)
    total_pixels = int(np.count_nonzero(inside))
    valid_pixels = int(np.count_nonzero(valid))
    valid_fraction = (
        valid_pixels / total_pixels if total_pixels > 0 else 0.0
    )

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


def make_plot(target_id, high, low, calculation, primary_summaries):
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))

    for column, (line_name, center) in enumerate(LINE_CENTERS.items()):
        spectrum_axis = axes[0, column]
        nsigma_axis = axes[1, column]
        display_half_width = 40.0

        for flux, label, color in (
            (calculation["high_flux"], f"High: {high['date']}", "tab:blue"),
            (calculation["low_flux"], f"Low: {low['date']}", "tab:orange"),
        ):
            spectrum_axis.plot(
                calculation["wave"], flux, color=color, linewidth=1.5,
                label=label,
            )

        for axis in (spectrum_axis, nsigma_axis):
            axis.axvspan(
                center - PRIMARY_HALF_WIDTH,
                center + PRIMARY_HALF_WIDTH,
                color="tab:green",
                alpha=0.12,
                label="Primary +/-10 A window" if axis is spectrum_axis else None,
            )
            axis.axvline(center, color="tab:green", linestyle=":", linewidth=1)
            axis.set_xlim(center - display_half_width, center + display_half_width)
            axis.grid(alpha=0.25)

        summary = primary_summaries[line_name]
        spectrum_axis.set_title(
            f"{line_name} | median N_sigma={summary['median_nsigma']:.2f} | "
            f"valid={summary['valid_fraction']:.0%}"
        )
        spectrum_axis.set_ylabel("Reconstructed flux")
        spectrum_axis.legend(fontsize="small")

        nsigma_axis.plot(
            calculation["wave"], calculation["n_sigma"],
            color="black", linewidth=1.3,
        )
        nsigma_axis.axhline(
            NSIGMA_THRESHOLD, color="tab:red", linestyle="--", linewidth=1,
        )
        nsigma_axis.set_xlabel(r"Rest-frame wavelength [$\AA$]")
        nsigma_axis.set_ylabel(r"$N_\sigma$")

    figure.suptitle(f"TARGETID {target_id}: C IV and C III] variability")
    figure.tight_layout()
    output_path = OUTPUT_DIR / f"civ_ciii_nsigma_{target_id}.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run(target_id):
    target_id = str(target_id)
    epochs = base.load_candidate_epochs(target_id)
    if len(epochs) < 2:
        raise RuntimeError(
            f"TARGETID {target_id} has only {len(epochs)} usable observing night(s)."
        )

    high_info, low_info = choose_common_high_low_epochs(epochs)
    _, high, high_civ_flux, high_ciii_flux = high_info
    _, low, low_civ_flux, low_ciii_flux = low_info
    calculation = calculate_nsigma_ivar_first(high, low)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {}
    result = {
        "TARGETID": target_id,
        "number_of_nights": len(epochs),
        "high_date": high["date"],
        "low_date": low["date"],
        "high_CIV_median_flux": high_civ_flux,
        "low_CIV_median_flux": low_civ_flux,
        "high_CIII_median_flux": high_ciii_flux,
        "low_CIII_median_flux": low_ciii_flux,
    }

    for line_name, center in LINE_CENTERS.items():
        print(f"\n{line_name} (center={center:.2f} A)")
        for half_width in CHECK_HALF_WIDTHS:
            summary = summarize_line(calculation, center, half_width)
            label = f"{int(2 * half_width)}A_window"
            for key, value in summary.items():
                result[f"{line_name}_{label}_{key}"] = value

            print(
                f"  +/-{half_width:.0f} A: "
                f"median N_sigma={summary['median_nsigma']:.6f}, "
                f"25th percentile={summary['p25_nsigma']:.6f}, "
                f"valid={summary['valid_pixels']}/{summary['total_pixels']} "
                f"({summary['valid_fraction']:.1%})"
            )

        summaries[line_name] = summarize_line(
            calculation, center, PRIMARY_HALF_WIDTH
        )

    line_passes = {}
    for line_name, summary in summaries.items():
        line_passes[line_name] = bool(
            summary["valid_fraction"] >= MIN_VALID_FRACTION
            and np.isfinite(summary["median_nsigma"])
            and summary["median_nsigma"] > NSIGMA_THRESHOLD
        )
        result[f"{line_name}_passes"] = line_passes[line_name]

    result["conservative_cl_both_lines"] = all(line_passes.values())

    plot_path = make_plot(target_id, high, low, calculation, summaries)
    result["plot_path"] = str(plot_path)

    result_path = OUTPUT_DIR / f"civ_ciii_nsigma_{target_id}.csv"
    pd.DataFrame([result]).to_csv(result_path, index=False)

    pixel_region = np.zeros(calculation["wave"].shape, dtype=bool)
    for center in LINE_CENTERS.values():
        pixel_region |= np.abs(calculation["wave"] - center) <= 20.0

    pixel_path = OUTPUT_DIR / f"civ_ciii_nsigma_pixels_{target_id}.csv"
    pd.DataFrame({
        "rest_wave": calculation["wave"][pixel_region],
        "high_flux": calculation["high_flux"][pixel_region],
        "low_flux": calculation["low_flux"][pixel_region],
        "high_ivar": calculation["high_ivar"][pixel_region],
        "low_ivar": calculation["low_ivar"][pixel_region],
        "denominator": calculation["denominator"][pixel_region],
        "n_sigma": calculation["n_sigma"][pixel_region],
    }).to_csv(pixel_path, index=False)

    print("\n========== CONSERVATIVE RESULT ==========")
    print(f"TARGETID = {target_id}")
    print(f"High/low dates = {high['date']} / {low['date']}")
    print(f"C IV passes = {line_passes['CIV']}")
    print(f"C III] passes = {line_passes['CIII]']}")
    print(f"Both lines pass = {result['conservative_cl_both_lines']}")
    print(f"Result CSV = {result_path}")
    print(f"Pixel CSV = {pixel_path}")
    print(f"Plot = {plot_path}")
    print("=========================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", default=DEFAULT_TARGET_ID)
    args = parser.parse_args()
    run(args.target_id)


if __name__ == "__main__":
    main()
