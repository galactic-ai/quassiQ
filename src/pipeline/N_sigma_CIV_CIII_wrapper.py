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
    good = np.isfinite(old_wave) & np.isfinite(values) & (old_wave > 0)
    if np.count_nonzero(good) < 2:
        return np.full(new_wave.shape, np.nan, dtype=float)

    wave_good = old_wave[good]
    values_good = values[good]
    order = np.argsort(wave_good)
    wave_good = wave_good[order]
    values_good = values_good[order]
    wave_good, unique_indices = np.unique(wave_good, return_index=True)
    values_good = values_good[unique_indices]

    return np.interp(
        new_wave,
        wave_good,
        values_good,
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


def choose_high_low_epochs(epochs, center):
    """Choose high and low epochs independently for one emission line."""
    scored = []

    for epoch in epochs:
        line_flux = median_line_flux(epoch, center)
        if np.isfinite(line_flux):
            scored.append((line_flux, epoch))

    if len(scored) < 2:
        raise RuntimeError(
            f"Fewer than two epochs have finite reconstructed flux within "
            f"{PRIMARY_HALF_WIDTH:.0f} A of {center:.2f} A."
        )

    scored.sort(key=lambda item: item[0])
    low = scored[0]
    high = scored[-1]
    high_flux, high_epoch = high
    low_flux, low_epoch = low
    return high_epoch, low_epoch, high_flux, low_flux


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
    n_sigma[valid] = np.abs(
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


def make_plot(target_id, line_results):
    """Match the original N_sigma plot style for both emission lines."""
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    redshift = base.redshift_map.get(str(target_id))

    for column, (line_name, center) in enumerate(LINE_CENTERS.items()):
        line_result = line_results[line_name]
        high = line_result["high"]
        low = line_result["low"]
        calculation = line_result["calculation"]
        summary = line_result["primary_summary"]
        spectrum_axis = axes[0, column]
        nsigma_axis = axes[1, column]
        display_half_width = 40.0

        for epoch, recon_flux, state, color in (
            (high, calculation["high_flux"], "High", "tab:blue"),
            (low, calculation["low_flux"], "Low", "tab:orange"),
        ):
            if redshift is not None and np.isfinite(redshift):
                original_wave, original_flux = base.load_original_coadd(
                    recon_path=epoch["fits_path"],
                    target_id=target_id,
                    redshift=redshift,
                )
                if original_wave.size > 0:
                    spectrum_axis.plot(
                        original_wave,
                        original_flux,
                        color=color,
                        linewidth=0.4,
                        alpha=0.08,
                    )
                    coarse_wave, coarse_flux = base.coarse_bin(
                        original_wave,
                        original_flux,
                        n_points=10,
                    )
                    spectrum_axis.plot(
                        coarse_wave,
                        coarse_flux,
                        color=color,
                        linewidth=1.5,
                        linestyle="--",
                        alpha=0.7,
                        label=f"{state} original coarse: {epoch['date']}",
                    )

            spectrum_axis.plot(
                calculation["wave"],
                recon_flux,
                color=color,
                linewidth=2.2,
                alpha=0.7,
                label=f"{state} reconstruction: {epoch['date']}",
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "TARGETID": target_id,
        "number_of_nights": len(epochs),
    }
    line_results = {}
    pixel_tables = []
    line_passes = {}

    for line_name, center in LINE_CENTERS.items():
        high, low, high_line_flux, low_line_flux = choose_high_low_epochs(
            epochs,
            center,
        )
        calculation = calculate_nsigma_ivar_first(high, low)

        result[f"{line_name}_high_date"] = high["date"]
        result[f"{line_name}_low_date"] = low["date"]
        result[f"{line_name}_high_median_flux"] = high_line_flux
        result[f"{line_name}_low_median_flux"] = low_line_flux

        print(f"\n{line_name} (center={center:.2f} A)")
        print(
            f"  high/low dates = {high['date']} / {low['date']} "
            f"(median flux {high_line_flux:.6f} / {low_line_flux:.6f})"
        )

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

        primary_summary = summarize_line(
            calculation, center, PRIMARY_HALF_WIDTH
        )
        line_passes[line_name] = bool(
            primary_summary["valid_fraction"] >= MIN_VALID_FRACTION
            and np.isfinite(primary_summary["median_nsigma"])
            and primary_summary["median_nsigma"] > NSIGMA_THRESHOLD
        )
        result[f"{line_name}_passes"] = line_passes[line_name]

        line_results[line_name] = {
            "high": high,
            "low": low,
            "calculation": calculation,
            "primary_summary": primary_summary,
        }

        pixel_region = (
            np.isfinite(calculation["wave"])
            & (np.abs(calculation["wave"] - center) <= 20.0)
        )
        pixel_tables.append(pd.DataFrame({
            "line": line_name,
            "rest_wave": calculation["wave"][pixel_region],
            "high_date": high["date"],
            "low_date": low["date"],
            "high_flux": calculation["high_flux"][pixel_region],
            "low_flux": calculation["low_flux"][pixel_region],
            "high_ivar": calculation["high_ivar"][pixel_region],
            "low_ivar": calculation["low_ivar"][pixel_region],
            "denominator": calculation["denominator"][pixel_region],
            "n_sigma": calculation["n_sigma"][pixel_region],
        }))

    result["conservative_cl_both_lines"] = all(line_passes.values())

    plot_path = make_plot(target_id, line_results)
    result["plot_path"] = str(plot_path)

    result_path = OUTPUT_DIR / f"civ_ciii_nsigma_{target_id}.csv"
    pd.DataFrame([result]).to_csv(result_path, index=False)

    pixel_path = OUTPUT_DIR / f"civ_ciii_nsigma_pixels_{target_id}.csv"
    pd.concat(pixel_tables, ignore_index=True).to_csv(pixel_path, index=False)

    print("\n========== CONSERVATIVE RESULT ==========")
    print(f"TARGETID = {target_id}")
    for line_name in LINE_CENTERS:
        print(
            f"{line_name} high/low dates = "
            f"{result[f'{line_name}_high_date']} / "
            f"{result[f'{line_name}_low_date']}"
        )
    print(f"C IV passes = {line_passes['CIV']}")
    print(f"C III] passes = {line_passes['CIII]']}")
    print(f"Both lines pass = {result['conservative_cl_both_lines']}")
    print(f"Result CSV = {result_path}")
    print(f"Pixel CSV = {pixel_path}")
    print(f"Plot = {plot_path}")
    print("=========================================")


def analyze_target_for_batch(target_id):
    """Return conservative C IV/C III] metrics without making a plot."""
    target_id = str(target_id)
    epochs = base.load_candidate_epochs(target_id)
    if len(epochs) < 2:
        raise RuntimeError(
            f"Only {len(epochs)} usable observing night(s) were found."
        )

    result = {
        "TARGETID": target_id,
        "number_of_nights": len(epochs),
        "status": "success",
        "failure_reason": "",
    }

    valid_line_scores = {}

    for line_name, center in LINE_CENTERS.items():
        high, low, high_flux, low_flux = choose_high_low_epochs(
            epochs,
            center,
        )
        calculation = calculate_nsigma_ivar_first(high, low)
        summary = summarize_line(
            calculation,
            center,
            PRIMARY_HALF_WIDTH,
        )

        result[f"{line_name}_high_date"] = high["date"]
        result[f"{line_name}_low_date"] = low["date"]
        result[f"{line_name}_high_median_flux"] = high_flux
        result[f"{line_name}_low_median_flux"] = low_flux
        result[f"{line_name}_median_nsigma"] = summary["median_nsigma"]
        result[f"{line_name}_p25_nsigma"] = summary["p25_nsigma"]
        result[f"{line_name}_valid_fraction"] = summary["valid_fraction"]
        result[f"{line_name}_valid_pixels"] = summary["valid_pixels"]
        result[f"{line_name}_total_pixels"] = summary["total_pixels"]

        if (
            summary["valid_fraction"] >= MIN_VALID_FRACTION
            and np.isfinite(summary["median_nsigma"])
        ):
            valid_line_scores[line_name] = summary["median_nsigma"]

    if valid_line_scores:
        trigger_line = max(valid_line_scores, key=valid_line_scores.get)
        selection_score = float(valid_line_scores[trigger_line])
    else:
        trigger_line = None
        selection_score = np.nan

    result["trigger_line"] = trigger_line
    result["selection_score"] = selection_score
    result["any_line_gt3"] = bool(selection_score > 3.0)
    result["any_line_gt2"] = bool(selection_score > 2.0)

    civ_score = result["CIV_median_nsigma"]
    ciii_score = result["CIII]_median_nsigma"]
    result["both_lines_gt3"] = bool(civ_score > 3.0 and ciii_score > 3.0)
    result["both_lines_gt2"] = bool(civ_score > 2.0 and ciii_score > 2.0)

    return result


def make_reconstruction_overlay(target_id, trigger_line, threshold):
    """Plot the high/low reconstructions in the style of the reference."""
    epochs = base.load_candidate_epochs(str(target_id))
    center = LINE_CENTERS[trigger_line]
    high, low, _, _ = choose_high_low_epochs(epochs, center)

    figure, axis = plt.subplots(figsize=(16, 6))

    for epoch, color, label in (
        (high, "tab:blue", f"High: {epoch_label(high)}"),
        (low, "tab:red", f"Low: {epoch_label(low)}"),
    ):
        wave = np.asarray(epoch["rest_wave"], dtype=float)
        flux = np.asarray(epoch["recon_flux"], dtype=float)
        valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
        order = np.argsort(wave[valid])
        axis.plot(
            wave[valid][order],
            flux[valid][order],
            color=color,
            linewidth=1.4,
            alpha=0.75,
            label=label,
        )

    axis.axvline(
        LINE_CENTERS["CIV"],
        color="tab:blue",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
    )
    axis.axvline(
        LINE_CENTERS["CIII]"],
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
    )
    axis.text(
        LINE_CENTERS["CIV"],
        0.98,
        "C IV",
        color="tab:blue",
        ha="center",
        va="top",
        transform=axis.get_xaxis_transform(),
    )
    axis.text(
        LINE_CENTERS["CIII]"],
        0.98,
        "C III]",
        color="tab:red",
        ha="center",
        va="top",
        transform=axis.get_xaxis_transform(),
    )

    axis.set_title(
        f"TARGETID {target_id} | Recon Overlay\n"
        f"Selected by {trigger_line}: median N_sigma > {threshold:g}"
    )
    axis.set_xlabel(r"Rest Wavelength [$\AA$]")
    axis.set_ylabel("Flux")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize="small")
    figure.tight_layout()

    path = (
        OUTPUT_DIR
        / "batch_first_100"
        / f"recon_overlay_{target_id}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def epoch_label(epoch):
    return f"{epoch['date']} | {epoch['fits_path'].name}"


def run_first_n(number_of_targets=100):
    """Analyze a deterministic sample and plot threshold-selected targets."""
    available_ids = [
        target_id
        for target_id in sorted(base.candidate_ids)
        if (base.RECON_BASE / target_id).is_dir()
    ]
    selected_ids = available_ids[:number_of_targets]

    if len(selected_ids) < number_of_targets:
        raise RuntimeError(
            f"Only {len(selected_ids)} accessible reconstruction directories "
            f"were found; requested {number_of_targets}."
        )

    rows = []
    for index, target_id in enumerate(selected_ids, start=1):
        try:
            result = analyze_target_for_batch(target_id)
        except Exception as error:
            result = {
                "TARGETID": target_id,
                "status": "failed",
                "failure_reason": str(error),
                "selection_score": np.nan,
                "trigger_line": None,
                "any_line_gt3": False,
                "any_line_gt2": False,
                "both_lines_gt3": False,
                "both_lines_gt2": False,
            }

        rows.append(result)
        score = result.get("selection_score", np.nan)
        score_text = f"{score:.3f}" if np.isfinite(score) else "NaN"
        print(
            f"[{index:03d}/{number_of_targets}] TARGETID={target_id} "
            f"max_line_median_N_sigma={score_text} "
            f"line={result.get('trigger_line')} status={result['status']}"
        )

    results = pd.DataFrame(rows)
    output_dir = OUTPUT_DIR / "batch_first_100"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results_path = output_dir / "civ_ciii_first_100_all_results.csv"
    results.to_csv(all_results_path, index=False)

    above_three = results[
        (results["status"] == "success")
        & (results["selection_score"] > 3.0)
    ].copy()

    if len(above_three) > 0:
        threshold = 3.0
        candidates = above_three
        fallback_used = False
    else:
        threshold = 2.0
        candidates = results[
            (results["status"] == "success")
            & (results["selection_score"] > 2.0)
        ].copy()
        fallback_used = True

    candidates = candidates.sort_values(
        "selection_score",
        ascending=False,
    )

    plot_paths = []
    for _, candidate in candidates.iterrows():
        plot_path = make_reconstruction_overlay(
            str(candidate["TARGETID"]),
            candidate["trigger_line"],
            threshold,
        )
        plot_paths.append(str(plot_path))

    if len(candidates) > 0:
        candidates["overlay_plot"] = plot_paths
    else:
        candidates["overlay_plot"] = pd.Series(dtype=str)

    candidates_path = output_dir / (
        f"civ_ciii_candidates_above_{int(threshold)}sigma.csv"
    )
    candidates.to_csv(candidates_path, index=False)

    print("\n========== BATCH SELECTION ==========")
    print(f"Targets attempted: {len(results)}")
    print(f"Targets with any line above 3 sigma: {len(above_three)}")
    print(f"Fallback to 2 sigma used: {fallback_used}")
    print(f"Selection threshold: > {threshold:g} sigma")
    print(f"Selected targets: {len(candidates)}")
    print(f"All results: {all_results_path}")
    print(f"Selected candidates: {candidates_path}")
    print(f"Overlay directory: {output_dir}")
    print("=====================================")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--target-id")
    mode.add_argument(
        "--first-n",
        type=int,
        help="Analyze the first N sorted targets with reconstructions.",
    )
    args = parser.parse_args()

    if args.first_n is not None:
        run_first_n(args.first_n)
    else:
        run(args.target_id or DEFAULT_TARGET_ID)


if __name__ == "__main__":
    main()
