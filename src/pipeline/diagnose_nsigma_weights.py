"""Diagnose why Ly-alpha N_sigma is NaN for one reconstructed TARGETID."""

from pathlib import Path
import argparse
import csv
import re

from astropy.io import fits
import numpy as np


RECON_BASE = Path(
    "/work/10579/prisha/ls6/desi_project/reconstructed_spectra"
)
OUTPUT_BASE = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/quassiQ/src/pipeline/"
    "clq_nsigma_results/weight_diagnostics"
)
LYA_WAVE = 1215.67


def date_from_name(path):
    match = re.search(r"-(\d{8})-\d+\.fits$", path.name)
    return match.group(1) if match else path.stem


def read_epoch(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        return {
            "path": path,
            "date": date_from_name(path),
            "wave": np.asarray(data["REST_WAVE"], dtype=float).squeeze(),
            "flux": np.asarray(data["RECON_FLUX"], dtype=float).squeeze(),
            "weights": np.asarray(data["WEIGHTS"], dtype=float).squeeze(),
            "final_weights": np.asarray(
                data["FINAL_WEIGHTS"], dtype=float
            ).squeeze(),
        }


def finite_interp(x_new, x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        return np.full(np.asarray(x_new).shape, np.nan, dtype=float)

    x_good = x[good]
    y_good = y[good]
    order = np.argsort(x_good)
    return np.interp(
        x_new,
        x_good[order],
        y_good[order],
        left=np.nan,
        right=np.nan,
    )


def scalar_interp(x_value, x, y):
    return float(finite_interp(np.asarray([x_value]), x, y)[0])


def weights_to_sigma(weights):
    sigma = np.full(weights.shape, np.nan, dtype=float)
    good = np.isfinite(weights) & (weights > 0)
    sigma[good] = 1.0 / np.sqrt(weights[good])
    return sigma


def current_nsigma_method(high, low, weight_key):
    """Reproduce the method currently used by N_sigma.py."""
    wave = high["wave"]
    high_flux = high["flux"]
    high_sigma = weights_to_sigma(high[weight_key])

    low_flux = np.interp(
        wave,
        low["wave"],
        low["flux"],
        left=np.nan,
        right=np.nan,
    )
    low_sigma = np.interp(
        wave,
        low["wave"],
        weights_to_sigma(low[weight_key]),
        left=np.nan,
        right=np.nan,
    )

    denominator = np.sqrt(high_sigma**2 + low_sigma**2)
    n_sigma = np.full(wave.shape, np.nan, dtype=float)
    good = (
        np.isfinite(high_flux)
        & np.isfinite(low_flux)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    n_sigma[good] = (high_flux[good] - low_flux[good]) / denominator[good]
    return wave, low_flux, high_sigma, low_sigma, denominator, n_sigma


def ivar_first_method(high, low, weight_key):
    """Interpolate inverse variance first, then convert it to sigma."""
    wave = high["wave"]
    high_flux = high["flux"]
    high_weight = high[weight_key]
    low_flux = finite_interp(wave, low["wave"], low["flux"])
    low_weight = finite_interp(wave, low["wave"], low[weight_key])

    denominator = np.full(wave.shape, np.nan, dtype=float)
    good_weight = (
        np.isfinite(high_weight)
        & np.isfinite(low_weight)
        & (high_weight > 0)
        & (low_weight > 0)
    )
    denominator[good_weight] = np.sqrt(
        1.0 / high_weight[good_weight]
        + 1.0 / low_weight[good_weight]
    )

    n_sigma = np.full(wave.shape, np.nan, dtype=float)
    good = (
        np.isfinite(high_flux)
        & np.isfinite(low_flux)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    n_sigma[good] = (high_flux[good] - low_flux[good]) / denominator[good]
    return low_weight, denominator, n_sigma


def print_epoch_summary(epoch):
    wave = epoch["wave"]
    nearest = int(np.nanargmin(np.abs(wave - LYA_WAVE)))
    window = np.isfinite(wave) & (np.abs(wave - LYA_WAVE) <= 5.0)

    print(f"\nEpoch {epoch['date']}: {epoch['path'].name}")
    print(f"  array length = {wave.size}")
    print(f"  wavelength range = {np.nanmin(wave):.3f} to {np.nanmax(wave):.3f} A")
    print(f"  wavelength strictly increasing = {bool(np.all(np.diff(wave) > 0))}")
    print(
        f"  nearest pixel to Ly-alpha = {wave[nearest]:.6f} A "
        f"(delta={wave[nearest] - LYA_WAVE:+.6f} A)"
    )
    print(f"  reconstructed flux at nearest pixel = {epoch['flux'][nearest]}")

    for key in ("weights", "final_weights"):
        values = epoch[key]
        positive = np.isfinite(values) & (values > 0)
        positive_window = positive & window
        print(f"  {key.upper()} at nearest pixel = {values[nearest]}")
        print(
            f"  {key.upper()} positive overall = "
            f"{np.count_nonzero(positive)}/{values.size}"
        )
        print(
            f"  {key.upper()} positive within +/-5 A = "
            f"{np.count_nonzero(positive_window)}/{np.count_nonzero(window)}"
        )


def write_epoch_arrays(epochs, output_dir):
    rows = []
    for epoch in epochs:
        region = np.isfinite(epoch["wave"]) & (
            np.abs(epoch["wave"] - LYA_WAVE) <= 20.0
        )
        for index in np.flatnonzero(region):
            rows.append({
                "date": epoch["date"],
                "filename": epoch["path"].name,
                "pixel_index": int(index),
                "rest_wave": epoch["wave"][index],
                "delta_from_lya": epoch["wave"][index] - LYA_WAVE,
                "recon_flux": epoch["flux"][index],
                "WEIGHTS": epoch["weights"][index],
                "FINAL_WEIGHTS": epoch["final_weights"][index],
            })

    path = output_dir / "epoch_weight_arrays_around_lya.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def diagnose(target_id):
    target_dir = RECON_BASE / str(target_id)
    paths = sorted(target_dir.glob("recon_*.fits"))
    if not paths:
        raise FileNotFoundError(f"No reconstruction FITS files found in {target_dir}")

    # Match N_sigma.py: use one reconstruction per observing date.
    epochs_by_date = {}
    for path in paths:
        date = date_from_name(path)
        if date not in epochs_by_date:
            epochs_by_date[date] = read_epoch(path)
    epochs = list(epochs_by_date.values())

    print(f"TARGETID = {target_id}")
    print(f"FITS files = {len(paths)}")
    print(f"Distinct observing dates = {len(epochs)}")

    for epoch in epochs:
        print_epoch_summary(epoch)

    output_dir = OUTPUT_BASE / str(target_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = write_epoch_arrays(epochs, output_dir)
    print(f"\nSaved wavelength/weight arrays: {arrays_path}")

    if len(epochs) < 2:
        print("FAILURE: N_sigma needs at least two distinct observing dates.")
        return

    fluxes = np.asarray([
        scalar_interp(LYA_WAVE, epoch["wave"], epoch["flux"])
        for epoch in epochs
    ])
    good_flux = np.isfinite(fluxes)
    if np.count_nonzero(good_flux) < 2:
        print("FAILURE: fewer than two epochs have flux coverage at Ly-alpha.")
        return

    valid_indices = np.flatnonzero(good_flux)
    high_index = valid_indices[np.argmax(fluxes[good_flux])]
    low_index = valid_indices[np.argmin(fluxes[good_flux])]
    high = epochs[high_index]
    low = epochs[low_index]

    print(f"\nHigh epoch = {high['date']}, flux(Ly-alpha) = {fluxes[high_index]}")
    print(f"Low epoch  = {low['date']}, flux(Ly-alpha) = {fluxes[low_index]}")

    comparison_rows = []
    final_values = {}

    for weight_key in ("weights", "final_weights"):
        (
            wave,
            low_flux,
            high_sigma,
            low_sigma,
            current_denominator,
            current_nsigma,
        ) = current_nsigma_method(high, low, weight_key)

        low_weight, ivar_denominator, ivar_nsigma = ivar_first_method(
            high, low, weight_key
        )
        current_at_lya = scalar_interp(LYA_WAVE, wave, current_nsigma)
        ivar_at_lya = scalar_interp(LYA_WAVE, wave, ivar_nsigma)
        final_values[weight_key] = (current_at_lya, ivar_at_lya)

        print(f"\nUsing {weight_key.upper()}:")
        print(f"  current sigma-first N_sigma(Ly-alpha) = {current_at_lya}")
        print(f"  IVAR-first N_sigma(Ly-alpha) = {ivar_at_lya}")
        print(
            "  finite current N_sigma pixels = "
            f"{np.count_nonzero(np.isfinite(current_nsigma))}/{wave.size}"
        )

        region = np.isfinite(wave) & (np.abs(wave - LYA_WAVE) <= 20.0)
        for index in np.flatnonzero(region):
            comparison_rows.append({
                "weight_source": weight_key.upper(),
                "rest_wave": wave[index],
                "delta_from_lya": wave[index] - LYA_WAVE,
                "high_flux": high["flux"][index],
                "low_flux_interpolated": low_flux[index],
                "high_weight": high[weight_key][index],
                "low_weight_interpolated": low_weight[index],
                "high_sigma": high_sigma[index],
                "low_sigma_interpolated": low_sigma[index],
                "current_denominator": current_denominator[index],
                "current_n_sigma": current_nsigma[index],
                "ivar_first_denominator": ivar_denominator[index],
                "ivar_first_n_sigma": ivar_nsigma[index],
            })

    comparison_path = output_dir / "nsigma_components_around_lya.csv"
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=comparison_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\nSaved N_sigma component arrays: {comparison_path}")
    print("\n========== DIAGNOSIS ==========")
    original_current, original_ivar = final_values["weights"]
    final_current, final_ivar = final_values["final_weights"]

    if np.isfinite(original_current):
        print("The current calculation works with WEIGHTS.")
    elif np.isfinite(original_ivar):
        print(
            "The failure is caused by interpolating sigma arrays containing "
            "NaN. Interpolate positive inverse variance first."
        )
    elif np.isfinite(final_current) or np.isfinite(final_ivar):
        print(
            "WEIGHTS are unusable at Ly-alpha, while FINAL_WEIGHTS work. "
            "Verify which FITS column represents the measurement IVAR."
        )
    else:
        print(
            "Both WEIGHTS and FINAL_WEIGHTS are non-positive/non-finite near "
            "Ly-alpha. The line is masked, so a measured N_sigma at 1215.67 A "
            "cannot be computed from these arrays."
        )
    print("===============================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    args = parser.parse_args()
    diagnose(args.target_id)


if __name__ == "__main__":
    main()
