#!/usr/bin/env python3
"""Turn the current quassiQ VI composite PNGs into spectra and diagnostics PNGs."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from PIL import Image


PLOT_NAME = re.compile(r"(\d+)_obs_(all_obs|\d{8}(?:_\d{8})?)\.png", re.IGNORECASE)
# Pixel coordinates on the example 2048 × 1280 figure. Scale to the source size.
REFERENCE_WIDTH, REFERENCE_HEIGHT = 2048, 1280
# Axis frame positions used to verify the layout before writing crops.
COLUMN_FRAMES = ((60, 480), (575, 990), (1085, 1500), (1600, 2010))
ROW_FRAMES = ((22, 234), (273, 485), (525, 737), (776, 988), (1028, 1240))
SPECTRA_AXES = ((0, 0), (0, 1), (1, 0), (1, 1))
COMMON_DIAGNOSTICS = ((2, 0), (2, 2), (3, 2))
PAIR_DIAGNOSTICS = ((0, 2), (0, 3), (1, 2), (1, 3), (4, 2), (4, 3))


def split_figure(source, is_pair):
    """Return the two figures at native pixel resolution."""
    width, height = source.size

    def x(value):
        return round(width * value / REFERENCE_WIDTH)

    def y(value):
        return round(height * value / REFERENCE_HEIGHT)

    def crop(left, top, right, bottom):
        return source.crop((x(left), y(top), x(right), y(bottom)))

    gray = source.convert("L")
    pixels = gray.load()
    required_axes = SPECTRA_AXES + COMMON_DIAGNOSTICS
    if is_pair:
        required_axes += PAIR_DIAGNOSTICS
    for row, col in required_axes:
        left, right = COLUMN_FRAMES[col]
        columns = [x(left + step * (right - left) / 32) for step in range(33)]
        for spine in ROW_FRAMES[row]:
            # A subplot border is dark across most of the axis width.
            if not any(
                sum(pixels[column, line] < 180 for column in columns) >= 28
                for line in range(max(0, y(spine - 12)), min(height, y(spine + 12) + 1))
            ):
                raise ValueError(f"Unexpected panel layout at row {row + 1}, column {col + 1}")

    spectra = Image.new("RGB", (x(1024), y(522)), "white")
    spectra.paste(crop(0, 0, 1024, 502))
    # Below the second row, the right xlabel overlaps the next title vertically.
    spectra.paste(crop(640, 502, 1024, 522), (x(640), y(502)))

    overlay = crop(0, 502, 560, 770)
    post_weights = crop(960, 502, 1620, 756)
    overlay_and_weights = Image.new("RGB", (overlay.width + post_weights.width, overlay.height), "white")
    overlay_and_weights.paste(overlay)
    overlay_and_weights.paste(post_weights, (overlay.width, 0))
    rows = [overlay_and_weights, crop(960, 756, 1620, 1009)]
    if is_pair:
        rows = [
            crop(1024, 0, 2048, 255),        # Reconstruction N sigma
            crop(1024, 255, 2048, 502),      # Coarse N sigma
            crop(1024, 1009, 2048, 1280),    # Original N sigma
            *rows,
        ]

    padding = max(1, y(8))
    diagnostics = Image.new(
        "RGB",
        (max(row.width for row in rows), sum(row.height for row in rows) + padding * (len(rows) - 1)),
        "white",
    )
    top = 0
    for row in rows:
        diagnostics.paste(row, ((diagnostics.width - row.width) // 2, top))
        top += row.height + padding
    return spectra, diagnostics


def save_png(figure, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        figure.save(temp_path, format="PNG", compress_level=3)
        temp_path.chmod(0o644)
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)


def process_plot(item, output, overwrite):
    source, target, observation = item
    name = f"{target}_obs_{observation}.png"
    spectra_path = output / "spectra" / target / name
    diagnostics_path = output / "diagnostics" / target / name
    if not overwrite and spectra_path.is_file() and diagnostics_path.is_file():
        return "skipped", None
    try:
        with Image.open(source) as img:
            rgba = img.convert("RGBA")
            white = Image.new("RGB", img.size, "white")
            white.paste(rgba, mask=rgba.getchannel("A"))
        spectra, diagnostics = split_figure(white, is_pair="_" in observation and observation != "all_obs")
        save_png(spectra, spectra_path)
        save_png(diagnostics, diagnostics_path)
    except (OSError, ValueError) as error:
        return "failed", f"{source}: {error}"
    return "written", None


def read_targets(csv_path):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        if "TARGETID" not in (rows.fieldnames or []):
            raise ValueError(f"{csv_path} needs a TARGETID column.")
        targets = {(row["TARGETID"] or "").strip() for row in rows}
    if not targets or any(not re.fullmatch(r"\d+", target) for target in targets):
        raise ValueError(f"{csv_path} must contain exact TARGETIDs as integers.")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", type=Path, required=True, help="Root of the existing VI PNGs")
    parser.add_argument("--out-dir", type=Path, required=True, help="Separate output root")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--target-id", help="Process one TARGETID")
    selection.add_argument("--targets-csv", type=Path, help="Process TARGETIDs listed in a CSV")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing split plots")
    args = parser.parse_args()
    source = args.plots.resolve()
    output = args.out_dir.resolve()
    if not source.is_dir():
        parser.error(f"Input plot directory does not exist: {source}")
    if output == source or source in output.parents:
        parser.error("Choose an output directory outside the input plot directory.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.target_id and not re.fullmatch(r"\d+", args.target_id):
        parser.error("--target-id must be an integer.")
    try:
        targets = read_targets(args.targets_csv) if args.targets_csv else {args.target_id} if args.target_id else None
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))

    plots = {}
    for path in source.rglob("*"):
        match = PLOT_NAME.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        target, observation = match.groups()
        if targets is not None and target not in targets:
            continue
        key = target, observation.lower()
        if key in plots:
            parser.error(f"Duplicate observation for {target} ({observation}): {plots[key]} and {path}")
        plots[key] = path
    if not plots:
        parser.error(f"No matching TARGETID_obs_*.png plots under {source}.")
    missing = sorted((targets or set()) - {target for target, _ in plots})
    if missing:
        print("Requested TARGETIDs with no PNGs:", ", ".join(missing), flush=True)

    print(f"Processing {len(plots):,} observation PNGs with {args.workers} worker(s).", flush=True)
    items = ((path, target, observation) for (target, observation), path in sorted(plots.items()))
    counts = {"written": 0, "skipped": 0, "failed": 0}
    run = partial(process_plot, output=output, overwrite=args.overwrite)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (status, error) in enumerate(pool.map(run, items, chunksize=4), 1):
            counts[status] += 1
            if error:
                print(error, flush=True)
            if index % 250 == 0:
                print(f"Processed {index:,}/{len(plots):,} PNGs", flush=True)
    print(f"Observations: {counts['written']:,} written, {counts['skipped']:,} skipped, {counts['failed']:,} failed.")
    return int(bool(counts["failed"] or missing))


if __name__ == "__main__":
    raise SystemExit(main())
