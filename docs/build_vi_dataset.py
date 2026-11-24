#!/usr/bin/env python3
"""Index existing quassiQ PNGs and merge target metadata for the static viewer."""

import argparse
import csv
import json
import math
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs, quote, unquote, urlsplit

PLOT_NAME = re.compile(r"(\d+)_obs_(all_obs|\d{8}(?:_\d{8})?)\.png", re.IGNORECASE)
LATENT_NAME = re.compile(r"Latent\d+", re.IGNORECASE)


def index_plots(root):
    """Return observation names and relative paths grouped by exact TARGETID."""
    if not root.is_dir():
        raise ValueError(f"Plot folder does not exist: {root}")
    targets = {}
    for path in sorted(root.rglob("*")):
        match = PLOT_NAME.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        target_id, observation = match.groups()
        plots = targets.setdefault(target_id, {})
        observation = observation.lower()
        if observation in plots:
            raise ValueError(f"Duplicate plot for {target_id} / {observation}: {path}")
        plots[observation] = quote(path.relative_to(root).as_posix(), safe="/")
    if not targets:
        raise ValueError(f"No TARGETID_obs_*.png files in {root}")
    return targets


def csv_rows(stream, path):
    rows = csv.DictReader(stream)
    rows.fieldnames = [
        name.strip().casefold() for name in (rows.fieldnames or [])
    ]
    if "targetid" not in rows.fieldnames:
        raise ValueError(f"{path} needs a TARGETID column")
    return rows


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_redshifts(path, targets):
    redshifts = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv_rows(stream, path)
        if "z" not in rows.fieldnames:
            raise ValueError(f"{path} needs a Z column")
        for row in rows:
            target_id = (row["targetid"] or "").strip()
            if target_id in targets and target_id not in redshifts:
                redshift = finite_float(row["z"])
                if redshift is not None:
                    redshifts[target_id] = redshift
    return redshifts


def read_subset(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            (row["targetid"] or "").strip()
            for row in csv_rows(stream, path)
            if (row["targetid"] or "").strip()
        }


def read_latent_ranges(path, targets):
    """Compute max minus min for each latent in groups with at least two rows."""
    ranges = {}
    epoch_counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv_rows(stream, path)
        columns = [name for name in rows.fieldnames if LATENT_NAME.fullmatch(name)]
        columns.sort(key=lambda name: int(name[6:]))
        if not columns:
            raise ValueError(f"{path} needs Latent1, Latent2, … columns")
        for row in rows:
            target_id = (row["targetid"] or "").strip()
            if target_id not in targets:
                continue
            epoch_counts[target_id] += 1
            for name in columns:
                value = finite_float(row[name])
                if value is None:
                    continue
                target_ranges = ranges.setdefault(target_id, {})
                low, high = target_ranges.get(name, (value, value))
                target_ranges[name] = [min(low, value), max(high, value)]
    return columns, {
        target_id: {
            name: [low, high, high - low]
            for name, (low, high) in ranges.get(target_id, {}).items()
        }
        for target_id, count in epoch_counts.items()
        if count >= 2
    }


def write_spreads(path, columns, latents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["TARGETID", *(name.title() for name in columns)])
        for target_id, values in sorted(latents.items()):
            spread_row = [values[name][2] if name in values else "" for name in columns]
            writer.writerow([target_id, *spread_row])


def write_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(dataset, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary.chmod(0o644)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def reviewer_name(value):
    if not isinstance(value, str):
        raise ValueError("Enter a reviewer name.")
    name = value.strip()
    if not name or len(name) > 80 or any(not character.isprintable() for character in name):
        raise ValueError("Enter a reviewer name of 1–80 printable characters.")
    return name


def valid_score(score):
    return type(score) in (int, float) and 1 <= score <= 5 and score * 2 == int(score * 2)


REVIEW_FLAGS = (
    "sky-subtraction-error",
    "redshift-error",
    "reconstruction-error",
    "continuum-offset",
    "lya",
    "intermediate-clq",
    "exceptional-clq",
)


def import_scores(connection, name, rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Include at least one score to import.")
    entries = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid score row.")
        target_id, score = row.get("targetId"), row.get("score")
        if not isinstance(target_id, str) or not re.fullmatch(r"[0-9]{1,20}", target_id):
            raise ValueError("Each TARGETID must be an exact integer written as text.")
        if not valid_score(score):
            raise ValueError("Scores must be from 1 to 5 in half-point steps.")
        if target_id in entries:
            raise ValueError("Each TARGETID must appear only once.")
        entries[target_id] = {"score": score, "observation": "", "reviewedAt": "", "notes": ""}

    inserted = {}
    for target_id, entry in entries.items():
        cursor = connection.execute(
            """INSERT INTO scores (reviewer_key, reviewer, target_id, score, observation, reviewed_at, notes)
               VALUES (?, ?, ?, ?, '', '', '')
               ON CONFLICT(reviewer_key, target_id) DO NOTHING""",
            (name.casefold(), name, target_id, entry["score"]),
        )
        if cursor.rowcount:
            inserted[target_id] = entry
    return {"inserted": inserted, "skipped": len(entries) - len(inserted)}


def serve_viewer(viewer, dataset_path, plot_root, plots, port, score_db):
    """Serve the viewer, index, and indexed PNGs on the local machine."""
    plot_root = plot_root.resolve()
    files = {"/": viewer, "/index.html": viewer, "/dataset.json": dataset_path}
    for observations in plots.values():
        for encoded_path in observations.values():
            relative = unquote(encoded_path)
            path = (plot_root / relative).resolve()
            if plot_root in path.parents:
                files["/plots/" + relative] = path
    score_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(score_db)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                reviewer_key TEXT NOT NULL, reviewer TEXT NOT NULL, target_id TEXT NOT NULL,
                score NUMERIC CHECK (score BETWEEN 1 AND 5),
                observation TEXT NOT NULL, reviewed_at TEXT NOT NULL, notes TEXT NOT NULL,
                flags TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (reviewer_key, target_id)
            )
        """)
        columns = {column[1] for column in connection.execute("PRAGMA table_info(scores)")}
        if "flags" not in columns:
            connection.execute("ALTER TABLE scores ADD COLUMN flags TEXT NOT NULL DEFAULT '[]'")
        connection.execute("CREATE INDEX IF NOT EXISTS scores_by_target ON scores(target_id)")

    class ViewerHandler(SimpleHTTPRequestHandler):
        def send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == "/api/scores/analysis":
                try:
                    reviewer_key = parse_qs(request.query).get("reviewerKey", [None])[0]
                    with closing(sqlite3.connect(score_db)) as connection:
                        reviewers = [
                            {"key": key, "name": name}
                            for key, name in connection.execute(
                                """SELECT reviewer_key, MIN(reviewer) FROM scores
                                   WHERE score IS NOT NULL GROUP BY reviewer_key ORDER BY reviewer_key"""
                            )
                        ]
                        if reviewer_key is None:
                            rows = connection.execute(
                                """SELECT target_id, AVG(score), COUNT(*) FROM scores
                                   WHERE score IS NOT NULL GROUP BY target_id"""
                            )
                        else:
                            rows = connection.execute(
                                """SELECT target_id, score, 1 FROM scores
                                   WHERE score IS NOT NULL AND reviewer_key = ?""",
                                (reviewer_key,),
                            )
                        self.send_json({"targets": {
                            target_id: {"score": score, "reviewCount": count}
                            for target_id, score, count in rows
                        }, "reviewers": reviewers})
                except sqlite3.Error:
                    self.send_json({"error": "Could not read analysis scores."}, 503)
                return
            if request.path != "/api/scores":
                return super().do_GET()
            params = parse_qs(request.query)
            if "reviewer" not in params:
                self.send_json({"shared": True})
                return
            try:
                name = reviewer_name(params["reviewer"][0])
                with closing(sqlite3.connect(score_db)) as connection:
                    if "target" in params:
                        rows = connection.execute(
                            """SELECT reviewer, score, observation, notes, flags FROM scores
                               WHERE target_id = ? AND reviewer_key != ? ORDER BY reviewer_key""",
                            (params["target"][0], name.casefold()),
                        )
                        self.send_json({"others": [
                            {"reviewer": reviewer, "score": score, "observation": observation,
                             "notes": notes, "flags": json.loads(flags)}
                            for reviewer, score, observation, notes, flags in rows
                        ]})
                    else:
                        rows = connection.execute(
                            """SELECT target_id, score, observation, reviewed_at, notes, flags FROM scores
                               WHERE reviewer_key = ?""", (name.casefold(),),
                        )
                        self.send_json({"scores": {
                            target_id: {"score": score, "observation": observation, "reviewedAt": reviewed_at,
                                        "notes": notes, "flags": json.loads(flags)}
                            for target_id, score, observation, reviewed_at, notes, flags in rows
                        }})
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            except sqlite3.Error:
                self.send_json({"error": "Could not read saved scores."}, 503)

        def do_POST(self):
            path = urlsplit(self.path).path
            importing = path == "/api/scores/import"
            if path != "/api/scores" and not importing:
                self.send_error(404)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json({"error": "Expected JSON."}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= (1_000_000 if importing else 16_000):
                    raise ValueError("Invalid score request size.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Invalid score request.")
                name = reviewer_name(payload.get("reviewer"))
                if importing:
                    with closing(sqlite3.connect(score_db)) as connection, connection:
                        result = import_scores(connection, name, payload.get("rows"))
                    self.send_json(result)
                    return
                flag_update = "flag" in payload
                flag = payload.get("flag")
                if flag_update and (
                    not isinstance(flag, dict)
                    or flag.get("name") not in REVIEW_FLAGS
                    or type(flag.get("enabled")) is not bool
                ):
                    raise ValueError("Invalid review flag.")
                if not flag_update and "score" not in payload:
                    raise ValueError("A score or flag is required.")
                target_id = payload.get("targetId")
                if not isinstance(target_id, str) or target_id not in plots:
                    raise ValueError("Unknown target.")
                score = payload.get("score")
                observation = payload.get("observation", "")
                notes = payload.get("notes", "")
                if not isinstance(notes, str) or len(notes) > 1500:
                    raise ValueError("Notes must be at most 1,500 characters.")
                notes = notes.strip()
                if score is not None and not valid_score(score):
                    raise ValueError("Scores must be from 1 to 5 in half-point steps.")
                entry = None
                with closing(sqlite3.connect(score_db)) as connection, connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = connection.execute(
                        "SELECT score, observation, notes, flags, reviewed_at FROM scores WHERE reviewer_key = ? AND target_id = ?",
                        (name.casefold(), target_id),
                    ).fetchone()
                    flags = set(json.loads(existing[3])) if existing else set()
                    if flag_update:
                        if existing:
                            score, observation, notes = existing[:3]
                        if flag["enabled"]:
                            flags.add(flag["name"])
                        else:
                            flags.discard(flag["name"])
                    ordered_flags = [value for value in REVIEW_FLAGS if value in flags]
                    reviewed_at = existing[4] if flag_update and existing else datetime.now(timezone.utc).isoformat()
                    if score is None and not notes and not flags:
                        connection.execute(
                            "DELETE FROM scores WHERE reviewer_key = ? AND target_id = ?",
                            (name.casefold(), target_id),
                        )
                    elif flag_update and existing:
                        connection.execute(
                            "UPDATE scores SET flags = ? WHERE reviewer_key = ? AND target_id = ?",
                            (json.dumps(ordered_flags), name.casefold(), target_id),
                        )
                    else:
                        if not isinstance(observation, str) or observation not in plots[target_id]:
                            raise ValueError("Unknown observation.")
                        connection.execute(
                            """INSERT INTO scores (reviewer_key, reviewer, target_id, score, observation, reviewed_at, notes, flags)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                               ON CONFLICT(reviewer_key, target_id) DO UPDATE SET
                               reviewer=excluded.reviewer, score=excluded.score,
                               observation=excluded.observation, reviewed_at=excluded.reviewed_at,
                               notes=excluded.notes, flags=excluded.flags""",
                            (name.casefold(), name, target_id, score, observation, reviewed_at, notes,
                             json.dumps(ordered_flags)),
                        )
                    if score is not None or notes or flags:
                        entry = {"score": score, "observation": observation, "reviewedAt": reviewed_at,
                                 "notes": notes, "flags": ordered_flags}
                self.send_json({"entry": entry})
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            except sqlite3.Error as error:
                self.log_error("Could not write review to SQLite: %s", error)
                self.send_json({"error": "The score database rejected the write. Check the viewer server log."}, 503)

        def send_head(self):
            path = files.get(unquote(urlsplit(self.path).path))
            if path is None:
                self.send_error(404)
                return None
            return super().send_head()

        def translate_path(self, request_path):
            return str(files[unquote(urlsplit(request_path).path)])

    with ThreadingHTTPServer(("127.0.0.1", port), ViewerHandler) as server:
        print(f"Open http://127.0.0.1:{server.server_port}/ in your browser.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", type=Path, required=True, help="Existing PNG directory on Lonestar6")
    parser.add_argument("--redshift-csv", type=Path, required=True, help="Catalog with TARGETID and Z")
    parser.add_argument("--latent-csv", type=Path, required=True, help="Epoch table with TARGETID and Latent1, …")
    parser.add_argument("--subset-csv", type=Path, help="Restrict the viewer and spread calculation to these TARGETIDs")
    parser.add_argument("--spread-csv", type=Path, help="Write one row of latent spreads per eligible TARGETID")
    parser.add_argument("--output", type=Path, default=Path("dataset.json"))
    parser.add_argument("--plot-url", default="plots/", help="PNG directory URL, relative to dataset.json or absolute HTTPS")
    parser.add_argument("--serve", action="store_true", help="Open the viewer through a local server with the existing PNGs")
    parser.add_argument("--viewer", type=Path, default=Path(__file__).with_name("index.html"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--score-db", type=Path, help="Shared SQLite score file (default: vi_scores.sqlite3 beside the dataset)")
    args = parser.parse_args()
    source = urlsplit(args.plot_url)
    if not args.plot_url.strip() or source.query or source.fragment or source.scheme not in ("", "http", "https"):
        parser.error("--plot-url must be a directory URL without a query or fragment.")
    plot_base = args.plot_url.rstrip("/") + "/"
    if args.serve and (plot_base != "plots/" or not args.viewer.is_file()):
        parser.error("--serve needs the default plot URL and an existing index.html (see --viewer).")
    score_db = args.score_db or args.output.with_name("vi_scores.sqlite3")
    input_paths = {path.resolve() for path in (args.redshift_csv, args.latent_csv, args.subset_csv, args.viewer) if path}
    if args.serve and score_db.resolve() in input_paths:
        parser.error("Choose a separate file for the score database.")
    if args.serve:
        input_paths.add(score_db.resolve())
    if args.output.resolve() in input_paths or (
        args.spread_csv
        and args.spread_csv.resolve() in input_paths | {args.output.resolve()}
    ):
        parser.error("Choose separate files for the inputs, dataset JSON, and spread CSV.")
    try:
        plots = index_plots(args.plots)
        eligible = read_subset(args.subset_csv) if args.subset_csv else plots.keys()
        plots = {target_id: files for target_id, files in plots.items() if target_id in eligible}
        if not plots:
            raise ValueError("No indexed plots match the subset.")
        redshifts = read_redshifts(args.redshift_csv, plots)
        columns, latents = read_latent_ranges(args.latent_csv, eligible)
        dataset = {
            "version": 1,
            "plotBase": plot_base,
            "targets": {
                target_id: {
                    "plots": observations,
                    "z": redshifts.get(target_id),
                    "latents": latents.get(target_id, {}),
                }
                for target_id, observations in sorted(plots.items())
            },
        }
        write_dataset(args.output, dataset)
        if args.spread_csv:
            write_spreads(args.spread_csv, columns, latents)
            print(f"Wrote {args.spread_csv} with {len(latents):,} targets.")
        plot_count = sum(len(observations) for observations in plots.values())
        matched_latents = sum(bool(latents.get(target_id)) for target_id in plots)
        print(f"Wrote {args.output} with {len(plots):,} targets and {plot_count:,} PNGs.")
        print(f"Matched redshift for {len(redshifts):,} targets and latent spreads for {matched_latents:,} targets.")
        if args.serve:
            serve_viewer(args.viewer, args.output, args.plots, plots, args.port, score_db)
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")

if __name__ == "__main__":
    main()
