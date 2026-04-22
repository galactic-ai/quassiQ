#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit chained Slurm jobs for SpenderQ until all coadds are processed. "
            "A coadd is considered done when recon/<coadd_tag>.done exists."
        )
    )
    parser.add_argument("--mode", choices=["start", "chain"], default="start")
    parser.add_argument(
        "--coadd-dir",
        default="/work/11161/kanyuni/ls6/quassiQ/coadds",
        help="Directory containing TARGETID subdirs with coadd-*.fits files.",
    )
    parser.add_argument(
        "--runner-script",
        default="/work/11161/kanyuni/ls6/quassiQ/src/clq_spenderq_tacc.py",
        help="Python script that processes coadds and writes done markers.",
    )
    parser.add_argument(
        "--runner-args",
        default="",
        help="Extra args passed to runner script, e.g. '--target-ids 123 456'.",
    )
    parser.add_argument("--account", default="AST25022")
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--time-hours", type=int, default=48)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--job-name", default="spenderq")
    parser.add_argument(
        "--workdir",
        default="/work/11161/kanyuni/ls6/quassiQ",
        help="Working directory used in Slurm jobs.",
    )
    parser.add_argument(
        "--venv-path",
        default="$WORK/venvs/astroenv/bin/activate",
        help="Virtualenv activation script used by Slurm jobs.",
    )
    parser.add_argument(
        "--slurm-output",
        default="/work/11161/kanyuni/ls6/quassiQ/logs/%x-%j.out",
        help="Slurm output file pattern.",
    )
    parser.add_argument(
        "--submit-dir",
        default=os.environ.get("WORK", os.getcwd()),
        help="Where the generated slurm script is written.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep checking periodically for future coadds after backlog is complete.",
    )
    parser.add_argument(
        "--watch-sleep-minutes",
        type=int,
        default=30,
        help="Sleep interval between checks when --watch is enabled.",
    )
    return parser.parse_args()


def find_pending_coadds(coadd_dir: Path) -> tuple[int, int]:
    coadd_files = sorted(coadd_dir.glob("*/coadd-*.fits"))
    pending = 0
    for coadd_path in coadd_files:
        coadd_tag = coadd_path.stem
        done_marker = coadd_path.parent / "recon" / f"{coadd_tag}.done"
        if not done_marker.exists():
            pending += 1
    return len(coadd_files), pending


def build_chain_cmd(script_path: Path, args: argparse.Namespace) -> str:
    cmd = [
        "python",
        str(script_path),
        "--mode",
        "chain",
        "--coadd-dir",
        args.coadd_dir,
        "--runner-script",
        args.runner_script,
        "--runner-args",
        args.runner_args,
        "--account",
        args.account,
        "--queue",
        args.queue,
        "--time-hours",
        str(args.time_hours),
        "--nodes",
        str(args.nodes),
        "--ntasks",
        str(args.ntasks),
        "--job-name",
        args.job_name,
        "--workdir",
        args.workdir,
        "--venv-path",
        args.venv_path,
        "--slurm-output",
        args.slurm_output,
        "--submit-dir",
        args.submit_dir,
        "--watch-sleep-minutes",
        str(args.watch_sleep_minutes),
    ]
    if args.watch:
        cmd.append("--watch")
    return " ".join(shlex.quote(c) for c in cmd)


def write_slurm_script(
    args: argparse.Namespace,
    script_path: Path,
    run_processing: bool,
) -> Path:
    submit_dir = Path(args.submit_dir)
    submit_dir.mkdir(parents=True, exist_ok=True)
    slurm_path = submit_dir / "spenderq_chain.slurm"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {args.job_name}",
        f"#SBATCH -o {args.slurm_output}",
        f"#SBATCH -p {args.queue}",
        f"#SBATCH -N {args.nodes}",
        f"#SBATCH -n {args.ntasks}",
        f"#SBATCH --time={str(args.time_hours).zfill(2)}:00:00",
        f"#SBATCH -A {args.account}",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(args.workdir)}",
        "mkdir -p logs",
        "module purge",
        f"source {args.venv_path}",
        "which python",
        "",
    ]

    if run_processing:
        runner_cmd = f"python -u {shlex.quote(args.runner_script)}"
        if args.runner_args.strip():
            runner_cmd += f" {args.runner_args.strip()}"
        lines.extend(
            [
                'echo "Starting SpenderQ processing..."',
                runner_cmd,
                'echo "SpenderQ processing step complete."',
                "",
            ]
        )
    else:
        sleep_seconds = max(1, args.watch_sleep_minutes * 60)
        lines.extend(
            [
                f'echo "No pending coadds. Sleeping {args.watch_sleep_minutes} minute(s) before next check..."',
                f"sleep {sleep_seconds}",
                "",
            ]
        )

    lines.extend(
        [
            'echo "Submitting next chained check..."',
            build_chain_cmd(script_path, args),
            "",
        ]
    )

    slurm_path.write_text("\n".join(lines), encoding="utf-8")
    return slurm_path


def submit_slurm(slurm_path: Path) -> None:
    result = subprocess.run(
        ["sbatch", str(slurm_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = result.stdout.strip()
    if stdout:
        print(stdout)


def maybe_submit_next(args: argparse.Namespace) -> int:
    coadd_dir = Path(args.coadd_dir)
    total, pending = find_pending_coadds(coadd_dir)
    print(f"Coadds found: {total} | Pending (no .done): {pending}")

    if pending > 0:
        run_processing = True
    elif args.watch:
        run_processing = False
    else:
        print("No pending coadds. Nothing to submit.")
        return 0

    script_path = Path(__file__).resolve()
    slurm_path = write_slurm_script(args, script_path, run_processing=run_processing)
    submit_slurm(slurm_path)
    return 0


def main() -> int:
    args = parse_args()

    # "start" and "chain" share submission logic; mode is retained for readability/CLI intent.
    if args.mode not in {"start", "chain"}:
        print(f"Unsupported mode: {args.mode}")
        return 1

    try:
        return maybe_submit_next(args)
    except subprocess.CalledProcessError as err:
        print("Failed to submit Slurm job.")
        if err.stdout:
            print(err.stdout)
        if err.stderr:
            print(err.stderr, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
