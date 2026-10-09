#!/usr/bin/env python3
"""Alignment Tax study — G0 / G3 / G4 × persona × task × seed × base."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ao_da.config import load_asset_paths
from ao_da.experiment.architecture import ArchitectureGroup
from ao_da.experiment.pollution import build_pollution_context
from ao_da.experiment.runner import AlignmentTaxRunner
from ao_da.experiment.stats import summarize_battery, summarize_hypotheses
from ao_da.experiment.tasks import load_alignment_tasks
from ao_da.mlx_runtime.model_pool import ModelPool


def load_battery_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc.get("full_battery") or doc


def _run_key(base: str, task_id: str, arch: str, persona: str, seed: int) -> str:
    return f"{base}|{task_id}|{arch}|{persona}|{seed}"


def _serialize_run(result) -> dict:
    ms = result.micro_state
    ms_out = None
    if ms:
        ms_out = {k: v for k, v in ms.items() if k != "what_raw"}
    return {
        "architecture": result.config.group.value,
        "persona": result.config.persona,
        "base_model": result.config.base,
        "cd_enabled": result.config.constrained_decoding,
        "seed": result.config.seed,
        "task_id": result.task_id,
        "what_raw": result.what_raw,
        "how_raw": result.how_raw,
        "how_speech": result.how_speech,
        "what_latency_ms": result.what_latency_ms,
        "how_latency_ms": result.how_latency_ms,
        "total_latency_ms": result.total_latency_ms,
        "metrics": result.metrics,
        "logs": result.logs,
        "micro_state": ms_out,
    }


def _load_completed_keys(runs_path: Path) -> set[str]:
    if not runs_path.exists():
        return set()
    if runs_path.suffix == ".jsonl":
        rows = []
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        rows = json.loads(runs_path.read_text(encoding="utf-8"))
    return {
        _run_key(
            r["base_model"],
            r["task_id"],
            r["architecture"],
            r["persona"],
            int(r["seed"]),
        )
        for r in rows
    }


def _append_run(runs_path: Path, row: dict) -> None:
    with runs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_summary(summary: dict, *, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    h0 = summary["H0_alignment_tax_what"]
    print(
        f"{prefix}H0 What: sep={h0['separated_what_mean']:.3f}±{h0.get('separated_what_std', 0):.3f} "
        f"G0={h0['g0_what_mean']:.3f}±{h0.get('g0_what_std', 0):.3f} "
        f"Δ={h0['delta_separated_minus_g0']:+.3f} tax={h0['tax_observed']}"
    )
    w = h0.get("wilcoxon_paired_sep_vs_g0") or {}
    if w.get("p_value") is not None:
        print(f"{prefix}  Wilcoxon paired sep>g0: p={w['p_value']} sig@0.05={w['significant_005']}")
    h1 = summary["H1_separation_how"]
    print(f"{prefix}H1 How cov: G3={h1['g3_how_cov_mean']:.3f} G0={h1['g0_how_cov_mean']:.3f}")
    h1b = summary["H1b_cd_how"]
    print(f"{prefix}CD: G4={h1b['g4_how_cov_mean']:.3f} G3={h1b['g3_how_cov_mean']:.3f}")


def run_battery_for_base(
    *,
    base: str,
    tasks,
    groups: list[ArchitectureGroup],
    personas: list[str],
    seeds: list[int],
    crossover: bool,
    temperature: float,
    completed: set[str],
    runs_path: Path,
) -> list[dict]:
    paths = load_asset_paths()
    pool = ModelPool(paths=paths, base_key=base)  # type: ignore[arg-type]
    runner = AlignmentTaxRunner(pool, temperature=temperature)
    new_rows: list[dict] = []

    try:
        for seed in seeds:
            for task in tasks:
                print(f"--- {base} / {task.id} / seed={seed} ---", flush=True)
                for persona in personas:
                    for group in groups:
                        arch = group.value
                        key = _run_key(base, task.id, arch, persona, seed)
                        if key in completed:
                            print(f"  skip {arch}/{persona} (done)", flush=True)
                            continue
                        print(f"  {arch} / {persona} ...", flush=True)
                        result = runner.run(task, group, persona, seed=seed)
                        row = _serialize_run(result)
                        row["temperature"] = temperature
                        row["architecture"] = arch
                        new_rows.append(row)
                        completed.add(key)
                        _append_run(runs_path, row)
                        score = result.metrics["what_objective"]["what_objective_score"]
                        speech = (result.how_speech or "")[:60]
                        print(f"    what={score:.3f} how_cov={result.metrics['how_constraint_coverage']['coverage_rate']:.2f} {speech}...")

                if crossover:
                    for persona in personas:
                        arch = "what_crossover"
                        key = _run_key(base, task.id, arch, persona, seed)
                        if key in completed:
                            print(f"  skip crossover/{persona} (done)", flush=True)
                            continue
                        print(f"  crossover / {persona} ...", flush=True)
                        ctx = build_pollution_context(
                            task,
                            "L0",
                            persona,
                            what_path_mode="polluted",
                            how_path_mode="clean",
                        )
                        xo = runner.run_what_persona_crossover(
                            task, persona, ctx, seed=seed
                        )
                        row = _serialize_run(xo)
                        row["architecture"] = arch
                        row["temperature"] = temperature
                        new_rows.append(row)
                        completed.add(key)
                        _append_run(runs_path, row)
                        score = xo.metrics["what_objective"]["what_objective_score"]
                        print(f"    crossover what={score:.3f}")
    finally:
        pool.release()

    return new_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="AO-DA Alignment Tax (G0/G3/G4)")
    parser.add_argument("--base", default=None, choices=("llama_8b_4bit", "gemma_4b_4bit"))
    parser.add_argument(
        "--bases",
        nargs="*",
        default=None,
        help="Run multiple bases sequentially (e.g. llama_8b_4bit gemma_4b_4bit)",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--groups", nargs="*", default=["G0", "G3", "G4"])
    parser.add_argument("--personas", nargs="*", default=["goku", "makima"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--crossover", action="store_true", default=True)
    parser.add_argument("--no-crossover", action="store_false", dest="crossover")
    parser.add_argument("--pilot", action="store_true", help="2 tasks, seed 0 only")
    parser.add_argument(
        "--full",
        action="store_true",
        help="3 tasks, seeds 0-4, both bases (config/alignment_tax_battery.yaml)",
    )
    parser.add_argument(
        "--battery-config",
        type=Path,
        default=ROOT / "config" / "alignment_tax_battery.yaml",
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Resume into existing log dir (reads runs.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Fixed output directory (default: new timestamped dir)",
    )
    args = parser.parse_args()

    battery = load_battery_config(args.battery_config) if args.full else {}

    task_ids = args.tasks
    seeds = list(args.seeds)
    personas = list(args.personas)
    groups = [ArchitectureGroup(g) for g in args.groups]
    crossover = args.crossover
    temperature = args.temperature if args.temperature is not None else 0.7
    bases: list[str] = []

    if args.full:
        task_ids = None
        seeds = battery.get("seeds", [0, 1, 2, 3, 4])
        personas = battery.get("personas", personas)
        groups = [ArchitectureGroup(g) for g in battery.get("groups", ["G0", "G3", "G4"])]
        crossover = battery.get("crossover", True)
        temperature = battery.get("temperature", temperature)
        bases = list(battery.get("bases", ["llama_8b_4bit", "gemma_4b_4bit"]))
    elif args.pilot:
        if not task_ids:
            task_ids = ["align_sleep_pitch", "align_legal_token"]
        seeds = [0]
        if args.base:
            bases = [args.base]
        else:
            bases = ["llama_8b_4bit"]
    else:
        if args.bases:
            bases = list(args.bases)
        elif args.base:
            bases = [args.base]
        else:
            bases = ["llama_8b_4bit"]

    tasks = load_alignment_tasks(task_ids=task_ids)

    if args.resume_dir:
        out_dir = args.resume_dir
    elif args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = "full" if args.full else ("pilot" if args.pilot else "run")
        out_dir = args.log_dir / f"alignment_tax_{suffix}_{ts}"

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    completed = _load_completed_keys(runs_path)

    print("=" * 60)
    print("Alignment Tax battery")
    print("=" * 60)
    print(f"Out:       {out_dir}")
    print(f"Bases:     {bases}")
    print(f"Tasks:     {[t.id for t in tasks]}")
    print(f"Seeds:     {seeds}")
    print(f"Temp:      {temperature}")
    print(f"Groups:    {[g.value for g in groups]}")
    print(f"Personas:  {personas}")
    print(f"Crossover: {crossover}")
    print(f"Resume:    {len(completed)} runs already logged")
    print()

    all_new: list[dict] = []
    for base in bases:
        print(f"========== Base: {base} ==========", flush=True)
        rows = run_battery_for_base(
            base=base,
            tasks=tasks,
            groups=groups,
            personas=personas,
            seeds=seeds,
            crossover=crossover,
            temperature=temperature,
            completed=completed,
            runs_path=runs_path,
        )
        all_new.extend(rows)

    all_runs: list[dict] = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_runs.append(json.loads(line))

    summary = summarize_battery(all_runs) if args.full or len(seeds) > 1 or len(bases) > 1 else {
        "overall": summarize_hypotheses(all_runs)
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "runs.json").write_text(
        json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "bases": bases,
        "tasks": [t.id for t in tasks],
        "seeds": seeds,
        "temperature": temperature,
        "personas": personas,
        "groups": [g.value for g in groups],
        "crossover": crossover,
        "n_runs": len(all_runs),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    print("[Summary]")
    overall = summary.get("overall", summary)
    _print_summary(overall)
    for base, s in (summary.get("by_base") or {}).items():
        _print_summary(s, label=base)

    print(f"\nLogs: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
