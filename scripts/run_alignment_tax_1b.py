#!/usr/bin/env python3
"""Run 1b — context pollution stress (G0 vs G3 clean-What vs G3 polluted-What)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ao_da.config import load_asset_paths
from ao_da.experiment.pollution import load_run_1b_arms, load_run_1b_defaults, list_pollution_levels
from ao_da.experiment.runner import AlignmentTaxRunner
from ao_da.experiment.stats import summarize_hypotheses
from ao_da.experiment.tasks import load_alignment_tasks
from ao_da.mlx_runtime.model_pool import ModelPool


def _run_key(row: dict) -> str:
    return "|".join(
        [
            str(row.get("base_model")),
            str(row.get("task_id")),
            str(row.get("arm_id")),
            str(row.get("persona")),
            str(row.get("pollution_level")),
            str(row.get("seed")),
        ]
    )


def _serialize(result) -> dict:
    ms = result.micro_state
    ms_out = {k: v for k, v in ms.items() if k != "what_raw"} if ms else None
    row = {
        "architecture": result.config.group.value,
        "arm_id": result.config.arm_id,
        "persona": result.config.persona,
        "base_model": result.config.base,
        "pollution_level": result.config.pollution_level,
        "what_path_mode": result.config.what_path_mode,
        "how_path_mode": result.config.how_path_mode,
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
    return row


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(_run_key(json.loads(line)))
    return keys


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_1b(rows: list[dict]) -> dict:
    """Per pollution level: G0 vs G3 clean vs G3_what_polluted What means."""
    levels = sorted({r.get("pollution_level") for r in rows})
    out: dict = {"by_level": {}, "n_runs": len(rows)}
    for level in levels:
        sub = [r for r in rows if r.get("pollution_level") == level]

        def what_mean(arm_id: str) -> float:
            scores = [
                r["metrics"]["what_objective"]["what_objective_score"]
                for r in sub
                if r.get("arm_id") == arm_id
            ]
            return sum(scores) / len(scores) if scores else 0.0

        def how_mean(arm_id: str) -> float:
            scores = [
                r["metrics"]["how_constraint_coverage"]["coverage_rate"]
                for r in sub
                if r.get("arm_id") == arm_id and r["metrics"].get("how_constraint_coverage")
            ]
            return sum(scores) / len(scores) if scores else 0.0

        def tokens_mean(arm_id: str, field: str) -> float:
            vals = [
                r["logs"][field]
                for r in sub
                if r.get("arm_id") == arm_id and r.get("logs", {}).get(field) is not None
            ]
            return sum(vals) / len(vals) if vals else 0.0

        g0_w = what_mean("G0")
        g3_w = what_mean("G3")
        g3p_w = what_mean("G3_what_polluted")
        out["by_level"][level] = {
            "what_objective_mean": {
                "G0": round(g0_w, 4),
                "G3_what_clean": round(g3_w, 4),
                "G3_what_polluted": round(g3p_w, 4),
            },
            "how_coverage_mean": {
                "G0": round(how_mean("G0"), 4),
                "G3": round(how_mean("G3"), 4),
                "G3_what_polluted": round(how_mean("G3_what_polluted"), 4),
            },
            "what_prompt_tokens_mean": {
                "G0": round(tokens_mean("G0", "what_prompt_tokens_est"), 1),
                "G3": round(tokens_mean("G3", "what_prompt_tokens_est"), 1),
                "G3_what_polluted": round(tokens_mean("G3_what_polluted", "what_prompt_tokens_est"), 1),
            },
            "tax_observed_g3_clean_gt_g0": g3_w > g0_w,
            "separation_protects_what": g3_w > g3p_w,
        }
    return out


def main() -> int:
    defaults = load_run_1b_defaults()
    parser = argparse.ArgumentParser(description="AO-DA Alignment Tax Run 1b (pollution)")
    parser.add_argument("--base", default="llama_8b_4bit", choices=("llama_8b_4bit", "gemma_4b_4bit"))
    parser.add_argument("--levels", nargs="*", default=defaults.get("default_levels"))
    parser.add_argument("--tasks", nargs="*", default=defaults.get("tasks"))
    parser.add_argument("--personas", nargs="*", default=defaults.get("personas"))
    parser.add_argument("--seeds", nargs="*", type=int, default=defaults.get("seeds"))
    parser.add_argument("--temperature", type=float, default=defaults.get("temperature", 0.7))
    parser.add_argument("--arms", nargs="*", default=None, help="Subset of G0,G3,G3_what_polluted")
    parser.add_argument("--crossover", action="store_true", help="Also run what_crossover per level")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = load_run_1b_arms()
    if args.arms:
        allow = set(args.arms)
        arms = [a for a in arms if a.id in allow]

    tasks = load_alignment_tasks(task_ids=list(args.tasks))
    levels = list(args.levels)

    if args.resume_dir:
        out_dir = args.resume_dir
    elif args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = args.log_dir / f"alignment_tax_1b_{ts}"

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    completed = _load_completed(runs_path)

    print("=" * 60)
    print("Alignment Tax Run 1b — context pollution")
    print("=" * 60)
    print(f"Out:      {out_dir}")
    print(f"Base:     {args.base}")
    print(f"Levels:   {levels}  (all: {list_pollution_levels()})")
    print(f"Tasks:    {[t.id for t in tasks]}")
    print(f"Arms:     {[a.id for a in arms]}")
    print(f"Personas: {args.personas}")
    print(f"Seeds:    {args.seeds}")
    print(f"Resume:   {len(completed)} done")
    print()

    if args.dry_run:
        total = len(tasks) * len(levels) * len(arms) * len(args.personas) * len(args.seeds)
        if args.crossover:
            total += len(tasks) * len(levels) * len(args.personas) * len(args.seeds)
        print(f"Dry-run: would execute ~{total} generations")
        return 0

    paths = load_asset_paths()
    pool = ModelPool(paths=paths, base_key=args.base)  # type: ignore[arg-type]
    runner = AlignmentTaxRunner(pool, temperature=args.temperature)

    try:
        for level in levels:
            for task in tasks:
                for seed in args.seeds:
                    print(f"--- {level} / {task.id} / seed={seed} ---", flush=True)
                    for arm in arms:
                        for persona in args.personas:
                            key = _run_key(
                                {
                                    "base_model": args.base,
                                    "task_id": task.id,
                                    "arm_id": arm.id,
                                    "persona": persona,
                                    "pollution_level": level,
                                    "seed": seed,
                                }
                            )
                            if key in completed:
                                continue
                            print(f"  {arm.id} / {persona} ...", flush=True)
                            result = runner.run_arm(
                                task, arm, persona, level, seed=seed
                            )
                            row = _serialize(result)
                            _append(runs_path, row)
                            completed.add(key)
                            w = result.metrics["what_objective"]["what_objective_score"]
                            cov = result.metrics.get("how_constraint_coverage", {}).get(
                                "coverage_rate", 0
                            )
                            tok = result.logs.get("what_prompt_tokens_est", "?")
                            print(f"    what={w:.3f} how_cov={cov:.2f} tok={tok}")

                    if args.crossover:
                        for persona in args.personas:
                            key = _run_key(
                                {
                                    "base_model": args.base,
                                    "task_id": task.id,
                                    "arm_id": "what_crossover",
                                    "persona": persona,
                                    "pollution_level": level,
                                    "seed": seed,
                                }
                            )
                            if key in completed:
                                continue
                            from ao_da.experiment.pollution import build_pollution_context

                            ctx = build_pollution_context(
                                task, level, persona, what_path_mode="polluted", how_path_mode="clean"
                            )
                            print(f"  crossover / {persona} ...", flush=True)
                            xo = runner.run_what_persona_crossover(
                                task, persona, ctx, seed=seed
                            )
                            row = _serialize(xo)
                            row["arm_id"] = "what_crossover"
                            row["architecture"] = "what_crossover"
                            _append(runs_path, row)
                            completed.add(key)
    finally:
        pool.release()

    rows = [
        json.loads(l)
        for l in runs_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    summary_1b = _summarize_1b(rows)
    g0_rows = [r for r in rows if r.get("arm_id") == "G0"]
    g3_rows = [r for r in rows if r.get("arm_id") == "G3"]
    summary_legacy = {
        "g0_as_run1": summarize_hypotheses(
            [{**r, "architecture": "G0"} for r in g0_rows]
        ),
        "g3_clean_as_run1": summarize_hypotheses(
            [{**r, "architecture": "G3"} for r in g3_rows]
        ),
    }
    payload = {"run_1b": summary_1b, "reference": summary_legacy}
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "runs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "base": args.base,
                "levels": levels,
                "tasks": [t.id for t in tasks],
                "arms": [a.id for a in arms],
                "personas": args.personas,
                "seeds": args.seeds,
                "temperature": args.temperature,
                "crossover": args.crossover,
                "n_runs": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[Run 1b summary by level]")
    for level, block in summary_1b.get("by_level", {}).items():
        w = block["what_objective_mean"]
        print(
            f"  {level}: G0={w['G0']:.3f} G3_clean={w['G3_what_clean']:.3f} "
            f"G3_poll={w['G3_what_polluted']:.3f} "
            f"tax@clean={block['tax_observed_g3_clean_gt_g0']} "
            f"protect={block['separation_protects_what']}"
        )
    print(f"\nLogs: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
