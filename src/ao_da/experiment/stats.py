from __future__ import annotations

import math
from typing import Any


def _scores(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        val: Any = r
        for part in key.split("."):
            val = val.get(part, {}) if isinstance(val, dict) else {}
        if isinstance(val, (int, float)):
            out.append(float(val))
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def cliffs_delta(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    greater = sum(1 for x in a for y in b if x > y)
    lesser = sum(1 for x in a for y in b if x < y)
    return (greater - lesser) / (len(a) * len(b))


def wilcoxon_signed_rank_paired(a: list[float], b: list[float]) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank (two-sided p via normal approx for n>=5)."""
    if len(a) != len(b) or len(a) < 2:
        return {"n_pairs": len(a), "p_value": None, "significant_005": None}
    diffs = [x - y for x, y in zip(a, b)]
    nonzero = [(i, d) for i, d in enumerate(diffs) if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"n_pairs": len(a), "p_value": 1.0, "significant_005": False, "median_diff": 0.0}

    ranked: list[tuple[float, int, float]] = []
    abs_diffs = sorted((abs(d), i, d) for i, d in nonzero)
    rank = 1
    i = 0
    while i < len(abs_diffs):
        j = i
        while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2
        for k in range(i, j):
            ranked.append((avg_rank, abs_diffs[k][1], abs_diffs[k][2]))
        rank += j - i
        i = j

    w_plus = sum(r for r, _, d in ranked if d > 0)
    w_minus = sum(r for r, _, d in ranked if d < 0)
    w_stat = min(w_plus, w_minus)

    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        p = 1.0
    else:
        z = (w_stat - mu) / sigma
        p = 2 * (0.5 * (1 + math.erf(-abs(z) / math.sqrt(2))))

    return {
        "n_pairs": len(a),
        "n_nonzero": n,
        "w_stat": w_stat,
        "median_diff": float(sorted(diffs)[len(diffs) // 2]),
        "p_value": round(p, 6),
        "significant_005": p < 0.05,
    }


def _run_key(r: dict) -> str:
    return "|".join(
        [
            str(r.get("base_model")),
            str(r.get("task_id")),
            str(r.get("architecture")),
            str(r.get("persona")),
            str(r.get("seed")),
        ]
    )


def summarize_hypotheses(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate runs into H0/H1/H3 comparison tables."""

    def filter_rows(arch: str, persona: str | None = None) -> list[dict]:
        rows = [r for r in runs if r.get("architecture") == arch]
        if persona:
            rows = [r for r in rows if r.get("persona") == persona]
        return rows

    def what_mean(rows: list[dict]) -> float:
        return mean(_scores(rows, "metrics.what_objective.what_objective_score"))

    def how_cov(rows: list[dict]) -> float:
        return mean(_scores(rows, "metrics.how_constraint_coverage.coverage_rate"))

    def polite_mean(rows: list[dict]) -> float:
        return mean(_scores(rows, "metrics.persona_politeness_rate"))

    g0 = filter_rows("G0")
    g3 = filter_rows("G3")
    g4 = filter_rows("G4")
    sep_what = filter_rows("G3")
    crossover = [r for r in runs if r.get("architecture") == "what_crossover"]

    sep_scores = _scores(sep_what, "metrics.what_objective.what_objective_score")
    g0_scores = _scores(g0, "metrics.what_objective.what_objective_score")

    h0 = {
        "claim": "G0.What < separated What-path (alignment tax on What)",
        "g0_what_mean": what_mean(g0),
        "g0_what_std": std(g0_scores),
        "separated_what_mean": what_mean(sep_what),
        "separated_what_std": std(sep_scores),
        "delta_separated_minus_g0": what_mean(sep_what) - what_mean(g0),
        "tax_observed": what_mean(sep_what) > what_mean(g0),
        "cliffs_delta_sep_vs_g0": cliffs_delta(sep_scores, g0_scores),
        "wilcoxon_paired_sep_vs_g0": wilcoxon_signed_rank_paired(
            _paired_g3_what(runs)[0], _paired_g3_what(runs)[1]
        ),
    }

    h1 = {
        "claim": "G3.How > G0.How (separation improves How)",
        "g0_how_cov_mean": how_cov(g0),
        "g3_how_cov_mean": how_cov(g3),
        "g0_polite_mean": polite_mean(g0),
        "g3_polite_mean": polite_mean(g3),
        "separation_improves_coverage": how_cov(g3) >= how_cov(g0),
        "separation_lowers_politeness": polite_mean(g3) <= polite_mean(g0),
        "cliffs_delta_g3_vs_g0_how": cliffs_delta(
            _scores(g3, "metrics.how_constraint_coverage.coverage_rate"),
            _scores(g0, "metrics.how_constraint_coverage.coverage_rate"),
        ),
    }

    h1_cd = {
        "claim": "G4.How > G3.How (CD contribution)",
        "g3_how_cov_mean": how_cov(g3),
        "g4_how_cov_mean": how_cov(g4),
        "g3_polite_mean": polite_mean(g3),
        "g4_polite_mean": polite_mean(g4),
        "cd_improves_coverage": how_cov(g4) >= how_cov(g3),
        "cliffs_delta_g4_vs_g3_how": cliffs_delta(
            _scores(g4, "metrics.how_constraint_coverage.coverage_rate"),
            _scores(g3, "metrics.how_constraint_coverage.coverage_rate"),
        ),
    }

    h3_rows: dict[str, Any] = {}
    for persona in ("goku", "makima"):
        sep_p = [r for r in sep_what if r.get("persona") == persona]
        xo_p = [r for r in crossover if r.get("persona") == persona]
        sep_scores_p = _scores(sep_p, "metrics.what_objective.what_objective_score")
        xo_scores_p = _scores(xo_p, "metrics.what_objective.what_objective_score")
        h3_rows[persona] = {
            "separated_what_mean": mean(sep_scores_p),
            "separated_what_std": std(sep_scores_p),
            "crossover_what_mean": mean(xo_scores_p),
            "crossover_what_std": std(xo_scores_p),
            "abs_gap": abs(mean(sep_scores_p) - mean(xo_scores_p)),
        }

    h3 = {
        "claim": "Separated What invariant across persona LoRA on What-path (H3)",
        "per_persona": h3_rows,
        "goku_makima_separated_gap": abs(
            h3_rows.get("goku", {}).get("separated_what_mean", 0)
            - h3_rows.get("makima", {}).get("separated_what_mean", 0)
        ),
    }

    g0_persona: dict[str, float] = {}
    for persona in ("goku", "makima"):
        g0_persona[persona] = what_mean(filter_rows("G0", persona))

    mechanism = {
        "claim": "Stronger persona pressure → larger G0.What degradation",
        "g0_what_by_persona": g0_persona,
        "goku_makima_g0_gap": abs(
            g0_persona.get("goku", 0) - g0_persona.get("makima", 0)
        ),
    }

    operation = {}
    for persona in ("goku", "makima"):
        rows = [r for r in g3 if r.get("persona") == persona]
        operation[persona] = mean(
            _scores(rows, "metrics.persona_operation.persona_signal_strength")
        )

    return {
        "H0_alignment_tax_what": h0,
        "H1_separation_how": h1,
        "H1b_cd_how": h1_cd,
        "H3_what_invariance": h3,
        "G0_persona_mechanism": mechanism,
        "persona_operation_check": operation,
        "n_runs": len(runs),
    }


def _paired_g3_what(runs: list[dict]) -> tuple[list[float], list[float]]:
    """Pair G3 What scores with G0 on same (base, task, persona, seed)."""
    g0_map: dict[tuple, float] = {}
    for r in runs:
        if r.get("architecture") != "G0":
            continue
        key = (
            r.get("base_model"),
            r.get("task_id"),
            r.get("persona"),
            r.get("seed"),
        )
        g0_map[key] = r["metrics"]["what_objective"]["what_objective_score"]

    sep: list[float] = []
    g0: list[float] = []
    for r in runs:
        if r.get("architecture") != "G3":
            continue
        key = (
            r.get("base_model"),
            r.get("task_id"),
            r.get("persona"),
            r.get("seed"),
        )
        if key in g0_map:
            sep.append(r["metrics"]["what_objective"]["what_objective_score"])
            g0.append(g0_map[key])
    return sep, g0


def summarize_by_task(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    task_ids = sorted({r.get("task_id") for r in runs if r.get("task_id")})
    for tid in task_ids:
        out[tid] = summarize_hypotheses([r for r in runs if r.get("task_id") == tid])
    return out


def summarize_battery(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Full-battery rollup: overall, per base, per task, cross-base."""
    bases = sorted({r.get("base_model") for r in runs if r.get("base_model")})
    by_base = {b: summarize_hypotheses([r for r in runs if r.get("base_model") == b]) for b in bases}
    by_task = summarize_by_task(runs)

    cross_base: dict[str, Any] = {}
    if len(bases) >= 2:
        b0, b1 = bases[0], bases[1]
        h0_0 = by_base[b0]["H0_alignment_tax_what"]["delta_separated_minus_g0"]
        h0_1 = by_base[b1]["H0_alignment_tax_what"]["delta_separated_minus_g0"]
        cross_base = {
            "bases": bases,
            "delta_tax_gap": h0_1 - h0_0,
            "smaller_model_larger_tax": abs(h0_1) > abs(h0_0),
        }

    return {
        "overall": summarize_hypotheses(runs),
        "by_base": by_base,
        "by_task": by_task,
        "cross_base": cross_base,
        "n_runs": len(runs),
        "n_seeds": len({r.get("seed") for r in runs}),
        "n_tasks": len({r.get("task_id") for r in runs}),
    }
