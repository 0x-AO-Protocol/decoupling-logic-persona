#!/usr/bin/env python3
"""Table 7 (system properties) and Section 6 figures from the June 2026 verification turns.

Usage:  python3 -I analysis/system_props.py <logs_dir> <out_dir>

Reads logs/turn_*.json and writes <out_dir>/table7_system.json. The memory row of
Table 7 is not computed here (it was read from a prototype session's terminal
output, which is not part of this release; see README), and the first-turn latency
row comes from analysis/out/table4_tokens_latency.csv (Run 1b).
"""
import glob
import json
import os
import statistics
import sys

LOGS = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def load(pattern):
    files = sorted(glob.glob(os.path.join(LOGS, pattern)))
    return [(os.path.basename(f), json.load(open(f, encoding="utf-8"))) for f in files]


res = {}

# Turn 2-A: persona switch Goku -> Makima with unchanged vitals (cache should hit).
t2a = load("turn_2a_*.json")
swaps = [d["turn_2a"]["logs"]["swap_ms_goku_to_makima"] for _, d in t2a]
warm = swaps[1:]  # files sort by timestamp; the first run is the cold call
how = [d["turn_2a"]["logs"]["how_path_latency_ms"] / 1000 for _, d in t2a]
res["turn_2a"] = dict(
    runs=len(t2a),
    swap_first_call_ms=round(swaps[0], 1),
    swap_warm_runs=len(warm),
    swap_warm_median_ms=round(statistics.median(warm), 2),
    swap_warm_min_ms=round(min(warm), 2),
    swap_warm_max_ms=round(max(warm), 2),
    cache_hits=sum(bool(d["turn_2a"]["logs"]["cache_hit"]) for _, d in t2a),
    what_path_invocations_zero=sum(d["turn_2a"]["logs"]["what_path_invocations"] == 0 for _, d in t2a),
    json_hash_match=sum(bool(d["turn_2a"]["kpi"]["json_hash_match"]) for _, d in t2a),
    how_only_latency_median_s=round(statistics.median(how), 1),
    how_only_latency_min_s=round(min(how), 1),
    how_only_latency_max_s=round(max(how), 1),
)

# Turn 2-B (vitals changed) and Turn 2-C (user override): cache should miss.
for key in ("turn_2b", "turn_2c"):
    runs = load(f"{key}_*.json")
    res[key] = dict(
        runs=len(runs),
        cache_misses=sum(not d[key]["logs"]["cache_hit"] for _, d in runs),
        what_path_reinvoked=sum(d[key]["logs"]["what_path_invocations"] >= 1 for _, d in runs),
        invalidate_reasons=sorted({d[key]["logs"]["cache_invalidate_reason"] for _, d in runs}),
        constraint_sets=sorted({tuple(d[key]["constraints"]) for _, d in runs}),
    )
    if key == "turn_2b":
        res[key]["recovery_pct_turn_1a"] = sorted({d["turn_1a"]["recovery_pct"] for _, d in runs})
        res[key]["recovery_pct_turn_2b"] = sorted({d[key]["recovery_pct"] for _, d in runs})

# Turn 3-A / 3-B: deterministic policy router, and the one LLM-classifier comparison.
t3 = load("turn_3a_*.json") + load("turn_3b_*.json")
router = [d["policy_router"]["router_latency_ms"] for _, d in t3]
cmp_runs = [(f, d["routing_baseline_comparison"]) for f, d in t3 if d.get("routing_baseline_comparison")]
res["turn_3"] = dict(
    runs=len(t3),
    router_latency_min_ms=round(min(router), 3),
    router_latency_max_ms=round(max(router), 3),
    llm_classifier_runs=len(cmp_runs),
    llm_classifier=[dict(
        file=f,
        latency_ms=round(c["llm_classifier"]["latency_ms"]),
        prompt_tokens_est=c["llm_classifier"]["prompt_tokens_est"],
        completion_tokens_est=c["llm_classifier"]["completion_tokens_est"],
        routing_match=c["routing_match"],
    ) for f, c in cmp_runs],
)

json.dump(res, open(os.path.join(OUT, "table7_system.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
