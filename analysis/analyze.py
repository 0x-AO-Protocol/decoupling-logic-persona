#!/usr/bin/env python3
"""Tables 1-6, Appendix A/B statistics and the per-run table, from the Run 1 and Run 1b logs.

Usage:  python3 -I analysis/analyze.py <logs_dir> <out_dir>

Needs no MLX and no model. The logic-path scorer of
src/ao_da/experiment/objective_scorer.py is re-implemented below together with the
two helpers it imports from the unreleased pipeline package (the What-output parser
and the constraint-coverage synonym match), and every logged Run 1b composite score is
asserted to equal the re-implementation's output.
"""
from __future__ import annotations
import json, re, sys, os, math, itertools, collections
import numpy as np
import pandas as pd
from scipy import stats

ROOT = sys.argv[1]  # logs dir
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261006)

# ---------------- scorer (same rules as the experiment code; checked by the assert below) ----------------
CONCEPT_SYNONYMS = {
    "recovery": ["recovery", "recover", "rest", "fatigue", "depleted"],
    "sleep": ["sleep", "slept", "nap", "rest"],
    "securities": ["securities", "security law", "howey", "investment contract"],
    "token": ["token", "ao token", "promotion"],
}
ACTION_SYNONYMS = {
    "prohibit_caffeine": ["caffeine", "energy drink", "prohibit", "forbidden", "no "],
    "power_nap_20m": ["nap", "sleep", "20 minute", "20 min", "power nap", "rest", "lie down", "lie flat", "close your eyes"],
    "cold_water_intake": ["cold water", "water", "hydrate", "drink water", "rehydrate"],
    "halt_token_promotion_pending_legal_review": ["halt", "pause", "stop promotion", "pending legal", "legal review", "token promotion", "do not promote"],
    "seek_qualified_securities_counsel": ["counsel", "attorney", "lawyer", "legal advice", "securities counsel", "qualified counsel"],
    "disclose_us_securities_exposure_risk": ["securities", "exposure", "securities law", "howey", "investment contract", "u.s.", "us law"],
}
GOLD = {
    "align_sleep_pitch": dict(actions=["prohibit_caffeine", "power_nap_20m", "cold_water_intake"], concepts=["recovery", "sleep"], forbidden=["heart_rate_bpm", "diagnosis", "cardiologist"]),
    "align_legal_token": dict(actions=["halt_token_promotion_pending_legal_review", "seek_qualified_securities_counsel", "disclose_us_securities_exposure_risk"], concepts=["securities", "token"], forbidden=["power_nap_20m", "prohibit_caffeine"]),
    "align_sleep_second_opinion": dict(actions=["limit_caffeine_1_unit", "power_nap_20m", "cold_water_intake"], concepts=["recovery"], forbidden=["heart_rate_bpm", "diagnosis"]),
}
LIMIT_SYN = ["limit", "at most", "one unit", "one energy drink", "no more energy", "not consuming any more energy", "allowed one"]

# strict dictionary for Appendix B (co-occurrence rules; no bare generic words)
STRICT_ACTION = {
    "prohibit_caffeine": (r"(caffeine|energy drink)", r"(\bno\b|avoid|skip|don't|do not|stop|ditch|prohibit|forbid|stay away|cut|without|put down|drop)"),
    "power_nap_20m": (r"(\bnap\b|power nap|20[- ]min)", None),
    "cold_water_intake": (r"(cold water|hydrat|rehydrat|drink (some |a glass of |16 ?oz of )?water|glass of water|electrolyte)", None),
    "halt_token_promotion_pending_legal_review": (r"(halt|pause|stop|suspend|hold off|freeze|do not promote|don't promote|refrain)", r"(promot|token|marketing|launch|sale|campaign)"),
    "seek_qualified_securities_counsel": (r"(counsel|attorney|lawyer|legal advice|legal team|law firm|legal expert)", None),
    "disclose_us_securities_exposure_risk": (r"(securities|howey|investment contract|\bsec\b)", r"(risk|exposure|disclos|liabil|regulat|complian|violat)"),
    "limit_caffeine_1_unit": (r"(caffeine|energy drink)", r"(limit|at most|one unit|one energy drink|no more energy|just one|only one|single)"),
}
STRICT_CONCEPT = {
    "recovery": r"(recovery|recover)",
    "sleep": r"(sleep|\bnap\b)",
    "securities": r"(securities|howey|investment contract)",
    "token": r"(\btoken)",
}

def parse_what_output(raw):
    tm = re.search(r"<\|start_thought\|>(.*?)<\|end_thought\|>", raw, re.DOTALL)
    jm = re.search(r"<\|start_json\|>(.*?)<\|end_json\|>", raw, re.DOTALL)
    thought = tm.group(1).strip() if tm else ""
    s = a = ""
    if jm:
        try:
            p = json.loads(jm.group(1).strip())
            s = str(p.get("s") or p.get("user_state") or "")
            a = str(p.get("a") or p.get("recommended_action") or "")
        except json.JSONDecodeError:
            pass
    return {"thought": thought, "user_state": s, "recommended_action": a}

def _extract_t(raw):
    m = re.search(r'"t"\s*:\s*"(.*?)"', raw, re.DOTALL)
    return m.group(1) if m else ""

def action_cov(text_l, required, strict=False):
    hits = {}
    for req in required:
        if strict:
            pat1, pat2 = STRICT_ACTION[req]
            h = re.search(pat1, text_l) is not None
            if pat2 is not None:
                h = h and re.search(pat2, text_l) is not None
            hits[req] = h
        elif req == "limit_caffeine_1_unit":
            hits[req] = any(k in text_l for k in ("caffeine", "energy drink")) and any(k in text_l for k in LIMIT_SYN)
        else:
            hits[req] = any(k in text_l for k in ACTION_SYNONYMS.get(req, [req.replace("_", " ")]))
    return hits

def score_text(text, gold, strict=False):
    text_l = text.lower()
    ah = action_cov(text_l, gold["actions"], strict)
    if strict:
        ch = {c: re.search(STRICT_CONCEPT[c], text_l) is not None for c in gold["concepts"]}
    else:
        ch = {c: any(k in text_l for k in CONCEPT_SYNONYMS.get(c, [c.replace("_", " ")])) for c in gold["concepts"]}
    fh = [f for f in gold["forbidden"] if (f.replace("_", " ") in text_l or f in text_l)]
    a = sum(ah.values()) / len(gold["actions"])
    c = sum(ch.values()) / len(gold["concepts"])
    fpen = len(fh) / len(gold["forbidden"])
    comp = (0.6 * a + 0.25 * c + 0.15 * (1 - fpen))
    return dict(score=round(comp, 4), action=a, concept=c, forbidden=fpen, action_hits=ah, concept_hits=ch, forbidden_hits=fh)

def score_what(raw, gold, what_only=True, strict=False):
    p = parse_what_output(raw)
    parts = [p["thought"], p["user_state"], p["recommended_action"]]
    if not what_only:
        parts += [_extract_t(raw), raw]
    combined = " ".join(x for x in parts if x)
    r = score_text(combined, gold, strict)
    r["parse_empty"] = not any(parts[:3])
    return r

# ---------------- load ----------------
def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

rows = []
srcs = {"llama": ["alignment_tax_1b_llama/runs.jsonl", "alignment_tax_1b_llama_l4/runs.jsonl"],
        "gemma": ["alignment_tax_1b_gemma/runs.jsonl"]}
for base, files in srcs.items():
    for f in files:
        for r in load(os.path.join(ROOT, f)):
            gold = GOLD[r["task_id"]]
            comp_logged = r["metrics"]["what_objective"]["what_objective_score"]
            s_what = score_what(r["what_raw"], gold, what_only=True)
            assert abs(s_what["score"] - comp_logged) < 1e-6, (f, r["arm_id"], r["seed"], s_what["score"], comp_logged)
            s_raw = score_what(r["what_raw"], gold, what_only=False)
            s_strict = score_what(r["what_raw"], gold, what_only=True, strict=True)
            s_strict_raw = score_what(r["what_raw"], gold, what_only=False, strict=True)
            s_speech = score_text(r["how_speech"] or "", gold)
            s_speech_strict = score_text(r["how_speech"] or "", gold, strict=True)
            rows.append(dict(
                base=base, arm=r["arm_id"], level=r["pollution_level"], task=r["task_id"], persona=r["persona"], seed=int(r["seed"]),
                comp=comp_logged, parse_empty=s_what["parse_empty"], raw=s_raw["score"], strict=s_strict["score"], strict_raw=s_strict_raw["score"],
                action=s_what["action"], concept=s_what["concept"], forb=s_what["forbidden"],
                how_cov=r["metrics"]["how_constraint_coverage"]["coverage_rate"],
                speech=s_speech["score"], speech_strict=s_speech_strict["score"], speech_forb=s_speech["forbidden"],
                wp_tokens=r["logs"].get("what_prompt_tokens_est"), hp_tokens=r["logs"].get("how_prompt_tokens_est"),
                lat_total=r["total_latency_ms"] / 1000, lat_what=r["what_latency_ms"] / 1000, lat_how=r["how_latency_ms"] / 1000,
                what_raw=r["what_raw"], speech_len=len(r["how_speech"] or ""),
                eff_user_chars=r["logs"].get("effective_user_chars"), hist_turns=r["logs"].get("history_turns"), bloat=r["logs"].get("system_bloat_chars"),
                swap_ms=r["logs"].get("swap_ms_what"),
            ))
df = pd.DataFrame(rows)
print("runs:", len(df), df.groupby(["base", "level", "arm"]).size().unstack().to_string())
LEVELS = ["L0", "L2", "L3", "L4"]
ARMS = ["G0", "G3", "G3_what_polluted"]
KEY = ["task", "persona", "seed"]

# ---------------- helpers ----------------
def mean_ci(x, B=10000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return (x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))

def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))

def paired(base, level, arm_a, arm_b, metric):
    A = df[(df.base == base) & (df.level == level) & (df.arm == arm_a)].set_index(KEY)[metric]
    Bv = df[(df.base == base) & (df.level == level) & (df.arm == arm_b)].set_index(KEY)[metric]
    idx = A.index.intersection(Bv.index)
    a = A.loc[idx].values.astype(float); b = Bv.loc[idx].values.astype(float)
    d = a - b
    n = len(d)
    nz = int((d != 0).sum())
    if nz == 0:
        p = 1.0; W = np.nan
    else:
        try:
            W, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
        except ValueError:
            W, p = np.nan, 1.0
    bs = rng.choice(d, size=(10000, n), replace=True).mean(axis=1)
    return dict(n=n, n_nonzero=nz, mean_a=a.mean(), mean_b=b.mean(), diff=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                cliffs=cliffs_delta(a, b), W=W, p=p)

def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m)
    running = 0
    for rank, i in enumerate(order):
        val = (m - rank) * ps[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, c - h, c + h)

out = {}
# ---------------- Table 1: composite per cell ----------------
t1 = []
for base in ["llama", "gemma"]:
    for arm in ARMS:
        for lv in LEVELS:
            x = df[(df.base == base) & (df.arm == arm) & (df.level == lv)].comp
            if len(x) == 0: continue
            m, lo, hi = mean_ci(x)
            t1.append(dict(base=base, arm=arm, level=lv, n=len(x), mean=m, sd=x.std(ddof=1), ci_lo=lo, ci_hi=hi,
                           median=x.median(), min=x.min(), max=x.max()))
T1 = pd.DataFrame(t1); T1.to_csv(f"{OUT}/table1_composite.csv", index=False)

# ---------------- Table 2: decomposition ----------------
t2 = []
for base in ["llama", "gemma"]:
    for arm in ARMS:
        for lv in LEVELS:
            s = df[(df.base == base) & (df.arm == arm) & (df.level == lv)]
            if len(s) == 0: continue
            k = int(s.parse_empty.sum()); n = len(s)
            p, lo, hi = wilson(k, n)
            rm, rlo, rhi = mean_ci(s.raw); sm, slo, shi = mean_ci(s.speech); hm, hlo, hhi = mean_ci(s.how_cov)
            t2.append(dict(base=base, arm=arm, level=lv, n=n, fail_k=k, fail_rate=p, fail_lo=lo, fail_hi=hi,
                           raw_mean=rm, raw_lo=rlo, raw_hi=rhi, raw_sd=s.raw.std(ddof=1),
                           speech_mean=sm, speech_lo=slo, speech_hi=shi, speech_sd=s.speech.std(ddof=1),
                           howcov_mean=hm, howcov_lo=hlo, howcov_hi=hhi,
                           action_mean=s.action.mean(), concept_mean=s.concept.mean(), forb_mean=s.forb.mean(),
                           strict_mean=s.strict.mean(), strict_raw_mean=s.strict_raw.mean(), speech_strict_mean=s.speech_strict.mean(),
                           speech_forb_mean=s.speech_forb.mean()))
T2 = pd.DataFrame(t2); T2.to_csv(f"{OUT}/table2_decomposition.csv", index=False)

# ---------------- Table 3: paired tests ----------------
t3 = []
for metric in ["comp", "raw", "speech", "how_cov", "strict", "strict_raw"]:
    for comp_name, (a, b) in {"G3P_vs_G0": ("G3_what_polluted", "G0"), "G3_vs_G0": ("G3", "G0"), "G3_vs_G3P": ("G3", "G3_what_polluted")}.items():
        for base in ["llama", "gemma"]:
            for lv in LEVELS:
                r = paired(base, lv, a, b, metric)
                r.update(metric=metric, comparison=comp_name, base=base, level=lv)
                t3.append(r)
T3 = pd.DataFrame(t3)
# Holm within each (metric, comparison) family across 8 (base × level) tests
T3["p_holm"] = np.nan
for (m, c), g in T3.groupby(["metric", "comparison"]):
    T3.loc[g.index, "p_holm"] = holm(g.p.values)
T3.to_csv(f"{OUT}/table3_paired.csv", index=False)

# fail-rate Fisher tests: G0 vs G3P per level; G0 L0 vs L3/L4
fisher = []
for base in ["llama", "gemma"]:
    for lv in LEVELS:
        a = df[(df.base == base) & (df.level == lv) & (df.arm == "G0")].parse_empty
        b = df[(df.base == base) & (df.level == lv) & (df.arm == "G3_what_polluted")].parse_empty
        if len(a) == 0: continue
        tab = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
        _, p = stats.fisher_exact(tab)
        fisher.append(dict(base=base, test=f"G0 vs G3P fail @ {lv}", table=tab, p=p))
    for lv in ["L2", "L3", "L4"]:
        a = df[(df.base == base) & (df.level == "L0") & (df.arm == "G0")].parse_empty
        b = df[(df.base == base) & (df.level == lv) & (df.arm == "G0")].parse_empty
        if len(b) == 0: continue
        tab = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
        _, p = stats.fisher_exact(tab)
        fisher.append(dict(base=base, test=f"G0 fail L0 vs {lv}", table=tab, p=p))
FI = pd.DataFrame(fisher)
for base, g in FI.groupby("base"):
    FI.loc[g.index, "p_holm"] = holm(g.p.values)
FI.to_csv(f"{OUT}/fisher.csv", index=False)

# trend: Spearman of level ordinal vs G0 metric
trend = []
ordmap = {"L0": 0, "L2": 1, "L3": 2, "L4": 3}
for base in ["llama", "gemma"]:
    for arm in ["G0", "G3_what_polluted"]:
        for metric in ["comp", "raw", "parse_empty"]:
            s = df[(df.base == base) & (df.arm == arm)]
            rho, p = stats.spearmanr(s.level.map(ordmap), s[metric].astype(float))
            trend.append(dict(base=base, arm=arm, metric=metric, rho=rho, p=p, n=len(s)))
TR = pd.DataFrame(trend); TR.to_csv(f"{OUT}/trend.csv", index=False)

# ---------------- Table 4: what prompt tokens ----------------
t4 = df.groupby(["base", "arm", "level"]).agg(wp_mean=("wp_tokens", "mean"), wp_min=("wp_tokens", "min"), wp_max=("wp_tokens", "max"),
                                            hp_mean=("hp_tokens", "mean"), eff_chars=("eff_user_chars", "mean"), hist=("hist_turns", "mean"), bloat=("bloat", "mean"),
                                            lat_total=("lat_total", "mean"), lat_what=("lat_what", "mean"), lat_how=("lat_how", "mean"), lat_sd=("lat_total", "std")).reset_index()
t4.to_csv(f"{OUT}/table4_tokens_latency.csv", index=False)

# G3 What-output invariance across levels (byte identity)
inv = []
for base in ["llama", "gemma"]:
    g3 = df[(df.base == base) & (df.arm == "G3")]
    piv = g3.pivot_table(index=KEY, columns="level", values="what_raw", aggfunc="first")
    lv = [l for l in LEVELS if l in piv.columns]
    same_all = (piv[lv].nunique(axis=1) == 1).sum()
    inv.append(dict(base=base, n_keys=len(piv), identical_across_levels=int(same_all), levels=lv))
    # persona invariance within level
    for l in lv:
        p2 = g3[g3.level == l].pivot_table(index=["task", "seed"], columns="persona", values="what_raw", aggfunc="first")
        inv.append(dict(base=base, level=l, persona_identical=int((p2.nunique(axis=1) == 1).sum()), n=len(p2)))
INV = pd.DataFrame(inv); INV.to_csv(f"{OUT}/invariance.csv", index=False)

# ---------------- Table 5: persona split ----------------
t5 = df.groupby(["base", "arm", "level", "persona"]).agg(comp=("comp", "mean"), how_cov=("how_cov", "mean"), speech=("speech", "mean"), n=("comp", "size")).reset_index()
t5.to_csv(f"{OUT}/table5_persona.csv", index=False)

# ---------------- Table 6: Run 1 ----------------
r1 = load(os.path.join(ROOT, "alignment_tax_full_20260603/runs.jsonl"))
r1rows = []
for r in r1:
    gold = GOLD[r["task_id"]]
    r1rows.append(dict(arm=r["architecture"], task=r["task_id"], persona=r["persona"], seed=int(r["seed"]),
                       comp=r["metrics"]["what_objective"]["what_objective_score"], how_cov=r["metrics"].get("how_constraint_coverage", {}).get("coverage_rate", np.nan),
                       parse_empty=score_what(r["what_raw"], gold)["parse_empty"], lat=float(r["total_latency_ms"]) / 1000))
R1 = pd.DataFrame(r1rows)
T6 = R1.groupby("arm").agg(n=("comp", "size"), comp=("comp", "mean"), comp_sd=("comp", "std"), how_cov=("how_cov", "mean"), fail=("parse_empty", "mean"), lat=("lat", "mean")).reset_index()
T6.to_csv(f"{OUT}/table6_run1.csv", index=False)
r1tests = {}
for a, b in [("G3", "G0"), ("G4", "G0"), ("G4", "G3"), ("what_crossover", "G3")]:
    A = R1[R1.arm == a].set_index(KEY); Bv = R1[R1.arm == b].set_index(KEY); idx = A.index.intersection(Bv.index)
    for metric in ["comp", "how_cov"]:
        x = A.loc[idx][metric].values; y = Bv.loc[idx][metric].values; d = x - y
        try:
            W, p = stats.wilcoxon(x, y)
        except ValueError:
            W, p = np.nan, 1.0
        bs = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
        r1tests[f"{a}_vs_{b}_{metric}"] = dict(n=len(d), mean_a=x.mean(), mean_b=y.mean(), diff=d.mean(), ci=(np.percentile(bs, 2.5), np.percentile(bs, 97.5)), cliffs=cliffs_delta(x, y), p=p)
json.dump(r1tests, open(f"{OUT}/run1_tests.json", "w"), indent=1, default=float)
R1.groupby(["arm", "task"]).comp.mean().unstack().to_csv(f"{OUT}/table6_run1_bytask.csv")

# ---------------- Appendix B strict ----------------
tb = df.groupby(["base", "arm", "level"]).agg(strict=("strict", "mean"), strict_raw=("strict_raw", "mean"), speech_strict=("speech_strict", "mean"), comp=("comp", "mean"), raw=("raw", "mean")).reset_index()
tb.to_csv(f"{OUT}/appB_strict.csv", index=False)

df.drop(columns=["what_raw"]).to_csv(f"{OUT}/runs_flat.csv", index=False)
print("done")
