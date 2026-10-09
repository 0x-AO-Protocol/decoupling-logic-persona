#!/usr/bin/env python3
"""Markdown versions of the paper's tables, rendered from the CSVs written by analyze.py.

Usage:  python3 -I analysis/tables.py <out_dir> <out_dir>/tables.json
"""
import sys, json
import pandas as pd, numpy as np
OUT = sys.argv[1]; DST = sys.argv[2]
LEVELS = ["L0", "L2", "L3", "L4"]
ARM = {"G0": "G0", "G3": "G3", "G3_what_polluted": "G3-P"}
BASE = {"llama": "Llama-8B", "gemma": "Gemma-4B"}
T = {}
T1 = pd.read_csv(f"{OUT}/table1_composite.csv"); T2 = pd.read_csv(f"{OUT}/table2_decomposition.csv"); T3 = pd.read_csv(f"{OUT}/table3_paired.csv")
T4 = pd.read_csv(f"{OUT}/table4_tokens_latency.csv"); T5 = pd.read_csv(f"{OUT}/table5_persona.csv"); T6 = pd.read_csv(f"{OUT}/table6_run1.csv")
FI = pd.read_csv(f"{OUT}/fisher.csv"); TR = pd.read_csv(f"{OUT}/trend.csv"); AB = pd.read_csv(f"{OUT}/appB_strict.csv")
r1 = json.load(open(f"{OUT}/run1_tests.json"))

def fmt_p(p):
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

# Table 1
rows = ["| Base | Arm | L0 | L2 | L3 | L4 |", "|---|---|---|---|---|---|"]
for b in ["llama", "gemma"]:
    for a in ["G0", "G3", "G3_what_polluted"]:
        cells = []
        for l in LEVELS:
            s = T1[(T1.base == b) & (T1.arm == a) & (T1.level == l)].iloc[0]
            cells.append(f"{s['mean']:.3f} ± {s['sd']:.3f}")
        rows.append(f"| {BASE[b]} | {ARM[a]} | " + " | ".join(cells) + " |")
T["TABLE1"] = "\n".join(rows)

# Table 2
rows = ["| Base | Arm | Lv | Fail % [95% CI] | Raw-text | Speech | How cov. |", "|---|---|---|---|---|---|---|"]
for b in ["llama", "gemma"]:
    for a in ["G0", "G3", "G3_what_polluted"]:
        for l in LEVELS:
            s = T2[(T2.base == b) & (T2.arm == a) & (T2.level == l)].iloc[0]
            rows.append(f"| {BASE[b]} | {a.replace('_what_polluted','-P')} | {l} | {s.fail_rate*100:.0f} [{abs(s.fail_lo)*100:.0f}, {s.fail_hi*100:.0f}] | {s.raw_mean:.3f} ± {s.raw_sd:.3f} | {s.speech_mean:.3f} ± {s.speech_sd:.3f} | {s.howcov_mean:.3f} |")
T["TABLE2"] = "\n".join(rows)

# Table 3 main: comp G3P vs G0, and raw G3P vs G0
def t3(metric, comp, title):
    rows = [f"| Base | Lv | {title} | G0 | Δ [95% CI] | δ | p | Holm p |", "|---|---|---|---|---|---|---|---|"]
    for b in ["llama", "gemma"]:
        for l in LEVELS:
            s = T3[(T3.metric == metric) & (T3.comparison == comp) & (T3.base == b) & (T3.level == l)].iloc[0]
            pw = "—" if s.n_nonzero == 0 else fmt_p(s.p)
            ph = "—" if s.n_nonzero == 0 else fmt_p(s.p_holm)
            rows.append(f"| {BASE[b]} | {l} | {s.mean_a:.3f} | {s.mean_b:.3f} | {s['diff']:+.3f} [{s.ci_lo:.2f}, {s.ci_hi:.2f}] | {s.cliffs:+.2f} | {pw} | {ph} |")
    return "\n".join(rows)
T["TABLE3A"] = t3("comp", "G3P_vs_G0", "G3-P")
T["TABLE3B"] = t3("raw", "G3P_vs_G0", "G3-P")
T["TABLE3C"] = t3("comp", "G3_vs_G0", "G3")
T["TABLE3D"] = t3("speech", "G3_vs_G0", "G3")
T["TABLE3E"] = t3("how_cov", "G3_vs_G0", "G3")

# Fisher table
rows = ["| Base | Comparison | Failures | Fisher p | Holm p |", "|---|---|---|---|---|"]
for _, s in FI.iterrows():
    tab = json.loads(s.table.replace("(", "[").replace(")", "]"))
    rows.append(f"| {BASE[s.base]} | {s.test.replace('G3P','G3-P')} | {tab[0][0]}/20 vs {tab[1][0]}/20 | {fmt_p(s.p)} | {fmt_p(s.p_holm)} |")
T["TABLE_FISHER"] = "\n".join(rows)

# Trend
rows = ["| Base | Arm | Metric | Spearman ρ | p | n |", "|---|---|---|---|---|---|"]
for _, s in TR.iterrows():
    rows.append(f"| {BASE[s.base]} | {s.arm.replace('_what_polluted','-P')} | {s.metric.replace('comp','composite').replace('parse_empty','failure')} | {s.rho:+.3f} | {fmt_p(s.p)} | {s.n} |")
T["TABLE_TREND"] = "\n".join(rows)

# Table 4 tokens + latency
rows = ["| Base | Arm | Lv | Logic prompt tokens [min–max] | Persona prompt tokens | Latency s (± SD) | Logic s | Persona s |", "|---|---|---|---|---|---|---|---|"]
for b in ["llama", "gemma"]:
    for a in ["G0", "G3", "G3_what_polluted"]:
        for l in LEVELS:
            s = T4[(T4.base == b) & (T4.arm == a) & (T4.level == l)].iloc[0]
            hp = "—" if pd.isna(s.hp_mean) else f"{s.hp_mean:,.0f}"
            rows.append(f"| {BASE[b]} | {a.replace('_what_polluted','-P')} | {l} | {s.wp_mean:,.0f} [{s.wp_min:,.0f}–{s.wp_max:,.0f}] | {hp} | {s.lat_total:.1f} ± {s.lat_sd:.1f} | {s.lat_what:.1f} | {s.lat_how:.1f} |")
T["TABLE4"] = "\n".join(rows)

# Table 5 persona split
rows = ["| Base | Arm | Lv | Comp. Goku | Comp. Makima | How Goku | How Makima | Speech Goku | Speech Makima |", "|---|---|---|---|---|---|---|---|---|"]
for b in ["llama", "gemma"]:
    for a in ["G0", "G3"]:
        for l in LEVELS:
            g = T5[(T5.base == b) & (T5.arm == a) & (T5.level == l) & (T5.persona == "goku")].iloc[0]
            m = T5[(T5.base == b) & (T5.arm == a) & (T5.level == l) & (T5.persona == "makima")].iloc[0]
            rows.append(f"| {BASE[b]} | {a} | {l} | {g.comp:.3f} | {m.comp:.3f} | {g.how_cov:.3f} | {m.how_cov:.3f} | {g.speech:.3f} | {m.speech:.3f} |")
T["TABLE5"] = "\n".join(rows)

# Table 6 run1
rows = ["| Arm | n | Composite logic (mean ± SD) | How coverage | Failure rate | Latency s |", "|---|---|---|---|---|---|"]
names = {"G0": "G0 mixed single pass", "G3": "G3 decoupled", "G4": "G4 decoupled + constraint echo", "what_crossover": "Persona LoRA on logic path (probe)"}
for _, s in T6.iterrows():
    hc = "—" if pd.isna(s.how_cov) else f"{s.how_cov:.3f}"
    rows.append(f"| {names[s.arm]} | {s.n} | {s.comp:.3f} ± {s.comp_sd:.3f} | {hc} | {s.fail*100:.1f}% | {s.lat:.1f} |")
T["TABLE6"] = "\n".join(rows)
T["RUN1_G3_G0_COMP"] = f"Δ = {r1['G3_vs_G0_comp']['diff']:+.3f} [95% CI {r1['G3_vs_G0_comp']['ci'][0]:+.3f}, {r1['G3_vs_G0_comp']['ci'][1]:+.3f}], Cliff's δ = {r1['G3_vs_G0_comp']['cliffs']:+.2f}, Wilcoxon p = {r1['G3_vs_G0_comp']['p']:.2f}"
T["RUN1_G3_G0_HOW"] = f"Δ = {r1['G3_vs_G0_how_cov']['diff']:+.3f} [95% CI {r1['G3_vs_G0_how_cov']['ci'][0]:+.3f}, {r1['G3_vs_G0_how_cov']['ci'][1]:+.3f}], Cliff's δ = {r1['G3_vs_G0_how_cov']['cliffs']:+.2f}, p = {fmt_p(r1['G3_vs_G0_how_cov']['p'])}"
T["RUN1_XO"] = f"Δ = {r1['what_crossover_vs_G3_comp']['diff']:+.3f} [95% CI {r1['what_crossover_vs_G3_comp']['ci'][0]:+.3f}, {r1['what_crossover_vs_G3_comp']['ci'][1]:+.3f}], p = {r1['what_crossover_vs_G3_comp']['p']:.2f}"

# Appendix B strict
rows = ["| Base | Arm | Lv | Comp. loose | Comp. strict | Raw loose | Raw strict | Speech strict |", "|---|---|---|---|---|---|---|---|"]
for b in ["llama", "gemma"]:
    for a in ["G0", "G3", "G3_what_polluted"]:
        for l in LEVELS:
            s = AB[(AB.base == b) & (AB.arm == a) & (AB.level == l)].iloc[0]
            t2 = T2[(T2.base == b) & (T2.arm == a) & (T2.level == l)].iloc[0]
            t1 = T1[(T1.base == b) & (T1.arm == a) & (T1.level == l)].iloc[0]
            # loose columns are taken from Tables 1 and 2 so that the same cell is rounded the same way in every table
            rows.append(f"| {BASE[b]} | {a.replace('_what_polluted','-P')} | {l} | {t1['mean']:.3f} | {s.strict:.3f} | {t2.raw_mean:.3f} | {s.strict_raw:.3f} | {t2.speech_strict_mean:.3f} |")
T["TABLE_B"] = "\n".join(rows)
T["TABLE_B_TEST"] = t3("strict", "G3P_vs_G0", "G3-P")

json.dump(T, open(DST, "w"), indent=1)
print("tables:", list(T))
