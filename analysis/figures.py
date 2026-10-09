#!/usr/bin/env python3
"""Figures 1-3 of the paper: fig3_architecture (Figure 1, a diagram that reads no data),
fig1_composite (Figure 2) and fig2_decomposition (Figure 3).

Usage:  python3 -I analysis/figures.py <out_dir> <figs_dir>
"""
import sys, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1]
FIG = sys.argv[2]
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
LEVELS = ["L0", "L2", "L3", "L4"]
ARMS = [("G0", "Mixed single pass (G0)", "#c0392b", "o"), ("G3", "Decoupled, logic path clean (G3)", "#1f5fa8", "s"), ("G3_what_polluted", "Decoupled, logic path polluted (G3-P)", "#7f8c8d", "^")]
BASES = [("llama", "Llama-3.1-8B-Instruct (INT4)"), ("gemma", "Gemma-3-4B-it (INT4)")]

T1 = pd.read_csv(f"{OUT}/table1_composite.csv")
T2 = pd.read_csv(f"{OUT}/table2_decomposition.csv")

# ---------- fig1_composite: Figure 2 of the paper ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, (base, title) in zip(axes, BASES):
    for arm, label, color, mk in ARMS:
        s = T1[(T1.base == base) & (T1.arm == arm)].set_index("level").reindex(LEVELS)
        x = np.arange(len(LEVELS)) + (0.06 if arm == "G3_what_polluted" else (-0.06 if arm == "G0" else 0))
        ax.errorbar(x, s["mean"], yerr=[s["mean"] - s.ci_lo, s.ci_hi - s["mean"]], fmt=mk + "-", color=color, label=label, capsize=2.5, ms=4.5, lw=1.3)
    ax.axhline(0.15, color="k", ls=":", lw=0.8)
    ax.text(3.35, 0.16, "floor 0.15", fontsize=7, ha="right", va="bottom")
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS); ax.set_xlabel("Pollution level")
    ax.set_title(title, fontsize=9); ax.set_ylim(0.05, 0.95)
axes[0].set_ylabel("Composite logic score")
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=3,fontsize=7,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(f"{FIG}/fig1_composite.png", dpi=300); fig.savefig(f"{FIG}/fig1_composite.pdf")

# ---------- fig2_decomposition: Figure 3 of the paper ----------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
for j, (base, title) in enumerate(BASES):
    ax = axes[0, j]
    w = 0.26
    for i, (arm, label, color, mk) in enumerate(ARMS):
        s = T2[(T2.base == base) & (T2.arm == arm)].set_index("level").reindex(LEVELS)
        x = np.arange(len(LEVELS)) + (i - 1) * w
        ax.bar(x, s.fail_rate * 100, w, color=color, label=label, yerr=[(s.fail_rate - s.fail_lo) * 100, (s.fail_hi - s.fail_rate) * 100], capsize=2, error_kw=dict(lw=0.8))
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 105)
    if j == 0: ax.set_ylabel("Structured-output failure (%)\n(thought, s, a all empty)")
    ax = axes[1, j]
    for arm, label, color, mk in ARMS:
        s = T2[(T2.base == base) & (T2.arm == arm)].set_index("level").reindex(LEVELS)
        x = np.arange(len(LEVELS)) + (0.06 if arm == "G3_what_polluted" else (-0.06 if arm == "G0" else 0))
        ax.errorbar(x, s.raw_mean, yerr=[s.raw_mean - s.raw_lo, s.raw_hi - s.raw_mean], fmt=mk + "-", color=color, label=label, capsize=2.5, ms=4.5, lw=1.3)
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS); ax.set_xlabel("Pollution level"); ax.set_ylim(0.2, 0.95)
    if j == 0: ax.set_ylabel("Raw-text logic score\n(format-independent)")
h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=3,fontsize=7,frameon=False,bbox_to_anchor=(0.5,-0.005))
fig.tight_layout(rect=(0,0.045,1,1)); fig.savefig(f"{FIG}/fig2_decomposition.png", dpi=300); fig.savefig(f"{FIG}/fig2_decomposition.pdf")

# ---------- fig3_architecture: Figure 1 of the paper (diagram) ----------
fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")
def box(x, y, w, h, text, fc="#f4f6f8", ec="#34495e", fs=7.8, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2, text=None, color="#34495e", ls="-", dx=0, dy=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10, color=color, lw=1.1, ls=ls))
    if text: ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", fontsize=6.8, color=color)
box(1, 44, 20, 11, "Core turn\n(user input + vitals,\nprofile, intent)", fc="#fdfefe")
box(1, 26, 20, 11, "Conversation history\n+ long input\n+ persona few-shot", fc="#fdebd0", ec="#b9770e")
box(1, 9, 20, 11, "Deterministic\npolicy router\n(signals + rules)", fc="#eafaf1", ec="#1e8449")
box(31, 44, 21, 11, "WHAT path\nlogic adapter (LoRA)\ninput bounded, no history", fc="#d6eaf8", ec="#1f5fa8", bold=True)
box(61, 44, 17, 11, "Micro-State\n(schema v1.1)\nstate, action,\nconstraints, risk", fc="#fef9e7", ec="#b7950b")
box(84, 44, 15, 11, "Micro-State\ncache\n(reuse /\ninvalidate)", fc="#fef9e7", ec="#b7950b", ls="--")
box(35, 20, 18, 11, "HOW path\npersona adapter (LoRA)\nreceives full history", fc="#fadbd8", ec="#c0392b", bold=True, fs=7.4)
box(61, 20, 17, 11, "Spoken reply\n(persona voice,\nconstraints echoed)", fc="#fdfefe")
box(31, 3, 68, 8, "Single INT4 base model resident in unified memory —\nlogic and persona adapters hot-swapped on the same weights (~ms)", fc="#eafaf1", ec="#1e8449", fs=7.4)
arrow(21, 49.5, 31, 49.5)
arrow(52, 49.5, 61, 49.5)
arrow(78, 49.5, 84, 49.5, color="#b7950b", style="<|-|>")
arrow(69.5, 44, 53, 29, color="#b7950b", text="verified\nstate", dx=4, dy=1)
arrow(21, 31.5, 35, 26.5, color="#b9770e")
arrow(21, 33.5, 30, 43, color="#b9770e", ls=":", text="not\nrouted", dx=-4.5, dy=0.5)
arrow(53, 25.5, 61, 25.5)
arrow(21, 14.5, 35, 22.5, color="#1e8449")
ax.text(22.5, 13.2, "select logic\nadapter + persona", fontsize=6.6, color="#1e8449", ha="left", va="top")
arrow(45, 11, 45, 20, color="#1e8449", ls="--")
ax.add_patch(FancyArrowPatch((31, 7), (32.5, 44), arrowstyle="-|>", mutation_scale=10, color="#1e8449", lw=1.1, ls="--", connectionstyle="angle,angleA=180,angleB=90,rad=0"))
ax.text(50, 0.3, "The logic path never shares a context window with the persona path; its input size does not depend on history length.", ha="center", fontsize=7.3, style="italic")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_architecture.png", dpi=300); fig.savefig(f"{FIG}/fig3_architecture.pdf")
print("figures written")
