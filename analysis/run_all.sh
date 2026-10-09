#!/usr/bin/env bash
# Recompute every table, statistic and figure of the paper from the released logs.
# Usage (from the repository root):  bash analysis/run_all.sh
# Requires: pip install -r analysis/requirements-analysis.txt   (no MLX, no model weights)
set -euo pipefail
export SOURCE_DATE_EPOCH=1791504000   # fixed PDF timestamps (2026-10-09 UTC) so reruns are byte-identical
cd "$(dirname "$0")/.."
LOGS=logs
OUT=analysis/out
FIGS=analysis/out/figs
mkdir -p "$OUT" "$FIGS"
python3 -I analysis/analyze.py "$LOGS" "$OUT"            # Tables 1-6, Appendix A/B CSVs, runs_flat.csv
python3 -I analysis/tables.py "$OUT" "$OUT/tables.json"  # Markdown versions of the tables
python3 -I analysis/system_props.py "$LOGS" "$OUT"       # Table 7 and Section 6 figures from the verification turns
python3 -I analysis/figures.py "$OUT" "$FIGS"            # Figures 1-3
echo "done: outputs in $OUT"
