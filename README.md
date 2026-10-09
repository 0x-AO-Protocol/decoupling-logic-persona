# Decoupling Logic from Persona — experiment code, logic adapters, run logs and analysis

Materials for:

> Masaaki Nakatsu and Reno Wang. 2026. *Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution.* arXiv:2610.09772 [cs.CL]. https://arxiv.org/abs/2610.09772

Companion paper: *Constitutional Gating and Deterministic Recovery for Multi-Agent LLM Negotiation*, arXiv:2610.11542 (code and logs: https://github.com/0x-AO-Protocol/constitutional-gating-testbed).

## What this repository lets you do

- **Recompute every table, statistic and figure in the paper from the raw logs** (all but the memory reading in Section 6.2; see below). This needs only Python with numpy, pandas, scipy and matplotlib: no MLX, no model weights, no Apple hardware. See [Reproduce the paper's numbers](#reproduce-the-papers-numbers).
- **Inspect every model output.** All runs of Run 1, Run 1b, the two pilot batteries and the verification turns are in `logs/`, with the model outputs and timings of every run, the Micro-State wherever the arm produces one, and (in Run 1b) prompt-size estimates.
- **Use the two logic-path (Mental Coach) LoRA adapters** with their base models (MLX), under the base models' licences.

What it does **not** let you do is re-run the experiment as released. The experiment code imports `ao_da.pipeline` (Micro-State assembly, the What-output parser, the constraint-coverage check and the signal collector), which belongs to the authors' orchestration layer and is not part of this release; `import ao_da.experiment` therefore fails with `ModuleNotFoundError: No module named 'ao_da.pipeline'`. The paper describes that layer at the level of its effects and reproduces the prompt templates and fixtures it assembles verbatim (Appendices D and E), and `analysis/analyze.py` re-implements the scoring rules it supplies. The persona adapters (Goku, Makima) are also withheld (see [Not included](#not-included)), so the persona path cannot be re-run either; all of its outputs are in the logs.

## Contents

| Path | What it is |
|---|---|
| `src/ao_da/experiment/` | Experiment code for Run 1 and Run 1b, as of commit `9c14bae`: arms (`architecture.py`), message assembly (`messages.py`, `g0_mixed.py` for the mixed single pass), pollution levels (`pollution.py`), tasks and gold (`tasks.py`), runner (`runner.py`), logic-path scorer (`objective_scorer.py`) and run summaries (`stats.py`). |
| `src/ao_da/mlx_runtime/` | MLX runtime: one resident base model with LoRA adapters held in memory and hot-swapped (`model_pool.py`, `lora.py`), memory snapshots (`memory.py`). `ModelPool` mounts a persona adapter at start-up and stops if none is found, so it cannot be started with this release alone; it is released as the reference for the hot-swap described in Section 3.3. |
| `src/ao_da/config.py`, `config/paths.local.yaml.example` | Where the runtime looks for base models and adapters. |
| `config/` | Battery definition (`alignment_tax_battery.yaml`), pollution levels (`alignment_tax_pollution.yaml`), tasks and gold labels (`alignment_tax_tasks.yaml`), arm definitions (`experiment_groups.yaml`), and the pollution fixtures (`fixtures/`, reproduced in Appendix D). |
| `schemas/micro_state_v1_1.schema.json` | JSON Schema of the Micro-State, version 1.1 (Appendix C). |
| `scripts/run_alignment_tax.py`, `scripts/run_alignment_tax_1b.py` | The drivers for Run 1 and Run 1b, as of commit `9c14bae` (see [Provenance](#provenance)). |
| `adapters/` | The two Mental Coach logic-path LoRA adapters, one per base, with their licence files (see [Adapters](#adapters)). |
| `logs/` | All run logs (see `logs/README.md`). |
| `analysis/` | Scripts that turn the logs into the paper's tables, statistics and figures, and their outputs in `analysis/out/`. |
| `PROVENANCE.tsv` | Every file taken from the authors' internal repository, with its git blob ID there and its SHA-256 here. |
| `requirements.txt` | Runtime dependencies of the experiment code (MLX). The analysis needs only `analysis/requirements-analysis.txt`. |

## Reproduce the paper's numbers

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r analysis/requirements-analysis.txt
bash analysis/run_all.sh
git diff --quiet -- analysis/out/tables.json analysis/out/table7_system.json && echo "paper numbers identical"
git status --short analysis/out     # empty on the platform named below; see the note on other platforms
```

| Output in `analysis/out/` | In the paper |
|---|---|
| `table1_composite.csv` | Table 1, Figure 2 |
| `table2_decomposition.csv` | Table 2, Figure 3 |
| `fisher.csv`, `trend.csv` | Tables S1 and S2 |
| `table3_paired.csv` | Tables 3, 3b, 3c, 3d, 3e and B2 (all paired tests, with Holm-adjusted p) |
| `table4_tokens_latency.csv` | Table 4; the first-turn latency row of Table 7 |
| `table5_persona.csv`, `invariance.csv` | Table 5; the byte-identity counts in Sections 5.5 and 5.6 |
| `table6_run1.csv`, `table6_run1_bytask.csv`, `run1_tests.json` | Table 6 and the Run 1 tests in Section 5.1 |
| `table7_system.json` | Table 7 and Sections 6.1 and 6.3 (from the verification turns) |
| `appB_strict.csv` | Table B1 |
| `tables.json` | Markdown renderings of all the tables above |
| `runs_flat.csv` | One row per Run 1b run with every score (Appendix A) |
| `figs/fig3_architecture.pdf` | Figure 1 (diagram) |
| `figs/fig1_composite.pdf` | Figure 2 |
| `figs/fig2_decomposition.pdf` | Figure 3 |

`analysis/analyze.py` re-implements the logic-path scorer and asserts, for each of the 480 Run 1b runs, that it reproduces the composite score recorded in the log. Bootstrap intervals use a fixed seed and figure timestamps are fixed. The committed outputs were produced on Linux x86-64 with Python 3.13.16, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1 and matplotlib 3.11.2, where a rerun is byte-identical. On other platforms the results agree up to floating-point rounding: on macOS on Apple silicon (Python 3.13.12, pandas 3.0.6, the other packages as above), a rerun leaves `tables.json` and `table7_system.json`, which hold every number printed in the paper, byte-identical, while four CSV/JSON files differ in the last digits of some values (relative differences of the order of 1e-15) and the figure files differ in their bytes.

The memory figures in Section 6.2 (4.25–4.29 GB active, 4.91–5.09 GB peak) were read from the terminal output of a single prototype session ("Phase 3 capture"), which is not part of this release.

## Adapters

| Directory | Model name | Base model | Licence |
|---|---|---|---|
| `adapters/mental_coach_llama/` | Llama-3.1-8B-Instruct-4bit-MentalCoach-LoRA | Meta-Llama-3.1-8B-Instruct, MLX 4-bit, group size 64 | Llama 3.1 Community License (`LICENSE`, `USE_POLICY.md`, `NOTICE`) |
| `adapters/mental_coach_gemma/` | Gemma-3-4B-it-4bit-MentalCoach-LoRA | Gemma-3-4B-it, text-only MLX 4-bit re-quantization, group size 64 | Gemma Terms of Use (`GEMMA_TERMS_OF_USE.md`, `NOTICE`) |

Both are rank-8 LoRA adapters (scale 20, 16 layers, 200 iterations, learning rate 1e-5, batch size 1) trained with `mlx_lm.lora` on 20 examples, as described in Section 4.2. The training data is not released. The base model weights are not included. Each directory's `README.md` gives the details and the terms that apply.

Built with Llama.

## Not included

- `src/ao_da/pipeline/` and everything else in the authors' orchestration layer (Micro-State construction, policy router, cache policy, LLM intent classifier, orchestrator), and the files that cannot run without it: the scripts that ran the verification turns in `logs/turn_*.json` and their scenario files (the Turn 1 and Turn 3 logs repeat their scenario in a `scenario` field; the Turn 2 logs record the follow-up prompt and the changed signal next to the Turn 1-A state), a Run 1b fixture check, a runtime smoke test, an adapter-swap benchmark not used in the paper, and the unit test.
- The persona adapters (Goku, Makima). Goku and Makima are characters from third-party works; the adapters were trained to imitate their speaking styles and are withheld for that reason. The short persona few-shot prompts in `config/fixtures/system_bloat/` were written by the authors as test stimuli. This project is not affiliated with or endorsed by the rights holders of those works.
- The adapter training data and the scripts that converted it and trained the adapters, intermediate adapter checkpoints, and an empty placeholder for a third persona adapter.
- Internal working documents and reports.

## Provenance

The 79 files listed in `PROVENANCE.tsv` come from the authors' internal repository at commit `9c14bae` (2026-06-07). Apart from the three files below, the code, configuration, schema and committed log files are byte-identical to that commit (`git hash-object <file>` returns the blob ID listed). The four `runs.jsonl` files and the two adapter weight files were never committed there (its `.gitignore` excluded them); they are byte-identical to the files in the authors' working copy. To check every file against the SHA-256 column:

```bash
awk -F'\t' 'NR>1 {print $3"  "$1}' PROVENANCE.tsv | shasum -a 256 -c    # or sha256sum -c on Linux
```

Three files were changed, and only in one way: the operator's home-directory prefix was replaced by `/Users/user/` in 9 places — `logs/alignment_tax_full_20260603/nohup.out` (5), `adapters/mental_coach_llama/adapter_config.json` (2) and `adapters/mental_coach_gemma/adapter_config.json` (2). No log line used by the analysis was changed.

The code is the state at commit `9c14bae`, made after both runs. The repository has one earlier commit, `3a5cfdf`, made on 4 June 2026 just before Run 1b started; in it, the released code and configuration differ only as follows: the fixture texts sat inside `config/alignment_tax_pollution.yaml` (they were moved to `config/fixtures/` unchanged); `ModelPool` looked up a third persona that has no Gemma entry (the error that stopped the Gemma part of Run 1, fixed in `model_pool.py`); `scripts/run_alignment_tax.py` called the crossover probe with an older signature; and two changes with no effect on the runs (an import moved out of `src/ao_da/experiment/__init__.py`, and a default in `pollution.py` that the configuration always overrides). Run 1 and the verification turns ran on 3 June 2026, before the repository's first commit, so the line numbers in Run 1's traceback do not match the released files.

Two names in the code and logs differ from the paper. `alignment_tax` in file and directory names is the internal name of this study (the paper calls the effect persona–logic interference). In Run 1, the G4 flag is called `cd_enabled` in the logs and `constrained_decoding` in the code and in `config/experiment_groups.yaml`; it switches on the constraint-echo instruction in the persona prompt (`strict_constraint_echo` in `messages.py`), not constrained decoding, which was not used.

Some strings are in Japanese and are kept because the files are released unmodified: the Micro-State risk label `label_ja` in the logs (the English `level` field next to it carries the same information), a few model outputs, and prompt lines in `g0_mixed.py` that tell the model not to use Japanese polite endings.

## Citation

```bibtex
@misc{nakatsu2026decoupling,
  title         = {Decoupling Logic from Persona: Structural Immunity of Edge {LLM} Agents to Context Pollution},
  author        = {Nakatsu, Masaaki and Wang, Reno},
  year          = {2026},
  eprint        = {2610.09772},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@misc{nakatsu2026constitutional,
  title         = {Constitutional Gating and Deterministic Recovery for Multi-Agent {LLM} Negotiation: Ablations Against a Stateful Adversarial Gatekeeper},
  author        = {Nakatsu, Masaaki and Wang, Reno},
  year          = {2026},
  eprint        = {2610.11542},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

## Licence

Code, configuration, fixtures, schema and analysis scripts: MIT (see `LICENSE`). The adapters are not covered by the MIT licence: each is distributed under the licence of its base model (see its directory). The logs contain text generated by Llama 3.1 and Gemma 3 models; their use is subject to those models' terms.

The tasks concern sleep, caffeine and securities law. The adapters and the logged outputs are research artefacts; they are not medical, health or legal advice.
