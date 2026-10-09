# Logs

All logs were written on one M2 MacBook Pro (24 GB) on 3–4 June 2026 (UTC). They are released as written, except for one anonymised file (see the end of this page). Base model keys: `llama_8b_4bit` = Meta-Llama-3.1-8B-Instruct 4-bit, `gemma_4b_4bit` = Gemma-3-4B-it 4-bit (text-only re-quantization).

## Run 1b — context pollution (Sections 5.2–5.7 of the paper)

Arms G0 (mixed single pass), G3 (decoupled, logic path clean) and `G3_what_polluted` (G3-P in the paper: decoupled, logic path polluted); two tasks, two personas, seeds 0–4, temperature 0.7.

| Directory | Base | Levels | Runs |
|---|---|---|---|
| `alignment_tax_1b_llama/` | Llama | L0, L2, L3 | 180 |
| `alignment_tax_1b_llama_l4/` | Llama | L4 (run afterwards as a separate battery) | 60 |
| `alignment_tax_1b_gemma/` | Gemma | L0, L2, L3, L4 | 240 |

Each directory holds `runs.jsonl` (one JSON record per run: arm, level, task, persona, seed, the raw logic-path output `what_raw`, the Micro-State, the persona-path output `how_raw` and spoken line `how_speech`, latencies, prompt-size estimates and the run-time scores), `runs.json` (the same records as one JSON array), `meta.json` (the battery definition), `summary.json` (the run-time summary written by `src/ao_da/experiment/stats.py`; the paper's statistics come from `analysis/`, not from this file) and `nohup.out` (console output).

## Run 1 — low pollution (Section 5.1)

`alignment_tax_full_20260603/runs.jsonl`: 120 runs on Llama — arms G0, G3, G4 (G3 with a constraint-echo instruction on the persona path) and `what_crossover` (the persona adapter on the logic path), three tasks, two personas, seeds 0–4. `nohup.out` is the console output. The battery was configured for both bases; after the Llama runs it stops with a traceback while mounting the Gemma base (the runtime looked up a third persona that has no Gemma entry), so Run 1 has Llama runs only, as reported in the paper. Run 1 ran before the repository's first commit; the released driver and runtime are later versions (the lookup that stopped the Gemma part has since been fixed), so the traceback's line numbers do not match the released files (README, Provenance). The flag `cd_enabled` marks G4's constraint-echo instruction, not constrained decoding.

## Pilot batteries (not analysed in the paper)

`alignment_tax_20260603T104621Z/` and `alignment_tax_20260603T105221Z/`: two 16-run batteries run on 3 June 2026 shortly before Run 1 (Llama; arms G0, G3, G4 and `what_crossover`; two tasks, two personas, seed 0). Each holds `runs.json` and `summary.json`.

## Verification turns (Section 6 and Table 7)

`turn_*.json`, one file per run, written by the authors' turn-verification scripts (not released) on the Llama base. The Turn 1 and Turn 3 files repeat their scenario (profile, vitals, user prompt) in a `scenario` field; the Turn 1-A file carries the logic-path output and the full Micro-State, the Turn 1-B files the logic-path output and the reduced Micro-State, the Turn 3 files a Micro-State summary (Turn 3-B also the logic-path output), and all of them the persona outputs, the run-time checks and timings. The Turn 2 files record their own Turn 1-A state and, next to it, the follow-up prompt, any changed signal, the cache decision, the persona outputs and timings.

| Files | Runs | Scenario | Used in the paper |
|---|---|---|---|
| `turn_1a_*` | 1 | First turn of the sleep/caffeine scenario (Goku) | Not directly; each Turn 2 file records its own Turn 1-A, which its cache check compares against |
| `turn_1b_*` | 2 | Turn 1-A with Micro-State fields removed before the persona path | No |
| `turn_2a_*` | 9 | "Second opinion": switch to Makima with unchanged vitals | Section 6.1 (hot-swap, cache hits) |
| `turn_2b_*` | 8 | Vitals change (recovery 42% → 58%) | Section 6.1 (cache invalidation) |
| `turn_2c_*` | 2 | User insists on overriding the advice | Section 6.1 (cache invalidation) |
| `turn_3a_*` | 3 | Deterministic policy router, sleep/caffeine scenario; one run also times an LLM classifier on the same input | Section 6.3 |
| `turn_3b_*` | 1 | Deterministic policy router, securities-law scenario | Section 6.3 |

`analysis/system_props.py` computes Table 7 and the figures in Sections 6.1 and 6.3 from these files.

## Anonymisation

In `alignment_tax_full_20260603/nohup.out` the operator's home-directory prefix in the traceback was replaced by `/Users/user/` (5 places). No other log file was changed; every other file is byte-identical to the authors' copy (`PROVENANCE.tsv`).

## Japanese strings

The Micro-State records carry a Japanese risk label (`risk.label_ja`) next to the English `risk.level`, and a few model outputs contain Japanese words. They are kept as written.
