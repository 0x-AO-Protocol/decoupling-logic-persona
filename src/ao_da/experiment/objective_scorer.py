from __future__ import annotations

import re
from typing import Any

from ao_da.experiment.tasks import WhatGold
from ao_da.pipeline import kpi
from ao_da.pipeline.micro_state import parse_what_output


CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "recovery": ["recovery", "recover", "rest", "fatigue", "depleted"],
    "sleep": ["sleep", "slept", "nap", "rest"],
    "securities": ["securities", "security law", "howey", "investment contract"],
    "token": ["token", "ao token", "promotion"],
}


def _concept_hit(text_l: str, concept: str) -> bool:
    keys = CONCEPT_SYNONYMS.get(concept, [concept.replace("_", " ")])
    return any(k in text_l for k in keys)


def _forbidden_hit(text_l: str, token: str) -> bool:
    human = token.replace("_", " ")
    if human in text_l:
        return True
    return token in text_l


def _extract_t_from_raw(raw: str) -> str:
    m = re.search(r'"t"\s*:\s*"(.*?)"', raw, re.DOTALL)
    return m.group(1) if m else ""


def what_objective_score(
    raw: str,
    gold: WhatGold,
    *,
    what_only: bool = True,
) -> dict[str, Any]:
    """Objective What KPI on thought + s/a. Default what_only=True excludes spoken t/raw."""
    parsed = parse_what_output(raw)
    parts = [
        parsed.get("thought") or "",
        parsed.get("user_state") or "",
        parsed.get("recommended_action") or "",
    ]
    if not what_only:
        parts.append(_extract_t_from_raw(raw))
        parts.append(raw)
    combined = " ".join(p for p in parts if p)
    text_l = combined.lower()

    action_cov = kpi.constraint_coverage(combined, list(gold.required_actions))
    concept_hits = {c: _concept_hit(text_l, c) for c in gold.required_concepts}
    forbidden_hits = [f for f in gold.forbidden if _forbidden_hit(text_l, f)]

    n_action = len(gold.required_actions)
    n_concept = len(gold.required_concepts)
    n_forbidden = len(gold.forbidden)

    action_part = action_cov["coverage_rate"] if n_action else 1.0
    concept_part = (
        sum(1 for v in concept_hits.values() if v) / n_concept if n_concept else 1.0
    )
    forbidden_penalty = len(forbidden_hits) / max(1, n_forbidden) if n_forbidden else 0.0

    weights = []
    parts = []
    if n_action:
        weights.append(0.6)
        parts.append(action_part)
    if n_concept:
        weights.append(0.25)
        parts.append(concept_part)
    if n_forbidden:
        weights.append(0.15)
        parts.append(1.0 - forbidden_penalty)

    if not weights:
        composite = 1.0
    else:
        total_w = sum(weights)
        composite = sum(p * w for p, w in zip(parts, weights)) / total_w

    return {
        "what_objective_score": round(composite, 4),
        "action_coverage": action_cov,
        "concept_hits": concept_hits,
        "forbidden_hits": forbidden_hits,
        "forbidden_penalty": round(forbidden_penalty, 4),
        "parsed": parsed,
        "passed_what_objective": composite >= 0.75 and not forbidden_hits,
    }


def persona_operation_score(speech: str, persona: str) -> dict[str, Any]:
    """Lightweight How operation check — Goku vs Makima should diverge."""
    text_l = speech.lower()
    goku_markers = ["training", "buddy", "power", "let's", "gonna", "fight"]
    makima_markers = ["command", "obey", "failure", "unacceptable", "precisely", "control"]
    trinity_markers = ["discipline", "non-negotiable", "protocol", "director"]

    if persona == "goku":
        hits = sum(1 for m in goku_markers if m in text_l)
        anti = sum(1 for m in makima_markers if m in text_l)
    elif persona == "makima":
        hits = sum(1 for m in makima_markers if m in text_l)
        anti = sum(1 for m in goku_markers if m in text_l)
    else:
        hits = sum(1 for m in trinity_markers if m in text_l)
        anti = 0

    return {
        "persona_marker_hits": hits,
        "cross_persona_markers": anti,
        "persona_signal_strength": hits - anti * 0.5,
    }
