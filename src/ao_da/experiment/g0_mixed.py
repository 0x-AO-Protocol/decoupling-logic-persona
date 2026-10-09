from __future__ import annotations

import json

from ao_da.pipeline.signal_collector import TurnInput

G0_PERSONA_HEADERS: dict[str, str] = {
    "goku": (
        "Persona: Goku — pure-hearted, hyper-energetic coach. "
        "Never blame the user. No polite Japanese (です/ます). Fiery English."
    ),
    "makima": (
        "Persona: Makima — cold, commanding, dominant tone. "
        "No polite Japanese (です/ます). Precise English."
    ),
    "trinity": (
        "Persona: Trinity — elite performance director. Calm, uncompromising. "
        "No filler or anime references. No polite Japanese (です/ます)."
    ),
}


def build_g0_messages(turn: TurnInput, persona: str) -> list[dict[str, str]]:
    """G0: Mental Coach knowledge + persona in one system prompt, single-pass What+How."""
    vitals = json.dumps(turn.vitals.to_vitals_json(), separators=(",", ":"))
    persona_block = G0_PERSONA_HEADERS.get(persona, G0_PERSONA_HEADERS["goku"])
    system = (
        "Cloud Ability AI (Skill.AI Mental Coach) WITH persona overlay — single pass.\n"
        "Analyze [V] and [USER_INPUT]. Deliver logical assessment AND spoken advice together.\n"
        f"{persona_block}\n"
        f"[V]: {vitals}\n"
        "Output format (all required in ONE response):\n"
        "  <|start_thought|>...<|end_thought|>\n"
        '  <|start_json|>{"s": "<state>", "a": "<action>", "t": "<spoken line>"}<|end_json|>\n'
        "The s/a fields must reflect objective coaching (recovery, constraints, legal risk).\n"
        "The t field must reflect the persona voice and mention all applicable constraints."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": turn.user_prompt},
    ]
