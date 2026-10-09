from __future__ import annotations

from ao_da.experiment.g0_mixed import build_g0_messages
from ao_da.experiment.pollution import PollutionContext
from ao_da.pipeline import micro_state
from ao_da.pipeline.signal_collector import TurnInput


def _append_system_bloat(system: str, bloat: str) -> str:
    if not bloat:
        return system
    return f"{system}\n\n{bloat}"


def assemble_g0_messages(
    turn: TurnInput,
    persona: str,
    ctx: PollutionContext,
) -> list[dict[str, str]]:
    """G0: all pollution in one pass (system bloat + history + effective user)."""
    base = build_g0_messages(turn, persona)
    system = _append_system_bloat(base[0]["content"], ctx.system_bloat)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(ctx.history_list)
    messages.append({"role": "user", "content": ctx.effective_user_prompt})
    return messages


def assemble_what_messages(
    turn: TurnInput,
    ctx: PollutionContext,
) -> list[dict[str, str]]:
    """What path: clean = core user only; polluted = history + effective user."""
    if ctx.what_path_mode == "clean":
        return micro_state.build_what_messages(turn)

    base = micro_state.build_what_messages(turn)
    system = _append_system_bloat(base[0]["content"], ctx.system_bloat)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(ctx.history_list)
    messages.append({"role": "user", "content": ctx.effective_user_prompt})
    return messages


def assemble_how_messages(
    turn: TurnInput,
    micro_state_dict: dict,
    persona_id,
    ctx: PollutionContext,
    *,
    strict_constraint_echo: bool = False,
) -> list[dict[str, str]]:
    """How path: polluted mode injects history + effective user after persona system."""
    base = micro_state.build_persona_messages(
        persona_id,
        turn,
        micro_state_dict,
        strict_constraint_echo=strict_constraint_echo,
    )
    if ctx.how_path_mode == "clean" or (not ctx.history_list and ctx.effective_user_prompt == ctx.core_user_prompt):
        return base

    system = _append_system_bloat(base[0]["content"], ctx.system_bloat)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(ctx.history_list)
    messages.append({"role": "user", "content": ctx.effective_user_prompt})
    return messages
