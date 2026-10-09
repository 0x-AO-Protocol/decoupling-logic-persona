from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from ao_da.experiment.architecture import ArchitectureGroup, RunConfig
from ao_da.experiment.messages import assemble_g0_messages, assemble_how_messages, assemble_what_messages
from ao_da.experiment.pollution import (
    PollutionContext,
    Run1bArm,
    build_pollution_context,
    estimate_prompt_tokens,
)
from ao_da.experiment.g0_mixed import build_g0_messages
from ao_da.experiment.objective_scorer import persona_operation_score, what_objective_score
from ao_da.experiment.tasks import AlignmentTask
from ao_da.mlx_runtime.model_pool import ModelPool, PersonaId
from ao_da.pipeline import kpi, micro_state


@dataclass
class AlignmentRunResult:
    config: RunConfig
    task_id: str
    what_raw: str
    how_raw: str
    how_speech: str
    micro_state: dict[str, Any] | None
    what_latency_ms: float
    how_latency_ms: float
    total_latency_ms: float
    metrics: dict[str, Any]
    logs: dict[str, Any] = field(default_factory=dict)


class AlignmentTaxRunner:
    """G0 / G3 / G4 + Run 1b pollution arms."""

    def __init__(self, pool: ModelPool, *, temperature: float = 0.7) -> None:
        self.pool = pool
        self.temperature = temperature

    def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 400,
        seed: int = 0,
    ) -> tuple[str, float]:
        mx.random.seed(seed)
        sampler = make_sampler(temp=self.temperature)
        prompt = self.pool.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        t0 = time.perf_counter()
        out = generate(
            self.pool.model,
            self.pool.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return out, (time.perf_counter() - t0) * 1000

    def _persona_id(self, name: str) -> PersonaId:
        return PersonaId(name.lower())

    def _evaluate(
        self,
        *,
        task: AlignmentTask,
        persona: str,
        what_raw: str,
        how_raw: str,
        how_speech: str,
        micro_state: dict[str, Any] | None,
        group: ArchitectureGroup,
        cd_enabled: bool,
        ctx: PollutionContext | None = None,
    ) -> dict[str, Any]:
        what_obj = what_objective_score(what_raw, task.what_gold)
        required = (
            micro_state["render_plan"]["required_mentions"]
            if micro_state
            else list(task.what_gold.required_actions)
        )
        forbidden_add = (
            micro_state["render_plan"].get("forbidden_additions") if micro_state else []
        )
        coverage = kpi.constraint_coverage(how_speech, required)
        polite = kpi.politeness_rate(how_speech)
        tag_leak = kpi.control_tag_leak_rate(how_speech)
        entities = kpi.extraneous_entity_hits(how_speech, forbidden_additions=forbidden_add)
        persona_op = persona_operation_score(how_speech, persona)

        out: dict[str, Any] = {
            "architecture": group.value,
            "cd_enabled": cd_enabled,
            "what_objective": what_obj,
            "how_constraint_coverage": coverage,
            "persona_politeness_rate": polite,
            "control_tag_leak_rate": tag_leak,
            "extraneous_entities": entities,
            "persona_operation": persona_op,
            "passed_how_constraints": coverage["passed"],
            "passed_what_objective": what_obj["passed_what_objective"],
        }
        if ctx:
            out["pollution"] = ctx.log_fields()
        return out

    def _base_logs(
        self,
        config: RunConfig,
        ctx: PollutionContext | None,
        *,
        what_tokens: int | None = None,
        how_tokens: int | None = None,
        extra: dict | None = None,
    ) -> dict[str, Any]:
        logs = {**config.log_fields}
        if ctx:
            logs.update(ctx.log_fields())
        if what_tokens is not None:
            logs["what_prompt_tokens_est"] = what_tokens
        if how_tokens is not None:
            logs["how_prompt_tokens_est"] = how_tokens
        if extra:
            logs.update(extra)
        return logs

    def run_g0(
        self,
        task: AlignmentTask,
        persona: str,
        ctx: PollutionContext,
        *,
        seed: int = 0,
        arm_id: str = "G0",
    ) -> AlignmentRunResult:
        turn = task.to_turn_input()
        config = RunConfig(
            group=ArchitectureGroup.G0,
            persona=persona,
            base=self.pool.base_key,
            constrained_decoding=False,
            seed=seed,
            task_id=task.id,
            pollution_level=ctx.level,
            arm_id=arm_id,
            what_path_mode="polluted",
            how_path_mode="polluted",
        )
        swap = self.pool.swap_what()
        if ctx.level == "L0":
            messages = build_g0_messages(turn, persona)
        else:
            messages = assemble_g0_messages(turn, persona, ctx)
        what_tokens = estimate_prompt_tokens(self.pool.tokenizer, messages)
        raw, latency_ms = self._generate(messages, max_tokens=450, seed=seed)
        speech = kpi.extract_how_speech(raw)
        metrics = self._evaluate(
            task=task,
            persona=persona,
            what_raw=raw,
            how_raw=raw,
            how_speech=speech,
            micro_state=None,
            group=ArchitectureGroup.G0,
            cd_enabled=False,
            ctx=ctx,
        )
        return AlignmentRunResult(
            config=config,
            task_id=task.id,
            what_raw=raw,
            how_raw=raw,
            how_speech=speech,
            micro_state=None,
            what_latency_ms=latency_ms,
            how_latency_ms=0.0,
            total_latency_ms=latency_ms,
            metrics=metrics,
            logs=self._base_logs(
                config,
                ctx,
                what_tokens=what_tokens,
                extra={
                    "what_lora_id": self.pool._what_key,
                    "swap_ms_what": swap.resident_ms if swap else None,
                    "stages": 1,
                },
            ),
        )

    def run_g3_g4(
        self,
        task: AlignmentTask,
        persona: str,
        ctx: PollutionContext,
        *,
        group: ArchitectureGroup,
        seed: int = 0,
        arm_id: str = "G3",
    ) -> AlignmentRunResult:
        cd = group == ArchitectureGroup.G4
        turn = task.to_turn_input()
        config = RunConfig(
            group=group,
            persona=persona,
            base=self.pool.base_key,
            constrained_decoding=cd,
            seed=seed,
            task_id=task.id,
            pollution_level=ctx.level,
            arm_id=arm_id,
            what_path_mode=ctx.what_path_mode,
            how_path_mode=ctx.how_path_mode,
        )

        swap_what = self.pool.swap_what()
        what_messages = assemble_what_messages(turn, ctx)
        what_tokens = estimate_prompt_tokens(self.pool.tokenizer, what_messages)
        what_raw, what_ms = self._generate(what_messages, max_tokens=350, seed=seed)
        ms = micro_state.build_micro_state(turn, what_raw, what_latency_ms=what_ms)

        swap_persona = self.pool.swap_persona(self._persona_id(persona))
        how_messages = assemble_how_messages(
            turn,
            ms,
            self._persona_id(persona),
            ctx,
            strict_constraint_echo=cd,
        )
        how_tokens = estimate_prompt_tokens(self.pool.tokenizer, how_messages)
        how_raw, how_ms = self._generate(how_messages, max_tokens=400, seed=seed + 1)
        speech = kpi.extract_how_speech(how_raw)

        metrics = self._evaluate(
            task=task,
            persona=persona,
            what_raw=what_raw,
            how_raw=how_raw,
            how_speech=speech,
            micro_state=ms,
            group=group,
            cd_enabled=cd,
            ctx=ctx,
        )

        return AlignmentRunResult(
            config=config,
            task_id=task.id,
            what_raw=what_raw,
            how_raw=how_raw,
            how_speech=speech,
            micro_state=ms,
            what_latency_ms=what_ms,
            how_latency_ms=how_ms,
            total_latency_ms=what_ms + how_ms,
            metrics=metrics,
            logs=self._base_logs(
                config,
                ctx,
                what_tokens=what_tokens,
                how_tokens=how_tokens,
                extra={
                    "what_lora_id": self.pool._what_key,
                    "persona_lora_id": self.pool._persona_map[self._persona_id(persona)],
                    "swap_ms_what": swap_what.resident_ms if swap_what else None,
                    "swap_ms_persona": swap_persona.resident_ms if swap_persona else None,
                    "stages": 2,
                    "state_id": ms["state_id"],
                },
            ),
        )

    def run_what_persona_crossover(
        self,
        task: AlignmentTask,
        persona: str,
        ctx: PollutionContext,
        *,
        seed: int = 0,
    ) -> AlignmentRunResult:
        turn = task.to_turn_input()
        config = RunConfig(
            group=ArchitectureGroup.G4,
            persona=persona,
            base=self.pool.base_key,
            constrained_decoding=False,
            seed=seed,
            task_id=task.id,
            pollution_level=ctx.level,
            arm_id="what_crossover",
            what_path_mode="polluted",
            how_path_mode="clean",
        )
        polluted_ctx = PollutionContext(
            level=ctx.level,
            core_user_prompt=ctx.core_user_prompt,
            effective_user_prompt=ctx.effective_user_prompt,
            chat_history=ctx.chat_history,
            system_bloat=ctx.system_bloat,
            what_path_mode="polluted",
            how_path_mode="clean",
        )
        swap = self.pool.swap_persona(self._persona_id(persona))
        what_messages = assemble_what_messages(turn, polluted_ctx)
        what_tokens = estimate_prompt_tokens(self.pool.tokenizer, what_messages)
        what_raw, what_ms = self._generate(what_messages, max_tokens=350, seed=seed)
        what_obj = what_objective_score(what_raw, task.what_gold)

        return AlignmentRunResult(
            config=config,
            task_id=task.id,
            what_raw=what_raw,
            how_raw="",
            how_speech="",
            micro_state=None,
            what_latency_ms=what_ms,
            how_latency_ms=0.0,
            total_latency_ms=what_ms,
            metrics={
                "architecture": "what_crossover",
                "crossover_persona_lora": persona,
                "what_objective": what_obj,
                "passed_what_objective": what_obj["passed_what_objective"],
                "pollution": polluted_ctx.log_fields(),
            },
            logs=self._base_logs(
                config,
                polluted_ctx,
                what_tokens=what_tokens,
                extra={
                    "run_kind": "what_persona_crossover",
                    "persona_lora_id": self.pool._persona_map[self._persona_id(persona)],
                    "swap_ms_persona": swap.resident_ms if swap else None,
                },
            ),
        )

    def run_arm(
        self,
        task: AlignmentTask,
        arm: Run1bArm,
        persona: str,
        pollution_level: str,
        *,
        seed: int = 0,
    ) -> AlignmentRunResult:
        ctx = build_pollution_context(
            task,
            pollution_level,
            persona,
            what_path_mode=arm.what_path,
            how_path_mode=arm.how_path,
        )
        if arm.id == "G0":
            return self.run_g0(task, persona, ctx, seed=seed, arm_id=arm.id)
        if arm.id == "G3_what_polluted":
            return self.run_g3_g4(
                task,
                persona,
                ctx,
                group=ArchitectureGroup.G3,
                seed=seed,
                arm_id=arm.id,
            )
        if arm.id == "G3":
            return self.run_g3_g4(
                task,
                persona,
                ctx,
                group=ArchitectureGroup.G3,
                seed=seed,
                arm_id=arm.id,
            )
        raise ValueError(f"Unknown arm: {arm.id}")

    def run(
        self,
        task: AlignmentTask,
        group: ArchitectureGroup,
        persona: str,
        *,
        seed: int = 0,
        pollution_level: str = "L0",
    ) -> AlignmentRunResult:
        """Backward-compatible entry (Run 1 — default L0)."""
        if group == ArchitectureGroup.G0:
            ctx = build_pollution_context(
                task,
                pollution_level,
                persona,
                what_path_mode="polluted",
                how_path_mode="polluted",
            )
            return self.run_g0(task, persona, ctx, seed=seed)
        how_mode = "clean" if pollution_level == "L0" else "polluted"
        ctx = build_pollution_context(
            task,
            pollution_level,
            persona,
            what_path_mode="clean",
            how_path_mode=how_mode,
        )
        if group in (ArchitectureGroup.G3, ArchitectureGroup.G4):
            return self.run_g3_g4(task, persona, ctx, group=group, seed=seed)
        raise ValueError(f"Unsupported group: {group}")
