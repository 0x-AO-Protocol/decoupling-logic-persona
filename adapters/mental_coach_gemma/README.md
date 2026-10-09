# Gemma-3-4B-it-4bit-MentalCoach-LoRA

The logic-path ("What" path) adapter on the Gemma base in *Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution* (arXiv:2610.09772), referred to there as the Mental Coach LoRA. It is a LoRA adapter for [MLX](https://github.com/ml-explore/mlx) / mlx-lm.

**Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms**

## Files

| File | Content |
|---|---|
| `adapters.safetensors` | LoRA weights (28,076,958 bytes; SHA-256 `379ab890934c506662fa3d8dbd99fb1019ea38f33afa1099ebac12fe34a01dbc`) |
| `adapter_config.json` | The mlx-lm training configuration (local paths anonymised; the `model` and `data` paths are not needed for loading) |
| `GEMMA_TERMS_OF_USE.md` | Copy of the Gemma Terms of Use as published at https://ai.google.dev/gemma/terms when this repository was released |
| `NOTICE` | Notice required by the Gemma Terms of Use |

## Base model

Gemma-3-4B-it as a text-only 4-bit MLX re-quantization with group size 64 (`./gemma-3-4b-it-4bit-quantized` in `adapter_config.json`). The base weights are not included; obtain Gemma under the Gemma Terms of Use. The adapter was evaluated only on the authors' copy of this re-quantization.

## Training

`mlx_lm.lora`, LoRA rank 8, scale 20, dropout 0, 16 layers, 200 iterations, Adam, learning rate 1e-5, batch size 1, maximum sequence length 2048, seed 0, on 20 training examples (not released). The released weights are those after the final (200th) iteration.

The adapter was trained to answer in the format `<|start_thought|>…<|end_thought|><|start_json|>{"s": "<state>", "a": "<recommended_action>"}<|end_json|>`; the system headers used with it in the paper are reproduced in its Appendix E.2.

## Loading

```python
from mlx_lm import load
model, tokenizer = load("<path to gemma-3-4b-it-4bit-quantized>", adapter_path="adapters/mental_coach_gemma")
```

## Intended use and limitations

For reproducing and extending the experiments in the paper. The adapter was trained on 20 examples and its advice is often incomplete or wrong; the paper scores it only on whether required actions are mentioned. It must not be used to give coaching, health, medical or legal advice to anyone.

## Licence and use restrictions

These adapter weights are a Model Derivative of Gemma as defined in the Gemma Terms of Use, and are distributed under and subject to the Gemma Terms of Use (`GEMMA_TERMS_OF_USE.md`). The MIT licence of the rest of this repository does not apply to this directory.

**Use restrictions.** As a condition of receiving and using these adapter weights, you may not use them, or any model derived from them, for the restricted uses set out in the Gemma Prohibited Use Policy at https://ai.google.dev/gemma/prohibited_use_policy (which is incorporated here by reference), or in violation of applicable laws and regulations. These are the use restrictions of Section 3.2 of the Gemma Terms of Use. If you distribute these weights or a model derived from them, you must include these use restrictions as an enforceable provision in the terms governing that distribution, give your recipients a copy of the Gemma Terms of Use, and include the `NOTICE` file.

**Modified files.** `adapters.safetensors` is a Model Derivative of Gemma-3-4B-it: LoRA weights created by the authors by training on that model. This notice marks it as such. No file distributed by Google is included in this directory.
