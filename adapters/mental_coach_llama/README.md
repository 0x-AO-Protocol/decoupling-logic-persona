# Llama-3.1-8B-Instruct-4bit-MentalCoach-LoRA

**Built with Llama.**

The logic-path ("What" path) adapter on the Llama base in *Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution* (arXiv:2610.09772), referred to there as the Mental Coach LoRA. It is a LoRA adapter for [MLX](https://github.com/ml-explore/mlx) / mlx-lm.

## Files

| File | Content |
|---|---|
| `adapters.safetensors` | LoRA weights (41,967,272 bytes; SHA-256 `ac652398a5016455d9d76545459cbf89a89dcbebc6fef167d1eb5936890842f5`) |
| `adapter_config.json` | The mlx-lm training configuration (local paths anonymised; the `model` and `data` paths are not needed for loading) |
| `LICENSE` | Copy of the Llama 3.1 Community License Agreement |
| `USE_POLICY.md` | Copy of the Llama 3.1 Acceptable Use Policy |
| `NOTICE` | Attribution notice required by the Llama 3.1 Community License |

## Base model

Meta-Llama-3.1-8B-Instruct, 4-bit MLX quantization with group size 64 (`./Meta-Llama-3.1-8B-Instruct-4bit` in `adapter_config.json`). The base weights are not included; obtain them under Meta's licence.

## Training

`mlx_lm.lora`, LoRA rank 8, scale 20, dropout 0, 16 layers, 200 iterations, Adam, learning rate 1e-5, batch size 1, maximum sequence length 2048, seed 0, on 20 training examples (not released). The released weights are those after the final (200th) iteration.

The adapter was trained to answer in the format `<|start_thought|>…<|end_thought|><|start_json|>{"s": "<state>", "a": "<recommended_action>"}<|end_json|>`; the system headers used with it in the paper are reproduced in its Appendix E.2.

## Loading

```python
from mlx_lm import load
model, tokenizer = load("<path to Meta-Llama-3.1-8B-Instruct-4bit>", adapter_path="adapters/mental_coach_llama")
```

## Intended use and limitations

For reproducing and extending the experiments in the paper. The adapter was trained on 20 examples and its advice is often incomplete or wrong; the paper scores it only on whether required actions are mentioned. It must not be used to give coaching, health, medical or legal advice to anyone.

## Licence

This adapter is a derivative of Llama 3.1 and is distributed under the Llama 3.1 Community License (`LICENSE`). Its use must comply with applicable laws and regulations and with the Llama 3.1 Acceptable Use Policy (`USE_POLICY.md`, current version at https://llama.meta.com/llama3_1/use-policy). The MIT licence of the rest of this repository does not apply to this directory.

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
