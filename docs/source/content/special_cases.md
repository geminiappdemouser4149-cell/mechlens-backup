# Special Cases

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.
```

## Mixture of Experts error rates
Due to the Top-K gating performed in the hidden layer of Mixture of Experts models, small errors can be amplified 
greatly in cases where a different expert is selected, which leads to a higher than normal variance in the error rate
of the final logits. In testing done on Mixtral running in half precision, the standard deviation of the absolute error 
rate of the logits compared to those from the default model was found to be around 2e-3.

There are two main ways to mitigate this:

1. **Skip weight preprocessing.** On the bridge, simply load with `TransformerBridge.boot_transformers(...)` and do not call `enable_compatibility_mode()` - the bridge preserves raw HF weights by default, so no additional flag is needed. On the legacy `HookedTransformer` path, use `HookedTransformer.from_pretrained_no_processing` instead of `HookedTransformer.from_pretrained`.
2. **Increase the precision of the data type used in the model.**

## Qwen3.5 text-only models

Qwen3.5 has no `HookedTransformer.from_pretrained` legacy path. Load it through `TransformerBridge`. Dense text-only checkpoints can be loaded with:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("Qwen/Qwen3.5-0.8B")
```

Qwen3.5 uses a hybrid stack. Full-attention layers expose the usual hooks under
`blocks.N.attn.*`; linear-attention layers expose GatedDeltaNet hooks under
`blocks.N.linear_attn.*`, including `hook_q_pre_conv`, `hook_k_pre_conv`,
`hook_v_pre_conv`, `hook_beta`, `hook_log_decay`, `hook_recurrence_out`, and
`hook_out`. Full multimodal `Qwen3_5ForConditionalGeneration`, image/video
inputs, and Qwen3.5 MoE checkpoints are not supported by this adapter.
## Cohere Command-R and Aya models

Cohere Command-R and Aya models (`CohereForCausalLM` — including `CohereLabs/c4ai-command-r-v01`, `CohereLabs/c4ai-command-r-plus`, `CohereLabs/c4ai-command-r-08-2024`, `CohereLabs/c4ai-command-r-plus-08-2024`, `CohereLabs/aya-23-*`, and `CohereLabs/aya-expanse-*`) are supported through `TransformerBridge` (no legacy `HookedTransformer.from_pretrained` path):

- **Parallel attention + MLP (`parallel_attn_mlp=True`)**: Single pre-norm `blocks.{i}.ln1` (`input_layernorm`) feeds both `attn` and `mlp`; no `ln2` or `hook_resid_mid`.
- **True mean-subtracting LayerNorm (`uses_rms_norm=False`)**: `CohereLayerNorm` subtracts the mean with a learnable `weight` and no `bias`.
- **Interleaved RoPE (`rotary_adjacent_pairs=True`) and optional 2D QK-norm (`use_qk_norm`)**: Command-R+ and 08-2024 checkpoints apply per-head `q_norm`/`k_norm` with shape `(n_heads, head_dim)` post-reshape before RoPE. Under `enable_compatibility_mode()` (`fold_ln=True`), `ln1` folds into `q_proj`/`k_proj`/`v_proj`/`mlp` while `q_norm`/`k_norm` remain active.
- **Idempotent `logit_scale` folding**: `preprocess_weights` folds `logit_scale` (default `0.0625`) into cloned `unembed.weight` (leaving tied `embed.W_E` unscaled) and resets `original_model.logit_scale = 1.0`.
- **`Cohere2ForCausalLM` scope**: Newer Cohere 2 models (Command-R7B / Command-A) use `Cohere2ForCausalLM` and are tracked separately.

## Gemma 3 RMSNorm epsilon upgrade note (v2.16.1+)

Gemma 3 (`Gemma3ForCausalLM` / `Gemma3ForConditionalGeneration`) applies RMSNorm across all 7 norm sites as `(1.0 + weight) * x * rsqrt(mean(x^2) + eps)` with `eps` (`1e-6`) **inside** `rsqrt`. Users on `v2.16.0` should upgrade to `transformer-lens >= 2.16.1` (or `v3.0+` `TransformerBridge`) for the `#1121` / PR `#1122` hotfix (`< 3e-5` max logit delta vs HuggingFace).

