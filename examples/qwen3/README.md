# Qwen3-0.6B as a TGIR program

`import.py` turns the open-weight [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
checkpoint into a plain Technograd program — `qwen3.tg`, where every operation
of the forward pass is explicit and editable — plus f32 weight files its
`file()` params load. Once imported, the model runs in the reference VM
(`tgc run`) and every tensor and operation can be inspected, instrumented, or
rewritten as ordinary TGIR.

```sh
sh examples/qwen3/run.sh build/tgc work      # download, import, verify vs Hugging Face
build/tgc run work/qwen3.tg "151643,785,3974,13876,11,323,279,1879,374,264,1602,2244,1992,311,3161,13"
```

## What the import covers

The whole dense transformer, in existing TGIR ops — no new operation was
added:

- token embedding via `take`, tied `lm_head`;
- 28 blocks, each pre-norm RMSNorm (`rmsnorm(x) * weight`), grouped-query
  attention (16 query heads over 8 key/value heads), SwiGLU MLP, residuals;
- Qwen3 per-head query/key RMSNorm (`q_norm`, `k_norm` over the 128-dim head);
- rotary position embedding in the Hugging Face rotate-half convention with
  theta 1e6, baked as exact cos/sin tables and a rotation matrix, so no
  `rope`-op convention mismatch;
- causal attention as a masked softmax.

Multi-head attention is emitted **without a batched-attention op**: each of
the 16 heads is a few rank-2 matmuls, and each head's `o_proj` contribution
is summed, so grouped-query attention falls out of baking per-head weight
slices at import time.

## Fidelity

For a fixed sequence length (the causal mask and rotary tables are sized at
import), `tgc run` reproduces Hugging Face's float32 logits:

| | |
|---|---|
| max abs logit difference | 4.1e-5 |
| mean abs difference | 1.1e-5 |
| relative (/ max abs logit) | 3.0e-6 |
| argmax / top-5 | identical |

`verify.py` feeds identical token ids to both and reports this. The
difference is float32 matmul reassociation, not a modelling gap.

## Scope

This is an importer and an execution/inspection substrate for small open
models (Qwen3-0.6B is ~0.6B parameters, ~2.4 GB as f32). The reference VM is
CPU and single-arena; it is not a runtime for multi-billion-parameter models.
No tokenizer is bundled — the program takes token ids in and returns logits,
which is what an editing/interpretability workflow needs. The next steps are
IR edit passes (head/layer ablation, activation dumps, steering vectors) over
the imported program.
