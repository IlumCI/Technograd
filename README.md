# Technograd

Technograd is a small, statically shaped language for building black-box
embedded models with latent reasoning loops in the style of Coconut. It has
two layers:

| Layer | Form | Audience |
|-------|------|----------|
| Surface (`.tg`) | Indentation-structured, Python-like | Programmers |
| TGIR (`.tgir`) | Lisp S-expressions, SSA, explicit shapes | Machines: generators, optimizers, verifiers, other models |

Both layers compile to the same IR. The compiler, `tgc`, can interpret a model
or emit one freestanding C file: weights baked in as `const` data, one
statically planned arena, no heap, no I/O, and a single entry point. That
file is the black box you ship to the target.

```
.tg  --parse-->  AST (S-expr)  --lower/inline/SSA-->  IR  <--read--  .tgir
                                                        |
                                    shape verify, static memory plan
                                                        |
                                 +----------------------+-----------------+
                                 |                                        |
                         tgc run (reference VM)                 tgc c (freestanding C99)
                                 \________ identical kernels: runtime/tg_rt.h ________/
```

## Quick start

```sh
make            # builds build/tgc (C11, no dependencies beyond libm; perl at build time)
build/tgc train hf:scikit-learn/iris            # any dataset, no configuration: detect, featurize, train, export
build/tgc train reviews.csv --model ssm         # adds stacked Mamba-3 blocks (MIMO, BCNorm) over the words
build/tgc pretrain corpus.jsonl -o lm             # next-token pretraining on unlabeled text ...
build/tgc train labeled.jsonl --model ssm --init lm   # ... then fine-tune from it
sh examples/military_purple/run.sh build/tgc work  # military science + purple teaming corpus, ATT&CK tactic task
build/tgc gen-train pairs.jsonl --aligned -o cipher   # learn a byte transform from (in,out) pairs; emit exact strings
build/tgc generate cipher --input hello               # -> the transformed string
sh examples/crypto_ctf/run.sh build/tgc work          # learn fixed-key classical ciphers, exact-match on held-out
sh examples/qwen3/run.sh build/tgc work             # import open-weight Qwen3-0.6B into TGIR; logits match Hugging Face
build/tgc predict iris_model new_flowers.csv     # predictions on raw new data
make test       # 291 checks (offline; TG_TEST_NETWORK=1 adds a Hugging Face run)
make fixer      # retrain the auto-fix forest (deterministic, ~15 s) and evaluate it on held-out programs

build/tgc run  examples/latent_reasoner.tg 1,0,0,1,0,1,1,0
# 0.225605994 0.24476327 0.162138939 0.367491812
# steps 0 9                 <- the latent loop halted after 9 of 32 allowed steps

build/tgc ir   examples/newton.tg            # canonical TGIR
build/tgc batch examples/xor.tg tests/io/xor.csv   # one sample per row; .bin/.f32 for raw f32 streams
build/tgc run examples/newton.tg 2,9,16 -vv --log run.jsonl   # stage/value/iteration trace + JSONL log
build/tgc batch examples/latent_reasoner.tg data.bin -j auto -o out.bin   # rows across all cores, bit-identical to -j 1
build/tgc batch examples/delta_memory.tg tests/state/mem.csv   # a model that rewrites its own memory every run
build/tgc batch examples/train_xor.tg xor.csv --save-state xor   # online SGD via grad(); writes xor.<weight>.bin
build/tgc batch examples/selective_ssm.tg seqs.csv   # a Mamba-style layer over 12-step sequences, trained by BPTT
build/tgc batch examples/kv_attention.tg tokens.csv   # streaming causal attention over a fixed key/value ring
sh examples/coconut/curriculum.sh build/tgc work     # Coconut curriculum: k latent steps for k-hop answers
sh examples/ponder/run.sh build/tgc work             # learned halting (PonderNet): steps taken = hops needed
build/tgc serve examples/xor.tg --listen 8080      # TCP rows and HTTP (GET /health, GET /, POST /run)
printf '1,0\n0,1\n' | build/tgc stream examples/xor.tg   # row in, row out; tcp://HOST:PORT works as source/sink
build/tgc plan examples/latent_reasoner.tg   # arena layout: 320 B instead of 624 B unshared
build/tgc c    examples/latent_reasoner.tg -o reasoner.c
build/tgc c    examples/latent_reasoner.tg --fixed -o reasoner_q16.c   # Q16.16 integers: no FPU needed
build/tgc quantize model.tg --bits 4 --method gptq --calib rows.csv   # int4 weights, per-channel scales
build/tgc asm  examples/latent_reasoner.tg --target cortex-m4f -o reasoner.s   # native: cortex-m3 (Q16.16), cortex-m4f, rv64gcv
build/tgc asm  examples/latent_reasoner.tg --target cortex-m4f --support -o support.c   # kernels + driver to link with it
cc -O2 -DTG_MAIN reasoner.c -lm -o reasoner && ./reasoner 1,0,0,1,0,1,1,0
```

## The language in one example

```python
model latent_reasoner

param w_emb  : f32[16, 8]  = rand(11, 0.35)      # or file("w_emb.bin"), zeros, ones, fill(v), [..literal..]
param w_core : f32[16, 16] = rand(23, 0.12)
param w_inj  : f32[16, 16] = rand(37, 0.10)
param w_out  : f32[4, 16]  = rand(41, 0.5)

def core(h: f32[16], e: f32[16]) -> f32[16]:
    return tanh(w_core @ h + w_inj @ e)

def forward(x: f32[8]) -> f32[4]:        # entry point: args = inputs, return = output
    e = rmsnorm(w_emb @ x)
    h = e
    think h for 32 until 0.0001:          # continuous thought: the hidden state is fed
        h = core(h, e)                    # back as the next input, never decoded
    return softmax(w_out @ h)
```

`think h for N [until eps]:` is the language's central construct. It runs the
body at most `N` times with `h` as the single loop-carried latent state. With
`until eps`, the loop stops as soon as `max|h_next - h| <= eps`, so easy
inputs use less compute. The number of iterations taken is reported in
`steps` by the VM and in `tg_<model>_steps[]` by compiled units.

The full `examples/latent_reasoner.tg` (which also has biases) in TGIR, param data elided:

```lisp
(tgir 1
  (model latent_reasoner)
  (param w_emb (f32 16 8) (data ...))
  ...
  (input x (f32 8))
  (block
    (%8 (f32 16) (matmul w_emb x))
    (%9 (f32 16) (add %8 b_emb))
    (%10 (f32 16) (rmsnorm %9))
    (%11 (f32 16) (think %10 32 9.99999975e-05
      (%12 (f32 16) (matmul w_core %11))
      (%13 (f32 16) (matmul w_inj %10))
      (%14 (f32 16) (add %12 %13))
      (%15 (f32 16) (add %14 b_core))
      (%16 (f32 16) (tanh %15))
      (yield %16)))
    (%17 (f32 4) (matmul w_out %11))
    (%18 (f32 4) (add %17 b_out))
    (%19 (f32 4) (softmax %18))
  )
  (output %19))
```

The full reference is in [docs/SPEC.md](docs/SPEC.md).

## Auto-fix: a neural decision forest that repairs compile errors

When compilation fails, `tgc fix` (or `--autofix` on any command) runs a
repair loop instead of stopping at the first error:

```
$ build/tgc fix tests/fix/def_typo.tg -o fixed.tg
tests/fix/def_typo.tg:32: error: undefined name 'e'
  autofix: rename at line 31 (confidence 1.00)
  -     ee = embed(x)
  +     e = embed(x)
autofix: 1 repair(s)

$ build/tgc run tests/fix/colon.tg 2,9,16 --autofix   # repairs in memory, source untouched
```

Each round of the loop does the following:

1. The diagnostic is classified, and repair operators chosen for that class
   propose candidate edits: rename (to similar names, or back to a name that
   is defined but never used), re-indent, swap or transpose matmul operands,
   insert or delete a token, close a bracket at every token boundary,
   make-return, delete a line, fix a character.
2. Every candidate is trial-compiled. The compiler's errors are trapped
   (`setjmp`/`longjmp`) and their allocations are released through a scoped
   allocator.
3. Each candidate, together with its diagnostic and trial outcome, becomes 35
   features. These include whether it compiles, whether the error moved, edit
   locality and size, name similarity (optimal string alignment) and usage,
   natural indentation, the number of candidates that compile (ambiguity), and
   the similarity margin over the other compiling candidates.
4. A forest of 16 soft decision trees of depth 3 scores each candidate. These
   are deep neural decision forests (Kontschieder et al., ICCV 2015) with
   sigmoid routing, trained end to end; each tree gets a bootstrap sample and
   a random feature subspace. The output is Platt-calibrated on out-of-bag
   predictions.
5. The best candidate is applied only if its calibrated probability is at
   least 0.95 and it makes progress. Otherwise the fixer abstains and the
   original error is reported.

The forest is a Technograd program: [`fixer/forest.tg`](fixer/forest.tg). The
compiler compiles it and runs it on its own VM. `tgc c fixer/forest.tg` turns
it into standalone C like any other model.

Training is self-supervised, as in DrRepair and Break-It-Fix-It. Breakers
corrupt valid programs with identifier typos, function-name typos,
indentation changes, dropped `:` and `)`, swapped matmul operands, removed
`return`, stray characters, and tabs. A candidate is labelled correct iff it
compiles to IR identical to the original program's, up to param and input
names. No labelled data is used. Training is deterministic: rerunning
`make fixer` reproduces `forest.tg` byte for byte, and the trainer checks
that the VM output matches its own to within 3e-7.

The acceptance threshold was chosen as the lowest one with zero wrong
repairs on fresh corruptions of the training programs. Results on programs
the forest never saw (`tests/cases`, 640 corruptions, `make fixer`):

| breaker | exact repair | wrong repair | abstained |
|---------|-------------:|-------------:|----------:|
| identifier typo | 81.8% | 0.0% | 18.2% |
| function-name typo | 100.0% | 0.0% | 0.0% |
| indentation | 87.8% | 0.0% | 12.2% |
| missing `:` | 100.0% | 0.0% | 0.0% |
| missing `)` | 53.5% | 0.0% | 46.5% |
| swapped matmul operands | 100.0% | 0.0% | 0.0% |
| missing `return` | 100.0% | 0.0% | 0.0% |
| stray character | 98.6% | 0.0% | 1.4% |
| tab indentation | 96.2% | 0.0% | 3.8% |
| **total** | **89.5%** | **0.0%** | **10.5%** |

Abstentions are mostly genuinely ambiguous cases. For example, with
`f(a, b + c` the closing parenthesis could go in more than one place and
still compile. The ambiguity feature makes the fixer decline these rather
than guess. You can trade safety for coverage with
`TG_FIXER_THRESHOLD=<p>`. Set `TG_FIXER_DEBUG` to have `fixer-eval` list
every failure.

## Why this shape

- **Low level enough for machines.** All shapes are static. Every user
  function is inlined. Every value is SSA with its type written next to it,
  and the IR is plain S-expressions. A model or a script can produce, diff,
  or transform TGIR with no knowledge of the surface grammar. The reader
  re-verifies every shape, so generated IR cannot get past the checker.
- **Abstract enough to program.** The surface syntax has functions, infix
  tensor algebra (`+ - * / @`), scalar and row broadcast, and `think` loops.
- **Black box by construction.** A compiled unit exposes only
  `tg_<model>_run(inputs..., out)` and size macros. Weights are static
  `const` data. Memory use is fixed when the unit is compiled
  (`TG_<model>_ARENA_BYTES`), and nothing is allocated at run time.
- **Implementation language.** The design took Lisp, Perl, and assembly into
  account. The Lisp idea lives on as the IR: the machine-facing language is
  S-expressions. The toolchain is C11 because it is the lowest portable layer
  above assembly, has no runtime, and can host the compiler on the same
  class of targets it compiles for. Perl is used only at build time, to embed
  the kernel library into emitted units. `tgc asm` is that code
  generator. It emits native code from the same planned IR for Cortex-M3 in
  fixed point, Cortex-M4F and RV64GCV, and hand-writes matmul for each.
- **One source of arithmetic.** The VM and the generated code call the same
  kernels (`runtime/tg_rt.h`). The test suite checks that their outputs match
  byte for byte.

## Research basis

- Coconut, continuous latent reasoning: Hao et al., *Training Large Language Models to Reason in a Continuous Latent Space*, [arXiv:2412.06769](https://arxiv.org/abs/2412.06769). This is the source of `think`: the hidden state is the next input. `examples/coconut/` runs its multi-stage curriculum in the language, by backpropagation through fixed-budget think loops.
- Learned halting: PonderNet, Banino et al., [arXiv:2107.05407](https://arxiv.org/abs/2107.05407), for `think ... halt p` and `examples/ponder/`.
- Native kernels: CMSIS-NN, Lai et al., [arXiv:1801.06601](https://arxiv.org/abs/1801.06601), for the Cortex-M `SMLAL` matmul; vector-length-agnostic RVV 1.0 GEMM, [arXiv:2311.05284](https://arxiv.org/abs/2311.05284), for `rv64gcv`.
- Quantization: GPTQ, Frantar et al., [arXiv:2210.17323](https://arxiv.org/abs/2210.17323), for `tgc quantize --method gptq`; per-channel weight-only int8/int4 with clipping search for `rtn`.
- Attention: rotary position embedding (RoFormer), [arXiv:2104.09864](https://arxiv.org/abs/2104.09864); sliding-window causal attention with a rolling key/value buffer as in Mistral 7B, [arXiv:2310.06825](https://arxiv.org/abs/2310.06825). This is the basis for `rope`, `attention` and `examples/kv_attention.tg`.
- Recurrent depth with input re-injection: Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*, [arXiv:2502.05171](https://arxiv.org/abs/2502.05171). This is the pattern used in `examples/latent_reasoner.tg`.
- Convergence-based adaptive halting: AdaAnchor, [arXiv:2603.15051](https://arxiv.org/abs/2603.15051). It reports 48-60% fewer latent steps at equal budget. This is the basis for `until eps`.
- Survey: *A Survey on Latent Reasoning*, [arXiv:2507.06203](https://arxiv.org/abs/2507.06203).
- Static arena planning, greedy by size: Pisarchyk & Lee, [arXiv:2001.03288](https://arxiv.org/abs/2001.03288).
- Sparse rows: feature hashing, Weinberger et al., [arXiv:0902.2206](https://arxiv.org/abs/0902.2206); sparse embedding-bag layers over hashed ids as in DLRM, [arXiv:1906.00091](https://arxiv.org/abs/1906.00091). This is the basis for `spmm` and the text path of `tgc train`. Row-sparse (lazy) optimizer steps on the touched rows follow the lazy Adam used for sparse embeddings (TensorFlow's TPU embedding optimizers, MXNet `lazy_update`), with a closed-form catch-up of the skipped steps.
- Sequence loops: `scan` follows the carry/sequence/stack form of JAX's `lax.scan`; its gradient is backpropagation through time (Werbos, 1990) as a reverse scan over a statically planned activation store. Selective state-space models: Mamba, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752); Mamba-3, [arXiv:2603.15569](https://arxiv.org/abs/2603.15569). This is the basis for `scan`, `examples/selective_ssm.tg` and the blocks of `tgc train --model ssm` (exponential-trapezoidal recurrence, complex state as data-dependent RoPE, MIMO, BCNorm, SwiGLU stacking).
- Pretraining then fine-tuning (`tgc pretrain`, `--init`): ULMFiT, Howard and Ruder, [arXiv:1801.06146](https://arxiv.org/abs/1801.06146); logit soft-capping as in Gemma 2, [arXiv:2408.00118](https://arxiv.org/abs/2408.00118).
- Open-weight import: Qwen3, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388); `examples/qwen3/` lowers Qwen3-0.6B into TGIR and matches Hugging Face float32 logits to 4e-5. The model becomes an editable program for inspection and IR-level edits.
- Byte-level sequence-to-sequence (`tgc gen-train`, `tgc generate`): a conditional byte transform over a 258-token vocabulary, greedy decoding, exact-match scoring. `examples/crypto_ctf/` learns fixed-key classical ciphers (the deterministic-transform shape of a cryptography CTF); classical/educational ciphers only, not attacks on modern cryptography.
- Self-updating state: test-time training layers, [arXiv:2407.04620](https://arxiv.org/abs/2407.04620); Titans, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663); DeltaNet, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484); Gated DeltaNet, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464); test-time regression as the unifying view, [arXiv:2501.12352](https://arxiv.org/abs/2501.12352). This is the basis for `state` and `update`.
- Optimizers: Muon, [arXiv:2502.16982](https://arxiv.org/abs/2502.16982); optimizer comparison for tabular MLPs, [arXiv:2604.15297](https://arxiv.org/abs/2604.15297). Text features: feature hashing, Weinberger et al., [arXiv:0902.2206](https://arxiv.org/abs/0902.2206).
- Automatic differentiation through `think` loops: Deep Equilibrium Models, [arXiv:1909.01377](https://arxiv.org/abs/1909.01377) (implicit differentiation at the fixed point); truncated adjoints are the Neumann-series phantom gradients related to Jacobian-free backpropagation, [arXiv:2103.12803](https://arxiv.org/abs/2103.12803).
- Neural decision forests: Kontschieder et al., *Deep Neural Decision Forests*, ICCV 2015. This is the ranker architecture used by the auto-fixer.
- Learning repair from compiler diagnostics with self-supervised corruption: Yasunaga & Liang, *DrRepair*, [arXiv:2005.10636](https://arxiv.org/abs/2005.10636), and *Break-It-Fix-It*, [arXiv:2106.06600](https://arxiv.org/abs/2106.06600). Repair as classification over diagnostics: SynShine, [arXiv:2104.14671](https://arxiv.org/abs/2104.14671).
- Embedded deployment baselines: TFLite Micro [arXiv:2010.08678](https://arxiv.org/abs/2010.08678), MicroFlow [arXiv:2409.19432](https://arxiv.org/abs/2409.19432).

## Layout

```
src/        compiler: parse.c (surface), lower.c (inline/SSA), ir.c (TGIR read/write/verify),
            ops.c (op table + shape inference), plan.c (memory planner), vm.c, cgen.c, main.c
            fixer.c (repair operators, features, forest training and fix loop)
            autodiff.c (grad: reverse mode, implicit differentiation of think loops), par.c (threads)
            optim.c (train statement, optimizers), data.c (dataset loading, inference, featurization),
            autotrain.c (tgc train / predict / data)
fixer/      forest.tg (trained ranker, a Technograd program), corpus/ (training programs)
runtime/    tg_rt.h: kernels shared by the VM and the generated code
examples/   xor.tg, newton.tg, latent_reasoner.tg, activations.tg + use_import.tg (cross-file),
            delta_memory.tg, drift_calibration.tg (self-updating state), train_xor.tg (train + adamw), sgd_by_hand.tg (the same, via grad + update),
            selective_ssm.tg (scan + backpropagation through time), kv_attention.tg (KV-ring attention),
            coconut/ (latent-reasoning curriculum), ponder/ (learned halting)
tests/      run.sh, positive cases, diagnostic cases, fix/ (auto-fix), imports/ (cross-file), data/ (datasets)
docs/       SPEC.md
```
