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
make test       # 82 checks: known answers, IR round trip, VM == compiled C, diagnostics, auto-fix, imports, file I/O
make fixer      # retrain the auto-fix forest (deterministic, ~15 s) and evaluate it on held-out programs

build/tgc run  examples/latent_reasoner.tg 1,0,0,1,0,1,1,0
# 0.225605994 0.24476327 0.162138939 0.367491812
# steps 0 9                 <- the latent loop halted after 9 of 32 allowed steps

build/tgc ir   examples/newton.tg            # canonical TGIR
build/tgc batch examples/xor.tg tests/io/xor.csv   # one sample per row; .bin/.f32 for raw f32 streams
build/tgc plan examples/latent_reasoner.tg   # arena layout: 320 B instead of 624 B unshared
build/tgc c    examples/latent_reasoner.tg -o reasoner.c
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
  tensor algebra (`+ - * / @`), scalar broadcast, and `think` loops.
- **Black box by construction.** A compiled unit exposes only
  `tg_<model>_run(inputs..., out)` and size macros. Weights are static
  `const` data. Memory use is fixed when the unit is compiled
  (`TG_<model>_ARENA_BYTES`), and nothing is allocated at run time.
- **Implementation language.** The design took Lisp, Perl, and assembly into
  account. The Lisp idea lives on as the IR: the machine-facing language is
  S-expressions. The toolchain is C11 because it is the lowest portable layer
  above assembly, has no runtime, and can host the compiler on the same
  class of targets it compiles for. Perl is used only at build time, to embed
  the kernel library into emitted units. An assembly backend fits naturally
  as a later code generator from the same planned IR (see roadmap).
- **One source of arithmetic.** The VM and the generated code call the same
  kernels (`runtime/tg_rt.h`). The test suite checks that their outputs match
  byte for byte.

## Research basis

- Coconut, continuous latent reasoning: Hao et al., *Training Large Language Models to Reason in a Continuous Latent Space*, [arXiv:2412.06769](https://arxiv.org/abs/2412.06769). This is the source of `think`: the hidden state is the next input.
- Recurrent depth with input re-injection: Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*, [arXiv:2502.05171](https://arxiv.org/abs/2502.05171). This is the pattern used in `examples/latent_reasoner.tg`.
- Convergence-based adaptive halting: AdaAnchor, [arXiv:2603.15051](https://arxiv.org/abs/2603.15051). It reports 48-60% fewer latent steps at equal budget. This is the basis for `until eps`.
- Survey: *A Survey on Latent Reasoning*, [arXiv:2507.06203](https://arxiv.org/abs/2507.06203).
- Static arena planning, greedy by size: Pisarchyk & Lee, [arXiv:2001.03288](https://arxiv.org/abs/2001.03288).
- Neural decision forests: Kontschieder et al., *Deep Neural Decision Forests*, ICCV 2015. This is the ranker architecture used by the auto-fixer.
- Learning repair from compiler diagnostics with self-supervised corruption: Yasunaga & Liang, *DrRepair*, [arXiv:2005.10636](https://arxiv.org/abs/2005.10636), and *Break-It-Fix-It*, [arXiv:2106.06600](https://arxiv.org/abs/2106.06600). Repair as classification over diagnostics: SynShine, [arXiv:2104.14671](https://arxiv.org/abs/2104.14671).
- Embedded deployment baselines: TFLite Micro [arXiv:2010.08678](https://arxiv.org/abs/2010.08678), MicroFlow [arXiv:2409.19432](https://arxiv.org/abs/2409.19432).

## Layout

```
src/        compiler: parse.c (surface), lower.c (inline/SSA), ir.c (TGIR read/write/verify),
            ops.c (op table + shape inference), plan.c (memory planner), vm.c, cgen.c, main.c
            fixer.c (repair operators, features, forest training and fix loop)
fixer/      forest.tg (trained ranker, a Technograd program), corpus/ (training programs)
runtime/    tg_rt.h: kernels shared by the VM and the generated code
examples/   xor.tg, newton.tg, latent_reasoner.tg, activations.tg + use_import.tg (cross-file)
tests/      run.sh, positive cases, diagnostic cases, fix/ (auto-fix), imports/ (cross-file)
docs/       SPEC.md
```
