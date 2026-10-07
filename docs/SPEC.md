# Technograd language reference (v0.1)

## 1. Model

A program defines one model. Its parts:

- `model NAME`: required, exactly once.
- `param NAME : TYPE = INIT`: constant tensors (weights). They are global and read-only.
- `state NAME : TYPE = INIT`: mutable tensors. They are global, readable like
  params, and change only through `update` (section 15).
- `def NAME(args) -> TYPE:`: pure functions. Each one is inlined at every call site.
- `def forward(...)`: the entry point. Its arguments are the model inputs, in
  order, and its return value is the single output.
- `import "path"`: merge another surface file's `param`s and `def`s into this
  one (section 1.1).

Execution is a static dataflow graph. The only control flow is the `think`
loop. There is no recursion, no data-dependent shape, and no allocation at
run time.

### 1.1 Imports

`import "path"` appears among the top-level declarations. `path` is resolved
relative to the directory of the file containing the `import`, or absolutely
if it begins with `/`.

An imported file is treated as a library: only its `param` declarations and
its `def`s **other than `forward`** are merged in. Its own `model` line and
`forward`, if any, are ignored, so one file can both be run on its own and be
imported elsewhere. Imports are transitive.

Each resolved file is merged at most once, so a file reachable by several
paths (a diamond) contributes its definitions once, and an import cycle
terminates rather than looping. A name defined in two merged files is a
`duplicate param`/`duplicate def` error.

Imports are a surface-file feature resolved before lowering. TGIR is already
flat and has no `import`. After merging, a shape error inside an imported
definition is reported at the imported line but under the root file's name.

## 2. Types

`f32` is a scalar (rank 0). `f32[d1, ..., dk]` is a row-major tensor with
`1 <= k <= 4` and every `di >= 1`. Shapes are checked at compile time.

## 3. Lexical structure

- Indentation (spaces only; tabs are rejected) delimits blocks. Newlines
  inside `(...)` and `[...]` are ignored.
- `#` starts a comment that runs to the end of the line.
- Reserved words: `model param def return think for until f32`.
- Numbers use C `strtod` syntax.

## 4. Parameter initializers

| Form | Meaning |
|------|---------|
| `zeros`, `ones`, `fill(v)` | constant fill |
| `rand(seed, scale)` | uniform in `[-scale, scale)` from xorshift32 seeded with `seed`; deterministic |
| `file("path")` | raw little-endian f32, exactly `4 * numel` bytes; relative paths resolve against the source file's directory |
| `[ ... ]` / `v` | literal values in row-major order; nesting is optional, the count must equal `numel` |

All initializers are evaluated at compile time. The IR always holds the values
themselves.

## 5. Statements

```
NAME = expr                      bind (SSA rename; rebinding allowed; params cannot be assigned)
return expr                      last statement of a def; not allowed inside think
update NAME = expr               set a state for the next run (section 15)
think S for N [until EPS]:       latent loop, see section 7
    block
```

## 6. Expressions

Precedence from low to high: `+ -`, then `* / @`, then unary `-`, then calls
and parentheses. All binary operators are left associative.

| Builtin | Shape rule |
|---------|------------|
| `a + b`, `a - b`, `a * b`, `a / b`, `max(a,b)`, `min(a,b)` | equal shapes; either operand scalar; or one operand's shape equal to the trailing dimensions of the other (row broadcast: `(B,H) + (H)` adds the vector to every row) |
| `-a`, `neg tanh relu sigmoid exp sqrt gelu silu` | elementwise, shape preserved; `gelu` uses the tanh approximation |
| `a @ b` (`matmul`, `dot`) | `(m,k)@(k)->(m)`, `(m,k)@(k,n)->(m,n)`, `(k)@(k,n)->(n)`, `(k)@(k)->()` |
| `softplus(x)`, `log(x)` | elementwise; softplus is overflow-free: `max(x,0) + log1p(exp(-|x|))` |
| `softmax(x)`, `rmsnorm(x)` | over the last axis, shape preserved, rank >= 1; rmsnorm eps = 1e-6, no gain |
| `sum(x)`, `mean(x)` | reduce to scalar |
| `transpose(x)` | rank 2 only |
| `outer(a, b)` | `(m),(n)->(m,n)`, `a b^T`; the rank-1 write of delta-rule and Hebbian updates |
| `step(x)` | elementwise `x > 0 ? 1 : 0`; the derivative of relu, max and min |
| `grad(y, x)` | `y` scalar, `x` a name; returns dy/dx with the shape of `x` (section 16) |
| `mse(p, t)`, `bce(p, t)`, `xent(logits, onehot)` | scalar losses (section 17) |

## 7. `think`: continuous latent loops

```
think h for N until EPS:
    ...
    h = f(h, ...)
```

Semantics:

1. `h` must already be bound. Its current value is the initial state `h0`.
2. Run the body with `h` bound to the current state. At the end of the body,
   the value bound to `h` is the next state.
3. Compute `delta = max_i |next_i - state_i|`, then set `state = next`.
4. Stop after `N` iterations, or, if `until EPS` is given, once `delta <= EPS`.
5. After the loop, `h` refers to the final state. The iteration count is
   stored in steps slot `t`, where `t` numbers think loops in source order
   after inlining.

Scoping rules:

- Names first bound inside the body are local to the body.
- Assigning to any outer name other than `h` is a compile error, because a
  loop carries exactly one latent state.
- Values from outside the loop (inputs, params, earlier bindings) can be read
  inside the body.

Loops can nest. Inner loops can come from inlined functions. The steps slot of
an inner loop holds the count from its last execution.

## 8. TGIR

TGIR is the canonical, machine-facing form. `tgc ir` prints it, and every
`tgc` command accepts it as input (the first non-blank character is `(` or
`;`). Comments start with `;`.

```
(tgir 1
  (model NAME)
  (param NAME TYPE (data v...))           ; zero or more, before block
  (input NAME TYPE)                       ; one or more, in signature order
  (block INSTR...)
  (output VALUE))

TYPE  := (f32 d...)
INSTR := (NAME TYPE (OP VALUE...))        ; OP from the builtin table, by name
       | (NAME TYPE (const NUMBER))
       | (NAME TYPE (think INIT MAX EPS|none INSTR... (yield VALUE)))
```

Rules enforced by the reader:

- Every value name is defined exactly once (SSA), before it is used.
- The declared `TYPE` of every instruction must equal the shape the reader infers.
- Inside a think body, the instruction's own `NAME` is the loop state. After
  the instruction, the same name is the final state.
- Names defined inside a body cannot be seen outside it.
- The shape of `yield` must equal the shape of the state.

Writing the reader's output back with `tgc ir` gives the same text again
(this is tested).

## 9. Compiled unit ABI

`tgc c` writes one C99 translation unit:

```c
#define TG_<m>_IN_<input> <numel>   /* one per input */
#define TG_<m>_NIN, TG_<m>_OUT, TG_<m>_NTHINK, TG_<m>_ARENA_BYTES
void tg_<m>_run(const float *tgi_<input0>, ..., float *tg_out);
extern int tg_<m>_steps[];          /* per think loop, filled by each run */
```

- Not reentrant: there is one static arena per unit.
- Depends only on libm: `expf tanhf sqrtf fabsf`, plus `log1pf logf` when `softplus` or `log` is used.
- `-DTG_MAIN` adds a command-line driver whose output format matches `tgc run`.

## 10. Memory planning

Temporaries share one arena. Each temporary has a lifetime interval over the
linearized program. A value read inside a `think` body but defined outside it
stays live until the loop exits. The loop state and the yield buffer are
always distinct from each other and from the body's temporaries. Placement is
greedy by size (arXiv:2001.03288), aligned to 16 bytes. `tgc plan` prints the
plan.

## 11. Auto-fixer

`tgc fix <file> [-o out]` and `--autofix` repair surface programs that fail
to compile. The loop, its features and its evaluation are described in the
README. These properties are part of the contract:

- The fixer never writes to the source file. `fix` prints the repaired
  source to stdout or to `-o`, and `--autofix` repairs in memory only.
- Every applied edit is reported, with the diagnostic it answers and its
  calibrated confidence.
- An edit is applied only if (a) its trial compilation succeeds, or moves or
  changes the error, (b) it does not revisit a program state the loop has
  already seen, and (c) its confidence is at least the threshold (default
  0.95, environment variable `TG_FIXER_THRESHOLD`). There are at most 8
  rounds.
- If no edit qualifies, the original diagnostic is reported unchanged and the
  exit status is non-zero.
- Genuine semantic errors, such as shape mismatches between existing values,
  produce no candidates.

## 12. Command-line data I/O

File formats are chosen by extension:

| Extension | Format |
|-----------|--------|
| `.bin`, `.f32` | raw little-endian IEEE-754 f32, no header (the same format `file("...")` params read) |
| anything else | text: numbers separated by commas and/or whitespace; `#` starts a comment to end of line |

Commands:

```
tgc run   <file> <in>... [-o out]   each <in> is inline values (1,2,3) or @path
tgc batch <file> <data> [-o out]    one sample per row, or a raw f32 stream
```

- **`@path` inputs.** The file must hold exactly `numel` values for that
  input. In text files the line structure is ignored.
- **`run -o`.** Writes the output instead of printing it. A text file gets one
  row: the output values, then one column per think loop with its step count.
  A binary file gets the output values only.
- **`batch`, text input.** Each non-blank, non-comment row is one sample: all
  inputs concatenated in signature order (`forward(v: f32[3], k: f32)` takes 4
  values per row). Each sample produces one output row in the `run -o` text
  format, written to stdout or `-o`.
- **`batch`, binary input.** The stream is consecutive samples of the same
  concatenated layout, and its length must be a multiple of the per-sample
  count. Binary output is the output tensors back to back.
- **Atomic failure.** The whole input is validated (row widths, numbers,
  stream length) before anything is computed or any output file is created.
  A bad row 1000 produces no output at all, not 999 rows.

## 13. Tracing and logs

```
-v, --verbose   stage trace to stderr
-vv             also every VM value and every think iteration
--log FILE      append JSON Lines events (works with or without -v)
```

Tracing writes only to stderr and the log. Program output on stdout is
identical at every verbosity level (this is tested).

| Stage | Level | Fields |
|-------|-------|--------|
| `import` | debug | `file`, `depth`, `role` (`root`/`library`); one per merged file |
| `parse` | info | `file`, `format` (`surface`/`tgir`), `decls`, `ms` |
| `lower` | info | `model`, `inputs`, `params`, `param_floats`, `values`, `instructions`, `think_loops`, `ms` |
| `plan` | info | `arena_bytes`, `unshared_bytes`, `ms` |
| `run` | info | `model`, `ms` |
| `think` | info | per loop after `run`: `loop`, `steps`, `budget`, `final_delta`, `stop` (`converged`/`budget`); after `batch`: `steps_min`, `steps_mean`, `steps_max` |
| `batch` | info | `rows`, `ms`, `us_per_row` |
| `cgen` | info | `output`, `arena_bytes` |
| `vm` | debug | per instruction: `value`, `op`, `shape`, `min`, `max`, `mean`, `nonfinite` |
| `think` | debug | per iteration: `loop`, `iteration`, `delta` |
| `autofix` | info | `file`, `line`, `diagnostic`, `decision` (`applied`/`abstained`), and for applied: `operator`, `edit_line`, `confidence`, `old`, `new` |
| `error` | error | `file`, `line` (when known); the message is in `msg` |

Every line also has `t` (Unix seconds), `level`, `stage` and `msg`, the
human-readable text.

- **Log level policy.** Error and info events are always logged. Debug
  events are logged only under `-vv`, so a large `batch` does not write
  per-instruction records unless asked.
- **Scope.** The VM traces only the user's model. The auto-fixer's internal
  forest scoring and trial compilations emit no events.
- **Non-finite values.** Under `-vv`, a value containing NaN or infinity is
  flagged `NONFINITE` and counted in `nonfinite`, which locates where a
  model first diverges.

## 14. Threads

```
-j N, --threads N    N worker threads; 0 or `auto` = one per online CPU; default 1
```

The thread pool (`src/par.c`, pthreads and C11 atomics) is shared by two kinds
of work:

- **Batch rows.** Rows are claimed dynamically in small chunks from an atomic
  counter, because adaptive `think` loops make rows cost different amounts.
  Each thread owns its arena and input pointers. Rows are computed in blocks
  of `1024 * threads` and written in input order, so memory stays bounded and
  output order never depends on scheduling.
- **Large matmuls.** A matmul with at least 2 output rows and at least 2^16
  multiply-adds splits its output rows across threads. Each output element is
  computed by the same operations in the same order as the serial kernel.

Guarantees:

- **Bit-identical results at every thread count.** Neither path reorders
  floating-point operations. Tested byte for byte on text and binary batches,
  on a single large run, and on the nested case.
- **No nested oversubscription.** A parallel loop started from inside a
  worker runs inline, so a batch of models with large matmuls uses exactly N
  threads.
- **Race-free.** Workers run an allocation-free VM entry point with no shared
  mutable state. The suite passes under ThreadSanitizer.

Scope:

- **`-vv` value traces** are produced by `run` only. `batch` reports aggregate
  statistics (`rows`, `us_per_row`, `threads`, step min/mean/max).
- **Generated C units stay single-threaded.** They target freestanding
  embedded builds with no thread library.

Measured on 4 cores:

| Workload | 1 thread | 4 threads | Speedup |
|----------|---------:|----------:|--------:|
| `latent_reasoner`, 20,000-row batch, binary I/O | 130 ms | 47 ms | 2.8x |
| `latent_reasoner`, 20,000-row batch, text I/O | 185 ms | 97 ms | 1.9x |
| `tests/par/wide.tg`, one run (1024x1024 matvec in a 17-step think loop, best of 5) | 12.5 ms | 4.1 ms | 3.1x |

The text batch is limited by serial parsing and formatting of numbers.

## 15. Self-updating state

A model can rewrite its own weights while it runs, for test-time training, fast
weights and on-device adaptation.

```python
state mem : f32[4, 4] = zeros            # mutable, starts from its initializer

def forward(k: f32[4], v: f32[4]) -> f32[4]:
    y = mem @ k                          # reads see the start-of-run value
    update mem = mem + outer(v - y, k)   # delta rule: one regression step
    return y
```

### Semantics

- **Declaration.** `state` declares a tensor like `param` does, with the same
  initializers. It can be read anywhere a param can be.
- **When updates apply.** `update NAME = expr` computes the next value. All
  updates of a run are committed after the output is computed, in two phases:
  every source is copied to a staging area, then into its state. Every read
  during a run, including reads inside other update expressions, therefore
  sees the start-of-run value. The graph stays static and single-assignment,
  and the order of updates does not matter.
- **Restrictions**, each a compile error:
  - `update` only in the body of `forward` (not in a `def`, not in a `think`);
  - each state updated at most once per run;
  - the value must have the state's shape;
  - a `param` cannot be updated;
  - a state cannot be assigned with `=`.
- **Persistence across runs:**
  - **VM:** state persists for the process. `batch` carries it from row to row
    and keeps rows in input order for such models; `-j` threads then only
    split large matmuls inside each row. Results are identical at any `-j`.
  - **Generated C:** each state is an exported RAM array
    `float tg_<m>_state_<name>[TG_<m>_STATE_<name>]` that firmware can read,
    save to flash and write back. `void tg_<m>_reset(void)` restores the
    initial values from a `const` copy. The `-DTG_MAIN` driver accepts several
    samples in one process, so adaptation across calls can be checked against
    the VM.
- **TGIR.** `(state NAME TYPE (data ...))` declares a state, and
  `(update NAME VALUE)` forms follow `(output ...)`. The reader re-checks every
  rule above.
- **Tracing.** Under `-v`, `run` reports each committed update as
  `[update] state NAME updated, max |change| ...` (log fields `state`,
  `max_change`).

### Update rules expressible today

These are written directly in the language with existing builtins:

| Rule | `update` expression |
|------|---------------------|
| Delta rule (DeltaNet, arXiv:2406.06484) | `mem + beta * outer(v - mem @ k, k)` |
| Gated delta rule (Gated DeltaNet, arXiv:2412.06464) | `alpha * mem + beta * outer(v - mem @ k, k)` |
| Hebbian / linear-attention write | `mem + outer(v, k)` |
| Exponential moving average (calibration) | `s + rate * (x - s)` |
| TTT-Linear inner step (arXiv:2407.04620), fixed rate | `W - lr * outer(W @ k - v, k)` |

All of these are steps of online regression on a key-value objective, the
unifying view of arXiv:2501.12352. Rules whose step size or gate is computed
from the input, such as Titans' surprise-based momentum and forgetting
(arXiv:2501.00663), can be written the same way: compute the gate as a value,
then use it in the `update` expression. A second `state` holds momentum.

## 16. Automatic differentiation

`grad(y, x)` returns the gradient of the scalar `y` with respect to `x`, which
must be a name: an input, a param, a state, or a local variable. The result
has the shape of `x`.

```python
update w = w - lr * grad(loss, w)          # one SGD step, inside the model
```

### How it works

Reverse mode, as a source-to-source transform during lowering:

1. The first `grad(y, \.)` in a block runs one full backward pass over the
   instructions that reach `y` (collected from the lowering tape). It computes
   the adjoint of every value on the way and caches the result.
2. Every later `grad` of the same `y` in that block reuses the cached pass, so
   `n` gradients of one loss cost one backward pass. The XOR trainer drops
   from 61 to 31 instructions.
3. Dead-code elimination then removes the adjoints nobody uses. It runs on
   every program; its roots are the output and the update sources.
4. TGIR is written canonically: temporaries are numbered in the order the
   reader recreates them. Write, read, write is therefore a fixed point for
   any module, including those with compiler-generated values.

The gradient becomes ordinary instructions, so TGIR has no `grad` form. The
planner, the VM, threading and the C backend are unchanged, and a training
model compiles to freestanding C like any other.

**Coverage.** Every builtin has a derivative:

- broadcasting operands are summed back to their shape: a scalar by `sum`,
  a row-broadcast vector over the rows of a matrix by `ones(B) @ g`;
- `max`/`min`/`relu` send the gradient through the selected operand, with ties
  going to the second;
- `softmax` and `rmsnorm` are supported on rank 1 and 2;
- all four `matmul` shape cases and `outer` are covered.

### Through `think` loops

Loops with `until` are differentiated implicitly, as in Deep Equilibrium
Models (arXiv:1909.01377). At the converged fixed point h* = f(h*, theta):

```
adjoint   u = g + (df/dh)^T u      solved by a generated think loop (same budget and threshold)
gradient  theta_bar = (df/dtheta)^T u
```

- **Memory** is constant in the number of iterations: no per-iteration
  activations are kept.
- **Truncation.** If the adjoint loop stops at its budget, the result is the
  truncated Neumann series, the "phantom gradient" approximation.
- **Initial state.** At a fixed point the result does not depend on the
  loop's initial state, so the init receives no gradient.
- **Extra steps lines.** The generated adjoint loop is an ordinary think
  loop, so it appears in `steps` output.
- **Not yet supported:** fixed-budget loops (no `until`) and nested think
  loops. Both are compile errors.

### Training and deployment

A model whose weights are `state` and which applies
`update w = w - lr * grad(loss, w)` takes one SGD step per run. So
`tgc batch` over a dataset is online training (`examples/sgd_by_hand.tg` learns
XOR in 6000 steps; `examples/train_xor.tg` is the same model using `train`).

`--save-state PREFIX` writes each state to `PREFIX.<name>.bin` as raw
little-endian f32, the format `param w : ... = file("PREFIX.w.bin")` loads. A
frozen inference model can then be compiled for deployment. The training
model itself also compiles to C (about 11.5 KB for the XOR MLP, using only
`expf` and `tanhf`), and trains on the device bit-identically to the VM.

### Verification

`tests/gradcheck.pl` compares every derivative rule, including both
implicit-differentiation cases, against central finite differences (22 cases,
worst relative error 2.6e-4). A planted bug in one rule makes 9 cases fail.

## 17. Training statement and optimizers

```python
train LOSS [with OPTIMIZER[(option=value, ...)]] [over w1, w2, ...]
```

`train` turns a scalar loss into one optimizer step per run. It has the same
placement rules as `update`: only in the body of `forward`, and not inside
`think`.

- **Weights.** Without `over`, the weights are every `state` the loss depends
  on that no explicit `update` already changes. The shared backward pass
  answers which ones those are.
- **Expansion.** Gradients come from one shared backward pass (section 16).
  Optimizer moments and step counters become hidden states named `__opt_*`.
  The step itself becomes ordinary `update`s, so a training model compiles to
  TGIR, the VM and C like any other.
- **Saved state.** `--save-state` writes the weights and skips `__opt_*`.

| Optimizer | Step | Defaults |
|-----------|------|----------|
| `sgd` | `w -= lr g` | lr 0.01 |
| `momentum` | `m = mu m + g; w -= lr m` | lr 0.01, momentum 0.9 |
| `adam` | Adam with bias correction | lr 0.001, beta1 0.9, beta2 0.999, eps 1e-8 |
| `adamw` (default) | Adam + decoupled weight decay | as adam, wd 0.01 |
| `muon` | rank-2 weights: Nesterov momentum orthogonalized by 5 quintic Newton–Schulz steps, scaled by 0.2·sqrt(max(rows, cols)) to share AdamW's learning rate (arXiv:2502.16982); other weights: AdamW | lr 0.001, momentum 0.95, wd 0.01 |

`clip=c` (any optimizer) rescales all gradients so that their global L2 norm
is at most `c`. An unknown optimizer or option is a compile error that lists
the valid ones.

Loss builtins expand to existing operations, so they differentiate like any
other expression:

| Builtin | Definition |
|---------|-----------|
| `mse(p, t)` | `mean((p - t)^2)` |
| `bce(p, t)` | `-mean(t log(p + 1e-7) + (1 - t) log(1 - p + 1e-7))` |
| `xent(logits, onehot)` | `-sum(onehot * log(softmax(logits) + 1e-12))`; for a batch of logits `(B, C)` the sum is divided by `B` (mean over rows) |

### Choosing a default (measured)

A step's cost is paid once per mini-batch: a model whose input is a batch
`(B, D)` takes one optimizer step per `B` rows (section 18).

- **Muon's cost.** Its Newton–Schulz orthogonalization costs about 40 M FLOPs
  per step on a 32×4096 layer. With one step per sample this was 112 ms per
  sample against AdamW's 4.5 ms (25×); with batch 32 the gap is 24 s against
  5 s for a full SST-2 run.
- **Quality.** On XOR all five optimizers converge (worst error < 0.01 over the
  last 400 samples). With batch 32, lr 0.01, 3 seeds:

  | Task | AdamW | Muon |
  |------|-------|------|
  | iris (accuracy) | 96.67% | similar |
  | regression (RMSE) | 0.48 | similar |
  | 3-class blobs | 100% | 100% |
  | SST-2 (accuracy, time) | 69.73%, 5 s | 69.94%, 24 s |

- **Decision.** AdamW at lr 0.01 is the default. Muon is fully supported and
  marginally more accurate on text at about 5× the time.

## 18. Datasets and one-command training

```
tgc train SOURCE [-o DIR] [--target COL] [--epochs N] [--hidden H] [--lr X]
                 [--batch B] [--text-dim N] [--optimizer NAME] [--val FRACTION] [--max-rows N] [--seed S]
tgc predict DIR SOURCE [-o OUT.csv]
tgc data inspect SOURCE [--target COL]
tgc data prep SOURCE -o OUT.csv [--target COL]
```

### Sources

The format is detected from the extension, falling back to the content.

| SOURCE | Read as |
|--------|---------|
| `*.csv`, `*.tsv`, other text | delimited text: delimiter sniffed among `,` tab `;` `\|`; header detected; RFC 4180 quoting; CRLF |
| `*.jsonl`, `*.ndjson` | one JSON object per line |
| `*.json` | an array of objects or arrays, or an object holding one (`rows`, `data`, `records`, ...); Hugging Face API pages are recognized |
| `*.npy` | NumPy arrays: f4/f8, signed and unsigned integers, 1-D or 2-D, C order |
| `hf:OWNER/NAME[/CONFIG[/SPLIT]]` | the Hugging Face datasets-server rows API |

Notes on `hf:` sources:

- **Fetching** uses `curl`, spawned directly with no shell, with retries for
  rate limits and server errors.
- **Defaults:** the first config, and the `train` split when one exists.
- **Paging and size:** the dataset is fetched in pages of 100 rows, up to
  `--max-rows` (default 20000).
- **Caching:** pages are cached under `$XDG_CACHE_HOME/technograd` (or
  `~/.cache/technograd`). The cache file is renamed into place only after a
  complete fetch, so an interrupted download is never mistaken for a dataset.
- **Gated datasets:** `HF_TOKEN` is sent when set.

### Inference

Each column is classified:

| Kind | When | Features |
|------|------|----------|
| numeric | every value parses as a number | standardized; plus a 0/1 missing-value indicator if the column has missing values |
| categorical | strings with few distinct values (≤ 64, or ≤ 5% of rows and short) | one-hot over the 32 most frequent values |
| text | longer strings with many distinct values | 1024 signed hashed word-unigram buckets (feature hashing; `--text-dim N`), L2-normalized |
| vector | fixed-length numeric lists | each component standardized |
| dropped | identifiers (`id`, `*_id`, ...; or unique increasing integers), nested objects (images, audio), lists of varying length, empty columns | none, and the reason is printed |

Two further rules:

- **Leak guard.** A low-cardinality column whose values determine the target
  exactly (for example `label_text` next to `label`) is dropped as
  *"determines the target exactly (would leak the answer)"*.
- **Target choice.** Without `--target`, a column named `label`, `target`,
  `class`, `y`, `species`, `category`, ... is preferred; otherwise the last
  usable column is taken. A categorical target, or one with few integer
  values, gives classification (softmax, `xent`). Anything else gives
  regression on the standardized target (`mse`), reported in original units.

### Training

1. **Split and fit.** A seeded split (default 20% validation) is made, and the
   featurization (means, standard deviations, vocabularies) is fitted on the
   training rows only.
2. **Model.** The generated model is `D -> H (gelu) -> C` over a batch:
   `h = gelu(x @ w1 + b1); return h @ w2 + b2` with `x : f32[B, D]`, weights
   stored `(in, out)` and Glorot-uniform initial weights. `H` is 16 for at
   most 8 inputs, 32 for 9 to 64 inputs, 64 for 65 to 512 inputs, and 32
   above that, since wide sparse inputs overfit a wide layer. The model trains with
   `train ... with adamw(lr=0.01)`. `B` is 32 by default (`--batch`), capped
   at the number of training rows.
3. **Epochs and stopping.**
   - Training runs `clamp(ceil(3000 B / training rows), 20, 500)` epochs, a
     budget of about 3000 optimizer steps. Rows are shuffled each epoch and
     packed into batches of `B`; the last partial batch is dropped.
   - After each epoch, validation runs on a generated single-row,
     forward-only model with no `train` statement, so it carries no backward
     pass; its weights point at the training states in place.
   - The weights with the best validation loss are kept. Training stops
     early after `max(10, epochs/5)` epochs without improvement.
4. **Artifacts.** `DIR` (default `<name>_model`) receives:

   | File | Contents |
   |------|----------|
   | `model.tg` | the training model, readable and editable |
   | `infer.tg` | the frozen inference model, loading `weights.*.bin` via `file()` |
   | `infer.c` | the inference model compiled to a freestanding C unit |
   | `features.tgf` | the featurization, as plain text |
   | `report.txt` | the run summary |

`tgc predict DIR SOURCE` applies `features.tgf` to new raw data, matching
columns by name in any order. A missing column counts as missing values. It
prints `prediction,confidence` (classification) or `prediction` (regression,
in original units), and reports accuracy or RMSE when the target column is
present.

### Measured

| Dataset (no configuration given) | Result |
|----------------------------------|--------|
| `hf:scikit-learn/iris` (150 rows) | 96.67% validation accuracy (mean of 3 seeds); `Id` dropped as an identifier |
| `hf:SetFit/sst2` (6920 sentences) | 69.73% validation accuracy in 5 s (36 s with one step per sample); `label_text` caught as a leak |
| SST-2 with `--text-dim 4096`, one step per sample | 72.4% (465 s); a linear bag-of-words model on the same features reaches 75.7% |
| synthetic `y = 3 x1 - 2 x2 + [red] 1.5 + noise`, 3% missing `x2` | predictions 2.99 and -0.52 for true 3.0 and -0.5 |
| synthetic 3-class blobs | 100% (regression-tested) |

On text, unigrams outperformed unigrams+bigrams at every bucket count tried
(256 to 4096). With a few thousand sentences, bigram collisions add more
noise than signal.

## 19. Roadmap

Ordered by importance for latent-reasoning models on embedded targets:

1. int8/int4 weights with per-channel scales, and fixed-point kernels for
   targets without an FPU.
2. `embed(table, token)` gather and causal attention over a fixed-size KV
   ring. These are needed for token-level Coconut models, where continuous
   thoughts are mixed with token embeddings.
3. Autodiff through fixed-budget and nested think loops: unrolled
   backpropagation through time with a statically planned activation store.
4. Learned halting heads (`until` driven by a predicate value) in addition to
   convergence halting.
5. Sparse tensors for hashed text features (about 30 active of 1024+ buckets
   per row), making wide text models proportionally cheaper.
6. Native backends from the planned IR: ARMv7-M/ARMv8-M assembly with
   CMSIS-NN style kernels, and RISC-V with the vector extension.
