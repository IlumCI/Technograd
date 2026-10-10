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
| `env("NAME")`, `env("NAME", v)` | an environment variable at compile time: `numel` numbers, or one number to fill; without it, the default `v` (or a compile error) |
| `[ ... ]` / `v` | literal values in row-major order; nesting is optional, the count must equal `numel` |

All initializers are evaluated at compile time. The IR always holds the values
themselves.

## 5. Statements

```
NAME = expr                      bind (SSA rename; rebinding allowed; params cannot be assigned)
return expr                      last statement of a def; not allowed inside think
update NAME = expr               set a state for the next run (section 15)
update NAME[rows] = expr         set only rows `rows` (n) of a state (D, ...) to expr (n, ...)
think S for N [until EPS]:       latent loop, see section 7
    block
scan C, ... over X in E, ...:    loop over the rows of sequences, see section 19
    block
emit N = expr                    in a scan body: stack expr over the steps as N
```

## 6. Expressions

Precedence from low to high: `+ -`, then `* / @`, then unary `-`, then calls
and parentheses. All binary operators are left associative.

| Builtin | Shape rule |
|---------|------------|
| `a + b`, `a - b`, `a * b`, `a / b`, `max(a,b)`, `min(a,b)` | NumPy broadcasting: dimensions are compared from the right, each pair equal or one of them 1, missing leading dimensions count as 1. `(B,H) + (H)` adds a vector to every row, `(G,1) * (G,K)` scales each row, `(N,1) * (1,P)` is an outer product |
| `-a`, `neg tanh relu sigmoid exp sqrt gelu silu` | elementwise, shape preserved; `gelu` uses the tanh approximation |
| `a @ b` (`matmul`, `dot`) | `(m,k)@(k)->(m)`, `(m,k)@(k,n)->(m,n)`, `(k)@(k,n)->(n)`, `(k)@(k)->()`; batched `(g,m,k)@(g,k,n)->(g,m,n)`; shared right operand `(g,m,k)@(k,n)->(g,m,n)`, `(g,m,k)@(k)->(g,m)` |
| `softplus(x)`, `log(x)` | elementwise; softplus is overflow-free: `max(x,0) + log1p(exp(-|x|))` |
| `sin(x)`, `cos(x)`, `floor(x)` | elementwise; `floor` has zero derivative |
| `rope(x, pos)` | rotary position embedding (arXiv:2104.09864): x (d) at a scalar position, or the rows of x (T, d) at positions (T); channel pair (2i, 2i+1) rotates by pos * 10000^(-2i/d); d even |
| `attention(Q, K, V, W)` | causal attention of the rows of Q (T, d) over K (T, d) and V (T, e) within a window of W rows (row t sees t-W+1 .. t): `softmax(Q K^T / sqrt(d) + M) V`, M a constant 0 / -1e9 mask; W a literal, W >= T is full causal attention (section 20) |
| `reshape(x, d0, d1, ...)` | the same elements in row-major order with new dimensions (integer literals, same element count); TGIR `(reshape X)` with the target as the declared type |
| `softmax(x)`, `rmsnorm(x)` | over the last axis, shape preserved, rank >= 1; rmsnorm eps = 1e-6, no gain |
| `sum(x)`, `mean(x)` | reduce to scalar |
| `transpose(x)` | rank 2, or rank 3 with the last two axes swapped |
| `sum_to(x, like)` | `x` summed over the axes along which `like` broadcasts to it; shape of `like` (the adjoint of broadcasting) |
| `outer(a, b)` | `(m),(n)->(m,n)`, `a b^T`; the rank-1 write of delta-rule and Hebbian updates |
| `step(x)` | elementwise `x > 0 ? 1 : 0`; the derivative of relu, max and min |
| `spmm(s, w)` | sparse rows times a matrix: `s` is `(K, 2)` or `(B, K, 2)`, each row K `(index, value)` pairs (ELLPACK form), `w` is `(D, H)`; returns `(H)` or `(B, H)`, `sum_k value_k * w[index_k]`. Cost O(B K H), independent of D. Indices round to the nearest integer; padding has value 0; an index outside `[0, D)` is skipped, so no input can address out of bounds. With one pair `(token, 1)` per row it is an embedding gather |
| `active(s, w)` | `(n)`, n = B K: the distinct rows of `w` that sparse rows `s` touch with a nonzero value, ascending, padded with -1. Not differentiable (indices) |
| `take(w, r)` | `(D, ...) , (n) -> (n, ...)`: rows `r` of `w`; zero for -1 or out-of-range indices; repeats allowed |
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

### Learned halting

```
think h for N halt p [at T]:
    h = f(h, ...)
    p = sigmoid(...)         # a scalar of the body: this step's halting probability
```

Instead of convergence, a learned head decides when to stop. After step n
the loop has halted with probability `1 - prod_{i<=n} (1 - p_i)`; it stops
at the first step where that exceeds T (default 0.5: the median of the
halting distribution, PonderNet's deterministic inference rule), or at N.
`p` is clamped to [0, 1]. VM, TGIR (`(halt V T)` after `(yield ...)`), C
and fixed-point C implement it identically.

Halting is a discrete decision, so `grad` through such a loop is a compile
error. Training uses PonderNet's objective (Banino et al., arXiv:2107.05407)
written with `scan` over the whole budget: at every step n a prediction and
`lam_n`, the halting distribution `p_n = lam_n prod_{i<n} (1 - lam_i)`
(normalized over the budget), and the loss `sum_n p_n L_n + beta KL(p ||
geometric prior)`. The trained weights then run in a `think ... halt` model
(`examples/ponder/`).

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
INSTR := (NAME TYPE (OP VALUE...))        ; OP from the builtin table, by name; 1 to 3 operands
       | (NAME TYPE (const NUMBER))
       | (NAME TYPE (think INIT MAX EPS|none INSTR... (yield VALUE) [(halt VALUE T)]))
       | (scan T forward|reverse [steps] (carry C TYPE INIT)... (in X TYPE SEQ)...
               (body INSTR...) (next C VALUE)... (emit Y TYPE VALUE)...)   ; section 19; `steps`: a fixed-budget think loop turned into a scan by `grad` reports T in the next steps slot
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

Two operations appear only in TGIR, as derivatives of `spmm` written by
autodiff: `(spmm_t S G W)`, the scatter-add `S^T G` with the shape of `W`
(gradient for the matrix), and `(spmm_dx S W G)`, with the shape of `S`
(0 for each index, `W[index] . G[row]` for each value). Likewise
`(take_t R G W)` scatter-adds the rows of `G` into the shape of `W` (the
derivative of `take`), and the optimizer writes `(spmm_tc S G R)`, the
gradient of an `spmm` table restricted to the row set `R`, as `(n, H)`.
A row update is `(update NAME VALUE ROWS)`.

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
tgc batch <file> <data> [-o out]    one sample per row, or a raw f32 stream; <data> - is stdin
tgc stream <file> [src] [-o dst]    one row in, one row out, flushed per row
tgc serve <file> --listen [HOST:]PORT
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

### Streams, sockets and signals

The model stays a pure function with a static memory plan; `stream` and
`serve` connect it to the operating system and the network.

- **`stream`.** Reads rows one at a time and writes each output row as soon
  as it is computed (flushed), so it works in pipelines and on endless
  inputs. `src` and `dst` are `-` (stdin/stdout, the defaults), a file, or
  `tcp://HOST:PORT`, a connection tgc opens. With a `tcp://` source and no
  `-o`, outputs go back over the same connection. A malformed row produces
  `error: ...` on the output and the stream continues.
- **`serve`.** A TCP server; HOST defaults to 127.0.0.1 (give 0.0.0.0 to
  accept other machines). Each connection uses one of two protocols, chosen
  by its first line:
  - the row protocol: send rows, receive one output row each, any number per
    connection;
  - HTTP/1.1: `GET /health` (`ok`), `GET /` (the signature as JSON: inputs
    and types, output type, values per row, think loops, whether the model
    is stateful), `POST /run` (body rows in, output rows out, text/plain).

  Stateless models serve connections in parallel, one thread and private
  arena each. A model with `update`s (or running with `-j` > 1) runs one row
  at a time in arrival order, so its state evolves exactly as in `batch`
  (tested: HTTP and row-protocol requests against `batch` on the same rows,
  outputs and saved state byte-identical).
- **Signals.** SIGINT and SIGTERM stop `batch` (after the current block),
  `stream` and `serve` (after the current row) cleanly: outputs are flushed,
  `--save-state` is written, and the exit status is 128 + the signal number
  (130, 143). SIGPIPE is ignored, so a disconnecting peer ends only its own
  connection.
- **Environment.** `env("NAME")` initializers read configuration from the
  environment when a model is compiled (section 4).
- **Datasets over HTTP.** `tgc train` and `tgc data` accept `http://` and
  `https://` URLs of CSV/TSV/JSON/JSON Lines files (fetched with curl;
  `HF_TOKEN` is sent only to Hugging Face).

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
- **Row updates.** `update NAME[rows] = expr` writes only the listed rows of
  a state: `rows` is `(n)`, `expr` is `(n, ...)` with the state's trailing
  dimensions. Indices round to the nearest integer; -1 and out-of-range
  indices are skipped; a repeated index takes the later row. Staging holds
  the value and the indices, so the commit costs O(n), not O(state). Under
  `-v` the trace reads `state NAME: k of D rows updated`.
- **When updates apply.** `update NAME = expr` computes the next value. All
  updates of a run are committed after the output is computed, in two phases:
  every source is copied to a staging area, then into its state. Every read
  during a run, including reads inside other update expressions, therefore
  sees the start-of-run value. The graph stays static and single-assignment,
  and the order of updates does not matter.
- **Restrictions**, each a compile error:
  - `update` only in the body of `forward` (not in a `def`, not in a `think`);
  - each state updated at most once per run;
  - the value must have the state's shape (a row update: rows `(n)` and a
    value `(n, ...)` matching the state's trailing dimensions);
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
  a row-broadcast vector over the rows of a matrix by `ones(B) @ g`, any
  other broadcast by `sum_to`;
- `max`/`min`/`relu` send the gradient through the selected operand, with ties
  going to the second;
- `softmax` and `rmsnorm` are supported on rank 1 and 2;
- every `matmul` shape case (batched and shared right operand included),
  `transpose` of rank 2 and 3, and `outer` are covered;
- `spmm` sends gradients to the matrix (a scatter-add over the active rows)
  and to the values of the sparse rows; indices are piecewise constant and
  get zero. Its two derivative operations are themselves differentiable, so
  second-order gradients through `spmm` work;
- `take` sends its gradient to the table as a scatter-add (`take_t`), whose
  own derivative is `take`; `active` returns indices and has none;
- `scan` loops are differentiated by backpropagation through time
  (section 19).

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
- **Fixed-budget loops** (no `until`) have no fixed point. `grad` turns
  such a loop into a scan of its budget and differentiates it through time
  (section 19); the loop still reports its step count.
- **Nested loops** (think or scan, in any order and depth) inside a body
  are cloned whole and differentiated by their own rule.

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
implicit-differentiation cases and both second-order `spmm` cases, against
central finite differences (46 cases, worst relative error 4.3e-4). A planted bug in one rule makes 9 cases fail.

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

### Row-sparse steps for `spmm` tables

A weight used only as the matrix of `spmm` has a gradient that is zero outside
the rows the batch touched. `train` detects this from the shared backward pass
and steps those rows alone (lazy updates, as for sparse embeddings):

1. `r = active(s, w)`, the touched rows; `gc = spmm_tc(s, g, r)`, the compact
   gradient, `(n, H)`;
2. the step runs on `take(w, r)` and `take` of the moments;
3. `update w[r] = ...` and `update m[r] = ...` write the rows back.

Per-step cost is O(n H) with n = B K, instead of O(D H). The dense gradient is
removed by dead-code elimination. Clipping uses the compact gradient, which
has the same norm.

**Catch-up.** Each table row records the step it was last updated
(`__opt_last_<w>`). When a row returns after k skipped steps, the k steps the
dense optimizer would have taken with a zero gradient are applied first, in
closed form:

| Optimizer | Catch-up for k skipped steps |
|-----------|------------------------------|
| `sgd` | none needed: the lazy step equals the dense step exactly |
| `momentum` | `w -= lr m mu (1 - mu^k) / (1 - mu)`; `m *= mu^k` (exact) |
| `adam`, `adamw` | `w -= lr m^/sqrt(v^) q (1 - q^k) / (1 - q)` with `q = beta1 / sqrt(beta2)`, holding the bias corrections at the current step and neglecting eps; `m *= beta1^k`; `v *= beta2^k`; `w *= (1 - lr wd)^k` |
| `muon` | tables take the AdamW step: Muon orthogonalizes whole matrices and leaves embeddings to AdamW |

A row's trajectory thus follows the dense one, deferred: the forward pass sees
a row as of its last update. When every row is touched at every step the lazy
and dense steps are identical; both properties are tested. `lazy=0` forces
dense steps.

Measured on SST-2 (8192 buckets, AdamW lr 0.01, 3 seeds, validation
accuracy):

| Step | Accuracy | Time per run |
|------|----------|--------------|
| dense | 76.16, 75.07, 74.86 (mean 75.36) | 11.0 s |
| lazy, no catch-up | 73.41, 74.28, 74.13 (mean 73.94) | 4.0 s |
| lazy, decay catch-up only | 74.86, 72.40, 74.86 (mean 74.04) | 4.6 s |
| lazy, full catch-up (default) | 74.93, 75.29, 75.87 (mean 75.36) | 5.3 s |

Most of the gap without catch-up comes from the drift a dense optimizer keeps
applying to a row on its stale momentum, not from weight decay.

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
  | SST-2, 1024 buckets, dense input (accuracy, time) | 69.73%, 5 s | 69.94%, 24 s |

- **Decision.** AdamW at lr 0.01 is the default. Muon is fully supported and
  marginally more accurate on text at about 5× the time.

## 18. Datasets and one-command training

```
tgc train SOURCE [-o DIR] [--target COL] [--epochs N] [--hidden H] [--lr X]
                 [--batch B] [--text-dim N] [--model bow|ssm] [--optimizer NAME] [--val FRACTION]
                 [--max-rows N] [--seed S]
tgc predict DIR SOURCE [-o OUT.csv]
tgc data inspect SOURCE [--target COL]
tgc data prep SOURCE -o OUT.csv [--target COL]
```

### Sources

The format is detected from the extension, falling back to the content.

| SOURCE | Read as |
|--------|---------|
| `*.csv`, `*.tsv`, other text | delimited text: delimiter sniffed among `,` tab `;` `\|`; header detected (a non-numeric field above a numeric one, or, for all-string columns, short name-like fields that never recur below); RFC 4180 quoting; CRLF |
| `*.jsonl`, `*.ndjson` | one JSON object per line |
| `*.json` | an array of objects or arrays, or an object holding one (`rows`, `data`, `records`, ...); Hugging Face API pages are recognized |
| `*.npy` | NumPy arrays: f4/f8, signed and unsigned integers, 1-D or 2-D, C order |
| `hf:OWNER/NAME[/CONFIG[/SPLIT]]` | the Hugging Face datasets-server rows API |
| `http://...`, `https://...` | a text-format file (the format from the URL path, then the content) |

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
| text | longer strings with many distinct values | 32768 signed hashed word-unigram buckets (feature hashing; `--text-dim N`), L2-normalized; enters the model as sparse rows |
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
   `h = gelu(x @ w1 + spmm(s, wt) + b1); return h @ w2 + b2`, weights stored
   `(in, out)`, Glorot-uniform initial weights.
   - Dense features (numeric, categorical, vector) form `x : f32[B, Dd]`.
   - Text features form `s : f32[B, K, 2]`: each row's nonzero hashed
     buckets as `(bucket, weight)` pairs, `K` the most any training row
     has. The first layer then costs O(K H) per row instead of O(D H). The
     inference model allows `max(2K, 64)` pairs (at most the bucket count);
     a longer row at `predict` keeps its largest weights and is reported.
   - Either part is left out when there are no such features.
   ` `H` is 16 for at
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

### Sequence model (`--model ssm`)

`--model ssm` adds a stack of Mamba-3 blocks (arXiv:2603.15569) over the
words of the first text column, in reading order, next to the bag of words:

```python
h1 = gelu(x @ w1 + spmm(s, wt) + rmsnorm(ssm(tok)) @ wp + b1)
```

- **Input.** `tok : f32[T B, 1, 2]` holds one `(word bucket, 1)` pair per
  position, time-major, `(0, 0)` for padding. `T` is the longest training
  row (at most 64 words); `predict` reads the first `T` words of longer
  rows and says how many were cut.
- **Embedding.** `spmm(tok, E)`, `E` of width D = 32 over the same 32768
  hashed buckets; `E` takes row-sparse optimizer steps (section 17).
- **Blocks** (`--layers`, default 2), each a Mamba-3 mixer and a SwiGLU MLP
  with pre-norm residuals, as in the paper's Llama-style stack. The mixer has
  4 heads of P = 16 channels (inner width 2D) and a state of N = 16 by P per
  head. With u = rmsnorm(e) g1:

  ```
  x = u Wx    dt = softplus(u Wdt + bdt)    lam = sigmoid(u Wl + bl)    al = exp(-dt softplus(la))
  B, C = rmsnorm(u WB), rmsnorm(u WC) (BCNorm, gains gB, gC) + per-head biases bB, bC
  phi_t = sum_{s<=t} dt_s (u_s Wth)          (cumulative rotation angles, one per channel pair)
  B~, C~ = B, C rotated by phi               (the complex state of Mamba-3, as data-dependent RoPE)
  X = x scaled per rank (Wxr)                (MIMO rank R, --mimo, default 4)
  h_t = al_t h_{t-1} + (1 - lam_t) dt_t al_t B~_{t-1} X_{t-1}^T + lam_t dt_t B~_t X_t^T
  y_t = C~_t^T h_t, mixed over ranks (Wyr), + dk x
  e  += (y silu(u Wz)) Wo;     e += SwiGLU(rmsnorm(e) g2)
  ```

  The cumulative sum is a lower-triangular matmul over time; every product
  B~ X^T is formed for all positions at once by a batched matmul
  (section 6), so the scan only carries the state and the previous product.
  Padding sets `dt = 0`, which leaves the state and the angles unchanged.
  The output is the mean of the final-normed `e` over the real words.
  `bdt` starts with step sizes log-spaced in [0.001, 0.1] and `la` with
  decay rates in [1, 16] per head, as in Mamba-2; output projections are
  scaled by 1/sqrt(2 L) (GPT-2 style residual scaling).
- **Training** is backpropagation through time (section 19) inside the same
  `train ... with adamw` step; `infer.tg` and `infer.c` run the stack over
  one row.

### Pretraining (`tgc pretrain`)

```
tgc pretrain SOURCE [-o DIR] [--column COL] [--epochs N] [--steps N] [--batch N] [--lr X]
                    [--layers N] [--mimo R] [--max-rows N] [--seed S]
tgc train LABELED --model ssm --init DIR
```

`pretrain` trains the embedding table and the blocks by next-token
prediction on unlabeled text. The rows of the text column are packed into
one token stream (hashed exactly as `train` hashes text) and cut into
windows of 64 + 1 tokens. The generated model (`DIR/lm.tg`, plain
Technograd) ends in:

```python
h = states(tok)                                   # the stack, per position
r = active(nxt, E)                                # distinct target buckets of the batch
L = 15 * tanh((h @ transpose(take(E, r))) / 15) - step(0 - r) * 10000
pos = 15 * tanh(((h * spmm(nxt, E)) @ onesD) / 15)
loss = sum((log(exp(L) @ onesN) - pos) * mk) / sum(mk)
train loss with adamw(lr=0.003, clip=1)
```

The softmax runs over the batch's distinct targets (in-batch sampled
softmax; a full softmax over 32768 buckets would cost 32x more), the table
is tied between input and output, and logits are soft-capped at 15 so `exp`
cannot overflow. 2% of the windows are held out; the loss on them is
printed before training and after every epoch. `--init DIR` then starts
every weight of a classifier that `DIR/weights.<name>.bin` holds with the
same size; the bag of words and the head start fresh. A size mismatch (a
different `--mimo`) is an error.

Measured:

| Task (seed 1 unless noted) | `--model bow` | `--model ssm` | `--model ssm --init` |
|------|---------------|---------------|---------------|
| order of two words decides the label (synthetic, regression-tested) | 52.67% (a bag cannot see order) | 100% | 100% |
| SST-2 (6,920 sentences) | 77.48% (3 seeds) | 77.38% in 296 s | |
| ATT&CK tactic of a procedure example (14,787 rows, 15 classes) | 82.45% in 5 s | 81.91% in 790 s | **82.65%** in 808 s |

The tactic task and its pretraining corpus come from
`examples/military_purple/` (`build.sh`, `run.sh`). The corpus (12.7 MB, 1.96M
words) joins two domains:

- **Military science**: 13 public-domain works from Project Gutenberg (Sun
  Tzu; Clausewitz, *On War*, and Murray's companion to it; Jomini; du Picq;
  Halleck; Mahan on sea power and the War of 1812; *Lectures on Land
  Warfare*; Lippitt on the three arms; Napoleon's maxims; the Naval War
  College's *Sound Military Decision*), one row per paragraph.
- **Purple teaming**: one document per MITRE ATT&CK technique (858 in
  v18, 741 live) that puts the red side (what the adversary does) next to
  the blue side (its detection strategies with their analytics, and every
  mitigation with its technique-specific guidance), plus the mitigations
  themselves and 4,835 MITRE D3FEND defensive definitions.

The labeled task uses ATT&CK's procedure examples (an intrusion set,
campaign or tool using a technique), labeled with the technique's tactic
where it has only one; they never enter the corpus. Pretraining took 4
epochs (1,876 steps each, 16 minutes); the held-out next-token loss fell
from 10.24 to 4.35 and was still falling. Fine-tuning from it gained 0.7
points over the same model from scratch and 0.2 over the bag of words, the
first configuration where the sequence model leads; with one seed this is
within run-to-run spread, and the from-scratch and pretrained runs both
overfit after epoch 2 (a lower fine-tuning rate, 0.003, reached 81.81%).
Longer pretraining is the open lever: the corpus is 2M words, three orders
of magnitude below what pretraining normally uses.

`tgc predict DIR SOURCE` applies `features.tgf` to new raw data, matching
columns by name in any order. A missing column counts as missing values. It
prints `prediction,confidence` (classification) or `prediction` (regression,
in original units), and reports accuracy or RMSE when the target column is
present.

### Measured

| Dataset (no configuration given) | Result |
|----------------------------------|--------|
| `hf:scikit-learn/iris` (150 rows) | 96.67% validation accuracy (mean of 3 seeds); `Id` dropped as an identifier |
| `hf:SetFit/sst2` (6920 sentences) | 77.48% validation accuracy (mean of 3 seeds) in 6.9 s; `label_text` caught as a leak |
| synthetic reviews: text + a numeric column, `--target mood` | 100% (regression-tested, including `predict` and compiling `infer.c`) |
| synthetic `y = 3 x1 - 2 x2 + [red] 1.5 + noise`, 3% missing `x2` | predictions 2.99 and -0.52 for true 3.0 and -0.5 |
| synthetic 3-class blobs | 100% (regression-tested) |

On text, unigrams outperformed unigrams+bigrams at every bucket count tried
(256 to 4096). With a few thousand sentences, bigram collisions add more
noise than signal.

SST-2, batch 32, AdamW lr 0.01, end-to-end time per run on one machine. Dense
input and sparse rows with dense steps are single runs (seed 1); row-sparse
steps are the mean of 3 seeds:

| Buckets | Dense input | Sparse rows, dense steps | Sparse rows, row-sparse steps | Accuracy (row-sparse) | `infer.c` |
|---------|-------------|--------------------------|-------------------------------|-----------------------|-----------|
| 1024 | 7.2 s | 2.2 s | | | |
| 4096 | 26.4 s | 6.4 s | | | |
| 8192 | | 11.0 s | 5.3 s | 75.36% | 4.8 MB |
| 16384 | | 22.6 s | 5.5 s | 76.37% | 9.5 MB |
| 32768 (default) | | 49.9 s | 6.9 s | 77.48% | 19 MB |
| 65536 | | 105.9 s | 9.2 s | 77.26% | 38 MB |
| 131072 | | | 13.2 s | 77.53% | 76 MB |

More buckets mean fewer hash collisions; accuracy passes the 75.7% that a
linear bag-of-words model reached on 4096 buckets and levels off near 77.5%
from 32768. What still grows with the bucket count is featurization, the
initial weights and the best-weights snapshot, not the training step. The
default of 32768 takes the plateau; `infer.c` then holds 4 MB of weights
(compiles in about 6 s with `gcc -O2`). `--text-dim` trades accuracy for size
on small targets.

## 19. `scan` and backpropagation through time

```python
h = h0
scan h, c over xt in x, gt in g:      # x : f32[T, ...], g : f32[T, ...]
    h = f(h, xt, gt)
    c = c + sum(h)
    emit hs = h                        # hs : f32[T, shape of h] after the loop
```

`scan` steps through the first axis of one or more sequences (the shape of
JAX's `lax.scan`: carries, sequences, stacked outputs).

1. **Carries** (`h, c`) must be bound before the loop; their values are the
   initial carries. At the end of each step the value bound to each carry
   name is its next value; it must keep the carry's shape. After the loop the
   names hold the final carries.
2. **Sequences** (`xt in x`) all have the same length T along their first
   axis. In step t the slice name is row t (`x[t]`, shape `x` without its
   first axis). Each sequence is an expression, so a projection such as
   `xt in x @ W` is computed for all steps at once before the loop.
3. **`emit N = v`**, directly in the scan body, binds N after the loop to the
   `(T, shape of v)` stack of v. Rows are written before the carries advance,
   so emitting a carry name before reassigning it stacks the start-of-step
   value.
4. Assigning any other outer name is a compile error, as are `update`,
   `train` and `return` inside the body. `think` and `scan` nest inside a
   scan body and inside defs.

TGIR spells a scan out with its carries, slices, body, next values and
stacks (section 8). In C it is a `for` loop over row offsets; nothing is
allocated: carries, slices and stacks are arena tensors.

### Backpropagation through time

`grad` through a scan builds a second scan that runs backwards:

1. **Activation store.** The forward scan is extended to stack each carry's
   start-of-step value, `(T, carry)`. It is planned like any tensor, so a
   training step still has a fixed memory size, known at compile time.
2. **Reverse scan** (TGIR `reverse`), t = T-1 .. 0, over the stored carries,
   the sequences and the adjoints of the stacked outputs. Its carries are
   the carry adjoints u, starting from the adjoints of the final carries,
   and one accumulator per outer value the body reads (weights, earlier
   values). Its body is the forward body cloned at the stored carries and
   swept in reverse:

   ```
   u      <- (d next / d c)^T u + (d y / d c)^T gy[t]
   acc_w  <- acc_w + (d next / d w)^T u + (d y / d w)^T gy[t]
   dx[t]  <- (d next / d x_t)^T u + (d y / d x_t)^T gy[t]      stacked
   ```
3. After the loop: u is the gradient of the initial carries, each
   accumulator the gradient of its value, and the stacks the gradients of
   the sequences.

Nothing is recomputed: memory is O(T x carry) for the store plus the forward
body's temporaries once. The reverse scan is itself a scan, so gradients of
gradients work (tested). Carries and accumulators nobody uses are removed by
dead-code elimination. `-vv` reports each transform:
`[grad] backpropagation through a 12-step scan: 1 carry, 6 accumulated
gradient(s), 1 stacked carry history`.

Loops nested in a scan body (scans, think loops with or without `until`)
are differentiated too: the reverse body clones them whole, and each is
differentiated by its own rule inside the reverse scan. Six further
finite-difference cases cover a fixed-budget think loop, a scan in a scan, a
think-until in a scan, a fixed think in a scan, a scan in a think-until, and
a second-order gradient through a fixed-budget loop; `tests/scan/nested_grad.tg`
runs several at once in the VM, from TGIR and as C.

**Verification.** Six finite-difference cases in `tests/gradcheck.pl`:
gradients with respect to the sequence, the weights and the initial carry;
two carries with a swap and a start-of-step `emit`; a selective SSM; and a
second-order gradient. `tests/scan/scan.tg` runs forward scans, a think loop
inside a scan, a scan in a def and two reverse scans in the VM, from TGIR
and as C, with identical output.

### Example: a selective state-space layer

`examples/selective_ssm.tg` is a Mamba-style layer (Gu & Dao,
arXiv:2312.00752) with an input-dependent step size and zero-order-hold
discretization, trained with `train mse(y, target) with adamw(lr=0.01)` by
BPTT through a 12-step scan:

```python
a = softplus(la)
h = h0
scan h over xt in x:
    dt = softplus(xt @ Wd + bd)
    e = exp(-(dt * a))
    h = e * h + (1 - e) / a * (B @ xt)
    emit y = dot(c, h) + d
train mse(y, target) with adamw(lr=0.01)
```

On the task in `tests/run.sh` (a running sum that a flag in the input
resets) one pass over 20,000 sequences brings the error to under 30% of
predicting zero (about 20% measured) in 0.8 s. The full training step
(forward scan, reverse scan, AdamW) needs a 1,936-byte arena. The same
model compiles to C and trains on the device exactly like the VM (tested
over 100 sequences).

## 20. Attention and latent reasoning

### Causal attention over a key/value ring

`attention(Q, K, V, W)` is the training form: every position at once, each
row attending to itself and the W-1 rows before it. For deployment the same
layer runs one token per call over a ring of W keys and values held in
`state` (`examples/kv_attention.tg`):

```python
slot = pos - 4 * floor(pos / 4)                       # ring slot of this token
hot = step(slot + 0.5 - idx) * step(idx + 0.5 - slot) # one-hot of the slot
Kc = K * outer(1 - hot, ones8) + outer(hot, k)        # ring with this token in it
filled = step(pos + 0.5 - idx)                        # slots written so far
p = softmax(Kc @ q * 0.353553391 + (filled - 1) * 1000000000)
update K[reshape(slot, 1)] = reshape(k, 1, 8)          # one row written per call
update pos = pos + 1
return p @ Vc
```

Keys are stored already rotated by `rope` at their absolute position, so the
order of slots in the ring does not matter and the streaming form computes
the windowed attention exactly. Memory is W x d whatever the stream length,
and the compiled C unit allocates nothing. Tested: the outputs of 7 streamed
tokens (wrapping a ring of 4) equal `attention(..., 4)` over the sequence
within 1e-5 (measured 6e-8), and the C unit equals the VM. Gradients of
`attention` and `rope` are finite-difference checked.

### Coconut curriculum, trained in the language

`examples/coconut/` trains a Coconut-style model (Hao et al.,
arXiv:2412.06769): the answer to a k-hop question over a graph given in the
input is read out after k continuous latent steps (`think h for k:`), with
no supervision of the intermediate hops. Fixed-budget think loops are
differentiated through time, so `train` handles it directly.
`curriculum.sh` trains it in stages as Coconut does: stage k uses k latent
steps on k-hop answers and starts from stage k-1's weights (`--save-state`,
then `file()` initializers). Accuracy over the last 500 of each stage's
fresh examples (AdamW lr 0.003):

| Hops | Rows per stage | Curriculum (final stage) | Direct, from scratch, same total rows |
|------|----------------|--------------------------|---------------------------------------|
| 4 | 1000 | 100% | 92.8% |
| 6 | 1000 | 100% | 100% |
| 8 | 2000 | 100% | 83.6% |

At lr 0.01 the curriculum broke at stage 3 (14.6%), so the rate matters
more than the schedule; at lr 0.003 the curriculum reached every depth
tried, while direct training fell short at 4 hops with 1000 rows per stage
and at 8 hops. The 4-hop curriculum is regression-tested.

### Learned halting, measured

`examples/ponder/` (graph in the input, start node, hop count k = 1..6,
answer k hops later; 8 latent steps at most) trained with the PonderNet
objective on 20,000 examples (AdamW lr 0.001), then run with
`think h for 8 halt lam` on 1,200 new examples:

| Hops | 1 | 2 | 3 | 4 | 5 | 6 |
|------|---|---|---|---|---|---|
| accuracy | 100% | 100% | 100% | 100% | 100% | 100% |
| mean steps taken | 1.00 | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 |

The model spends exactly as many steps as the question needs: 3.5 on
average against a budget of 8. Run for all 8 steps, the same weights reach
56.5%, because they overshoot the answer. At lr 0.003 the halting was less
exact (94.6% overall, hop 5 at 63.2%), and at 60,000 examples and lr 0.001
it was 99.5%. The fixed-point C unit halts at the same step as the VM.

## 21. Quantization

```
tgc quantize MODEL [-o OUT.tgir] [--bits 8|4] [--method rtn|gptq] [--calib DATA] [--min-size N]
```

Weights become signed int8 codes, or int4 codes packed two per byte, with
one f32 scale per output channel: rows of `W` in `W @ x` and of `spmm`
tables, columns of `W` in `x @ W` (the layout `tgc train` writes).
Activations and accumulation stay f32 (weight-only, W8A32 / W4A32): the
size of an embedded model is its weights. The output is self-contained
TGIR; `tgc run`, `batch`, `c`, and `tgc predict DIR SOURCE --model
OUT.tgir` take it directly.

- **Eligible params:** rank 2, at least `--min-size` (256) elements, every
  use a matmul or spmm operand reading codes on one axis. Others stay f32.
- **TGIR:** `(param NAME (f32 R C) (quant BITS AXIS (scale s...) (codes q...)))`,
  one code per element; the reader checks the code range and counts.
- **Kernels:** `tg_matmul_qa` (quantized left operand), `tg_matmul_qb`
  (right), `tg_spmm_q` (tables), shared by the VM and C, so a quantized model
  gives identical outputs in both (tested). A C unit stores the codes as
  `signed char` arrays and drops the f32 copy when no other use needs it.
- **rtn:** round to nearest; each channel's scale is the clipping of its
  absmax (1.0 down to 0.55) with the smallest rounding error.
- **gptq:** GPTQ (Frantar et al., arXiv:2210.17323). For each matmul weight,
  the calibration rows (`--calib`, the `tgc batch` format; `tgc predict DIR
  SOURCE --inputs ROWS.csv` writes them for a trained model) give the
  Hessian H = X X^T of its inputs. Each channel is quantized one input at a
  time, and the rounding error is pushed onto the remaining inputs through
  the upper Cholesky factor of H^-1 (1% dampening; unused inputs zeroed).
  The clipping search is weighted by diag(H). spmm tables and layers wider
  than 2048 inputs use rtn.
- **Report:** per param the relative weight error, the size before and
  after, and with `--calib` the relative change of the model's outputs on
  those rows.

Measured:

| Model | f32 | int8 rtn | int4 rtn | int4 gptq |
|-------|-----|----------|----------|-----------|
| breast cancer MLP (30 -> 256 -> 2), relative output error on 569 rows | | 0.00022 | 0.00406 | 0.00222 |
| same, accuracy | 99.12% | 99.12% | 99.12% | 99.12% |
| SST-2 bag of words (32768 x 32 table), validation accuracy | 77.18% | 77.18% | 77.29% | (table: rtn) |
| SST-2 weights | 4.19 MB | 1.18 MB | 0.66 MB | |

GPTQ has the larger weight error (0.11 against 0.08) and half the output
error: it trades weight fidelity for fidelity of what the layer computes on
real inputs. The per-row scales of a 32768-row table are a fifth of its int4
size; fp16 scales are a later step.

### Fixed point for targets without an FPU

`tgc c MODEL --fixed` emits the unit in Q16.16 integer arithmetic: every
value is a `tg_t` (int32_t, 1.0 = 65536, range [-32768, 32768),
resolution 1.5e-5), and the runtime (`runtime/tg_rtq.h`) has the same
kernels as the float one under the same names.

- **Arithmetic.** Products accumulate in 64 bits and round once; results
  saturate instead of wrapping; division rounds to nearest.
- **Functions.** exp: `x = k ln2 + r`, a degree-8 polynomial in Q30, shifted
  by k (saturates above 10.397, 0 below -12). log: `m 2^e` with
  `log m = 2 atanh((m-1)/(m+1))` to the 13th power in Q30. sin/cos: reduced
  to [-pi/2, pi/2], Taylor to x^13 in Q30. sqrt: exact integer square root.
  tanh, sigmoid, softplus, silu and gelu are built from these without
  overflow; softmax subtracts the row maximum; rmsnorm uses a 64-bit sum of
  squares.
- **Everything else** (matmul, sparse rows, take, active, row updates,
  scan, think halting on the integer delta) runs on the same integers;
  sparse indices are Q16.16 too. Quantized weights keep their int8/int4
  codes; scales become Q7.24.
- **The test driver** (`-DTG_MAIN`) parses and prints decimal text with
  integer arithmetic only, so a whole program runs with no FPU.

Measured against the float VM (max |fixed - float| over the outputs):
newton 1.5e-6, latent_reasoner 1.6e-5, `tests/cases/ops.tg` 3.3e-5,
broadcast 1.4e-5, windowed attention with rope 2.3e-5, sparse rows 2.2e-6,
a scan model that takes two gradients through time 1.4e-4 (outputs up to
9.7), the int4 GPTQ model 5.7e-6. The suite requires < 5e-4.

On a Cortex-M3 (no FPU; `tests/mcu/run.sh` builds with arm-none-eabi-gcc
and runs on QEMU's mps2-an385 board with semihosting), latent_reasoner's
fixed-point unit references no floating-point routine (only 64-bit integer
division helpers), runs, and matches the float VM within 1e-4 with the
same think step count. The float unit for the same target needs the
soft-float library (`__aeabi_fadd`, `fmul`, `fdiv`, ...) and `expf`,
`tanhf`, `sqrtf`; its code is 3.8 KB against 5.2 KB fixed-point, before
the soft-float and libm code it pulls in.

Training in fixed point is not supported: optimizer updates (lr x
gradient, Adam's second moments) fall below the 1.5e-5 resolution. Train
in float and deploy the fixed-point unit.

## 22. Native backends

`tgc asm MODEL --target T` emits assembly for the whole model function
from the planned IR. Every operand address is fixed at compile time: arena
offsets, weight and state symbols, and the input pointers saved in the
frame. Native code runs the `think` and `scan` loops (counters in frame
slots, learned-halting survival in a static word) and the two-phase update
commit. Weights go to `.rodata`. A quantized parameter keeps only its codes
and scales unless a use needs f32. Each IR instruction becomes one call
with the operand addresses as arguments.

| target | ISA and ABI | arithmetic | hand-written `tg_matmul` |
|---|---|---|---|
| `cortex-m3` | ARMv7-M Thumb-2, soft float | Q16.16, same as `tgc c --fixed` | `SMLAL` 32x32->64 accumulate, one rounding, saturation |
| `cortex-m4f` | ARMv7E-M + FPv4-SP, hard float | f32 | `VFMA.F32` |
| `rv64gcv` | RV64GC + V 1.0, LP64D | f32 | vector-length agnostic: `vfmacc.vf` over strips of output columns; `vfmacc.vv` + `vfredusum` for matrix-vector |

The other kernels come from the support unit (`tgc asm MODEL --target T
--support -o support.c`). It holds the runtime of `runtime/tg_rt.h` (or
`tg_rtq.h` for `cortex-m3`) compiled as extern functions, with
`TG_ASM_MATMUL` removing the C matmul. It also has the `TG_MAIN` test
driver. The assembly uses no floating-point registers outside `tg_matmul`.
Loop termination (`tg_k_think`: convergence delta, PonderNet survival) and
broadcasting (`tg_k_bin`) are calls. One generator therefore covers the
soft-float, hard-float and fixed-point ABIs.

```sh
tgc asm model.tg --target cortex-m4f -o model.s
tgc asm model.tg --target cortex-m4f --support -o support.c
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 \
    -ffunction-sections -Wl,--gc-sections firmware.c support.c model.s -lm
```

The exported symbols are the same as for a C unit: `tg_<m>_run`,
`tg_<m>_steps` and `tg_<m>_state_<name>`. The one exception is
`tg_<m>_reset`, which is not emitted.

### Verification

`tests/mcu/native.sh` builds each unit and runs it. ARM targets run on
QEMU `mps2-an385` (Cortex-M3, no FPU) and `mps2-an386` (Cortex-M4F) with
semihosting. RISC-V runs under `qemu-riscv64` with `v=true` at VLEN 128 and
256. The script compares each run with the VM. The suite covers
`latent_reasoner`, every op class (`tests/cases/ops.tg`), `scan`, learned
halting, row-sparse `spmm`, the selective SSM, int4 GPTQ weights, a
self-updating memory over ten runs, the KV ring with row updates, and
`tests/native/matmul.tg`. That last model covers m > 1, n = 37 (not a
multiple of any VL), k = 1 and five inputs, which puts arguments on the
stack on every target.

| target | largest difference vs VM (relative to max(1, \|y\|)) | think/scan steps |
|---|---|---|
| `cortex-m4f` | 1.2e-7 (FMA contraction) | equal |
| `rv64gcv` | 1.2e-7 (VLEN 128, 256, 1024) | equal |
| `cortex-m3` | 4.6e-5 (Q16.16) | equal on the suite |

The Cortex-M3 output is bit-identical to the `--fixed` C unit compiled for
the host. The `SMLAL` kernel rounds and saturates exactly like
`tg_rtq.h`, so a convergence loop stops at the same step as the C unit. In
three of the eighteen sweep models, Q16.16 deltas move the stopping step of
a convergence loop relative to the f32 VM. The same happens with
`tgc c --fixed`. The M3 objects reference no floating-point helpers, only
`__aeabi_ldivmod`. For `latent_reasoner` the native object is 3756 bytes
of text against 5172 for the `-O2` C unit (kernels excluded from both).

References: CMSIS-NN, Lai et al.,
[arXiv:1801.06601](https://arxiv.org/abs/1801.06601) (Cortex-M kernels
with 64-bit `SMLAL` accumulation). Vector-length-agnostic RVV 1.0 GEMM
kernels: [arXiv:2311.05284](https://arxiv.org/abs/2311.05284).

## 23. Roadmap

Ordered by importance for latent-reasoning models on embedded targets:

1. Sequence models at scale: the chunked (SSD) form of the recurrence, which
   turns the per-step scan into batched matmuls over chunks, and pretraining
   on corpora of 10^8 words or more; at about 8,000 tokens a second on 4
   cores the VM is the limit (section 18).
2. Distributed execution: multiple processes over `serve`/`stream`
   transports, then data-parallel training with gradients and optimizer
   state partitioned across them (ZeRO, arXiv:1910.02054).
