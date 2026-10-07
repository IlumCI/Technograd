# Technograd language reference (v0.1)

## 1. Model

A program defines one model. Its parts:

- `model NAME`: required, exactly once.
- `param NAME : TYPE = INIT`: constant tensors (weights). They are global and read-only.
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
think S for N [until EPS]:       latent loop, see section 7
    block
```

## 6. Expressions

Precedence from low to high: `+ -`, then `* / @`, then unary `-`, then calls
and parentheses. All binary operators are left associative.

| Builtin | Shape rule |
|---------|------------|
| `a + b`, `a - b`, `a * b`, `a / b`, `max(a,b)`, `min(a,b)` | equal shapes, or either operand scalar (broadcast) |
| `-a`, `neg tanh relu sigmoid exp sqrt gelu silu` | elementwise, shape preserved; `gelu` uses the tanh approximation |
| `a @ b` (`matmul`, `dot`) | `(m,k)@(k)->(m)`, `(m,k)@(k,n)->(m,n)`, `(k)@(k,n)->(n)`, `(k)@(k)->()` |
| `softplus(x)`, `log(x)` | elementwise; softplus is overflow-free: `max(x,0) + log1p(exp(-|x|))` |
| `softmax(x)`, `rmsnorm(x)` | over the last axis, shape preserved, rank >= 1; rmsnorm eps = 1e-6, no gain |
| `sum(x)`, `mean(x)` | reduce to scalar |
| `transpose(x)` | rank 2 only |

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

## 14. Roadmap

Ordered by importance for latent-reasoning models on embedded targets:

1. int8/int4 weights with per-channel scales, and fixed-point kernels for
   targets without an FPU.
2. `embed(table, token)` gather and causal attention over a fixed-size KV
   ring. These are needed for token-level Coconut models, where continuous
   thoughts are mixed with token embeddings.
3. Reverse-mode autodiff over TGIR, including backprop through `think`, so
   that Coconut's multi-stage curriculum can be trained in-language.
4. Learned halting heads (`until` driven by a predicate value) in addition to
   convergence halting.
5. Native backends from the planned IR: ARMv7-M/ARMv8-M assembly with
   CMSIS-NN style kernels, and RISC-V with the vector extension.
