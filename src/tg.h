/* Technograd compiler: shared declarations. */
#ifndef TG_H
#define TG_H

#include <setjmp.h>
#include <stddef.h>
#include <stdio.h>

#define TG_MAXRANK 4
#define TG_ALIGN   4 /* arena alignment in floats (16 bytes) */
#define TG_MAXARGS 3 /* operands per instruction */

/* ---- utilities ---------------------------------------------------------- */
void *xmalloc(size_t n);
void *xrealloc(void *p, size_t n);
char *xstrdup(const char *s);
void xfree(void *p);
size_t alloc_mark(void);
void alloc_release(size_t mark); /* free everything allocated since the mark */
void *alloc_keep(void *p);        /* exempt one block from release */
char *read_file(const char *path, size_t *len);
void die(const char *file, int line, const char *fmt, ...);

/* When tg_trap is set, die() records the diagnostic in tg_diag and longjmps
 * instead of exiting. Used by the auto-fixer for trial compilation. */
typedef struct {
	char file[256];
	int line;
	char msg[512];
} Diag;
extern jmp_buf *tg_trap;
extern Diag tg_diag;

/* ---- shapes ------------------------------------------------------------- */
typedef struct {
	int rank;
	int dim[TG_MAXRANK];
} Shape;

int shape_numel(const Shape *s);
int shape_eq(const Shape *a, const Shape *b);
int shape_suffix(const Shape *small, const Shape *big); /* row broadcasting: (H) into (B,H) */
void shape_str(const Shape *s, char *buf, size_t n); /* "(f32 8 4)" */

/* ---- S-expressions (TGIR carrier and surface AST) ----------------------- */
typedef enum { SX_SYM, SX_NUM, SX_STR, SX_LIST } SxKind;

typedef struct Sx {
	SxKind k;
	char *s;
	double n;
	struct Sx **v;
	int len, cap;
	int line;
} Sx;

Sx *sx_new(SxKind k, int line);
Sx *sx_sym(const char *s, int line);
Sx *sx_num(double n, int line);
Sx *sx_str(const char *s, int line);
Sx *sx_list(int line, int n, ...);
void sx_push(Sx *l, Sx *e);
int sx_issym(const Sx *x, const char *s);
Sx *sx_read(const char *src, const char *file); /* list of top-level forms */

/* ---- operations --------------------------------------------------------- */
typedef enum {
	OP_CONST,
	OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MAX, OP_MIN,
	OP_NEG, OP_TANH, OP_RELU, OP_SIGMOID, OP_EXP, OP_SQRT, OP_GELU, OP_SILU, OP_SOFTPLUS, OP_LOG,
	OP_MATMUL,
	OP_SOFTMAX, OP_RMSNORM,
	OP_SUM, OP_MEAN,
	OP_TRANSPOSE,
	OP_OUTER,
	OP_STEP,
	OP_SPMM, OP_SPMM_T, OP_SPMM_DX,
	OP_ACTIVE, OP_TAKE, OP_TAKE_T, OP_SPMM_TC,
	OP_SIN, OP_COS, OP_RESHAPE, OP_FLOOR,
	OP_THINK,
	OP_SCAN,
	OP_COUNT
} Op;

typedef enum { CLS_CONST, CLS_BIN, CLS_UN, CLS_MATMUL, CLS_ROW, CLS_RED, CLS_TRANS, CLS_OUTER, CLS_SPMM, CLS_ROWS, CLS_RESHAPE, CLS_THINK, CLS_SCAN } OpClass;

typedef struct {
	const char *name;
	int arity;
	OpClass cls;
} OpInfo;

extern const OpInfo tg_ops[OP_COUNT];
int op_lookup(const char *name); /* -1 if unknown; never returns OP_CONST/OP_THINK/OP_SCAN */
int op_infer(Op op, const Shape *a, int na, Shape *out, char *err, size_t errn);

/* matmul operand geometry: o[m,n] = a[m,k] @ b[k,n] */
void matmul_dims(const Shape *a, const Shape *b, int *m, int *k, int *n);
void row_dims(const Shape *s, int *rows, int *cols);
/* sparse rows in ELLPACK form: x[rows, K, 2] of (index, value) pairs against w[D, H] */
void spmm_dims(const Shape *x, const Shape *w, int *rows, int *k, int *d, int *h);

/* ---- IR ----------------------------------------------------------------- */
/* V_STATE: like a param, but mutable; changed only by `update`, committed after a run. */
typedef enum { V_PARAM, V_INPUT, V_TMP, V_STATE } VKind;

typedef struct {
	VKind kind;
	Shape sh;
	char *name;  /* params/inputs; NULL for temporaries */
	float *data; /* params */
	int index;   /* inputs: position in the entry signature */
	int dead;    /* compiler-generated param left unused after dead-code elimination */
	int def, last, off; /* filled by the planner (temporaries only) */
} Value;

typedef struct Block Block;

/* OP_SCAN: a loop over the leading axis of T-row sequences (JAX-style scan).
 * Carries c[k] start at init[k] and become next[k] after every step; after the
 * loop c[k] holds the final value. Each step sees row t of every sequence x[j]
 * as the slice value xt[j], and row t of every stack ys[m] receives the body
 * value y[m] (read before the carries advance). reverse runs t = T-1 .. 0. */
typedef struct {
	int T, reverse;
	int tid; /* >= 0: a fixed-budget think loop turned into a scan reports T in this steps slot */
	int nc, *c, *init, *next;
	int nx, *x, *xt;
	int ny, *y, *ys;
} Scan;

typedef struct GradCache {
	int y;
	Block *b;
	int *adj, nadj; /* value -> adjoint value, or -1 */
} GradCache;

typedef struct {
	Op op;
	int out;
	int a[TG_MAXARGS];
	int na;
	float k;      /* OP_CONST */
	int init;     /* OP_THINK: initial state value */
	int maxit;    /* OP_THINK: iteration budget */
	float eps;    /* OP_THINK: halting threshold, < 0 = fixed budget */
	Block *body;  /* OP_THINK */
	int yield;    /* OP_THINK: next-state value computed by body */
	int tid;      /* OP_THINK: index into the steps vector */
	Scan *sc;     /* OP_SCAN (body in `body`; `out` is one of its outputs) */
	int pbeg, pend;
} Ins;

struct Block {
	Ins *v;
	int len, cap;
};

typedef struct {
	char *name;
	Value *val;
	int nval, capval;
	int *inputs;
	int ninputs;
	int output;
	Block top;
	int nthink;
	int *upd_state, *upd_src; /* update i: upd_state[i] <- upd_src[i] after the run */
	int *upd_rows;            /* -1, or a row-index vector: only those rows of the state are written */
	int nupd;
	int stage;       /* arena offset of the commit staging area (floats) */
	/* Lowering-time tape for autodiff: the instruction defining each value, and
	 * the think loops whose bodies are being lowered (their state is a leaf). */
	Block **def_blk;
	int *def_idx, ndef;
	int *open_think, nopen;
	struct GradCache *gcache; /* shared backward passes, see autodiff.c */
	int ngcache;
	int arena; /* floats */
	int planned;
	int trace; /* VM tracing enabled (the user's model only) */
} Module;

Module *mod_new(const char *name);
int mod_value(Module *m, VKind k, const Shape *sh, const char *name);
Ins *block_push(Block *b);
Scan *scan_new(int T, int reverse);
void scan_carry(Scan *s, int c, int init, int next);
void scan_seq(Scan *s, int x, int xt);
void scan_stack(Scan *s, int y, int ys);
int ins_outs(const Ins *in, int *buf); /* every value an instruction defines; buf may be NULL */
void mod_update(Module *m, int state, int src); /* record `update state = src` */
void mod_update_rows(Module *m, int state, int src, int rows); /* `update state[rows] = src` */
int upd_check(const Module *m, int state, int src, int rows, char *err, size_t n); /* shape rules of an update */
void mod_note_def(Module *m, int v, Block *b, int idx); /* record b->v[idx] as v's definition */
int ad_grad(Module *m, Block *b, int y, int x, const char *file, int line); /* autodiff.c */
int ad_depends(Module *m, Block *b, int y, int x, const char *file, int line);
int ir_op(Module *m, Block *b, Op op, int p, int q, const char *file, int line); /* emit, shape-checked */
int ir_op3(Module *m, Block *b, Op op, int p, int q, int r, const char *file, int line);
int ir_k(Module *m, Block *b, float k);
int ir_reshape(Module *m, Block *b, int x, const Shape *to); /* same data, new shape (equal element count) */
void optim_train(Module *m, Block *b, int loss, const char *opt, const char **keys, const double *vals, int nkv,
		 const int *over, int nover, const char *file, int line); /* optim.c */
void value_name(const Module *m, int v, char *buf, size_t n);

/* frontends */
Sx *surface_parse(const char *src, const char *file);
Sx *surface_load(const char *path); /* parse + merge `import` declarations */
Module *lower(Sx *ast, const char *file);
Module *ir_read(Sx *forms, const char *file);
Module *load_module(const char *path);

/* middle/backends */
void ir_write(const Module *m, FILE *f);
void plan(Module *m);
void plan_dump(const Module *m, FILE *f);
void vm_run(const Module *m, const float **in, float *out, int *steps);
float vm_last_delta(int tid); /* halting delta of think loop tid in the last run */
void vm_run_into(const Module *m, const float **in, float *out, int *steps, float *arena, float *delta); /* thread-safe */

/* thread pool (par.c) */
int par_cpus(void);
void par_init(int threads); /* 0 = one per online CPU */
int par_threads(void);
void par_for(int n, int grain, void (*fn)(void *ctx, int lo, int hi, int tid), void *ctx);
void cgen(const Module *m, FILE *f);

/* tracing and structured logs (trace.c): levels 0 error, 1 info (-v), 2 debug (-vv) */
extern int tg_verbose;
double tr_now_ms(void);
void tr_open_log(const char *path);
void tr_close_log(void);
int tr_on(int level);
void tr_begin(int level, const char *stage);
void tr_str(const char *key, const char *val);
void tr_num(const char *key, double val);
void tr_end(const char *fmt, ...);
void tr_quiet(void);
void tr_error(const char *file, int line, const char *msg);
int block_count(const Block *b); /* instructions, think bodies included */

/* tensor file I/O (io.c): .bin/.f32 = raw little-endian f32, else text */
int io_is_binary(const char *path);
int io_parse_row(const char *s, float *v, int max, const char **bad);
float *io_load(const char *path, int *count);
void io_write_row(FILE *f, int binary, const float *v, int n, const int *steps, int nsteps);

/* auto-fixer (fixer.c) */
char *autofix(const char *src, const char *path, int verbose, int *nfixed); /* NULL if unrepaired */
int fixer_train(const char *out, char **corpus, int n);
int fixer_eval(char **corpus, int n);
int autotrain_main(int argc, char **argv); /* autotrain.c: train, predict, data */

/* host I/O (serve.c): streams, sockets, signals */
void tg_signals(void);  /* SIGINT/SIGTERM request a clean stop; SIGPIPE ignored */
int tg_exit_status(void); /* 0, or 128 + the signal that requested a stop */
int tg_stopping(void);
int stream_main(const Module *m, const char *src, const char *dst);
int serve_main(const Module *m, const char *listen_spec);

#endif
