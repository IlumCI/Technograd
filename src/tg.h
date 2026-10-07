/* Technograd compiler: shared declarations. */
#ifndef TG_H
#define TG_H

#include <setjmp.h>
#include <stddef.h>
#include <stdio.h>

#define TG_MAXRANK 4
#define TG_ALIGN   4 /* arena alignment in floats (16 bytes) */

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
	OP_THINK,
	OP_COUNT
} Op;

typedef enum { CLS_CONST, CLS_BIN, CLS_UN, CLS_MATMUL, CLS_ROW, CLS_RED, CLS_TRANS, CLS_THINK } OpClass;

typedef struct {
	const char *name;
	int arity;
	OpClass cls;
} OpInfo;

extern const OpInfo tg_ops[OP_COUNT];
int op_lookup(const char *name); /* -1 if unknown; never returns OP_CONST/OP_THINK */
int op_infer(Op op, const Shape *a, int na, Shape *out, char *err, size_t errn);

/* matmul operand geometry: o[m,n] = a[m,k] @ b[k,n] */
void matmul_dims(const Shape *a, const Shape *b, int *m, int *k, int *n);
void row_dims(const Shape *s, int *rows, int *cols);

/* ---- IR ----------------------------------------------------------------- */
typedef enum { V_PARAM, V_INPUT, V_TMP } VKind;

typedef struct {
	VKind kind;
	Shape sh;
	char *name;  /* params/inputs; NULL for temporaries */
	float *data; /* params */
	int index;   /* inputs: position in the entry signature */
	int def, last, off; /* filled by the planner (temporaries only) */
} Value;

typedef struct Block Block;

typedef struct {
	Op op;
	int out;
	int a[2];
	int na;
	float k;      /* OP_CONST */
	int init;     /* OP_THINK: initial state value */
	int maxit;    /* OP_THINK: iteration budget */
	float eps;    /* OP_THINK: halting threshold, < 0 = fixed budget */
	Block *body;  /* OP_THINK */
	int yield;    /* OP_THINK: next-state value computed by body */
	int tid;      /* OP_THINK: index into the steps vector */
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
	int arena; /* floats */
	int planned;
	int trace; /* VM tracing enabled (the user's model only) */
} Module;

Module *mod_new(const char *name);
int mod_value(Module *m, VKind k, const Shape *sh, const char *name);
Ins *block_push(Block *b);
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

#endif
