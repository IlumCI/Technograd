/* Reverse-mode automatic differentiation, as a source-to-source transform.
 *
 * `grad(y, x)` is lowered into ordinary first-order instructions appended to
 * the current block, so TGIR, the planner, the VM and the C backend need no
 * knowledge of differentiation.
 *
 *   1. depends(v) marks the values on a path from x.
 *   2. The instructions between x and y are collected by depth-first search
 *      over the lowering-time tape (Module.def_blk/def_idx) and processed in
 *      reverse definition order.
 *   3. Each instruction contributes vector-Jacobian products to the adjoints
 *      of its operands; operands that do not depend on x get nothing.
 *
 * think loops with `until` are differentiated implicitly (Deep Equilibrium
 * Models, arXiv:1909.01377). At a converged fixed point h* = f(h*, theta):
 *   adjoint      u = g + (df/dh)^T u          solved by a new think loop
 *   gradient     theta_bar = (df/dtheta)^T u
 * Both Jacobian products come from cloning the loop body at h* and running
 * this same reverse pass over the clone. Memory is constant in the number of
 * iterations; a truncated adjoint solve is the Neumann-series ("phantom")
 * approximation. Fixed-budget loops have no fixed point and are rejected.
 *
 * scan loops are differentiated by backpropagation through time, as another
 * scan. The forward scan is extended to stack each carry's start-of-step
 * value (the activation store, T x carry, planned statically like any other
 * tensor). The reverse scan runs t = T-1 .. 0 over those stacks, the
 * sequences and the adjoints of the stacked outputs. Its carries are the
 * carry adjoints u (starting from the adjoints of the final carries) and one
 * accumulator per outer value the body reads; its body is the forward body
 * cloned at the stored carries and swept in reverse, so
 *   u      <- (d next / d c)^T u + (d y / d c)^T gy[t]
 *   acc_w  <- acc_w + (d next / d w)^T u + (d y / d w)^T gy[t]
 * and the per-step adjoints of the sequence rows are stacked. Memory is
 * O(T x carry); no recomputation. */
#include "tg.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
	Block *b;
	int i;
} Ref;

typedef struct {
	int *v;
	int n;
} Adj;

typedef struct {
	Module *m;
	Block *b; /* emission target */
	const char *file;
	int line;
	int x;
	signed char *dep; /* 0 unknown, 1 depends on x, 2 does not */
	int ndep;
	/* which operands receive adjoints */
	int clo, chi;  /* cloned values [clo, chi) */
	int hs;        /* a value always wanted (h* in the adjoint solve), or -1 */
	int excl;      /* a value never wanted (h* in the parameter pass), or -1 */
	int use_dep;   /* also want values that depend on x */
	int all;       /* full backward pass: every value on the tape gets its adjoint */
} AD;

/* ---- maps --------------------------------------------------------------- */

static int adj_get(const Adj *a, int v) { return v < a->n ? a->v[v] : -1; }

static void adj_put(Adj *a, int v, int g)
{
	if (v >= a->n) {
		int n = v + 64;
		a->v = xrealloc(a->v, (size_t)n * sizeof *a->v);
		for (int i = a->n; i < n; i++) a->v[i] = -1;
		a->n = n;
	}
	a->v[v] = g;
}

static const Ins *def_of(const Module *m, int v)
{
	if (v >= m->ndef || !m->def_blk[v]) return NULL;
	for (int i = 0; i < m->nopen; i++)
		if (m->open_think[i] == v) return NULL; /* state of a loop being lowered: an input */
	return &m->def_blk[v]->v[m->def_idx[v]];
}

/* ---- emission ----------------------------------------------------------- */

static int E3(AD *a, Op op, int p, int q, int r)
{
	Module *m = a->m;
	int na = tg_ops[op].arity, args[TG_MAXARGS] = { p, q, r };
	Shape in[TG_MAXARGS] = { { 0 } }, out;
	char err[256];
	for (int i = 0; i < na; i++) in[i] = m->val[args[i]].sh;
	if (!op_infer(op, in, na, &out, err, sizeof err)) die(a->file, a->line, "autodiff: %s", err);
	int o = mod_value(m, V_TMP, &out, NULL);
	Ins *ins = block_push(a->b);
	ins->op = op;
	ins->out = o;
	ins->na = na;
	for (int i = 0; i < na; i++) ins->a[i] = args[i];
	mod_note_def(m, o, a->b, a->b->len - 1);
	return o;
}

static int E(AD *a, Op op, int p, int q) { return E3(a, op, p, q, -1); }

#define E1(a, op, p) E(a, op, p, -1)

static int K(AD *a, float k)
{
	Shape s = { 0 };
	int o = mod_value(a->m, V_TMP, &s, NULL);
	Ins *ins = block_push(a->b);
	ins->op = OP_CONST;
	ins->out = o;
	ins->k = k;
	mod_note_def(a->m, o, a->b, a->b->len - 1);
	return o;
}

/* A constant all-ones tensor, shared per shape (compiler-generated param). */
static int ONES(AD *a, const Shape *sp)
{
	Shape s0 = *sp, *s = &s0; /* sp may point into m->val, which may move */
	char name[64];
	size_t o = (size_t)snprintf(name, sizeof name, "__ones");
	for (int i = 0; i < s->rank; i++) o += (size_t)snprintf(name + o, sizeof name - o, "_%d", s->dim[i]);
	for (int v = 0; v < a->m->nval; v++)
		if (a->m->val[v].kind == V_PARAM && a->m->val[v].name && strcmp(a->m->val[v].name, name) == 0) return v;
	int v = mod_value(a->m, V_PARAM, s, name);
	int n = shape_numel(s);
	a->m->val[v].data = xmalloc((size_t)n * sizeof(float));
	for (int i = 0; i < n; i++) a->m->val[v].data[i] = 1.0f;
	return v;
}

/* A constant selecting one slot of ELLPACK pairs: 1 at slot (0 index, 1 value), 0 elsewhere. */
static int ELL_SLOT(AD *a, const Shape *sp, int slot)
{
	Shape s0 = *sp, *s = &s0; /* sp may point into m->val, which may move */
	char name[64];
	size_t o = (size_t)snprintf(name, sizeof name, "__ell%d", slot);
	for (int i = 0; i < s->rank; i++) o += (size_t)snprintf(name + o, sizeof name - o, "_%d", s->dim[i]);
	for (int v = 0; v < a->m->nval; v++)
		if (a->m->val[v].kind == V_PARAM && a->m->val[v].name && strcmp(a->m->val[v].name, name) == 0) return v;
	int v = mod_value(a->m, V_PARAM, s, name);
	int n = shape_numel(s);
	a->m->val[v].data = xmalloc((size_t)n * sizeof(float));
	for (int i = 0; i < n; i++) a->m->val[v].data[i] = i % 2 == slot ? 1.0f : 0.0f;
	return v;
}

static const Shape *SH(AD *a, int v) { return &a->m->val[v].sh; }

/* ---- dependence --------------------------------------------------------- */

static void mark_defs(const Block *b, char *in)
{
	for (int i = 0; i < b->len; i++) {
		if (b->v[i].op == OP_SCAN) {
			const Scan *s = b->v[i].sc;
			for (int k = 0; k < s->nc; k++) in[s->c[k]] = 1;
			for (int j = 0; j < s->nx; j++) in[s->xt[j]] = 1;
			for (int y = 0; y < s->ny; y++) in[s->ys[y]] = 1;
		}
		in[b->v[i].out] = 1;
		if (b->v[i].op == OP_THINK || b->v[i].op == OP_SCAN) mark_defs(b->v[i].body, in);
	}
}

/* Calls f(ctx, v) for every value the body reads from outside itself. */
static void outer_reads_rec(const Block *b, const char *in, int state, void (*f)(void *, int), void *ctx)
{
	for (int i = 0; i < b->len; i++) {
		const Ins *ins = &b->v[i];
		int r[TG_MAXARGS + 1] = { -1, -1, -1, -1 };
		if (ins->op == OP_SCAN) {
			const Scan *s = ins->sc;
			int lists[4] = { s->nc, s->nx, s->nc, s->ny };
			const int *vals[4] = { s->init, s->x, s->next, s->y };
			for (int l = 0; l < 4; l++)
				for (int k = 0; k < lists[l]; k++)
					if (vals[l][k] != state && !in[vals[l][k]]) f(ctx, vals[l][k]);
			outer_reads_rec(ins->body, in, state, f, ctx);
			continue;
		}
		if (ins->op == OP_THINK) {
			r[0] = ins->init;
			r[1] = ins->yield;
			outer_reads_rec(ins->body, in, state, f, ctx);
		} else {
			for (int j = 0; j < ins->na; j++) r[j] = ins->a[j];
		}
		for (int j = 0; j < TG_MAXARGS + 1; j++)
			if (r[j] >= 0 && r[j] != state && !in[r[j]]) f(ctx, r[j]);
	}
}

static void outer_reads(AD *a, const Ins *t, void (*f)(void *, int), void *ctx)
{
	char *in = xmalloc((size_t)a->m->nval);
	mark_defs(t->body, in);
	outer_reads_rec(t->body, in, t->out, f, ctx);
	if (t->yield != t->out && !in[t->yield]) f(ctx, t->yield);
	xfree(in);
}

/* Values the body of scan t reads from outside it (carries and slices excluded). */
static void scan_outer_reads(AD *a, const Ins *t, void (*f)(void *, int), void *ctx)
{
	const Scan *s = t->sc;
	char *in = xmalloc((size_t)a->m->nval);
	mark_defs(t->body, in);
	for (int k = 0; k < s->nc; k++) in[s->c[k]] = 1;
	for (int j = 0; j < s->nx; j++) in[s->xt[j]] = 1;
	outer_reads_rec(t->body, in, -1, f, ctx);
	for (int k = 0; k < s->nc; k++)
		if (!in[s->next[k]]) { in[s->next[k]] = 1; f(ctx, s->next[k]); }
	for (int y = 0; y < s->ny; y++)
		if (!in[s->y[y]]) { in[s->y[y]] = 1; f(ctx, s->y[y]); }
	xfree(in);
}

static int depends(AD *a, int v);

typedef struct {
	AD *a;
	int any;
} DepScan;

static void dep_cb(void *ctx, int v)
{
	DepScan *s = ctx;
	if (!s->any && depends(s->a, v)) s->any = 1;
}

static int depends(AD *a, int v)
{
	if (v == a->x) return 1;
	if (v >= a->ndep) {
		int n = a->m->nval + 64;
		a->dep = xrealloc(a->dep, (size_t)n);
		memset(a->dep + a->ndep, 0, (size_t)(n - a->ndep));
		a->ndep = n;
	}
	if (a->dep[v]) return a->dep[v] == 1;
	a->dep[v] = 2;
	int r = 0;
	const Ins *in = def_of(a->m, v);
	if (in && in->op == OP_THINK) {
		Ins t = *in;
		DepScan s = { a, depends(a, t.init) };
		if (!s.any) outer_reads(a, &t, dep_cb, &s);
		r = s.any;
	} else if (in && in->op == OP_SCAN) { /* any output depends if any input does */
		Ins t = *in;
		DepScan s = { a, 0 };
		for (int k = 0; k < t.sc->nc && !s.any; k++) s.any = depends(a, t.sc->init[k]);
		for (int j = 0; j < t.sc->nx && !s.any; j++) s.any = depends(a, t.sc->x[j]);
		if (!s.any) scan_outer_reads(a, &t, dep_cb, &s);
		r = s.any;
	} else if (in) {
		for (int j = 0; j < in->na && !r; j++) r = depends(a, in->a[j]);
	}
	a->dep[v] = r ? 1 : 2;
	return r;
}

static int want(AD *a, int v)
{
	if (v == a->excl) return 0;
	if (v >= a->clo && v < a->chi) return 1;
	if (v == a->hs) return 1;
	return a->use_dep && (a->all || depends(a, v));
}

static void acc(AD *a, Adj *adj, int v, int c)
{
	int prev = adj_get(adj, v);
	adj_put(adj, v, prev < 0 ? c : E(a, OP_ADD, prev, c));
}

/* Sum a broadcast contribution back to its operand's shape: to a scalar, or
 * over the rows a row-broadcast operand was repeated across. */
static int red(AD *a, int c, int operand)
{
	const Shape *so = SH(a, operand), *sc = SH(a, c);
	if (shape_eq(so, sc)) return c;
	if (so->rank == 0) return E1(a, OP_SUM, c);
	if (sc->rank == 2 && so->rank == 1) {
		Shape rows = { 1, { sc->dim[0] } };
		return E(a, OP_MATMUL, ONES(a, &rows), c); /* ones(R) @ g: column sums */
	}
	return E(a, OP_SUM_TO, c, operand); /* general broadcasting */
}

/* ---- row helpers for rank-2 softmax/rmsnorm ------------------------------ */

static int rowsum(AD *a, int M) /* (r,c) -> (r) */
{
	Shape c = { 1, { SH(a, M)->dim[1] } };
	return E(a, OP_MATMUL, M, ONES(a, &c));
}

static int bcast_rows(AD *a, int v, int cols) /* (r) -> (r,c) */
{
	Shape c = { 1, { cols } };
	return E(a, OP_OUTER, v, ONES(a, &c));
}

/* ---- vector-Jacobian products ------------------------------------------- */

static void think_vjp(AD *a, Ins t, int g, Adj *adj);
static int RESHAPE(AD *a, int x, const Shape *to);
static int ZEROS(AD *a, const Shape *sp);
static void scan_vjp(AD *a, Ins t, Adj *adj);

static int scan_has_adj(const Ins *in, const Adj *adj)
{
	for (int k = 0; k < in->sc->nc; k++)
		if (adj_get(adj, in->sc->c[k]) >= 0) return 1;
	for (int y = 0; y < in->sc->ny; y++)
		if (adj_get(adj, in->sc->ys[y]) >= 0) return 1;
	return 0;
}

static void vjp(AD *a, const Ins *in, int g, Adj *adj)
{
	int x0 = in->na > 0 ? in->a[0] : -1, x1 = in->na > 1 ? in->a[1] : -1, x2 = in->na > 2 ? in->a[2] : -1, y = in->out;
	int w0 = x0 >= 0 && want(a, x0), w1 = x1 >= 0 && want(a, x1), w2 = x2 >= 0 && want(a, x2);
	if (!w0 && !w1 && !w2) return;
	switch (in->op) {
	case OP_CONST:
	case OP_STEP:
	case OP_FLOOR: /* piecewise constant */
		return;
	case OP_ADD:
		if (w0) acc(a, adj, x0, red(a, g, x0));
		if (w1) acc(a, adj, x1, red(a, g, x1));
		return;
	case OP_SUB:
		if (w0) acc(a, adj, x0, red(a, g, x0));
		if (w1) acc(a, adj, x1, red(a, E1(a, OP_NEG, g), x1));
		return;
	case OP_MUL:
		if (w0) acc(a, adj, x0, red(a, E(a, OP_MUL, g, x1), x0));
		if (w1) acc(a, adj, x1, red(a, E(a, OP_MUL, g, x0), x1));
		return;
	case OP_DIV:
		if (w0) acc(a, adj, x0, red(a, E(a, OP_DIV, g, x1), x0));
		if (w1) acc(a, adj, x1, red(a, E1(a, OP_NEG, E(a, OP_DIV, E(a, OP_MUL, g, y), x1)), x1));
		return;
	case OP_MAX:
	case OP_MIN: {
		/* s = 1 where x0 is selected; ties go to x1 */
		int s = in->op == OP_MAX ? E1(a, OP_STEP, E(a, OP_SUB, x0, x1)) : E1(a, OP_STEP, E(a, OP_SUB, x1, x0));
		if (w0) acc(a, adj, x0, red(a, E(a, OP_MUL, g, s), x0));
		if (w1) acc(a, adj, x1, red(a, E(a, OP_MUL, g, E(a, OP_SUB, K(a, 1), s)), x1));
		return;
	}
	case OP_NEG:
		acc(a, adj, x0, E1(a, OP_NEG, g));
		return;
	case OP_TANH:
		acc(a, adj, x0, E(a, OP_MUL, g, E(a, OP_SUB, K(a, 1), E(a, OP_MUL, y, y))));
		return;
	case OP_RELU:
		acc(a, adj, x0, E(a, OP_MUL, g, E1(a, OP_STEP, x0)));
		return;
	case OP_SIGMOID:
		acc(a, adj, x0, E(a, OP_MUL, g, E(a, OP_MUL, y, E(a, OP_SUB, K(a, 1), y))));
		return;
	case OP_EXP:
		acc(a, adj, x0, E(a, OP_MUL, g, y));
		return;
	case OP_SQRT:
		acc(a, adj, x0, E(a, OP_DIV, E(a, OP_MUL, g, K(a, 0.5f)), y));
		return;
	case OP_LOG:
		acc(a, adj, x0, E(a, OP_DIV, g, x0));
		return;
	case OP_SOFTPLUS:
		acc(a, adj, x0, E(a, OP_MUL, g, E1(a, OP_SIGMOID, x0)));
		return;
	case OP_SILU: {
		int s = E1(a, OP_SIGMOID, x0);
		int d = E(a, OP_MUL, s, E(a, OP_ADD, K(a, 1), E(a, OP_MUL, x0, E(a, OP_SUB, K(a, 1), s))));
		acc(a, adj, x0, E(a, OP_MUL, g, d));
		return;
	}
	case OP_GELU: {
		/* y = 0.5 x (1 + tanh(c (x + k x^3))) */
		const float c = 0.7978845608f, k = 0.044715f;
		int x2 = E(a, OP_MUL, x0, x0);
		int u = E(a, OP_MUL, K(a, c), E(a, OP_ADD, x0, E(a, OP_MUL, K(a, k), E(a, OP_MUL, x0, x2))));
		int t = E1(a, OP_TANH, u);
		int left = E(a, OP_MUL, K(a, 0.5f), E(a, OP_ADD, K(a, 1), t));
		int sech2 = E(a, OP_SUB, K(a, 1), E(a, OP_MUL, t, t));
		int du = E(a, OP_MUL, K(a, c), E(a, OP_ADD, K(a, 1), E(a, OP_MUL, K(a, 3 * k), x2)));
		int right = E(a, OP_MUL, E(a, OP_MUL, K(a, 0.5f), x0), E(a, OP_MUL, sech2, du));
		acc(a, adj, x0, E(a, OP_MUL, g, E(a, OP_ADD, left, right)));
		return;
	}
	case OP_MATMUL: {
		int ra = SH(a, x0)->rank, rb = SH(a, x1)->rank;
		if (ra == 3 && rb == 3) { /* batched */
			if (w0) acc(a, adj, x0, E(a, OP_MATMUL, g, E1(a, OP_TRANSPOSE, x1)));
			if (w1) acc(a, adj, x1, E(a, OP_MATMUL, E1(a, OP_TRANSPOSE, x0), g));
		} else if (ra == 3) { /* (G,M,K) @ (K,N) or (K): rows of all groups share the right operand */
			const Shape *sa = SH(a, x0);
			Shape flat = { 2, { sa->dim[0] * sa->dim[1], sa->dim[2] } }, gf = { rb == 2 ? 2 : 1, { sa->dim[0] * sa->dim[1], rb == 2 ? SH(a, x1)->dim[1] : 0 } };
			int af = RESHAPE(a, x0, &flat), gl = RESHAPE(a, g, &gf);
			if (rb == 2) {
				if (w0) acc(a, adj, x0, E(a, OP_MATMUL, g, E1(a, OP_TRANSPOSE, x1)));
				if (w1) acc(a, adj, x1, E(a, OP_MATMUL, E1(a, OP_TRANSPOSE, af), gl));
			} else {
				if (w0) acc(a, adj, x0, RESHAPE(a, E(a, OP_OUTER, gl, x1), SH(a, x0)));
				if (w1) acc(a, adj, x1, E(a, OP_MATMUL, E1(a, OP_TRANSPOSE, af), gl));
			}
		} else if (ra == 2 && rb == 1) {
			if (w0) acc(a, adj, x0, E(a, OP_OUTER, g, x1));
			if (w1) acc(a, adj, x1, E(a, OP_MATMUL, E1(a, OP_TRANSPOSE, x0), g));
		} else if (ra == 2 && rb == 2) {
			if (w0) acc(a, adj, x0, E(a, OP_MATMUL, g, E1(a, OP_TRANSPOSE, x1)));
			if (w1) acc(a, adj, x1, E(a, OP_MATMUL, E1(a, OP_TRANSPOSE, x0), g));
		} else if (ra == 1 && rb == 2) {
			if (w0) acc(a, adj, x0, E(a, OP_MATMUL, x1, g));
			if (w1) acc(a, adj, x1, E(a, OP_OUTER, x0, g));
		} else { /* dot */
			if (w0) acc(a, adj, x0, E(a, OP_MUL, x1, g));
			if (w1) acc(a, adj, x1, E(a, OP_MUL, x0, g));
		}
		return;
	}
	case OP_OUTER:
		if (w0) acc(a, adj, x0, E(a, OP_MATMUL, g, x1));
		if (w1) acc(a, adj, x1, E(a, OP_MATMUL, x0, g));
		return;
	case OP_SPMM: /* indices are piecewise constant: only values and weights get gradients */
		if (w0) acc(a, adj, x0, E3(a, OP_SPMM_DX, x0, x1, g));
		if (w1) acc(a, adj, x1, E3(a, OP_SPMM_T, x0, g, x1));
		return;
	case OP_SPMM_T: /* o = x^T g, linear in g and in the values of x; x2 gives only the shape */
		if (w0) acc(a, adj, x0, E3(a, OP_SPMM_DX, x0, g, x1));
		if (w1) acc(a, adj, x1, E(a, OP_SPMM, x0, g));
		return;
	case OP_SPMM_DX: { /* o.value = w[index] . g; independent of the values of x */
		int z = E(a, OP_ADD, E(a, OP_MUL, x0, ELL_SLOT(a, SH(a, x0), 0)), E(a, OP_MUL, g, ELL_SLOT(a, SH(a, x0), 1)));
		if (w1) acc(a, adj, x1, E3(a, OP_SPMM_T, z, x2, x1));
		if (w2) acc(a, adj, x2, E(a, OP_SPMM, z, x1));
		return;
	}
	case OP_RESHAPE:
		acc(a, adj, x0, RESHAPE(a, g, SH(a, x0)));
		return;
	case OP_SIN:
		acc(a, adj, x0, E(a, OP_MUL, g, E1(a, OP_COS, x0)));
		return;
	case OP_COS:
		acc(a, adj, x0, E1(a, OP_NEG, E(a, OP_MUL, g, E1(a, OP_SIN, x0))));
		return;
	case OP_ACTIVE: /* row indices: piecewise constant */
		return;
	case OP_TAKE: /* rows are piecewise constant */
		if (w0) acc(a, adj, x0, E3(a, OP_TAKE_T, x1, g, x0));
		return;
	case OP_TAKE_T: /* linear in g */
		if (w1) acc(a, adj, x1, E(a, OP_TAKE, g, x0));
		return;
	case OP_SUM_TO: /* the gradient broadcasts back over the reduced axes */
		if (w0) acc(a, adj, x0, E(a, OP_ADD, g, ZEROS(a, SH(a, x0))));
		return;
	case OP_TRANSPOSE:
		acc(a, adj, x0, E1(a, OP_TRANSPOSE, g));
		return;
	case OP_SUM:
		acc(a, adj, x0, E(a, OP_MUL, ONES(a, SH(a, x0)), g));
		return;
	case OP_MEAN:
		acc(a, adj, x0, E(a, OP_MUL, ONES(a, SH(a, x0)), E(a, OP_MUL, g, K(a, 1.0f / (float)shape_numel(SH(a, x0))))));
		return;
	case OP_SOFTMAX:
	case OP_RMSNORM: {
		Shape xs = *SH(a, x0);
		int rank = xs.rank, cols = xs.dim[rank - 1], X = x0, Y = y;
		if (rank > 2) { /* over the last axis: work on (rows, cols) */
			Shape flat = { 2, { shape_numel(&xs) / cols, cols } };
			X = RESHAPE(a, x0, &flat);
			Y = RESHAPE(a, y, &flat);
			g = RESHAPE(a, g, &flat);
			rank = 2;
		}
		int dx;
		if (in->op == OP_SOFTMAX) { /* dx = s*g - s*rowsum(s*g) */
			int t = E(a, OP_MUL, g, Y);
			int rs = rank == 1 ? E1(a, OP_SUM, t) : bcast_rows(a, rowsum(a, t), cols);
			dx = E(a, OP_SUB, t, E(a, OP_MUL, Y, rs));
		} else { /* dx = inv * (g - y * rowmean(g*y)), inv = 1/sqrt(rowmean(x^2) + 1e-6) */
			int x0 = X, y = Y;
			int x2 = E(a, OP_MUL, x0, x0), gy = E(a, OP_MUL, g, y), inv, mg;
			if (rank == 1) {
				inv = E(a, OP_DIV, K(a, 1), E1(a, OP_SQRT, E(a, OP_ADD, E1(a, OP_MEAN, x2), K(a, 1e-6f))));
				mg = E1(a, OP_MEAN, gy);
			} else {
				int ms = E(a, OP_DIV, rowsum(a, x2), K(a, (float)cols));
				inv = bcast_rows(a, E(a, OP_DIV, K(a, 1), E1(a, OP_SQRT, E(a, OP_ADD, ms, K(a, 1e-6f)))), cols);
				mg = bcast_rows(a, E(a, OP_DIV, rowsum(a, gy), K(a, (float)cols)), cols);
			}
			dx = E(a, OP_MUL, inv, E(a, OP_SUB, g, E(a, OP_MUL, y, mg)));
		}
		acc(a, adj, x0, xs.rank > 2 ? RESHAPE(a, dx, &xs) : dx);
		return;
	}
	case OP_SPMM_TC: /* optimizer-internal compact gradient */
	case OP_THINK:
	case OP_SCAN:
	case OP_COUNT:
		break;
	}
	die(a->file, a->line, "autodiff: no derivative for '%s'", tg_ops[in->op].name);
}

/* Reverse pass over instructions [lo, hi) of block b, in place. */
static void sweep(AD *a, Block *b, int lo, int hi, Adj *adj)
{
	for (int i = hi - 1; i >= lo; i--) {
		Ins in = b->v[i]; /* copy: emission may reallocate b */
		if (in.op == OP_SCAN) {
			if (scan_has_adj(&in, adj)) scan_vjp(a, in, adj);
			continue;
		}
		int g = adj_get(adj, in.out);
		if (g < 0) continue;
		if (in.op == OP_THINK) think_vjp(a, in, g, adj);
		else vjp(a, &in, g, adj);
	}
}

/* ---- think: implicit differentiation ------------------------------------- */

/* Re-emit a loop body into a->b with the values from[i] replaced by to[i].
 * Returns the map from body values to their clones (-1 outside the body);
 * the caller frees it. Nested think and scan loops are cloned whole, with
 * fresh values for everything they define. */
typedef struct {
	int *map, n;
	const int *from, *to;
	int nsub;
} Clone;

static int cl_get(const Clone *c, int v)
{
	for (int i = 0; i < c->nsub; i++)
		if (c->from[i] == v) return c->to[i];
	return v < c->n && c->map[v] >= 0 ? c->map[v] : v;
}

static int fresh(AD *a, Clone *c, int v) /* a new value standing for loop-defined v */
{
	int o = mod_value(a->m, V_TMP, &a->m->val[v].sh, NULL);
	c->map[v] = o;
	return o;
}

static void clone_into(AD *a, const Block *body, Clone *c)
{
	Module *m = a->m;
	for (int i = 0; i < body->len; i++) {
		Ins in = body->v[i];
		if (in.op == OP_THINK) {
			int st = fresh(a, c, in.out);
			Ins *ni = block_push(a->b);
			int idx = a->b->len - 1;
			ni->op = OP_THINK;
			ni->out = st;
			ni->init = cl_get(c, in.init);
			ni->maxit = in.maxit;
			ni->eps = in.eps;
			ni->tid = m->nthink++;
			mod_note_def(m, st, a->b, idx);
			Block *outer = a->b, *nb = xmalloc(sizeof *nb);
			a->b = nb;
			clone_into(a, in.body, c);
			a->b = outer;
			outer->v[idx].body = nb;
			outer->v[idx].yield = cl_get(c, in.yield);
			outer->v[idx].halt = in.halt >= 0 ? cl_get(c, in.halt) : -1;
			outer->v[idx].hthr = in.hthr;
		} else if (in.op == OP_SCAN) {
			const Scan *s = in.sc;
			Scan *ns = scan_new(s->T, s->reverse);
			if (s->tid >= 0) ns->tid = m->nthink++;
			for (int k = 0; k < s->nc; k++) scan_carry(ns, fresh(a, c, s->c[k]), cl_get(c, s->init[k]), -1);
			for (int j = 0; j < s->nx; j++) scan_seq(ns, cl_get(c, s->x[j]), fresh(a, c, s->xt[j]));
			Ins *ni = block_push(a->b);
			int idx = a->b->len - 1;
			ni->op = OP_SCAN;
			ni->sc = ns;
			ni->out = ns->c[0];
			for (int k = 0; k < ns->nc; k++) mod_note_def(m, ns->c[k], a->b, idx);
			Block *outer = a->b, *nb = xmalloc(sizeof *nb);
			a->b = nb;
			clone_into(a, in.body, c);
			a->b = outer;
			outer->v[idx].body = nb;
			for (int k = 0; k < s->nc; k++) ns->next[k] = cl_get(c, s->next[k]);
			for (int y = 0; y < s->ny; y++) {
				int ys = fresh(a, c, s->ys[y]);
				scan_stack(ns, cl_get(c, s->y[y]), ys);
				mod_note_def(m, ys, outer, idx);
			}
		} else {
			c->map[in.out] = in.op == OP_CONST ? K(a, in.k)
				       : in.op == OP_RESHAPE ? RESHAPE(a, cl_get(c, in.a[0]), &m->val[in.out].sh)
				       : E3(a, in.op, cl_get(c, in.a[0]), in.na > 1 ? cl_get(c, in.a[1]) : -1, in.na > 2 ? cl_get(c, in.a[2]) : -1);
		}
	}
}

static Clone clone_block(AD *a, const Block *body, const int *from, const int *to, int nsub)
{
	Clone c = { NULL, a->m->nval, from, to, nsub };
	c.map = xmalloc((size_t)c.n * sizeof *c.map);
	for (int i = 0; i < c.n; i++) c.map[i] = -1;
	clone_into(a, body, &c);
	return c;
}

/* Re-emit the body of t into a->b with its state replaced by h. Returns the
 * clone of the yielded value. */
static int clone_body(AD *a, const Ins *t, int h)
{
	Clone c = clone_block(a, t->body, &t->out, &h, 1);
	int y = cl_get(&c, t->yield);
	xfree(c.map);
	return y;
}

static int RESHAPE(AD *a, int x, const Shape *to)
{
	Shape s = *to;
	if (shape_numel(&s) != shape_numel(SH(a, x))) die(a->file, a->line, "reshape: element counts differ");
	int o = mod_value(a->m, V_TMP, &s, NULL);
	Ins *ins = block_push(a->b);
	ins->op = OP_RESHAPE;
	ins->out = o;
	ins->na = 1;
	ins->a[0] = x;
	mod_note_def(a->m, o, a->b, a->b->len - 1);
	return o;
}

static int ZEROS(AD *a, const Shape *sp)
{
	Shape s = *sp;
	return E(a, OP_MUL, ONES(a, &s), K(a, 0));
}

/* ---- scan: backpropagation through time ---------------------------------- */

typedef struct {
	int *v, n;
} List;

static void list_cb(void *ctx, int v)
{
	List *l = ctx;
	for (int i = 0; i < l->n; i++)
		if (l->v[i] == v) return;
	l->v = xrealloc(l->v, (size_t)(l->n + 1) * sizeof *l->v);
	l->v[l->n++] = v;
}

static void scan_vjp(AD *a, Ins t, Adj *adj)
{
	Module *m = a->m;
	Scan *s = t.sc; /* shared with the instruction in its block: history is added in place */
	AD save = *a;
	Block *outer = a->b;
	int T = s->T, nc = s->nc, nx = s->nx, ny0 = s->ny;

	/* 1. activation store: the forward scan stacks every carry's start-of-step value */
	int *hist = xmalloc((size_t)nc * sizeof *hist);
	Block *fb = m->def_blk[t.out];
	int fi = m->def_idx[t.out];
	for (int k = 0; k < nc; k++) {
		hist[k] = -1;
		for (int y = 0; y < s->ny; y++)
			if (s->y[y] == s->c[k]) hist[k] = s->ys[y];
		if (hist[k] < 0) {
			Shape q = m->val[s->c[k]].sh;
			if (q.rank == TG_MAXRANK) die(a->file, a->line, "grad through scan: carry rank %d leaves no room for the time axis", q.rank);
			for (int d = q.rank; d > 0; d--) q.dim[d] = q.dim[d - 1];
			q.dim[0] = T;
			q.rank++;
			hist[k] = mod_value(m, V_TMP, &q, NULL);
			scan_stack(s, s->c[k], hist[k]);
			mod_note_def(m, hist[k], fb, fi);
		}
	}

	/* 2. the reverse scan's carries (carry adjoints) and sequences */
	Scan *r = scan_new(T, !s->reverse);
	int lo = m->nval;
	for (int k = 0; k < nc; k++) scan_carry(r, mod_value(m, V_TMP, &m->val[s->c[k]].sh, NULL), -1, -1);
	int *hk = xmalloc((size_t)nc * sizeof *hk), *xs = xmalloc((size_t)nx * sizeof *xs), *gs = xmalloc((size_t)(ny0 ? ny0 : 1) * sizeof *gs);
	for (int k = 0; k < nc; k++) {
		hk[k] = mod_value(m, V_TMP, &m->val[s->c[k]].sh, NULL);
		scan_seq(r, hist[k], hk[k]);
	}
	for (int j = 0; j < nx; j++) {
		xs[j] = mod_value(m, V_TMP, &m->val[s->xt[j]].sh, NULL);
		scan_seq(r, s->x[j], xs[j]);
	}
	for (int y = 0; y < ny0; y++) {
		int g = adj_get(adj, s->ys[y]);
		gs[y] = -1;
		if (g < 0) continue;
		gs[y] = mod_value(m, V_TMP, &m->val[s->y[y]].sh, NULL);
		scan_seq(r, g, gs[y]);
	}

	/* 3. the reverse body: the forward body at the stored carries, swept backwards */
	Block *rb = xmalloc(sizeof *rb);
	a->b = rb;
	int nsub = nc + nx, *from = xmalloc((size_t)nsub * sizeof *from), *to = xmalloc((size_t)nsub * sizeof *to);
	for (int k = 0; k < nc; k++) { from[k] = s->c[k]; to[k] = hk[k]; }
	for (int j = 0; j < nx; j++) { from[nc + j] = s->xt[j]; to[nc + j] = xs[j]; }
	Clone cl = clone_block(a, t.body, from, to, nsub);
	int ncl = rb->len;
	a->clo = lo;
	a->chi = m->nval;
	a->hs = -1;
	a->excl = -1;
	a->use_dep = 1;
	Adj in = { 0 };
	for (int k = 0; k < nc; k++) acc(a, &in, cl_get(&cl, s->next[k]), r->c[k]);
	for (int y = 0; y < ny0; y++)
		if (gs[y] >= 0) acc(a, &in, cl_get(&cl, s->y[y]), gs[y]);
	sweep(a, rb, 0, ncl, &in);
	for (int k = 0; k < nc; k++) {
		int u = adj_get(&in, hk[k]);
		r->next[k] = u >= 0 ? u : ZEROS(a, &m->val[s->c[k]].sh);
	}
	/* per-step adjoints of the sequence rows, restored to the outer want() */
	a->clo = save.clo;
	a->chi = save.chi;
	a->hs = save.hs;
	a->excl = save.excl;
	a->use_dep = save.use_dep;
	int *dx = xmalloc((size_t)(nx ? nx : 1) * sizeof *dx);
	for (int j = 0; j < nx; j++) {
		dx[j] = -1;
		int g = adj_get(&in, xs[j]);
		if (g < 0 || !want(a, s->x[j])) continue;
		dx[j] = mod_value(m, V_TMP, &m->val[s->x[j]].sh, NULL);
		scan_stack(r, g, dx[j]);
	}
	/* accumulators for the outer values the body reads */
	List rd = { NULL, 0 };
	scan_outer_reads(a, &t, list_cb, &rd);
	int *accv = xmalloc((size_t)(rd.n ? rd.n : 1) * sizeof *accv), nacc = 0;
	int *accw = xmalloc((size_t)(rd.n ? rd.n : 1) * sizeof *accw);
	for (int i = 0; i < rd.n; i++) {
		int w = rd.v[i], g = adj_get(&in, w);
		if (g < 0 || !want(a, w)) continue;
		int st = mod_value(m, V_TMP, &m->val[w].sh, NULL);
		scan_carry(r, st, -1, E(a, OP_ADD, st, g));
		accv[nacc] = st;
		accw[nacc++] = w;
	}
	xfree(in.v);
	xfree(cl.map);

	/* 4. initial values in the outer block, then the reverse scan itself */
	a->b = outer;
	for (int k = 0; k < r->nc; k++) {
		if (k < nc) {
			int g = adj_get(adj, s->c[k]);
			r->init[k] = g >= 0 ? g : ZEROS(a, &m->val[s->c[k]].sh);
		} else {
			r->init[k] = ZEROS(a, &m->val[r->c[k]].sh);
		}
	}
	Ins *ri = block_push(outer);
	int idx = outer->len - 1;
	ri->op = OP_SCAN;
	ri->sc = r;
	ri->body = rb;
	ri->out = r->c[0];
	for (int k = 0; k < r->nc; k++) mod_note_def(m, r->c[k], outer, idx);
	for (int y = 0; y < r->ny; y++) mod_note_def(m, r->ys[y], outer, idx);

	/* 5. hand the results to the outer adjoints */
	for (int k = 0; k < nc; k++)
		if (want(a, s->init[k])) acc(a, adj, s->init[k], r->c[k]);
	for (int i = 0; i < nacc; i++) acc(a, adj, accw[i], accv[i]);
	for (int j = 0, y = 0; j < nx; j++)
		if (dx[j] >= 0) acc(a, adj, s->x[j], r->ys[y++]);

	if (tr_on(2)) {
		tr_begin(2, "grad");
		tr_num("scan_steps", T);
		tr_num("carries", nc);
		tr_num("accumulators", nacc);
		tr_end("backpropagation through a %d-step scan: %d carr%s, %d accumulated gradient(s), %d stacked carry history", T, nc,
		       nc == 1 ? "y" : "ies", nacc, nc);
	}
	xfree(hist);
	xfree(hk);
	xfree(xs);
	xfree(gs);
	xfree(from);
	xfree(to);
	xfree(dx);
	xfree(rd.v);
	xfree(accv);
	xfree(accw);
}

static void think_vjp(AD *a, Ins t, int g, Adj *adj)
{
	Module *m = a->m;
	if (t.halt >= 0)
		die(a->file, a->line, "grad through a think loop with learned halting: train with the expected loss over a fixed "
				      "budget (a scan; see examples/ponder) and halt only at inference");
	if (t.eps < 0) { /* a fixed budget has no fixed point: it is a T-step scan, differentiated through time */
		Ins *in = &m->def_blk[t.out]->v[m->def_idx[t.out]];
		Scan *s = scan_new(in->maxit, 0);
		s->tid = in->tid;
		scan_carry(s, in->out, in->init, in->yield);
		in->op = OP_SCAN;
		in->sc = s;
		in->init = in->yield = -1;
		(void)g;
		scan_vjp(a, *in, adj);
		return;
	}
	AD save = *a;
	int hstar = t.out;
	Shape s = m->val[hstar].sh;

	/* 1. adjoint fixed point u = g + (df/dh)^T u, as a think loop */
	int u = mod_value(m, V_TMP, &s, NULL);
	Block *outer = a->b;
	Ins *ui = block_push(outer);
	int idx = outer->len - 1;
	ui->op = OP_THINK;
	ui->out = u;
	ui->init = g;
	ui->maxit = t.maxit;
	ui->eps = t.eps;
	ui->tid = m->nthink++;
	ui->yield = -1;
	Block *body = xmalloc(sizeof *body);
	ui->body = body;
	mod_note_def(m, u, outer, idx);
	m->open_think = xrealloc(m->open_think, (size_t)(m->nopen + 1) * sizeof *m->open_think);
	m->open_think[m->nopen++] = u;

	a->b = body;
	a->clo = m->nval;
	int yc = clone_body(a, &t, hstar);
	a->chi = m->nval;
	a->hs = hstar;
	a->excl = -1;
	a->use_dep = 0;
	int ncl = body->len;
	Adj in = { 0 };
	if (yc == hstar) adj_put(&in, hstar, u); /* f(h) = h */
	else if (yc >= a->clo) adj_put(&in, yc, u);
	sweep(a, body, 0, ncl, &in);
	int r = adj_get(&in, hstar);
	int next = r < 0 ? g : E(a, OP_ADD, g, r);
	xfree(in.v);
	outer->v[idx].yield = next;
	m->nopen--;

	/* 2. parameter gradient (df/dtheta)^T u* at h*; h* itself gets nothing */
	a->b = outer;
	int lo = outer->len;
	a->clo = m->nval;
	int yc2 = clone_body(a, &t, hstar);
	a->chi = m->nval;
	a->hs = -1;
	a->excl = hstar;
	a->use_dep = 1;
	if (yc2 != hstar && want(a, yc2)) acc(a, adj, yc2, u);
	sweep(a, outer, lo, outer->len, adj);

	a->clo = save.clo;
	a->chi = save.chi;
	a->hs = save.hs;
	a->excl = save.excl;
	a->use_dep = save.use_dep;
}

/* ---- entry --------------------------------------------------------------- */

typedef struct {
	AD *a;
	char *seen;
	Ref *v;
	int n, cap;
} Tape;

static void collect(Tape *tp, int v);

static void collect_cb(void *ctx, int v) { collect(ctx, v); }

static void collect(Tape *tp, int v)
{
	AD *a = tp->a;
	if (v == a->x || tp->seen[v] || (!a->all && !depends(a, v))) return;
	tp->seen[v] = 1;
	const Ins *in = def_of(a->m, v);
	if (!in) return;
	if (tp->n == tp->cap) tp->v = xrealloc(tp->v, (size_t)(tp->cap = tp->cap ? tp->cap * 2 : 32) * sizeof *tp->v);
	tp->v[tp->n++] = (Ref){ a->m->def_blk[v], a->m->def_idx[v] };
	Ins t = *in;
	if (t.op == OP_SCAN) { /* one tape entry for all its outputs */
		int *outs = xmalloc((size_t)ins_outs(&t, NULL) * sizeof *outs), no = ins_outs(&t, outs);
		for (int i = 0; i < no; i++) tp->seen[outs[i]] = 1;
		xfree(outs);
		for (int k = 0; k < t.sc->nc; k++) collect(tp, t.sc->init[k]);
		for (int j = 0; j < t.sc->nx; j++) collect(tp, t.sc->x[j]);
		scan_outer_reads(a, &t, collect_cb, tp);
		return;
	}
	if (t.op == OP_THINK && t.eps < 0) collect(tp, t.init); /* a fixed budget: the result depends on the start */
	if (t.op == OP_THINK) outer_reads(a, &t, collect_cb, tp); /* the fixed point does not depend on init */
	else
		for (int j = 0; j < t.na; j++) collect(tp, t.a[j]);
}

static int by_out_desc(const void *p, const void *q)
{
	const Ref *r = p, *s = q;
	return s->b->v[s->i].out - r->b->v[r->i].out;
}

/* One full backward pass per (objective, block), shared by every grad() of
 * that objective: the first call computes the adjoint of every value on the
 * tape and caches the map; later calls look theirs up. Adjoints nobody uses
 * are removed by dead-code elimination after lowering. */
static GradCache *gc_get(AD *a0, int y)
{
	AD a = *a0;
	Module *m = a.m;
	Block *b = a.b;
	GradCache *c = NULL;
	for (int i = 0; i < m->ngcache; i++)
		if (m->gcache[i].y == y && m->gcache[i].b == b) c = &m->gcache[i];
	if (!c) {
		Tape tp = { &a, xmalloc((size_t)m->nval), NULL, 0, 0 };
		collect(&tp, y);
		qsort(tp.v, (size_t)tp.n, sizeof *tp.v, by_out_desc);
		Adj adj = { 0 };
		adj_put(&adj, y, K(&a, 1));
		for (int i = 0; i < tp.n; i++) {
			Ins in = tp.v[i].b->v[tp.v[i].i];
			if (in.op == OP_SCAN) {
				if (scan_has_adj(&in, &adj)) scan_vjp(&a, in, &adj);
				continue;
			}
			int g = adj_get(&adj, in.out);
			if (g < 0) continue;
			if (in.op == OP_THINK) think_vjp(&a, in, g, &adj);
			else vjp(&a, &in, g, &adj);
		}
		xfree(tp.v);
		xfree(tp.seen);
		m->gcache = xrealloc(m->gcache, (size_t)(m->ngcache + 1) * sizeof *m->gcache);
		c = &m->gcache[m->ngcache++];
		c->y = y;
		c->b = b;
		c->adj = adj.v;
		c->nadj = adj.n;
		if (tr_on(2)) {
			tr_begin(2, "grad");
			tr_num("objective", y);
			tr_num("tape", tp.n);
			tr_end("backward pass for objective %%%d over %d instruction(s), shared by later grad() calls", y, tp.n);
		}
	}
	xfree(a.dep);
	return c;
}

int ad_grad(Module *m, Block *b, int y, int x, const char *file, int line)
{
	AD a = { m, b, file, line, -1, NULL, 0, 0, 0, -1, -1, 1, 1 };
	if (y == x) return K(&a, 1);
	GradCache *c = gc_get(&a, y);
	int r = x < c->nadj ? c->adj[x] : -1;
	if (r < 0) r = E(&a, OP_MUL, ONES(&a, SH(&a, x)), K(&a, 0)); /* y does not depend on x */
	xfree(a.dep);
	return r;
}

/* Does y depend on x? Answered from the shared backward pass. */
int ad_depends(Module *m, Block *b, int y, int x, const char *file, int line)
{
	if (y == x) return 1;
	AD a = { m, b, file, line, -1, NULL, 0, 0, 0, -1, -1, 1, 1 };
	GradCache *c = gc_get(&a, y);
	return x < c->nadj && c->adj[x] >= 0;
}

/* Shape-checked emission for other compiler passes (optimizers, losses). */
int ir_op(Module *m, Block *b, Op op, int p, int q, const char *file, int line)
{
	AD a = { m, b, file, line, -1, NULL, 0, 0, 0, -1, -1, 1, 1 };
	return E(&a, op, p, q);
}

int ir_op3(Module *m, Block *b, Op op, int p, int q, int r, const char *file, int line)
{
	AD a = { m, b, file, line, -1, NULL, 0, 0, 0, -1, -1, 1, 1 };
	return E3(&a, op, p, q, r);
}

int ir_reshape(Module *m, Block *b, int x, const Shape *to)
{
	AD a = { m, b, "", 0, -1, NULL, 0, 0, 0, -1, -1, 1, 1 };
	return RESHAPE(&a, x, to);
}

int ir_k(Module *m, Block *b, float k)
{
	AD a = { m, b, "", 0, -1, NULL, 0, 0, 0, -1, -1, 1, 1 };
	return K(&a, k);
}
