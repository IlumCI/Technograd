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
 * approximation. Fixed-budget loops have no fixed point and are rejected. */
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
static int ONES(AD *a, const Shape *s)
{
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
static int ELL_SLOT(AD *a, const Shape *s, int slot)
{
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
		in[b->v[i].out] = 1;
		if (b->v[i].op == OP_THINK) mark_defs(b->v[i].body, in);
	}
}

/* Calls f(ctx, v) for every value the body reads from outside itself. */
static void outer_reads_rec(const Block *b, const char *in, int state, void (*f)(void *, int), void *ctx)
{
	for (int i = 0; i < b->len; i++) {
		const Ins *ins = &b->v[i];
		int r[TG_MAXARGS + 1] = { -1, -1, -1, -1 };
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
	if (so->rank == sc->rank) return c;
	if (so->rank == 0) return E1(a, OP_SUM, c);
	if (sc->rank != 2 || so->rank != 1)
		die(a->file, a->line, "autodiff of row broadcasting supports a vector over the rows of a matrix");
	Shape rows = { 1, { sc->dim[0] } };
	return E(a, OP_MATMUL, ONES(a, &rows), c); /* ones(R) @ g: column sums */
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

static void vjp(AD *a, const Ins *in, int g, Adj *adj)
{
	int x0 = in->na > 0 ? in->a[0] : -1, x1 = in->na > 1 ? in->a[1] : -1, x2 = in->na > 2 ? in->a[2] : -1, y = in->out;
	int w0 = x0 >= 0 && want(a, x0), w1 = x1 >= 0 && want(a, x1), w2 = x2 >= 0 && want(a, x2);
	if (!w0 && !w1 && !w2) return;
	switch (in->op) {
	case OP_CONST:
	case OP_STEP:
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
		if (ra == 2 && rb == 1) {
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
		int rank = SH(a, x0)->rank;
		if (rank > 2) die(a->file, a->line, "autodiff of %s supports rank 1 and 2", tg_ops[in->op].name);
		int cols = SH(a, x0)->dim[rank - 1];
		if (in->op == OP_SOFTMAX) { /* dx = s*g - s*rowsum(s*g) */
			int t = E(a, OP_MUL, g, y);
			int rs = rank == 1 ? E1(a, OP_SUM, t) : bcast_rows(a, rowsum(a, t), cols);
			acc(a, adj, x0, E(a, OP_SUB, t, E(a, OP_MUL, y, rs)));
		} else { /* dx = inv * (g - y * rowmean(g*y)), inv = 1/sqrt(rowmean(x^2) + 1e-6) */
			int x2 = E(a, OP_MUL, x0, x0), gy = E(a, OP_MUL, g, y), inv, mg;
			if (rank == 1) {
				inv = E(a, OP_DIV, K(a, 1), E1(a, OP_SQRT, E(a, OP_ADD, E1(a, OP_MEAN, x2), K(a, 1e-6f))));
				mg = E1(a, OP_MEAN, gy);
			} else {
				int ms = E(a, OP_DIV, rowsum(a, x2), K(a, (float)cols));
				inv = bcast_rows(a, E(a, OP_DIV, K(a, 1), E1(a, OP_SQRT, E(a, OP_ADD, ms, K(a, 1e-6f)))), cols);
				mg = bcast_rows(a, E(a, OP_DIV, rowsum(a, gy), K(a, (float)cols)), cols);
			}
			acc(a, adj, x0, E(a, OP_MUL, inv, E(a, OP_SUB, g, E(a, OP_MUL, y, mg))));
		}
		return;
	}
	case OP_THINK:
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
		int g = adj_get(adj, in.out);
		if (g < 0) continue;
		if (in.op == OP_THINK) think_vjp(a, in, g, adj);
		else vjp(a, &in, g, adj);
	}
}

/* ---- think: implicit differentiation ------------------------------------- */

static int has_think(const Block *b)
{
	for (int i = 0; i < b->len; i++)
		if (b->v[i].op == OP_THINK) return 1;
	return 0;
}

/* Re-emit the body of t into a->b with its state replaced by h. Returns the
 * clone of the yielded value. */
static int clone_body(AD *a, const Ins *t, int h)
{
	int n = a->m->nval;
	int *map = xmalloc((size_t)n * sizeof *map);
	for (int i = 0; i < n; i++) map[i] = -1;
#define MAP(v) ((v) == t->out ? h : (v) < n && map[v] >= 0 ? map[v] : (v))
	for (int i = 0; i < t->body->len; i++) {
		Ins in = t->body->v[i];
		map[in.out] = in.op == OP_CONST ? K(a, in.k)
			    : E3(a, in.op, MAP(in.a[0]), in.na > 1 ? MAP(in.a[1]) : -1, in.na > 2 ? MAP(in.a[2]) : -1);
	}
	int y = MAP(t->yield);
#undef MAP
	xfree(map);
	return y;
}

static void think_vjp(AD *a, Ins t, int g, Adj *adj)
{
	Module *m = a->m;
	if (t.eps < 0)
		die(a->file, a->line, "grad through a fixed-budget think loop is not supported: add 'until' so the loop "
				      "converges (implicit differentiation needs a fixed point)");
	if (has_think(t.body)) die(a->file, a->line, "grad through nested think loops is not supported yet");
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

int ir_k(Module *m, Block *b, float k)
{
	AD a = { m, b, "", 0, -1, NULL, 0, 0, 0, -1, -1, 1, 1 };
	return K(&a, k);
}
