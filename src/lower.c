/* Lowering: surface AST -> IR.
 *
 * - Every user `def` is inlined at its call site (static graph, no call stack).
 * - Variables are renamed into SSA values.
 * - Parameter initializers are materialized so the IR is self-contained.
 * - `think` blocks become OP_THINK with exactly one loop-carried state.
 * - `scan` blocks become OP_SCAN: carries, per-step slices of sequences, and
 *   `emit`ted values stacked over the steps. */
#include "tg.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	const char *name;
	int val;
} Bind;

typedef struct {
	Bind *b;
	int n, cap;
} Env;

typedef struct {
	Module *m;
	const char *file;
	Sx **defs;
	int ndefs;
	Env globals;
	int depth;    /* def-inlining depth; 0 = the body of forward */
	int in_think; /* nesting of think and scan bodies being lowered */
	const char *loop_kw; /* the innermost of them: "think" or "scan" */
	struct Emits *emits; /* the scan whose body is at the current level, if any */
	int ntmp;
} L;

typedef struct Emits {
	int level; /* in_think of the scan body */
	const char **name;
	int *val, n;
} Emits;

static int env_get(const Env *e, const char *name)
{
	for (int i = e->n - 1; i >= 0; i--)
		if (strcmp(e->b[i].name, name) == 0) return e->b[i].val;
	return -1;
}

static void env_set(Env *e, const char *name, int v)
{
	for (int i = e->n - 1; i >= 0; i--)
		if (strcmp(e->b[i].name, name) == 0) { e->b[i].val = v; return; }
	if (e->n == e->cap) {
		e->cap = e->cap ? e->cap * 2 : 16;
		e->b = xrealloc(e->b, (size_t)e->cap * sizeof *e->b);
	}
	e->b[e->n].name = name;
	e->b[e->n].val = v;
	e->n++;
}

static Env env_copy(const Env *e)
{
	Env c = { 0 };
	c.cap = c.n = e->n;
	c.b = xmalloc((size_t)(c.cap ? c.cap : 1) * sizeof *c.b);
	if (e->n) memcpy(c.b, e->b, (size_t)e->n * sizeof *e->b);
	return c;
}

static Sx *find_def(L *l, const char *name)
{
	for (int i = 0; i < l->ndefs; i++)
		if (strcmp(l->defs[i]->v[1]->s, name) == 0) return l->defs[i];
	return NULL;
}

static Shape to_shape(Sx *ty)
{
	Shape s = { 0 };
	s.rank = ty->len - 1;
	for (int i = 1; i < ty->len; i++) s.dim[i - 1] = (int)ty->v[i]->n;
	return s;
}

static int lookup(L *l, Env *e, Sx *name)
{
	int v = env_get(e, name->s);
	if (v < 0) v = env_get(&l->globals, name->s);
	if (v < 0) die(l->file, name->line, "undefined name '%s'", name->s);
	return v;
}

static int emit_op(L *l, Block *b, Op op, const int *args, int na, int line)
{
	Shape in[TG_MAXARGS], out;
	char err[256];
	if (na != tg_ops[op].arity)
		die(l->file, line, "'%s' takes %d operand(s), got %d", tg_ops[op].name, tg_ops[op].arity, na);
	for (int i = 0; i < na; i++) in[i] = l->m->val[args[i]].sh;
	if (!op_infer(op, in, na, &out, err, sizeof err)) die(l->file, line, "%s", err);
	int o = mod_value(l->m, V_TMP, &out, NULL);
	Ins *ins = block_push(b);
	ins->op = op;
	ins->out = o;
	ins->na = na;
	for (int i = 0; i < na; i++) ins->a[i] = args[i];
	mod_note_def(l->m, o, b, b->len - 1);
	return o;
}

static int lower_block(L *l, Block *b, Env *e, Sx *stmts, int is_fn);

static int lower_expr(L *l, Block *b, Env *e, Sx *x)
{
	const char *h = x->v[0]->s;
	if (strcmp(h, "num") == 0) {
		Shape s = { 0 };
		int o = mod_value(l->m, V_TMP, &s, NULL);
		Ins *ins = block_push(b);
		ins->op = OP_CONST;
		ins->out = o;
		ins->k = (float)x->v[1]->n;
		mod_note_def(l->m, o, b, b->len - 1);
		return o;
	}
	if (strcmp(h, "ref") == 0) return lookup(l, e, x->v[1]);

	/* call */
	Sx *fn = x->v[1];
	if (strcmp(fn->s, "grad") == 0) {
		if (x->len != 4) die(l->file, x->line, "'grad' takes (objective, name)");
		Sx *wrt = x->v[3];
		if (!sx_issym(wrt->v[0], "ref")) die(l->file, x->line, "second argument of 'grad' must be a name, not an expression");
		int xv = lookup(l, e, wrt->v[1]);
		int yv = lower_expr(l, b, e, x->v[2]);
		if (l->m->val[yv].sh.rank != 0)
			die(l->file, x->line, "'grad' needs a scalar objective; reduce it with sum(...) or mean(...)");
		return ad_grad(l->m, b, yv, xv, l->file, x->line);
	}
	if (strcmp(fn->s, "reshape") == 0) { /* reshape(x, d0, d1, ...): literal dimensions */
		Shape to = { 0 };
		if (x->len < 4 || x->len - 3 > TG_MAXRANK) die(l->file, x->line, "'reshape' takes a tensor and 1 to %d dimensions", TG_MAXRANK);
		for (int i = 3; i < x->len; i++) {
			Sx *d = x->v[i];
			if (!sx_issym(d->v[0], "num") || d->v[1]->n < 1 || d->v[1]->n != (double)(int)d->v[1]->n)
				die(l->file, x->line, "'reshape' dimensions must be positive integer literals");
			to.dim[to.rank++] = (int)d->v[1]->n;
		}
		int v = lower_expr(l, b, e, x->v[2]);
		if (shape_numel(&to) != shape_numel(&l->m->val[v].sh)) {
			char s0[64], s1[64];
			shape_str(&l->m->val[v].sh, s0, sizeof s0);
			shape_str(&to, s1, sizeof s1);
			die(l->file, x->line, "'reshape' from %s to %s changes the element count", s0, s1);
		}
		return ir_reshape(l->m, b, v, &to);
	}
	int nargs = x->len - 2;
	int args[16];
	if (nargs > 16) die(l->file, x->line, "too many arguments");
	for (int i = 0; i < nargs; i++) args[i] = lower_expr(l, b, e, x->v[i + 2]);

	Sx *def = find_def(l, fn->s);
	if (def) {
		Sx *params = def->v[2];
		if (params->len != nargs)
			die(l->file, x->line, "'%s' takes %d argument(s), got %d", fn->s, params->len, nargs);
		if (++l->depth > 64) die(l->file, x->line, "inlining depth exceeded (recursive def '%s'?)", fn->s);
		Env fe = { 0 };
		for (int i = 0; i < nargs; i++) {
			Shape want = to_shape(params->v[i]->v[1]);
			if (!shape_eq(&want, &l->m->val[args[i]].sh)) {
				char s0[64], s1[64];
				shape_str(&want, s0, sizeof s0);
				shape_str(&l->m->val[args[i]].sh, s1, sizeof s1);
				die(l->file, x->line, "argument %d of '%s': expected %s, got %s", i + 1, fn->s, s0, s1);
			}
			env_set(&fe, params->v[i]->v[0]->s, args[i]);
		}
		int r = lower_block(l, b, &fe, def->v[4], 1);
		Shape rt = to_shape(def->v[3]);
		if (!shape_eq(&rt, &l->m->val[r].sh)) {
			char s0[64], s1[64];
			shape_str(&rt, s0, sizeof s0);
			shape_str(&l->m->val[r].sh, s1, sizeof s1);
			die(l->file, def->line, "'%s' declared to return %s, returns %s", fn->s, s0, s1);
		}
		xfree(fe.b);
		l->depth--;
		return r;
	}

	if (!strcmp(fn->s, "mse") || !strcmp(fn->s, "bce") || !strcmp(fn->s, "xent")) {
		if (nargs != 2) die(l->file, x->line, "'%s' takes (prediction, target)", fn->s);
		if (!shape_eq(&l->m->val[args[0]].sh, &l->m->val[args[1]].sh)) {
			char s0[64], s1[64];
			shape_str(&l->m->val[args[0]].sh, s0, sizeof s0);
			shape_str(&l->m->val[args[1]].sh, s1, sizeof s1);
			die(l->file, x->line, "'%s' prediction %s and target %s differ in shape", fn->s, s0, s1);
		}
		int p = args[0], t = args[1];
#define OP2(k, a, c) ir_op(l->m, b, k, a, c, l->file, x->line)
#define CK(v) ir_k(l->m, b, v)
		if (!strcmp(fn->s, "mse")) { /* mean((p - t)^2) */
			int d = OP2(OP_SUB, p, t);
			return OP2(OP_MEAN, OP2(OP_MUL, d, d), -1);
		}
		if (!strcmp(fn->s, "bce")) { /* -mean(t log p + (1-t) log(1-p)) */
			int lp = OP2(OP_LOG, OP2(OP_ADD, p, CK(1e-7f)), -1);
			int lq = OP2(OP_LOG, OP2(OP_ADD, OP2(OP_SUB, CK(1), p), CK(1e-7f)), -1);
			int s = OP2(OP_ADD, OP2(OP_MUL, t, lp), OP2(OP_MUL, OP2(OP_SUB, CK(1), t), lq));
			return OP2(OP_NEG, OP2(OP_MEAN, s, -1), -1);
		}
		int rk = l->m->val[p].sh.rank;
		if (rk != 1 && rk != 2) die(l->file, x->line, "'xent' needs logits of shape (C) or a batch (B, C), with one-hot targets");
		/* -sum(t * log softmax(logits)), averaged over the rows of a batch */
		int ls = OP2(OP_LOG, OP2(OP_ADD, OP2(OP_SOFTMAX, p, -1), CK(1e-12f)), -1);
		int s = OP2(OP_SUM, OP2(OP_MUL, t, ls), -1);
		if (rk == 2) s = OP2(OP_DIV, s, CK((float)l->m->val[p].sh.dim[0]));
		return OP2(OP_NEG, s, -1);
#undef OP2
#undef CK
	}
	const char *opname = strcmp(fn->s, "dot") == 0 ? "matmul" : fn->s;
	int op = op_lookup(opname);
	if (op < 0) die(l->file, fn->line, "unknown function '%s'", fn->s);
	return emit_op(l, b, (Op)op, args, nargs, x->line);
}

/* scan C1, C2 over X1 in E1, X2 in E2: body
 * Carries must be bound before the loop and keep their shapes; after it they
 * hold the final values. Each `emit N = v` in the body binds N to the (T, ...)
 * stack of v over the steps. */
static void lower_scan(L *l, Block *b, Env *e, Sx *s)
{
	Module *m = l->m;
	Sx *cs = s->v[1], *xs = s->v[2];
	int T = -1, nc = cs->len, nx = xs->len;
	int *seq = xmalloc((size_t)nx * sizeof *seq);
	for (int j = 0; j < nx; j++) {
		seq[j] = lower_expr(l, b, e, xs->v[j]->v[1]);
		const Shape *q = &m->val[seq[j]].sh;
		if (q->rank < 1) die(l->file, s->line, "scan sequence '%s' must have rank >= 1 (steps along the first axis)", xs->v[j]->v[0]->s);
		if (T >= 0 && q->dim[0] != T)
			die(l->file, s->line, "scan sequences differ in length: %d and %d steps", T, q->dim[0]);
		T = q->dim[0];
	}
	Scan *sc = scan_new(T, 0);
	for (int k = 0; k < nc; k++) {
		const char *n = cs->v[k]->s;
		for (int j = 0; j < k; j++)
			if (!strcmp(cs->v[j]->s, n)) die(l->file, s->line, "scan carry '%s' listed twice", n);
		int init = env_get(e, n);
		if (init < 0) {
			if (env_get(&l->globals, n) >= 0) die(l->file, s->line, "scan carry '%s' is a parameter", n);
			die(l->file, s->line, "scan carry '%s' must be assigned before the loop", n);
		}
		Shape sh = m->val[init].sh;
		scan_carry(sc, mod_value(m, V_TMP, &sh, NULL), init, -1);
	}
	for (int j = 0; j < nx; j++) {
		Shape q = m->val[seq[j]].sh;
		for (int d = 1; d < q.rank; d++) q.dim[d - 1] = q.dim[d];
		q.rank--;
		scan_seq(sc, seq[j], mod_value(m, V_TMP, &q, NULL));
	}
	Ins *ins = block_push(b);
	int idx = (int)(ins - b->v);
	ins->op = OP_SCAN;
	ins->sc = sc;
	ins->out = sc->c[0];
	for (int k = 0; k < nc; k++) mod_note_def(m, sc->c[k], b, idx);
	Block *body = xmalloc(sizeof *body);

	Env inner = env_copy(e);
	for (int k = 0; k < nc; k++) env_set(&inner, cs->v[k]->s, sc->c[k]);
	for (int j = 0; j < nx; j++) env_set(&inner, xs->v[j]->v[0]->s, sc->xt[j]);
	int outer_n = e->n;
	const char *kw = l->loop_kw;
	Emits em = { l->in_think + 1, NULL, NULL, 0 }, *save = l->emits;
	l->loop_kw = "scan";
	l->in_think++;
	l->emits = &em;
	m->open_think = xrealloc(m->open_think, (size_t)(m->nopen + nc + nx) * sizeof *m->open_think);
	for (int k = 0; k < nc; k++) m->open_think[m->nopen++] = sc->c[k]; /* the body sees carries and slices as inputs */
	for (int j = 0; j < nx; j++) m->open_think[m->nopen++] = sc->xt[j];
	lower_block(l, body, &inner, s->v[3], 0);
	m->nopen -= nc + nx;
	l->in_think--;
	l->loop_kw = kw;
	l->emits = save;
	for (int j = 0; j < outer_n; j++) {
		const char *n = e->b[j].name;
		int carried = 0;
		for (int k = 0; k < nc; k++) carried |= !strcmp(n, cs->v[k]->s);
		for (int j2 = 0; j2 < nx; j2++) carried |= !strcmp(n, xs->v[j2]->v[0]->s);
		if (!carried && inner.b[j].val != e->b[j].val)
			die(l->file, s->line, "scan block assigns outer variable '%s'; only the carries are carried (use emit to collect per-step values)", n);
	}
	for (int k = 0; k < nc; k++) {
		int nv = env_get(&inner, cs->v[k]->s);
		if (!shape_eq(&m->val[nv].sh, &m->val[sc->c[k]].sh)) {
			char s0[64], s1[64];
			shape_str(&m->val[sc->c[k]].sh, s0, sizeof s0);
			shape_str(&m->val[nv].sh, s1, sizeof s1);
			die(l->file, s->line, "scan carry '%s' changes shape from %s to %s", cs->v[k]->s, s0, s1);
		}
		for (int j = 0; j < nc; j++) /* another carry's value: copy it, so carries can advance in any order */
			if (j != k && nv == sc->c[j]) nv = ir_op(m, body, OP_MUL, nv, ir_k(m, body, 1), l->file, s->line);
		sc->next[k] = nv;
	}
	for (int i = 0; i < em.n; i++) {
		Shape q = m->val[em.val[i]].sh;
		if (q.rank == TG_MAXRANK) die(l->file, s->line, "emit '%s': a stack of rank-%d values exceeds rank %d", em.name[i], q.rank, TG_MAXRANK);
		for (int d = q.rank; d > 0; d--) q.dim[d] = q.dim[d - 1];
		q.dim[0] = T;
		q.rank++;
		int ys = mod_value(m, V_TMP, &q, NULL);
		scan_stack(sc, em.val[i], ys);
		mod_note_def(m, ys, b, idx);
	}
	xfree(inner.b);
	b->v[idx].body = body; /* block may have been reallocated */
	for (int k = 0; k < nc; k++) env_set(e, cs->v[k]->s, sc->c[k]);
	for (int i = 0; i < em.n; i++) env_set(e, em.name[i], sc->ys[i]);
	xfree(em.name);
	xfree(em.val);
	xfree(seq);
}

/* Returns the value of the `return` statement for function bodies, or -1. */
static int lower_block(L *l, Block *b, Env *e, Sx *stmts, int is_fn)
{
	for (int i = 0; i < stmts->len; i++) {
		Sx *s = stmts->v[i];
		const char *h = s->v[0]->s;
		if (strcmp(h, "return") == 0) {
			if (!is_fn) die(l->file, s->line, "'return' inside %s block", l->loop_kw);
			if (i != stmts->len - 1) die(l->file, stmts->v[i + 1]->line, "unreachable statement after return");
			return lower_expr(l, b, e, s->v[1]);
		}
		if (strcmp(h, "set") == 0) {
			int v = lower_expr(l, b, e, s->v[2]);
			const char *n = s->v[1]->s;
			int g = env_get(&l->globals, n);
			if (g >= 0 && l->m->val[g].kind == V_STATE)
				die(l->file, s->line, "cannot assign to state '%s'; use 'update %s = ...'", n, n);
			if (g >= 0) die(l->file, s->line, "cannot assign to parameter '%s'", n);
			env_set(e, n, v);
			continue;
		}
		if (strcmp(h, "train") == 0) {
			if (l->depth > 0) die(l->file, s->line, "'train' is only allowed in the body of 'forward', not in a def");
			if (l->in_think) die(l->file, s->line, "'train' inside %s block", l->loop_kw);
			int y = lower_expr(l, b, e, s->v[1]);
			if (l->m->val[y].sh.rank != 0) die(l->file, s->line, "'train' needs a scalar loss; reduce it with sum(...) or mean(...)");
			Sx *kv = s->v[3], *ov = s->v[4];
			const char **keys = xmalloc((size_t)(kv->len + 1) * sizeof *keys);
			double *vals = xmalloc((size_t)(kv->len + 1) * sizeof *vals);
			for (int k = 0; k < kv->len; k++) {
				keys[k] = kv->v[k]->v[0]->s;
				vals[k] = kv->v[k]->v[1]->n;
			}
			int *over = xmalloc((size_t)(ov->len + 1) * sizeof *over);
			for (int k = 0; k < ov->len; k++) {
				int sv = env_get(&l->globals, ov->v[k]->s);
				if (sv < 0 || l->m->val[sv].kind != V_STATE) die(l->file, s->line, "'%s' in 'over' is not a state", ov->v[k]->s);
				over[k] = sv;
			}
			optim_train(l->m, b, y, s->v[2]->s, keys, vals, kv->len, over, ov->len, l->file, s->line);
			xfree(keys);
			xfree(vals);
			xfree(over);
			continue;
		}
		if (strcmp(h, "update") == 0) {
			const char *n = s->v[1]->s;
			if (l->depth > 0) die(l->file, s->line, "'update' is only allowed in the body of 'forward', not in a def");
			if (l->in_think) die(l->file, s->line, "'update' inside %s block (a state changes once per run)", l->loop_kw);
			int sv = env_get(&l->globals, n);
			if (sv >= 0 && l->m->val[sv].kind == V_PARAM)
				die(l->file, s->line, "'%s' is a read-only param; declare it with 'state' to update it", n);
			if (sv < 0 || l->m->val[sv].kind != V_STATE) die(l->file, s->line, "'%s' is not a state", n);
			for (int k = 0; k < l->m->nupd; k++)
				if (l->m->upd_state[k] == sv) die(l->file, s->line, "state '%s' updated twice in one run", n);
			int rows = s->len > 3 ? lower_expr(l, b, e, s->v[3]) : -1; /* update NAME[rows] = v */
			int v = lower_expr(l, b, e, s->v[2]);
			char err[300];
			if (!upd_check(l->m, sv, v, rows, err, sizeof err)) die(l->file, s->line, "%s", err);
			mod_update_rows(l->m, sv, v, rows);
			continue;
		}
		if (strcmp(h, "think") == 0) {
			Sx *st = s->v[1];
			int init = env_get(e, st->s);
			if (init < 0) {
				if (env_get(&l->globals, st->s) >= 0) die(l->file, s->line, "think state '%s' is a parameter", st->s);
				die(l->file, s->line, "think state '%s' must be assigned before the loop", st->s);
			}
			Shape sh = l->m->val[init].sh;
			int state = mod_value(l->m, V_TMP, &sh, NULL);
			Ins *ins = block_push(b);
			int idx = (int)(ins - b->v);
			ins->op = OP_THINK;
			ins->out = state;
			ins->init = init;
			ins->maxit = (int)s->v[2]->n;
			ins->eps = (float)s->v[3]->n;
			ins->tid = l->m->nthink++;
			mod_note_def(l->m, state, b, idx);
			Block *body = xmalloc(sizeof *body);

			Env inner = env_copy(e);
			env_set(&inner, st->s, state);
			int outer_n = e->n;
			const char *kw = l->loop_kw;
			l->loop_kw = "think";
			l->in_think++;
			l->m->open_think = xrealloc(l->m->open_think, (size_t)(l->m->nopen + 1) * sizeof *l->m->open_think);
			l->m->open_think[l->m->nopen++] = state; /* its body sees the state as an input */
			lower_block(l, body, &inner, s->v[4], 0);
			l->m->nopen--;
			l->in_think--;
			l->loop_kw = kw;
			/* Only the state may be carried: reject assignments to other outer names. */
			for (int j = 0; j < outer_n; j++) {
				const char *n = e->b[j].name;
				if (strcmp(n, st->s) != 0 && inner.b[j].val != e->b[j].val)
					die(l->file, s->line, "think block assigns outer variable '%s'; only the state '%s' is carried", n, st->s);
			}
			int y = env_get(&inner, st->s);
			xfree(inner.b);
			ins = &b->v[idx]; /* block may have been reallocated */
			ins->body = body;
			ins->yield = y;
			env_set(e, st->s, state);
			continue;
		}
		if (strcmp(h, "emit") == 0) {
			Emits *em = l->emits;
			if (!em || em->level != l->in_think) die(l->file, s->line, "'emit' is only allowed directly in a scan body");
			const char *n = s->v[1]->s;
			for (int k = 0; k < em->n; k++)
				if (!strcmp(em->name[k], n)) die(l->file, s->line, "'%s' emitted twice in one scan", n);
			int v = lower_expr(l, b, e, s->v[2]);
			em->name = xrealloc(em->name, (size_t)(em->n + 1) * sizeof *em->name);
			em->val = xrealloc(em->val, (size_t)(em->n + 1) * sizeof *em->val);
			em->name[em->n] = n;
			em->val[em->n++] = v;
			continue;
		}
		if (strcmp(h, "scan") == 0) {
			lower_scan(l, b, e, s);
			continue;
		}
		die(l->file, s->line, "bad statement '%s'", h);
	}
	if (is_fn) die(l->file, stmts->line, "function body has no 'return'");
	return -1;
}

static uint32_t xorshift(uint32_t *s)
{
	uint32_t x = *s;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return *s = x;
}

static char *dirname_of(const char *path)
{
	const char *slash = strrchr(path, '/');
	if (!slash) return xstrdup(".");
	size_t n = (size_t)(slash - path);
	char *d = xmalloc(n + 1);
	memcpy(d, path, n);
	return d;
}

static char *path_join(const char *dir, const char *rel)
{
	if (rel[0] == '/') return xstrdup(rel);
	char *full = xmalloc(strlen(dir) + strlen(rel) + 2);
	sprintf(full, "%s/%s", dir, rel);
	return full;
}

/* Recursively merge `import "path"` declarations into one flat AST.
 *
 * The root file contributes everything. An imported file is a library: only
 * its `param`s and its non-`forward` `def`s are pulled in, so the same file
 * can be run on its own and imported elsewhere. Imports are idempotent and
 * their resolved paths are tracked, so diamonds and cycles terminate. */
typedef struct {
	char **p;
	int n, cap;
} Seen;

static int seen_add(Seen *s, const char *path)
{
	for (int i = 0; i < s->n; i++)
		if (strcmp(s->p[i], path) == 0) return 0;
	if (s->n == s->cap) s->p = xrealloc(s->p, (size_t)(s->cap = s->cap ? s->cap * 2 : 8) * sizeof *s->p);
	s->p[s->n++] = xstrdup(path);
	return 1;
}

static void load_into(Sx *out, const char *path, Seen *seen, int depth, int root, const char *importer, int iline)
{
	if (depth > 64) die(importer, iline, "import nesting too deep (cycle involving '%s'?)", path);
	if (!seen_add(seen, path)) return; /* already merged */
	FILE *probe = fopen(path, "rb");
	if (!probe) {
		if (importer) die(importer, iline, "cannot open import '%s'", path);
		die(NULL, 0, "cannot open '%s'", path);
	}
	fclose(probe);
	char *src = read_file(path, NULL);
	Sx *ast = surface_parse(src, path);
	xfree(src);
	if (tr_on(2)) {
		tr_begin(2, "import");
		tr_str("file", path);
		tr_num("depth", depth);
		tr_str("role", root ? "root" : "library");
		tr_end("%s %s (depth %d)", root ? "root" : "library", path, depth);
	}
	for (int i = 0; i < ast->len; i++) {
		Sx *d = ast->v[i];
		const char *h = d->v[0]->s;
		if (strcmp(h, "import") == 0) {
			char *dir = dirname_of(path);
			char *full = path_join(dir, d->v[1]->s);
			xfree(dir);
			load_into(out, full, seen, depth + 1, 0, path, d->line);
			xfree(full);
		} else if (root) {
			sx_push(out, d);
		} else if (strcmp(h, "param") == 0 || strcmp(h, "state") == 0 || (strcmp(h, "def") == 0 && strcmp(d->v[1]->s, "forward") != 0)) {
			sx_push(out, d); /* library: skip the imported file's model and its forward */
		}
	}
}

Sx *surface_load(const char *path)
{
	Seen seen = { 0 };
	Sx *merged = sx_new(SX_LIST, 1);
	load_into(merged, path, &seen, 0, 1, NULL, 0);
	for (int i = 0; i < seen.n; i++) xfree(seen.p[i]);
	xfree(seen.p);
	return merged;
}

static float *materialize(L *l, Sx *init, const Shape *sh, const char *pname)
{
	int n = shape_numel(sh);
	float *d = xmalloc((size_t)n * sizeof *d);
	const char *k = init->v[0]->s;
	if (strcmp(k, "zeros") == 0) return d;
	if (strcmp(k, "ones") == 0 || strcmp(k, "fill") == 0) {
		float v = strcmp(k, "ones") == 0 ? 1.0f : (float)init->v[1]->n;
		for (int i = 0; i < n; i++) d[i] = v;
		return d;
	}
	if (strcmp(k, "rand") == 0) {
		uint32_t s = (uint32_t)(int64_t)init->v[1]->n;
		if (!s) s = 0x9e3779b9u;
		float scale = (float)init->v[2]->n;
		for (int i = 0; i < n; i++) {
			float u = (float)(xorshift(&s) >> 8) / 16777216.0f;
			d[i] = (2.0f * u - 1.0f) * scale;
		}
		return d;
	}
	if (strcmp(k, "data") == 0) {
		if (init->len - 1 != n)
			die(l->file, init->line, "param '%s' needs %d values, literal has %d", pname, n, init->len - 1);
		for (int i = 0; i < n; i++) d[i] = (float)init->v[i + 1]->n;
		return d;
	}
	if (strcmp(k, "file") == 0) {
		const char *p = init->v[1]->s;
		char *full;
		if (p[0] == '/') full = xstrdup(p);
		else {
			char *dir = dirname_of(l->file);
			full = xmalloc(strlen(dir) + strlen(p) + 2);
			sprintf(full, "%s/%s", dir, p);
			xfree(dir);
		}
		size_t len;
		char *raw = read_file(full, &len);
		if (len != (size_t)n * 4)
			die(l->file, init->line, "'%s' has %zu bytes, param '%s' needs %d (raw little-endian f32)", full, len, pname, n * 4);
		const unsigned char *b = (const unsigned char *)raw;
		for (int i = 0; i < n; i++) {
			uint32_t w = (uint32_t)b[4 * i] | (uint32_t)b[4 * i + 1] << 8 | (uint32_t)b[4 * i + 2] << 16 | (uint32_t)b[4 * i + 3] << 24;
			memcpy(&d[i], &w, 4);
		}
		xfree(raw);
		xfree(full);
		return d;
	}
	die(l->file, init->line, "bad initializer '%s'", k);
	return NULL;
}

/* Dead-code elimination. Roots are the output and the update sources; a think
 * loop that is live keeps its init and whatever its yield needs. A scan keeps
 * the carries that are read after the loop or by its live body, and the
 * stacks read after it; the rest are removed from the scan. Instructions are
 * pure, so anything unreachable from a root is removed. Compiler-generated
 * params (autodiff constants) that end up unreferenced are marked dead so the
 * writers skip them. */
static void mark_block(const Block *b, char *live);

/* Liveness of one scan; returns whether any output is live. */
static int mark_scan(const Ins *in, char *live)
{
	const Scan *s = in->sc;
	int any = 0;
	for (int m = 0; m < s->ny; m++)
		if (live[s->ys[m]]) { live[s->y[m]] = 1; any = 1; }
	for (int changed = 1; changed;) { /* a carry read by the live body is needed too */
		mark_block(in->body, live);
		changed = 0;
		for (int k = 0; k < s->nc; k++)
			if (live[s->c[k]] && !live[s->next[k]]) { live[s->next[k]] = 1; changed = 1; }
	}
	for (int k = 0; k < s->nc; k++)
		if (live[s->c[k]]) { live[s->init[k]] = 1; any = 1; }
	if (any)
		for (int j = 0; j < s->nx; j++) live[s->x[j]] = 1;
	return any;
}

static void mark_block(const Block *b, char *live)
{
	for (int i = b->len - 1; i >= 0; i--) {
		const Ins *in = &b->v[i];
		if (in->op == OP_SCAN) {
			mark_scan(in, live);
		} else if (live[in->out]) {
			if (in->op == OP_THINK) {
				live[in->init] = 1;
				live[in->yield] = 1;
				mark_block(in->body, live);
			} else {
				for (int j = 0; j < in->na; j++) live[in->a[j]] = 1;
			}
		}
	}
}

static void dce_block(Block *b, char *live)
{
	mark_block(b, live);
	int k = 0;
	for (int i = 0; i < b->len; i++) {
		Ins *in = &b->v[i];
		if (in->op == OP_SCAN) {
			Scan *s = in->sc;
			int nc = 0, ny = 0;
			for (int c = 0; c < s->nc; c++)
				if (live[s->c[c]]) { s->c[nc] = s->c[c]; s->init[nc] = s->init[c]; s->next[nc++] = s->next[c]; }
			for (int m = 0; m < s->ny; m++)
				if (live[s->ys[m]]) { s->y[ny] = s->y[m]; s->ys[ny++] = s->ys[m]; }
			s->nc = nc;
			s->ny = ny;
			if (!nc && !ny) continue;
			in->out = nc ? s->c[0] : s->ys[0];
			dce_block(in->body, live);
		} else {
			if (!live[in->out]) continue;
			if (in->op == OP_THINK) dce_block(in->body, live);
		}
		b->v[k++] = *in;
	}
	b->len = k;
}

static void dce(Module *m)
{
	char *live = xmalloc((size_t)m->nval);
	live[m->output] = 1;
	for (int i = 0; i < m->nupd; i++) {
		live[m->upd_src[i]] = 1;
		if (m->upd_rows[i] >= 0) live[m->upd_rows[i]] = 1;
	}
	dce_block(&m->top, live);
	for (int v = 0; v < m->nval; v++)
		if (m->val[v].kind == V_PARAM && m->val[v].name && strncmp(m->val[v].name, "__", 2) == 0 && !live[v]) m->val[v].dead = 1;
	xfree(live);
	m->ndef = 0; /* the lowering tape is invalid after compaction */
}

Module *lower(Sx *ast, const char *file)
{
	L l = { 0 };
	l.file = file;
	const char *name = NULL;
	Sx *entry = NULL;

	for (int i = 0; i < ast->len; i++) {
		Sx *d = ast->v[i];
		if (sx_issym(d->v[0], "model")) {
			if (name) die(file, d->line, "duplicate 'model' declaration");
			name = d->v[1]->s;
		}
	}
	if (!name) die(file, 1, "missing 'model <name>' declaration");
	l.m = mod_new(name);

	for (int i = 0; i < ast->len; i++) {
		Sx *d = ast->v[i];
		if (sx_issym(d->v[0], "param") || sx_issym(d->v[0], "state")) {
			const char *pn = d->v[1]->s;
			if (env_get(&l.globals, pn) >= 0) die(file, d->line, "duplicate param or state '%s'", pn);
			Shape sh = to_shape(d->v[2]);
			int v = mod_value(l.m, sx_issym(d->v[0], "state") ? V_STATE : V_PARAM, &sh, pn);
			l.m->val[v].data = materialize(&l, d->v[3], &sh, pn);
			env_set(&l.globals, pn, v);
		} else if (sx_issym(d->v[0], "def")) {
			const char *dn = d->v[1]->s;
			if (find_def(&l, dn)) die(file, d->line, "duplicate def '%s'", dn);
			if (op_lookup(dn) >= 0 || strcmp(dn, "dot") == 0 || strcmp(dn, "grad") == 0 || !strcmp(dn, "mse") || !strcmp(dn, "bce") || !strcmp(dn, "xent")) die(file, d->line, "'%s' shadows a builtin", dn);
			l.defs = xrealloc(l.defs, (size_t)(l.ndefs + 1) * sizeof *l.defs);
			l.defs[l.ndefs++] = d;
			if (strcmp(dn, "forward") == 0) entry = d;
		}
	}
	for (int i = 0; i < l.ndefs; i++) {
		Sx *ps = l.defs[i]->v[2];
		for (int j = 0; j < ps->len; j++)
			if (env_get(&l.globals, ps->v[j]->v[0]->s) >= 0)
				die(file, ps->v[j]->line, "argument '%s' shadows a param", ps->v[j]->v[0]->s);
	}
	if (!entry) die(file, 1, "missing entry point 'def forward(...)'");

	Env e = { 0 };
	Sx *ps = entry->v[2];
	if (ps->len == 0) die(file, entry->line, "'forward' needs at least one input");
	for (int i = 0; i < ps->len; i++) {
		Shape sh = to_shape(ps->v[i]->v[1]);
		const char *an = ps->v[i]->v[0]->s;
		if (env_get(&e, an) >= 0) die(file, ps->v[i]->line, "duplicate input '%s'", an);
		env_set(&e, an, mod_value(l.m, V_INPUT, &sh, an));
	}
	int r = lower_block(&l, &l.m->top, &e, entry->v[4], 1);
	Shape rt = to_shape(entry->v[3]);
	if (!shape_eq(&rt, &l.m->val[r].sh)) die(file, entry->line, "'forward' return shape does not match its declaration");
	l.m->output = r;
	dce(l.m);
	xfree(e.b);
	return l.m;
}
