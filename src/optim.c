/* `train LOSS [with OPT(k=v, ...)] [over w, ...]` expansion.
 *
 * Gradients come from one shared backward pass (autodiff.c). Optimizer state
 * (moments, step counters) becomes hidden `state` named __opt_*, and every
 * step becomes ordinary `update`s, so training compiles everywhere models do.
 *
 *   sgd        w -= lr g
 *   momentum   m = mu m + g;  w -= lr m
 *   adam       Adam with bias correction (Kingma & Ba)
 *   adamw      Adam + decoupled weight decay (Loshchilov & Hutter)
 *   muon       matrices: Nesterov momentum orthogonalized by 5 quintic
 *              Newton-Schulz steps, scaled by 0.2*sqrt(max(rows, cols)) so it
 *              shares AdamW's learning rate (arXiv:2502.16982); vectors and
 *              scalars: AdamW
 *
 * clip=c rescales all gradients so their global L2 norm is at most c.
 *
 * Row-sparse (lazy) steps: a weight used only as the table of spmm has a
 * gradient that is zero outside the rows the batch touched. Its step then
 * runs on those rows alone: the row set r = active(s, w), the compact
 * gradient spmm_tc(s, g, r), take() of the weight and its moments, and row
 * updates `w[r] = ...`. Cost O(|r| H) instead of O(D H).
 *
 * Catch-up: each table row records the step it was last updated
 * (__opt_last_<w>). When a row comes back after k skipped steps, the k steps
 * the dense optimizer would have taken with a zero gradient are applied in
 * closed form first:
 *   momentum  w -= lr m mu (1 - mu^k) / (1 - mu);  m *= mu^k        (exact)
 *   adam(w)   w -= lr m^/sqrt(v^) q (1 - q^k) / (1 - q), q = b1/sqrt(b2);
 *             m *= b1^k; v *= b2^k; w *= (1 - lr wd)^k
 * (the Adam drift holds the bias corrections at the current step and
 * neglects eps). So a row's trajectory follows the dense one, deferred: the
 * forward pass sees a row as of its last update. For sgd the lazy step
 * equals the dense step exactly.
 * Muon orthogonalizes whole matrices, so its sparse tables take the AdamW
 * step, as Muon does for embeddings. lazy=0 forces dense steps. */
#include "tg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	Module *m;
	Block *b;
	const char *file;
	int line;
	const char **keys;
	const double *vals;
	int nkv;
	char *used;
} O;

static int op(O *o, Op k, int p, int q) { return ir_op(o->m, o->b, k, p, q, o->file, o->line); }
static int K(O *o, float k) { return ir_k(o->m, o->b, k); }
#define ADD(a, b) op(o, OP_ADD, a, b)
#define SUB(a, b) op(o, OP_SUB, a, b)
#define MUL(a, b) op(o, OP_MUL, a, b)
#define DIV(a, b) op(o, OP_DIV, a, b)

static double opt(O *o, const char *key, double dflt)
{
	for (int i = 0; i < o->nkv; i++)
		if (strcmp(o->keys[i], key) == 0) {
			o->used[i] = 1;
			return o->vals[i];
		}
	return dflt;
}

static int hidden(O *o, const char *what, int w)
{
	char name[160];
	snprintf(name, sizeof name, "__opt_%s_%s", what, o->m->val[w].name);
	for (int v = 0; v < o->m->nval; v++)
		if (o->m->val[v].name && strcmp(o->m->val[v].name, name) == 0) die(o->file, o->line, "state \x27%s\x27 is trained twice", o->m->val[w].name);
	int v = mod_value(o->m, V_STATE, &o->m->val[w].sh, name);
	o->m->val[v].data = xmalloc((size_t)shape_numel(&o->m->val[w].sh) * sizeof(float));
	return v;
}

static void set_rows(O *o, int state, int value, int rows)
{
	for (int k = 0; k < o->m->nupd; k++)
		if (o->m->upd_state[k] == state) die(o->file, o->line, "state \x27%s\x27 updated twice in one run", o->m->val[state].name);
	mod_update_rows(o->m, state, value, rows);
}

static void set(O *o, int state, int value) { set_rows(o, state, value, -1); }

/* The spmm_t instruction that is the whole gradient of table w, or NULL. */
static const Ins *sparse_grad(Module *m, int g, int w)
{
	if (g >= m->ndef || !m->def_blk[g]) return NULL;
	const Ins *in = &m->def_blk[g]->v[m->def_idx[g]];
	return in->op == OP_SPMM_T && in->a[2] == w ? in : NULL;
}

/* Orthogonalize a matrix: X <- a X + (b A + c A^2) X with A = X X^T, 5 times. */
static int newton_schulz(O *o, int g)
{
	const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
	const Shape *s = &o->m->val[g].sh;
	int tall = s->dim[0] > s->dim[1];
	int x = tall ? op(o, OP_TRANSPOSE, g, -1) : g;
	int nrm = op(o, OP_SQRT, op(o, OP_SUM, MUL(x, x), -1), -1);
	x = DIV(x, ADD(nrm, K(o, 1e-7f)));
	for (int i = 0; i < 5; i++) {
		int A = op(o, OP_MATMUL, x, op(o, OP_TRANSPOSE, x, -1));
		int B = ADD(MUL(K(o, b), A), MUL(K(o, c), op(o, OP_MATMUL, A, A)));
		x = ADD(MUL(K(o, a), x), op(o, OP_MATMUL, B, x));
	}
	return tall ? op(o, OP_TRANSPOSE, x, -1) : x;
}

void optim_train(Module *m, Block *b, int loss, const char *name, const char **keys, const double *vals, int nkv,
		 const int *over, int nover, const char *file, int line)
{
	O oo = { m, b, file, line, keys, vals, nkv, xmalloc((size_t)(nkv ? nkv : 1)) };
	O *o = &oo;
	if (strcmp(name, "default") == 0) name = "adamw";
	int kind = !strcmp(name, "sgd") ? 0 : !strcmp(name, "momentum") ? 1 : !strcmp(name, "adam") ? 2
		: !strcmp(name, "adamw") ? 3 : !strcmp(name, "muon") ? 4 : -1;
	if (kind < 0) die(file, line, "unknown optimizer \x27%s\x27 (sgd, momentum, adam, adamw, muon)", name);

	double lr = opt(o, "lr", kind <= 1 ? 0.01 : 0.001);
	double mu = opt(o, "momentum", kind == 4 ? 0.95 : 0.9);
	double b1 = opt(o, "beta1", 0.9), b2 = opt(o, "beta2", 0.999), eps = opt(o, "eps", 1e-8);
	double wd = opt(o, "wd", kind >= 3 ? 0.01 : 0.0);
	double clip = opt(o, "clip", 0.0);
	int lazy = opt(o, "lazy", 1.0) != 0;
	for (int i = 0; i < nkv; i++)
		if (!o->used[i]) die(file, line, "unknown option \x27%s\x27 for %s (lr, momentum, beta1, beta2, eps, wd, clip, lazy)", keys[i], name);

	/* weights: the listed states, or every state the loss depends on */
	int *w = xmalloc((size_t)(m->nval + 1) * sizeof *w), nw = 0;
	if (nover) {
		for (int i = 0; i < nover; i++) w[nw++] = over[i];
	} else {
		for (int v = 0; v < m->nval; v++) {
			const Value *x = &m->val[v];
			if (x->kind != V_STATE || (x->name && strncmp(x->name, "__", 2) == 0)) continue;
			int updated = 0;
			for (int k = 0; k < m->nupd; k++) updated |= m->upd_state[k] == v;
			if (!updated && ad_depends(m, b, loss, v, file, line)) w[nw++] = v;
		}
		if (!nw) die(file, line, "train: the loss depends on no trainable state; declare the weights with \x27state\x27");
	}

	int *g = xmalloc((size_t)nw * sizeof *g), *rows = xmalloc((size_t)nw * sizeof *rows), nsparse = 0;
	for (int i = 0; i < nw; i++) {
		g[i] = ad_grad(m, b, loss, w[i], file, line);
		rows[i] = -1;
		const Ins *sp = lazy ? sparse_grad(m, g[i], w[i]) : NULL;
		if (sp) { /* compact gradient over the touched rows; the dense one is left to DCE */
			int s = sp->a[0], go = sp->a[1];
			rows[i] = op(o, OP_ACTIVE, s, w[i]);
			g[i] = ir_op3(m, b, OP_SPMM_TC, s, go, rows[i], file, line);
			nsparse++;
		}
	}

	if (clip > 0) { /* global-norm clipping */
		int n2 = -1;
		for (int i = 0; i < nw; i++) {
			int s = op(o, OP_SUM, MUL(g[i], g[i]), -1);
			n2 = n2 < 0 ? s : ADD(n2, s);
		}
		int scale = op(o, OP_MIN, K(o, 1), DIV(K(o, (float)clip), op(o, OP_SQRT, ADD(n2, K(o, 1e-12f)), -1)));
		for (int i = 0; i < nw; i++) g[i] = MUL(g[i], scale);
	}

	/* one step counter per train statement for bias correction */
	int t1 = -1, bc1 = -1, bc2 = -1;
	if (kind >= 2 || (kind == 1 && nsparse)) {
		char tn[32];
		snprintf(tn, sizeof tn, "__opt_t%d", m->nupd);
		Shape sc = { 0 };
		int t = mod_value(m, V_STATE, &sc, tn);
		m->val[t].data = xmalloc(sizeof(float));
		t1 = ADD(t, K(o, 1));
		set(o, t, t1);
		if (kind >= 2) {
			bc1 = SUB(K(o, 1), op(o, OP_EXP, MUL(t1, K(o, (float)log(b1))), -1));
			bc2 = SUB(K(o, 1), op(o, OP_EXP, MUL(t1, K(o, (float)log(b2))), -1));
		}
	}

	for (int i = 0; i < nw; i++) {
		int gi = g[i], next, r = rows[i];
		int wst = w[i], wv = r < 0 ? wst : op(o, OP_TAKE, wst, r); /* the weight, or its touched rows */
		const Shape sv = m->val[wst].sh, *s = &sv; /* a copy: emission reallocates m->val */
#define TAKE(h) (r < 0 ? (h) : op(o, OP_TAKE, (h), r))
		/* catch-up factor base^k per touched row, k = steps since its last update, as (n, H) */
		int skipped = -1, ones_h = -1;
		if (r >= 0 && kind >= 1) {
			char ln[160];
			snprintf(ln, sizeof ln, "__opt_last_%s", m->val[wst].name);
			Shape d = { 1, { s->dim[0] } }, h = { 1, { s->dim[1] } };
			int last = mod_value(m, V_STATE, &d, ln);
			m->val[last].data = xmalloc((size_t)s->dim[0] * sizeof(float));
			int lr_ = op(o, OP_TAKE, last, r);
			skipped = SUB(SUB(t1, K(o, 1)), lr_);
			set_rows(o, last, ADD(MUL(lr_, K(o, 0)), t1), r);
			char on[64];
			snprintf(on, sizeof on, "__ones_%d", s->dim[1]); /* shared with autodiff's constants */
			for (int v = 0; v < m->nval; v++)
				if (m->val[v].kind == V_PARAM && m->val[v].name && !strcmp(m->val[v].name, on)) ones_h = v;
			if (ones_h < 0) {
				ones_h = mod_value(m, V_PARAM, &h, on);
				m->val[ones_h].data = xmalloc((size_t)s->dim[1] * sizeof(float));
				for (int j = 0; j < s->dim[1]; j++) m->val[ones_h].data[j] = 1.0f;
			}
		}
#define CATCH(x, base) (skipped < 0 || (base) == 1.0 ? (x) \
	: MUL((x), op(o, OP_OUTER, op(o, OP_EXP, MUL(skipped, K(o, (float)log(base))), -1), ones_h)))
		if (kind == 0) {
			next = SUB(wv, MUL(K(o, (float)lr), gi));
		} else if (kind == 1) {
			int mm = hidden(o, "m", wst);
			if (skipped >= 0) { /* the k skipped steps moved the row by lr m sum_j mu^j */
				int geo = MUL(K(o, (float)(mu / (1 - mu))), SUB(K(o, 1), op(o, OP_EXP, MUL(skipped, K(o, (float)log(mu))), -1)));
				wv = SUB(wv, MUL(K(o, (float)lr), MUL(TAKE(mm), op(o, OP_OUTER, geo, ones_h))));
			}
			int m1 = ADD(MUL(K(o, (float)mu), CATCH(TAKE(mm), mu)), gi);
			set_rows(o, mm, m1, r);
			next = SUB(wv, MUL(K(o, (float)lr), m1));
		} else if (kind == 4 && s->rank == 2 && r < 0) {
			int mm = hidden(o, "m", wv);
			int m1 = ADD(MUL(K(o, (float)mu), mm), gi);
			set(o, mm, m1);
			int u = ADD(gi, MUL(K(o, (float)mu), m1)); /* Nesterov */
			float scale = 0.2f * sqrtf((float)(s->dim[0] > s->dim[1] ? s->dim[0] : s->dim[1]));
			int dir = ADD(MUL(K(o, scale), newton_schulz(o, u)), MUL(K(o, (float)wd), wv));
			next = SUB(wv, MUL(K(o, (float)lr), dir));
		} else { /* adam, adamw, and muon's non-matrix and sparse weights */
			int mm = hidden(o, "m", wst), vv = hidden(o, "v", wst);
			int m1 = ADD(MUL(K(o, (float)b1), CATCH(TAKE(mm), b1)), MUL(K(o, (float)(1 - b1)), gi));
			int v1 = ADD(MUL(K(o, (float)b2), CATCH(TAKE(vv), b2)), MUL(K(o, (float)(1 - b2)), MUL(gi, gi)));
			if (skipped >= 0) { /* the k skipped Adam steps moved the row by lr m/sqrt(v) sum_j q^j, q = b1/sqrt(b2) */
				double q = b1 / sqrt(b2);
				int mo = TAKE(mm), vo = TAKE(vv);
				int ratio = DIV(DIV(mo, bc1), ADD(op(o, OP_SQRT, DIV(vo, bc2), -1), K(o, (float)eps)));
				int geo = MUL(K(o, (float)(q / (1 - q))), SUB(K(o, 1), op(o, OP_EXP, MUL(skipped, K(o, (float)log(q))), -1)));
				wv = SUB(wv, MUL(K(o, (float)lr), MUL(ratio, op(o, OP_OUTER, geo, ones_h))));
			}
			if (wd > 0 && kind >= 3) wv = CATCH(wv, 1.0 - lr * wd);
			set_rows(o, mm, m1, r);
			set_rows(o, vv, v1, r);
			int dir = DIV(DIV(m1, bc1), ADD(op(o, OP_SQRT, DIV(v1, bc2), -1), K(o, (float)eps)));
			if (wd > 0 && kind >= 3) dir = ADD(dir, MUL(K(o, (float)wd), wv));
			next = SUB(wv, MUL(K(o, (float)lr), dir));
		}
#undef TAKE
#undef CATCH
		set_rows(o, wst, next, r);
	}

	if (tr_on(1)) {
		char list[512];
		size_t n = 0;
		list[0] = 0;
		for (int i = 0; i < nw && n < sizeof list - 64; i++)
			n += (size_t)snprintf(list + n, sizeof list - n, "%s%s", i ? ", " : "", m->val[w[i]].name);
		tr_begin(1, "train");
		tr_str("optimizer", name);
		tr_str("weights", list);
		tr_num("lr", lr);
		tr_num("clip", clip);
		tr_num("row_sparse", nsparse);
		tr_end("%s over %s (lr %g%s%s)", name, list, lr, clip > 0 ? ", clipped" : "", nsparse ? ", row-sparse steps for spmm tables" : "");
	}
	xfree(w);
	xfree(g);
	xfree(rows);
	xfree(o->used);
}
