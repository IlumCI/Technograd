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
 * clip=c rescales all gradients so their global L2 norm is at most c. */
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

static void set(O *o, int state, int value)
{
	for (int k = 0; k < o->m->nupd; k++)
		if (o->m->upd_state[k] == state) die(o->file, o->line, "state \x27%s\x27 updated twice in one run", o->m->val[state].name);
	mod_update(o->m, state, value);
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
	for (int i = 0; i < nkv; i++)
		if (!o->used[i]) die(file, line, "unknown option \x27%s\x27 for %s (lr, momentum, beta1, beta2, eps, wd, clip)", keys[i], name);

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

	int *g = xmalloc((size_t)nw * sizeof *g);
	for (int i = 0; i < nw; i++) g[i] = ad_grad(m, b, loss, w[i], file, line);

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
	if (kind >= 2) {
		char tn[32];
		snprintf(tn, sizeof tn, "__opt_t%d", m->nupd);
		Shape sc = { 0 };
		int t = mod_value(m, V_STATE, &sc, tn);
		m->val[t].data = xmalloc(sizeof(float));
		t1 = ADD(t, K(o, 1));
		set(o, t, t1);
		bc1 = SUB(K(o, 1), op(o, OP_EXP, MUL(t1, K(o, (float)log(b1))), -1));
		bc2 = SUB(K(o, 1), op(o, OP_EXP, MUL(t1, K(o, (float)log(b2))), -1));
	}

	for (int i = 0; i < nw; i++) {
		int wv = w[i], gi = g[i], next;
		const Shape *s = &m->val[wv].sh;
		if (kind == 0) {
			next = SUB(wv, MUL(K(o, (float)lr), gi));
		} else if (kind == 1) {
			int mm = hidden(o, "m", wv);
			int m1 = ADD(MUL(K(o, (float)mu), mm), gi);
			set(o, mm, m1);
			next = SUB(wv, MUL(K(o, (float)lr), m1));
		} else if (kind == 4 && s->rank == 2) {
			int mm = hidden(o, "m", wv);
			int m1 = ADD(MUL(K(o, (float)mu), mm), gi);
			set(o, mm, m1);
			int u = ADD(gi, MUL(K(o, (float)mu), m1)); /* Nesterov */
			float scale = 0.2f * sqrtf((float)(s->dim[0] > s->dim[1] ? s->dim[0] : s->dim[1]));
			int dir = ADD(MUL(K(o, scale), newton_schulz(o, u)), MUL(K(o, (float)wd), wv));
			next = SUB(wv, MUL(K(o, (float)lr), dir));
		} else { /* adam, adamw, and muon's non-matrix weights */
			int mm = hidden(o, "m", wv), vv = hidden(o, "v", wv);
			int m1 = ADD(MUL(K(o, (float)b1), mm), MUL(K(o, (float)(1 - b1)), gi));
			int v1 = ADD(MUL(K(o, (float)b2), vv), MUL(K(o, (float)(1 - b2)), MUL(gi, gi)));
			set(o, mm, m1);
			set(o, vv, v1);
			int dir = DIV(DIV(m1, bc1), ADD(op(o, OP_SQRT, DIV(v1, bc2), -1), K(o, (float)eps)));
			if (wd > 0 && kind >= 3) dir = ADD(dir, MUL(K(o, (float)wd), wv));
			next = SUB(wv, MUL(K(o, (float)lr), dir));
		}
		set(o, wv, next);
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
		tr_end("%s over %s (lr %g%s)", name, list, lr, clip > 0 ? ", clipped" : "");
	}
	xfree(w);
	xfree(g);
	xfree(o->used);
}
