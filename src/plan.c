/* Static memory planning.
 *
 * Every temporary gets a fixed offset in one arena. Lifetimes are intervals
 * over a linear program order; a value defined outside a think loop and read
 * inside it stays live until the loop ends. Offsets are assigned with the
 * greedy-by-size strategy of Pisarchyk & Lee (arXiv:2001.03288): largest
 * tensors first, each placed at the lowest offset that does not collide with
 * an already-placed tensor whose lifetime overlaps. */
#include "tg.h"

#include <stdlib.h>
#include <string.h>

static int number(Block *b, int pos)
{
	for (int i = 0; i < b->len; i++) {
		Ins *in = &b->v[i];
		in->pbeg = pos++;
		if (in->op == OP_THINK) pos = number(in->body, pos);
		in->pend = in->op == OP_THINK ? pos++ : in->pbeg;
	}
	return pos;
}

static void use(Module *m, int v, int p, Ins **loops, int nl)
{
	Value *x = &m->val[v];
	if (x->kind != V_TMP) return;
	int last = p;
	for (int i = 0; i < nl; i++)
		if (x->def < loops[i]->pbeg && loops[i]->pend > last) last = loops[i]->pend;
	if (last > x->last) x->last = last;
}

static void def(Module *m, int v, int p)
{
	m->val[v].def = p;
	if (m->val[v].last < p) m->val[v].last = p;
}

static void live(Module *m, Block *b, Ins **loops, int nl)
{
	for (int i = 0; i < b->len; i++) {
		Ins *in = &b->v[i];
		if (in->op == OP_THINK) {
			use(m, in->init, in->pbeg, loops, nl);
			def(m, in->out, in->pbeg);
			loops[nl] = in;
			live(m, in->body, loops, nl + 1);
			use(m, in->yield, in->pend, loops, nl + 1);
			use(m, in->out, in->pend, loops, nl);
		} else {
			for (int j = 0; j < in->na; j++) use(m, in->a[j], in->pbeg, loops, nl);
			def(m, in->out, in->pbeg);
		}
	}
}

static int depth(const Block *b)
{
	int d = 0;
	for (int i = 0; i < b->len; i++)
		if (b->v[i].op == OP_THINK) {
			int k = 1 + depth(b->v[i].body);
			if (k > d) d = k;
		}
	return d;
}

static int asize(const Value *v)
{
	int n = shape_numel(&v->sh);
	return (n + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
}

static const Module *sort_m;

static int by_size(const void *pa, const void *pb)
{
	const Value *a = &sort_m->val[*(const int *)pa], *b = &sort_m->val[*(const int *)pb];
	int sa = asize(a), sb = asize(b);
	if (sa != sb) return sb - sa;
	return a->def - b->def;
}

static int by_off(const void *pa, const void *pb)
{
	return sort_m->val[*(const int *)pa].off - sort_m->val[*(const int *)pb].off;
}

void plan(Module *m)
{
	if (m->planned) return;
	int end = number(&m->top, 0);
	Ins **loops = xmalloc((size_t)(depth(&m->top) + 1) * sizeof *loops);
	live(m, &m->top, loops, 0);
	use(m, m->output, end, loops, 0);
	for (int i = 0; i < m->nupd; i++) { /* read by the commit */
		use(m, m->upd_src[i], end, loops, 0);
		if (m->upd_rows[i] >= 0) use(m, m->upd_rows[i], end, loops, 0);
	}
	xfree(loops);

	int *order = xmalloc((size_t)m->nval * sizeof *order), n = 0;
	for (int v = 0; v < m->nval; v++)
		if (m->val[v].kind == V_TMP && m->val[v].def >= 0) order[n++] = v;
	sort_m = m;
	qsort(order, (size_t)n, sizeof *order, by_size);

	int *placed = xmalloc((size_t)(n ? n : 1) * sizeof *placed), np = 0;
	int *ov = xmalloc((size_t)(n ? n : 1) * sizeof *ov);
	m->arena = 0;
	for (int i = 0; i < n; i++) {
		Value *v = &m->val[order[i]];
		int k = 0;
		for (int j = 0; j < np; j++) {
			Value *w = &m->val[placed[j]];
			if (w->def <= v->last && v->def <= w->last) ov[k++] = placed[j];
		}
		qsort(ov, (size_t)k, sizeof *ov, by_off);
		int off = 0, sz = asize(v);
		for (int j = 0; j < k; j++) {
			Value *w = &m->val[ov[j]];
			if (off + sz <= w->off) break;
			int e = w->off + asize(w);
			if (e > off) off = e;
		}
		v->off = off;
		if (off + sz > m->arena) m->arena = off + sz;
		placed[np++] = order[i];
	}
	/* Commit staging: every update source is copied here first, then into its
	 * state, so an update reading another state sees the start-of-run value. */
	m->stage = m->arena;
	for (int i = 0; i < m->nupd; i++) {
		m->arena += asize(&m->val[m->upd_src[i]]);
		if (m->upd_rows[i] >= 0) m->arena += asize(&m->val[m->upd_rows[i]]);
	}
	xfree(order);
	xfree(placed);
	xfree(ov);
	m->planned = 1;
}

void plan_dump(const Module *m, FILE *f)
{
	int naive = 0, params = 0, states = 0;
	char n[64], sh[64];
	fprintf(f, "; memory plan for model %s\n; value        shape            live         offset  floats\n", m->name);
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind == V_PARAM) params += shape_numel(&x->sh);
		if (x->kind == V_STATE) states += shape_numel(&x->sh);
		if (x->kind != V_TMP || x->def < 0) continue;
		naive += asize(x);
		value_name(m, v, n, sizeof n);
		shape_str(&x->sh, sh, sizeof sh);
		fprintf(f, "%-14s %-16s [%4d,%4d]  %6d  %6d\n", n, sh, x->def, x->last, x->off, shape_numel(&x->sh));
	}
	fprintf(f, "; arena: %d floats (%d bytes); unshared: %d floats; params: %d floats (%d bytes)\n",
		m->arena, m->arena * 4, naive, params, params * 4);
	if (states) fprintf(f, "; state: %d floats (%d bytes, mutable, updated after each run; commit staging at arena offset %d)\n", states, states * 4, m->stage);
}
