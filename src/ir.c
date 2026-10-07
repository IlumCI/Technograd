/* TGIR: the canonical, machine-readable form of a Technograd program.
 *
 * (tgir 1
 *   (model NAME)
 *   (param NAME (f32 d...) (data v...))
 *   (input NAME (f32 d...))
 *   (block
 *     (%N (f32 d...) (OP ARG...))
 *     (%N (f32) (const V))
 *     (%N (f32 d...) (think INIT MAX EPS|none
 *        ...body instructions...
 *        (yield V))))
 *   (output V))
 *
 * Values are SSA. Inside a think body the instruction's own name denotes the
 * loop-carried state; after the loop it denotes the final state. Names
 * defined inside a body are not visible outside it. */
#include "tg.h"

#include <stdlib.h>
#include <string.h>

/* ---- writer ------------------------------------------------------------- */

static void indent(FILE *f, int d)
{
	for (int i = 0; i < d; i++) fputs("  ", f);
}

static void write_block(const Module *m, const Block *b, FILE *f, int d)
{
	char sh[64], n0[64], n1[64], n2[64];
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		value_name(m, in->out, n0, sizeof n0);
		shape_str(&m->val[in->out].sh, sh, sizeof sh);
		indent(f, d);
		fprintf(f, "(%s %s ", n0, sh);
		if (in->op == OP_CONST) {
			fprintf(f, "(const %.9g))\n", (double)in->k);
		} else if (in->op == OP_THINK) {
			value_name(m, in->init, n1, sizeof n1);
			fprintf(f, "(think %s %d ", n1, in->maxit);
			if (in->eps < 0) fputs("none\n", f);
			else fprintf(f, "%.9g\n", (double)in->eps);
			write_block(m, in->body, f, d + 1);
			value_name(m, in->yield, n2, sizeof n2);
			indent(f, d + 1);
			fprintf(f, "(yield %s)))\n", n2);
		} else {
			fprintf(f, "(%s", tg_ops[in->op].name);
			for (int j = 0; j < in->na; j++) {
				value_name(m, in->a[j], n1, sizeof n1);
				fprintf(f, " %s", n1);
			}
			fputs("))\n", f);
		}
	}
}

void ir_write(const Module *m, FILE *f)
{
	char sh[64], n[64];
	fprintf(f, "(tgir 1\n  (model %s)\n", m->name);
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_PARAM) continue;
		shape_str(&x->sh, sh, sizeof sh);
		fprintf(f, "  (param %s %s\n    (data", x->name, sh);
		int k = shape_numel(&x->sh);
		for (int i = 0; i < k; i++) {
			if (i && i % 8 == 0) fputs("\n         ", f);
			fprintf(f, " %.9g", (double)x->data[i]);
		}
		fputs("))\n", f);
	}
	for (int i = 0; i < m->ninputs; i++) {
		const Value *x = &m->val[m->inputs[i]];
		shape_str(&x->sh, sh, sizeof sh);
		fprintf(f, "  (input %s %s)\n", x->name, sh);
	}
	fputs("  (block\n", f);
	write_block(m, &m->top, f, 2);
	value_name(m, m->output, n, sizeof n);
	fprintf(f, "  )\n  (output %s))\n", n);
}

/* ---- reader / verifier -------------------------------------------------- */

typedef struct {
	Module *m;
	const char *file;
	const char **names;
	int *vals;
	int n, cap;
} R;

static int r_get(R *r, const Sx *x)
{
	if (x->k != SX_SYM) die(r->file, x->line, "expected value name");
	for (int i = r->n - 1; i >= 0; i--)
		if (strcmp(r->names[i], x->s) == 0) return r->vals[i];
	die(r->file, x->line, "undefined value '%s'", x->s);
	return -1;
}

static void r_def(R *r, const Sx *x, int v)
{
	if (x->k != SX_SYM) die(r->file, x->line, "expected value name");
	for (int i = 0; i < r->n; i++)
		if (strcmp(r->names[i], x->s) == 0) die(r->file, x->line, "redefinition of '%s'", x->s);
	if (r->n == r->cap) {
		r->cap = r->cap ? r->cap * 2 : 64;
		r->names = xrealloc(r->names, (size_t)r->cap * sizeof *r->names);
		r->vals = xrealloc(r->vals, (size_t)r->cap * sizeof *r->vals);
	}
	r->names[r->n] = x->s;
	r->vals[r->n++] = v;
}

static Shape r_shape(R *r, const Sx *x)
{
	Shape s = { 0 };
	if (x->k != SX_LIST || x->len < 1 || !sx_issym(x->v[0], "f32") || x->len - 1 > TG_MAXRANK)
		die(r->file, x->line, "expected type (f32 d...)");
	s.rank = x->len - 1;
	for (int i = 1; i < x->len; i++) {
		if (x->v[i]->k != SX_NUM || x->v[i]->n < 1 || x->v[i]->n != (double)(int)x->v[i]->n)
			die(r->file, x->line, "bad dimension");
		s.dim[i - 1] = (int)x->v[i]->n;
	}
	return s;
}

static void need(R *r, const Sx *x, int ok, const char *what)
{
	if (!ok) die(r->file, x->line, "malformed %s", what);
}

static void r_block(R *r, Block *b, const Sx *forms, int from, int to)
{
	char err[256], s0[64], s1[64];
	for (int i = from; i < to; i++) {
		const Sx *x = forms->v[i];
		need(r, x, x->k == SX_LIST && x->len == 3 && x->v[2]->k == SX_LIST && x->v[2]->len >= 1, "instruction");
		Shape decl = r_shape(r, x->v[1]);
		const Sx *e = x->v[2];
		const Sx *head = e->v[0];
		need(r, e, head->k == SX_SYM, "instruction");
		Ins tmp;
		memset(&tmp, 0, sizeof tmp);
		Shape got = { 0 };

		if (sx_issym(head, "const")) {
			need(r, e, e->len == 2 && e->v[1]->k == SX_NUM, "const");
			tmp.op = OP_CONST;
			tmp.k = (float)e->v[1]->n;
		} else if (sx_issym(head, "think")) {
			need(r, e, e->len >= 5, "think");
			tmp.op = OP_THINK;
			tmp.init = r_get(r, e->v[1]);
			need(r, e, e->v[2]->k == SX_NUM && e->v[2]->n >= 1, "think budget");
			tmp.maxit = (int)e->v[2]->n;
			if (sx_issym(e->v[3], "none")) tmp.eps = -1;
			else {
				need(r, e, e->v[3]->k == SX_NUM && e->v[3]->n >= 0, "think threshold");
				tmp.eps = (float)e->v[3]->n;
			}
			got = r->m->val[tmp.init].sh;
		} else {
			int op = op_lookup(head->s);
			if (op < 0) die(r->file, head->line, "unknown op '%s'", head->s);
			tmp.op = (Op)op;
			tmp.na = e->len - 1;
			if (tmp.na > 2) die(r->file, e->line, "too many operands");
			Shape in[2];
			for (int j = 0; j < tmp.na; j++) {
				tmp.a[j] = r_get(r, e->v[j + 1]);
				in[j] = r->m->val[tmp.a[j]].sh;
			}
			if (!op_infer(tmp.op, in, tmp.na, &got, err, sizeof err)) die(r->file, e->line, "%s", err);
		}
		if (!shape_eq(&got, &decl)) {
			shape_str(&decl, s0, sizeof s0);
			shape_str(&got, s1, sizeof s1);
			die(r->file, x->line, "declared %s but operation yields %s", s0, s1);
		}

		int out = mod_value(r->m, V_TMP, &decl, NULL);
		if (tmp.op == OP_THINK) {
			const Sx *y = e->v[e->len - 1];
			need(r, y, y->k == SX_LIST && y->len == 2 && sx_issym(y->v[0], "yield"), "think (missing yield)");
			tmp.tid = r->m->nthink++;
			tmp.body = xmalloc(sizeof *tmp.body);
			int mark = r->n;
			r_def(r, x->v[0], out); /* state visible in body */
			r_block(r, tmp.body, e, 4, e->len - 1);
			tmp.yield = r_get(r, y->v[1]);
			if (!shape_eq(&r->m->val[tmp.yield].sh, &decl)) die(r->file, y->line, "yield shape differs from state shape");
			r->n = mark; /* drop body scope, state re-bound below */
		}
		r_def(r, x->v[0], out);
		Ins *in = block_push(b);
		int a0 = tmp.a[0], a1 = tmp.a[1];
		*in = tmp;
		in->out = out;
		if (tmp.na < 1) a0 = -1;
		if (tmp.na < 2) a1 = -1;
		in->a[0] = a0;
		in->a[1] = a1;
		if (tmp.op != OP_THINK) in->init = in->yield = -1;
	}
}

Module *ir_read(Sx *forms, const char *file)
{
	R r = { 0 };
	r.file = file;
	if (forms->len != 1) die(file, 1, "expected exactly one (tgir ...) form");
	Sx *t = forms->v[0];
	if (t->k != SX_LIST || t->len < 2 || !sx_issym(t->v[0], "tgir")) die(file, t->line, "expected (tgir 1 ...)");
	if (t->v[1]->k != SX_NUM || t->v[1]->n != 1) die(file, t->line, "unsupported TGIR version");

	int seen_block = 0, seen_out = 0;
	for (int i = 2; i < t->len; i++) {
		Sx *f = t->v[i];
		need(&r, f, f->k == SX_LIST && f->len >= 1 && f->v[0]->k == SX_SYM, "top-level form");
		const char *h = f->v[0]->s;
		if (strcmp(h, "model") == 0) {
			need(&r, f, f->len == 2 && f->v[1]->k == SX_SYM && !r.m, "model");
			r.m = mod_new(f->v[1]->s);
			continue;
		}
		if (!r.m) die(file, f->line, "(model ...) must come first");
		if (strcmp(h, "param") == 0) {
			need(&r, f, !seen_block && f->len == 4 && f->v[3]->k == SX_LIST && f->v[3]->len >= 1 && sx_issym(f->v[3]->v[0], "data"), "param");
			Shape sh = r_shape(&r, f->v[2]);
			int n = shape_numel(&sh);
			Sx *d = f->v[3];
			if (d->len - 1 != n) die(file, d->line, "param '%s' needs %d values, has %d", f->v[1]->s, n, d->len - 1);
			int v = mod_value(r.m, V_PARAM, &sh, f->v[1]->s);
			r.m->val[v].data = xmalloc((size_t)n * sizeof(float));
			for (int j = 0; j < n; j++) {
				need(&r, d, d->v[j + 1]->k == SX_NUM, "data");
				r.m->val[v].data[j] = (float)d->v[j + 1]->n;
			}
			r_def(&r, f->v[1], v);
		} else if (strcmp(h, "input") == 0) {
			need(&r, f, !seen_block && f->len == 3, "input");
			Shape sh = r_shape(&r, f->v[2]);
			r_def(&r, f->v[1], mod_value(r.m, V_INPUT, &sh, f->v[1]->s));
		} else if (strcmp(h, "block") == 0) {
			need(&r, f, !seen_block, "block (duplicate)");
			seen_block = 1;
			r_block(&r, &r.m->top, f, 1, f->len);
		} else if (strcmp(h, "output") == 0) {
			need(&r, f, seen_block && !seen_out && f->len == 2, "output");
			seen_out = 1;
			r.m->output = r_get(&r, f->v[1]);
		} else {
			die(file, f->line, "unknown top-level form '%s'", h);
		}
	}
	if (!r.m) die(file, 1, "missing (model ...)");
	if (!r.m->ninputs) die(file, 1, "module has no inputs");
	if (!seen_out) die(file, 1, "missing (output ...)");
	free(r.names);
	free(r.vals);
	return r.m;
}
