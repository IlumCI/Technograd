/* TGIR: the canonical, machine-readable form of a Technograd program.
 *
 * (tgir 1
 *   (model NAME)
 *   (param NAME (f32 d...) (data v...))
 *   (param NAME (f32 r c) (quant BITS AXIS (scale s...) (codes q...)))   ; tgc quantize
 *   (input NAME (f32 d...))
 *   (block
 *     (%N (f32 d...) (OP ARG...))
 *     (%N (f32) (const V))
 *     (%N (f32 d...) (think INIT MAX EPS|none
 *        ...body instructions...
 *        (yield V)))
 *     (scan T forward|reverse [steps]           ; steps: report T in the next steps slot
 *       (carry C (f32 d...) INIT) ...      ; C: the carry in the body, the final value after
 *       (in XT (f32 d...) SEQ) ...         ; XT: row t of SEQ, visible in the body only
 *       (body ...instructions...)
 *       (next C V) ...                     ; one per carry, in order
 *       (emit YS (f32 T d...) V) ...))     ; YS: the stack of V over the steps
 *   (output V))
 *
 * Values are SSA. Inside a think body the instruction's own name denotes the
 * loop-carried state; after the loop it denotes the final state. Names
 * defined inside a body are not visible outside it. */
#include "tg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int tg_qget_host(const signed char *q, int bits, int i)
{
	if (bits == 8) return q[i];
	int b = q[i >> 1];
	return (i & 1) ? (b >> 4) : (int)(signed char)(unsigned char)((unsigned)b << 4) >> 4;
}

/* ---- writer ------------------------------------------------------------- */

static void indent(FILE *f, int d)
{
	for (int i = 0; i < d; i++) fputs("  ", f);
}

/* Canonical numbering: temporaries are numbered in the order the reader will
 * recreate them (named values first, then instructions in pre-order, a think
 * result before its body), so write(read(write(m))) == write(m) for any module,
 * however it was built. */
static int *ren;

static void number(const Block *b, int *next)
{
	for (int i = 0; i < b->len; i++) {
		if (b->v[i].op == OP_SCAN) { /* reader order: carries, slices, body, stacks */
			const Scan *s = b->v[i].sc;
			for (int k = 0; k < s->nc; k++) ren[s->c[k]] = (*next)++;
			for (int j = 0; j < s->nx; j++) ren[s->xt[j]] = (*next)++;
			number(b->v[i].body, next);
			for (int y = 0; y < s->ny; y++) ren[s->ys[y]] = (*next)++;
			continue;
		}
		ren[b->v[i].out] = (*next)++;
		if (b->v[i].op == OP_THINK) number(b->v[i].body, next);
	}
}

static void vname(const Module *m, int v, char *buf, size_t n)
{
	if (m->val[v].name) snprintf(buf, n, "%s", m->val[v].name);
	else snprintf(buf, n, "%%%d", ren[v]);
}

static void write_block(const Module *m, const Block *b, FILE *f, int d)
{
	char sh[64], n0[64], n1[64], n2[64];
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		indent(f, d);
		if (in->op == OP_SCAN) {
			const Scan *s = in->sc;
			fprintf(f, "(scan %d %s%s\n", s->T, s->reverse ? "reverse" : "forward", s->tid >= 0 ? " steps" : "");
			for (int k = 0; k < s->nc; k++) {
				vname(m, s->c[k], n0, sizeof n0);
				vname(m, s->init[k], n1, sizeof n1);
				shape_str(&m->val[s->c[k]].sh, sh, sizeof sh);
				indent(f, d + 1);
				fprintf(f, "(carry %s %s %s)\n", n0, sh, n1);
			}
			for (int j = 0; j < s->nx; j++) {
				vname(m, s->xt[j], n0, sizeof n0);
				vname(m, s->x[j], n1, sizeof n1);
				shape_str(&m->val[s->xt[j]].sh, sh, sizeof sh);
				indent(f, d + 1);
				fprintf(f, "(in %s %s %s)\n", n0, sh, n1);
			}
			indent(f, d + 1);
			fputs("(body\n", f);
			write_block(m, in->body, f, d + 2);
			indent(f, d + 1);
			fputs(")\n", f);
			for (int k = 0; k < s->nc; k++) {
				vname(m, s->c[k], n0, sizeof n0);
				vname(m, s->next[k], n1, sizeof n1);
				indent(f, d + 1);
				fprintf(f, "(next %s %s)\n", n0, n1);
			}
			for (int y = 0; y < s->ny; y++) {
				vname(m, s->ys[y], n0, sizeof n0);
				vname(m, s->y[y], n1, sizeof n1);
				shape_str(&m->val[s->ys[y]].sh, sh, sizeof sh);
				indent(f, d + 1);
				fprintf(f, "(emit %s %s %s)\n", n0, sh, n1);
			}
			indent(f, d);
			fputs(")\n", f);
			continue;
		}
		vname(m, in->out, n0, sizeof n0);
		shape_str(&m->val[in->out].sh, sh, sizeof sh);
		fprintf(f, "(%s %s ", n0, sh);
		if (in->op == OP_CONST) {
			fprintf(f, "(const %.9g))\n", (double)in->k);
		} else if (in->op == OP_THINK) {
			vname(m, in->init, n1, sizeof n1);
			fprintf(f, "(think %s %d ", n1, in->maxit);
			if (in->eps < 0) fputs("none\n", f);
			else fprintf(f, "%.9g\n", (double)in->eps);
			write_block(m, in->body, f, d + 1);
			vname(m, in->yield, n2, sizeof n2);
			indent(f, d + 1);
			fprintf(f, "(yield %s)", n2);
			if (in->halt >= 0) {
				vname(m, in->halt, n2, sizeof n2);
				fprintf(f, "\n");
				indent(f, d + 1);
				fprintf(f, "(halt %s %.9g)", n2, (double)in->hthr);
			}
			fputs("))\n", f);
		} else {
			fprintf(f, "(%s", tg_ops[in->op].name);
			for (int j = 0; j < in->na; j++) {
				vname(m, in->a[j], n1, sizeof n1);
				fprintf(f, " %s", n1);
			}
			fputs("))\n", f);
		}
	}
}

void ir_write(const Module *m, FILE *f)
{
	char sh[64], n[64];
	ren = xmalloc((size_t)(m->nval ? m->nval : 1) * sizeof *ren);
	int next = 0;
	for (int v = 0; v < m->nval; v++)
		if ((m->val[v].kind != V_TMP) && !m->val[v].dead) next++;
	number(&m->top, &next);
	fprintf(f, "(tgir 1\n  (model %s)\n", m->name);
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if ((x->kind != V_PARAM && x->kind != V_STATE) || x->dead) continue;
		shape_str(&x->sh, sh, sizeof sh);
		if (x->qbits) { /* codes are written one per element (int4 unpacked) */
			int nch = x->sh.dim[x->qaxis], k = shape_numel(&x->sh);
			fprintf(f, "  (param %s %s\n    (quant %d %d\n      (scale", x->name, sh, x->qbits, x->qaxis);
			for (int i = 0; i < nch; i++) fprintf(f, "%s %.9g", i && i % 8 == 0 ? "\n            " : "", (double)x->qs[i]);
			fputs(")\n      (codes", f);
			for (int i = 0; i < k; i++) fprintf(f, "%s %d", i && i % 24 == 0 ? "\n            " : "", tg_qget_host(x->q, x->qbits, i));
			fputs(")))\n", f);
			continue;
		}
		fprintf(f, "  (%s %s %s\n    (data", x->kind == V_STATE ? "state" : "param", x->name, sh);
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
	vname(m, m->output, n, sizeof n);
	fprintf(f, "  )\n  (output %s)", n);
	for (int i = 0; i < m->nupd; i++) {
		char s[64];
		vname(m, m->upd_src[i], s, sizeof s);
		fprintf(f, "\n  (update %s %s", m->val[m->upd_state[i]].name, s);
		if (m->upd_rows[i] >= 0) {
			vname(m, m->upd_rows[i], s, sizeof s);
			fprintf(f, " %s", s);
		}
		fputc(')', f);
	}
	fputs(")\n", f);
	xfree(ren);
	ren = NULL;
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

static void r_block(R *r, Block *b, const Sx *forms, int from, int to);

static void r_scan(R *r, Block *b, const Sx *x)
{
	Module *m = r->m;
	char s0[64], s1[64];
	need(r, x, x->len >= 4 && x->v[1]->k == SX_NUM && x->v[1]->n >= 1 && x->v[1]->n == (double)(int)x->v[1]->n, "scan (steps)");
	need(r, x, sx_issym(x->v[2], "forward") || sx_issym(x->v[2], "reverse"), "scan (direction)");
	Scan *s = scan_new((int)x->v[1]->n, sx_issym(x->v[2], "reverse"));
	int i = 3, mark = r->n;
	if (i < x->len && sx_issym(x->v[i], "steps")) { /* a fixed-budget think loop: reports T in a steps slot */
		s->tid = r->m->nthink++;
		i++;
	}
	const Sx **cname = xmalloc((size_t)x->len * sizeof *cname);
	for (; i < x->len && x->v[i]->k == SX_LIST && x->v[i]->len == 4 && sx_issym(x->v[i]->v[0], "carry"); i++) {
		const Sx *c = x->v[i];
		int init = r_get(r, c->v[3]);
		Shape sh = r_shape(r, c->v[2]);
		if (!shape_eq(&sh, &m->val[init].sh)) die(r->file, c->line, "carry type differs from its initial value");
		cname[s->nc] = c->v[1];
		scan_carry(s, mod_value(m, V_TMP, &sh, NULL), init, -1);
	}
	need(r, x, s->nc >= 1, "scan (needs a carry)");
	int *seqv = xmalloc((size_t)x->len * sizeof *seqv);
	const Sx **xname = xmalloc((size_t)x->len * sizeof *xname);
	int nx = 0;
	for (; i < x->len && x->v[i]->k == SX_LIST && x->v[i]->len == 4 && sx_issym(x->v[i]->v[0], "in"); i++) {
		const Sx *c = x->v[i];
		int q = r_get(r, c->v[3]);
		Shape sh = r_shape(r, c->v[2]), want = m->val[q].sh;
		if (want.rank < 1 || want.dim[0] != s->T) die(r->file, c->line, "sequence does not have %d steps along its first axis", s->T);
		for (int d = 1; d < want.rank; d++) want.dim[d - 1] = want.dim[d];
		want.rank--;
		if (!shape_eq(&sh, &want)) die(r->file, c->line, "slice type differs from a row of its sequence");
		seqv[nx] = q;
		xname[nx++] = c->v[1];
		scan_seq(s, q, mod_value(m, V_TMP, &sh, NULL));
	}
	for (int k = 0; k < s->nc; k++) r_def(r, cname[k], s->c[k]); /* body scope: carries and slices */
	for (int j = 0; j < nx; j++) r_def(r, xname[j], s->xt[j]);
	need(r, x, i < x->len && x->v[i]->k == SX_LIST && x->v[i]->len >= 1 && sx_issym(x->v[i]->v[0], "body"), "scan (body)");
	Ins *ins = block_push(b);
	int idx = (int)(ins - b->v);
	Block *body = xmalloc(sizeof *body);
	r_block(r, body, x->v[i], 1, x->v[i]->len);
	i++;
	for (int k = 0; k < s->nc; k++, i++) {
		need(r, x, i < x->len && x->v[i]->k == SX_LIST && x->v[i]->len == 3 && sx_issym(x->v[i]->v[0], "next"), "scan (next)");
		if (r_get(r, x->v[i]->v[1]) != s->c[k]) die(r->file, x->v[i]->line, "'next' entries must follow the carries in order");
		int nv = r_get(r, x->v[i]->v[2]);
		if (!shape_eq(&m->val[nv].sh, &m->val[s->c[k]].sh)) die(r->file, x->v[i]->line, "next value shape differs from its carry");
		for (int j = 0; j < s->nc; j++)
			if (j != k && nv == s->c[j]) die(r->file, x->v[i]->line, "a carry's next value cannot be another carry");
		s->next[k] = nv;
	}
	int *yv = xmalloc((size_t)x->len * sizeof *yv), ny = 0;
	const Sx **yname = xmalloc((size_t)x->len * sizeof *yname);
	for (; i < x->len; i++) {
		const Sx *c = x->v[i];
		need(r, c, c->k == SX_LIST && c->len == 4 && sx_issym(c->v[0], "emit"), "scan (emit)");
		int v = r_get(r, c->v[3]);
		Shape sh = r_shape(r, c->v[2]), want = m->val[v].sh;
		if (want.rank == TG_MAXRANK) die(r->file, c->line, "stack rank exceeds %d", TG_MAXRANK);
		for (int d = want.rank; d > 0; d--) want.dim[d] = want.dim[d - 1];
		want.dim[0] = s->T;
		want.rank++;
		if (!shape_eq(&sh, &want)) {
			shape_str(&sh, s0, sizeof s0);
			shape_str(&want, s1, sizeof s1);
			die(r->file, c->line, "emit declared %s but stacks to %s", s0, s1);
		}
		yv[ny] = v;
		yname[ny++] = c->v[1];
	}
	r->n = mark; /* drop the body scope */
	for (int k = 0; k < s->nc; k++) r_def(r, cname[k], s->c[k]);
	for (int y = 0; y < ny; y++) {
		Shape sh = r_shape(r, x->v[x->len - ny + y]->v[2]);
		int ys = mod_value(m, V_TMP, &sh, NULL);
		scan_stack(s, yv[y], ys);
		r_def(r, yname[y], ys);
	}
	ins = &b->v[idx];
	ins->op = OP_SCAN;
	ins->sc = s;
	ins->body = body;
	ins->out = s->c[0];
	ins->init = ins->yield = -1;
	xfree(cname);
	xfree(seqv);
	xfree(xname);
	xfree(yv);
	xfree(yname);
}

static void r_block(R *r, Block *b, const Sx *forms, int from, int to)
{
	char err[256], s0[64], s1[64];
	for (int i = from; i < to; i++) {
		const Sx *x = forms->v[i];
		if (x->k == SX_LIST && x->len >= 1 && sx_issym(x->v[0], "scan")) {
			r_scan(r, b, x);
			continue;
		}
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
			if (op == OP_RESHAPE) { /* the declared type is the target; only the element count is checked */
				need(r, e, e->len == 2, "reshape");
				tmp.op = OP_RESHAPE;
				tmp.na = 1;
				tmp.a[0] = r_get(r, e->v[1]);
				if (shape_numel(&r->m->val[tmp.a[0]].sh) != shape_numel(&decl)) die(r->file, e->line, "reshape changes the element count");
				got = decl;
				goto checked;
			}
			tmp.op = (Op)op;
			tmp.na = e->len - 1;
			if (tmp.na > TG_MAXARGS) die(r->file, e->line, "too many operands");
			Shape in[TG_MAXARGS];
			for (int j = 0; j < tmp.na; j++) {
				tmp.a[j] = r_get(r, e->v[j + 1]);
				in[j] = r->m->val[tmp.a[j]].sh;
			}
			if (!op_infer(tmp.op, in, tmp.na, &got, err, sizeof err)) die(r->file, e->line, "%s", err);
		}
	checked:
		if (!shape_eq(&got, &decl)) {
			shape_str(&decl, s0, sizeof s0);
			shape_str(&got, s1, sizeof s1);
			die(r->file, x->line, "declared %s but operation yields %s", s0, s1);
		}

		int out = mod_value(r->m, V_TMP, &decl, NULL);
		if (tmp.op == OP_THINK) {
			const Sx *hl = e->v[e->len - 1];
			int has_halt = hl->k == SX_LIST && hl->len >= 1 && sx_issym(hl->v[0], "halt");
			const Sx *y = e->v[e->len - 1 - has_halt];
			need(r, y, y->k == SX_LIST && y->len == 2 && sx_issym(y->v[0], "yield"), "think (missing yield)");
			tmp.tid = r->m->nthink++;
			tmp.body = xmalloc(sizeof *tmp.body);
			int mark = r->n;
			r_def(r, x->v[0], out); /* state visible in body */
			r_block(r, tmp.body, e, 4, e->len - 1 - has_halt);
			tmp.yield = r_get(r, y->v[1]);
			tmp.halt = -1;
			if (has_halt) {
				need(r, hl, hl->len == 3 && hl->v[2]->k == SX_NUM && hl->v[2]->n > 0 && hl->v[2]->n < 1 && tmp.eps < 0, "think halt (value, threshold in (0, 1); no until)");
				tmp.halt = r_get(r, hl->v[1]);
				if (r->m->val[tmp.halt].sh.rank != 0) die(r->file, hl->line, "halting value must be a scalar");
				tmp.hthr = (float)hl->v[2]->n;
			}
			if (!shape_eq(&r->m->val[tmp.yield].sh, &decl)) die(r->file, y->line, "yield shape differs from state shape");
			r->n = mark; /* drop body scope, state re-bound below */
		}
		r_def(r, x->v[0], out);
		Ins *in = block_push(b);
		*in = tmp;
		in->out = out;
		for (int j = tmp.op == OP_THINK ? 0 : tmp.na; j < TG_MAXARGS; j++) in->a[j] = -1;
		if (tmp.op != OP_THINK) in->init = in->yield = in->halt = -1;
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
		if (strcmp(h, "param") == 0 && f->len == 4 && f->v[3]->k == SX_LIST && f->v[3]->len == 5 && sx_issym(f->v[3]->v[0], "quant")) {
			need(&r, f, !seen_block, h);
			Shape sh = r_shape(&r, f->v[2]);
			Sx *qd = f->v[3];
			need(&r, qd, qd->v[1]->k == SX_NUM && (qd->v[1]->n == 8 || qd->v[1]->n == 4), "quant bits (8 or 4)");
			need(&r, qd, qd->v[2]->k == SX_NUM && (qd->v[2]->n == 0 || qd->v[2]->n == 1) && sh.rank == 2, "quant axis (0 or 1, rank-2 param)");
			int bits = (int)qd->v[1]->n, axis = (int)qd->v[2]->n, n = shape_numel(&sh), nch = sh.dim[axis];
			Sx *sc = qd->v[3], *cd = qd->v[4];
			need(&r, sc, sc->k == SX_LIST && sc->len == nch + 1 && sx_issym(sc->v[0], "scale"), "quant scales (one per channel)");
			need(&r, cd, cd->k == SX_LIST && cd->len == n + 1 && sx_issym(cd->v[0], "codes"), "quant codes (one per element)");
			int v = mod_value(r.m, V_PARAM, &sh, f->v[1]->s);
			Value *x = &r.m->val[v];
			x->qbits = bits;
			x->qaxis = axis;
			x->qs = xmalloc((size_t)nch * sizeof(float));
			x->q = xmalloc((size_t)(bits == 8 ? n : (n + 1) / 2));
			x->data = xmalloc((size_t)n * sizeof(float));
			for (int j = 0; j < nch; j++) {
				need(&r, sc, sc->v[j + 1]->k == SX_NUM, "scale");
				x->qs[j] = (float)sc->v[j + 1]->n;
			}
			int lim = bits == 8 ? 127 : 7;
			for (int j = 0; j < n; j++) {
				need(&r, cd, cd->v[j + 1]->k == SX_NUM && cd->v[j + 1]->n == (double)(int)cd->v[j + 1]->n && fabs(cd->v[j + 1]->n) <= lim, "code");
				int c = (int)cd->v[j + 1]->n;
				if (bits == 8) x->q[j] = (signed char)c;
				else x->q[j / 2] = (signed char)(j % 2 ? (unsigned char)((unsigned char)x->q[j / 2] & 0x0f) | (unsigned char)((unsigned)c << 4)
							       : (unsigned char)((unsigned char)x->q[j / 2] & 0xf0) | (unsigned char)(c & 0x0f));
				int ch = axis == 0 ? j / sh.dim[1] : j % sh.dim[1];
				x->data[j] = (float)c * x->qs[ch];
			}
			r_def(&r, f->v[1], v);
		} else if (strcmp(h, "param") == 0 || strcmp(h, "state") == 0) {
			need(&r, f, !seen_block && f->len == 4 && f->v[3]->k == SX_LIST && f->v[3]->len >= 1 && sx_issym(f->v[3]->v[0], "data"), h);
			Shape sh = r_shape(&r, f->v[2]);
			int n = shape_numel(&sh);
			Sx *d = f->v[3];
			if (d->len - 1 != n) die(file, d->line, "param '%s' needs %d values, has %d", f->v[1]->s, n, d->len - 1);
			int v = mod_value(r.m, strcmp(h, "state") == 0 ? V_STATE : V_PARAM, &sh, f->v[1]->s);
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
		} else if (strcmp(h, "update") == 0) {
			need(&r, f, seen_block && (f->len == 3 || f->len == 4), "update");
			int sv = r_get(&r, f->v[1]), src = r_get(&r, f->v[2]), rows = f->len == 4 ? r_get(&r, f->v[3]) : -1;
			if (r.m->val[sv].kind != V_STATE) die(file, f->line, "'%s' is not a state", f->v[1]->s);
			for (int k = 0; k < r.m->nupd; k++)
				if (r.m->upd_state[k] == sv) die(file, f->line, "state '%s' updated twice", f->v[1]->s);
			char err[300];
			if (!upd_check(r.m, sv, src, rows, err, sizeof err)) die(file, f->line, "%s", err);
			mod_update_rows(r.m, sv, src, rows);
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
	xfree(r.names);
	xfree(r.vals);
	return r.m;
}
