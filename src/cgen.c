/* C backend: emits one freestanding translation unit per model.
 *
 * The unit embeds the runtime kernels and all weights, owns a statically
 * sized arena, and exposes a single entry point:
 *
 *   void tg_<model>_run(const float *<input>..., float *out);
 *   extern int tg_<model>_steps[];   iterations taken by each think loop
 *
 * No heap, no I/O, no globals other than the arena and the steps vector.
 * Compile with -DTG_MAIN to get a command-line driver for testing. */
#include "tg.h"
#include "rt_embed.h"
#include "rtq_embed.h"

#include <stdint.h>

#include <math.h>

typedef struct {
	const Module *m;
	FILE *f;
} G;

static void ref(G *g, int v, char *buf, size_t n)
{
	const Value *x = &g->m->val[v];
	switch (x->kind) {
	case V_PARAM: snprintf(buf, n, "tgp_%s", x->name); break;
	case V_INPUT: snprintf(buf, n, "tgi_%s", x->name); break;
	case V_TMP: snprintf(buf, n, "(A + %d)", x->off); break;
	case V_STATE: snprintf(buf, n, "tg_%s_state_%s", g->m->name, x->name); break;
	}
}

/* Does any use of param v need its f32 values (not quantized codes)? */
static int needs_f32_block(const Module *m, int v, const Block *b)
{
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		if (in->op == OP_THINK) {
			if (in->init == v || in->yield == v || needs_f32_block(m, v, in->body)) return 1;
		} else if (in->op == OP_SCAN) {
			const Scan *s = in->sc;
			for (int k = 0; k < s->nc; k++)
				if (s->init[k] == v || s->next[k] == v) return 1;
			for (int j = 0; j < s->nx; j++)
				if (s->x[j] == v) return 1;
			for (int y = 0; y < s->ny; y++)
				if (s->y[y] == v) return 1;
			if (needs_f32_block(m, v, in->body)) return 1;
		} else {
			for (int j = 0; j < in->na; j++)
				if (in->a[j] == v && q_axis_for(m, v, in) != m->val[v].qaxis) return 1;
		}
	}
	return 0;
}

static int needs_f32(const Module *m, int v)
{
	if (!m->val[v].qbits || m->output == v) return 1;
	for (int i = 0; i < m->nupd; i++)
		if (m->upd_src[i] == v || m->upd_rows[i] == v) return 1;
	return needs_f32_block(m, v, &m->top);
}

static void ind(G *g, int d)
{
	for (int i = 0; i < d; i++) fputc('\t', g->f);
}

/* Fixed-point mode (tgc c --fixed): every value is Q16.16 in a tg_t (int32_t). */
static int FX;
#define TY (FX ? "tg_t" : "float")

static long long to_fixed(double v, int frac)
{
	double s = v * (double)(1LL << frac);
	if (s >= 2147483647.0) return 2147483647LL;
	if (s <= -2147483648.0) return -2147483647LL - 1;
	return llround(s);
}

static void flt(G *g, float v)
{
	if (!isfinite(v)) die(NULL, 0, "non-finite constant in model");
	if (FX) fprintf(g->f, "%lld", to_fixed(v, 16));
	else fprintf(g->f, "%.9ef", (double)v);
}

static void block(G *g, const Block *b, int d)
{
	const Module *m = g->m;
	char o[64], a[64], c[64], e[64];
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		int n = shape_numel(&m->val[in->out].sh);
		ref(g, in->out, o, sizeof o);
		if (in->na > 0) ref(g, in->a[0], a, sizeof a);
		if (in->na > 1) ref(g, in->a[1], c, sizeof c);
		if (in->na > 2) ref(g, in->a[2], e, sizeof e);
		ind(g, d);
		switch (tg_ops[in->op].cls) {
		case CLS_CONST:
			fprintf(g->f, "%s[0] = ", o);
			flt(g, in->k);
			fputs(";\n", g->f);
			break;
		case CLS_BIN: {
			int na = shape_numel(&m->val[in->a[0]].sh), nb = shape_numel(&m->val[in->a[1]].sh);
			if ((na == n || na == 1) && (nb == n || nb == 1)) {
				fprintf(g->f, "tg_%s(%s, %s, %d, %s, %d, %d);\n", tg_ops[in->op].name, o, a, na == n, c, nb == n, n);
			} else { /* row broadcast */
				int w = na < n ? na : nb;
				char step[32];
				snprintf(step, sizeof step, " + r * %d", w);
				fprintf(g->f, "for (int r = 0; r < %d; r++) tg_%s(%s%s, %s%s, 1, %s%s, 1, %d);\n", n / w, tg_ops[in->op].name, o, step,
					a, na < n ? "" : step, c, nb < n ? "" : step, w);
			}
			break;
		}
		case CLS_UN:
			fprintf(g->f, "tg_%s(%s, %s, %d);\n", tg_ops[in->op].name, o, a, n);
			break;
		case CLS_MATMUL: {
			int mm, kk, nn;
			matmul_dims(&m->val[in->a[0]].sh, &m->val[in->a[1]].sh, &mm, &kk, &nn);
			const Value *qa = &m->val[in->a[0]], *qb = &m->val[in->a[1]];
			if (qa->qbits && q_axis_for(m, in->a[0], in) == qa->qaxis)
				fprintf(g->f, "tg_matmul_qa(%s, tgq_%s, %d, tgqs_%s, %s, %d, %d, %d);\n", o, qa->name, qa->qbits, qa->name, c, mm, kk, nn);
			else if (qb->qbits && q_axis_for(m, in->a[1], in) == qb->qaxis)
				fprintf(g->f, "tg_matmul_qb(%s, %s, tgq_%s, %d, tgqs_%s, %d, %d, %d);\n", o, a, qb->name, qb->qbits, qb->name, mm, kk, nn);
			else
				fprintf(g->f, "tg_matmul(%s, %s, %s, %d, %d, %d);\n", o, a, c, mm, kk, nn);
			break;
		}
		case CLS_ROW: {
			int rows, cols;
			row_dims(&m->val[in->a[0]].sh, &rows, &cols);
			fprintf(g->f, "tg_%s(%s, %s, %d, %d);\n", tg_ops[in->op].name, o, a, rows, cols);
			break;
		}
		case CLS_RED:
			fprintf(g->f, "tg_%s(%s, %s, %d);\n", tg_ops[in->op].name, o, a, shape_numel(&m->val[in->a[0]].sh));
			break;
		case CLS_OUTER:
			fprintf(g->f, "tg_outer(%s, %s, %s, %d, %d);\n", o, a, c, shape_numel(&m->val[in->a[0]].sh), shape_numel(&m->val[in->a[1]].sh));
			break;
		case CLS_SPMM: {
			int r, k, dd, h;
			spmm_dims(&m->val[in->a[0]].sh, &m->val[in->a[in->op == OP_SPMM_T ? 2 : 1]].sh, &r, &k, &dd, &h);
			if (in->op == OP_SPMM_DX) fprintf(g->f, "tg_spmm_dx(%s, %s, %s, %s, %d, %d, %d, %d);\n", o, a, c, e, r, k, dd, h);
			else if (in->op == OP_SPMM && m->val[in->a[1]].qbits && m->val[in->a[1]].qaxis == 0)
				fprintf(g->f, "tg_spmm_q(%s, %s, tgq_%s, %d, tgqs_%s, %d, %d, %d, %d);\n", o, a, m->val[in->a[1]].name, m->val[in->a[1]].qbits,
					m->val[in->a[1]].name, r, k, dd, h);
			else fprintf(g->f, "tg_%s(%s, %s, %s, %d, %d, %d, %d);\n", tg_ops[in->op].name, o, a, c, r, k, dd, h);
			break;
		}
		case CLS_ROWS: {
			const Shape *s0 = &m->val[in->a[0]].sh;
			if (in->op == OP_TAKE) {
				fprintf(g->f, "tg_take(%s, %s, %s, %d, %d, %d);\n", o, a, c, m->val[in->a[1]].sh.dim[0], s0->dim[0], shape_numel(s0) / s0->dim[0]);
			} else if (in->op == OP_TAKE_T) {
				const Shape *w = &m->val[in->a[2]].sh;
				fprintf(g->f, "tg_take_t(%s, %s, %s, %d, %d, %d);\n", o, a, c, s0->dim[0], w->dim[0], shape_numel(w) / w->dim[0]);
			} else {
				int r = s0->rank == 3 ? s0->dim[0] : 1, k = s0->dim[s0->rank - 2];
				if (in->op == OP_ACTIVE) fprintf(g->f, "tg_active(%s, %s, %d, %d, %d);\n", o, a, r, k, m->val[in->a[1]].sh.dim[0]);
				else fprintf(g->f, "tg_spmm_tc(%s, %s, %s, %s, %d, %d, %d, %d);\n", o, a, c, e, r, k, r * k, m->val[in->out].sh.dim[1]);
			}
			break;
		}
		case CLS_TRANS: {
			const Shape *s = &m->val[in->a[0]].sh;
			fprintf(g->f, "tg_transpose(%s, %s, %d, %d);\n", o, a, s->dim[0], s->dim[1]);
			break;
		}
		case CLS_RESHAPE:
			fprintf(g->f, "tg_copy(%s, %s, %d);\n", o, a, n);
			break;
		case CLS_SCAN: {
			const Scan *s = in->sc;
			char p[64], q[64];
			fprintf(g->f, "/* scan: %d steps%s, %d carr%s, %d sequence(s), %d stack(s) */\n", s->T, s->reverse ? " in reverse" : "",
				s->nc, s->nc == 1 ? "y" : "ies", s->nx, s->ny);
			for (int k = 0; k < s->nc; k++) {
				ref(g, s->c[k], p, sizeof p);
				ref(g, s->init[k], q, sizeof q);
				ind(g, d);
				fprintf(g->f, "tg_copy(%s, %s, %d);\n", p, q, shape_numel(&m->val[s->c[k]].sh));
			}
			ind(g, d);
			fprintf(g->f, "for (int s%d = 0; s%d < %d; s%d++) {\n", d, d, s->T, d);
			ind(g, d + 1);
			if (s->reverse) fprintf(g->f, "const int t%d = %d - s%d;\n", d, s->T - 1, d);
			else fprintf(g->f, "const int t%d = s%d;\n", d, d);
			for (int j = 0; j < s->nx; j++) {
				int w = shape_numel(&m->val[s->xt[j]].sh);
				ref(g, s->xt[j], p, sizeof p);
				ref(g, s->x[j], q, sizeof q);
				ind(g, d + 1);
				fprintf(g->f, "tg_copy(%s, %s + t%d * %d, %d);\n", p, q, d, w, w);
			}
			block(g, in->body, d + 1);
			for (int y = 0; y < s->ny; y++) {
				int w = shape_numel(&m->val[s->y[y]].sh);
				ref(g, s->ys[y], p, sizeof p);
				ref(g, s->y[y], q, sizeof q);
				ind(g, d + 1);
				fprintf(g->f, "tg_copy(%s + t%d * %d, %s, %d);\n", p, d, w, q, w);
			}
			for (int k = 0; k < s->nc; k++) {
				ref(g, s->c[k], p, sizeof p);
				ref(g, s->next[k], q, sizeof q);
				ind(g, d + 1);
				fprintf(g->f, "tg_copy(%s, %s, %d);\n", p, q, shape_numel(&m->val[s->c[k]].sh));
			}
			ind(g, d);
			fputs("}\n", g->f);
			if (s->tid >= 0) {
				ind(g, d);
				fprintf(g->f, "tg_%s_steps[%d] = %d;\n", m->name, s->tid, s->T);
			}
			break;
		}
		case CLS_THINK: {
			char init[64], y[64];
			ref(g, in->init, init, sizeof init);
			ref(g, in->yield, y, sizeof y);
			fprintf(g->f, "/* think: latent state %s, budget %d */\n", o, in->maxit);
			ind(g, d);
			fprintf(g->f, "tg_copy(%s, %s, %d);\n", o, init, n);
			if (in->halt >= 0) {
				ind(g, d);
				fprintf(g->f, "%s sv%d = %s; /* probability of not having halted yet */\n", TY, in->tid, FX ? "65536" : "1.0f");
			}
			ind(g, d);
			fprintf(g->f, "for (it[%d] = 0; it[%d] < %d;) {\n", in->tid, in->tid, in->maxit);
			block(g, in->body, d + 1);
			ind(g, d + 1);
			fprintf(g->f, "it[%d]++;\n", in->tid);
			ind(g, d + 1);
			fprintf(g->f, "%s d = tg_delta(%s, %s, %d);\n", TY, y, o, n);
			ind(g, d + 1);
			fprintf(g->f, "tg_copy(%s, %s, %d);\n", o, y, n);
			if (in->eps >= 0) {
				ind(g, d + 1);
				fputs("if (d <= ", g->f);
				if (FX) fprintf(g->f, "%lld", to_fixed(in->eps, 16) > 0 || in->eps == 0 ? to_fixed(in->eps, 16) : 1); /* a positive threshold stays positive */
				else flt(g, in->eps);
				fputs(") break;\n", g->f);
			} else {
				ind(g, d + 1);
				fputs("(void)d;\n", g->f);
			}
			if (in->halt >= 0) {
				char hp[64];
				ref(g, in->halt, hp, sizeof hp);
				ind(g, d + 1);
				if (FX)
					fprintf(g->f, "{ tg_t p = %s[0] < 0 ? 0 : %s[0] > 65536 ? 65536 : %s[0]; sv%d = tg_fmul(sv%d, 65536 - p); if (65536 - sv%d > %lld) break; }\n",
						hp, hp, hp, in->tid, in->tid, in->tid, to_fixed(in->hthr, 16));
				else
					fprintf(g->f, "{ float p = %s[0] < 0.0f ? 0.0f : %s[0] > 1.0f ? 1.0f : %s[0]; sv%d *= 1.0f - p; if (1.0f - sv%d > %.9ef) break; }\n",
						hp, hp, hp, in->tid, in->tid, (double)in->hthr);
			}
			ind(g, d);
			fputs("}\n", g->f);
			ind(g, d);
			fprintf(g->f, "tg_%s_steps[%d] = it[%d];\n", m->name, in->tid, in->tid);
			break;
		}
		}
	}
}

static void signature(G *g)
{
	const Module *m = g->m;
	fprintf(g->f, "void tg_%s_run(", m->name);
	for (int i = 0; i < m->ninputs; i++)
		fprintf(g->f, "const %s *tgi_%s, ", TY, m->val[m->inputs[i]].name);
	fprintf(g->f, "%s *tg_out)", TY);
}

/* Test driver for fixed-point units: decimal text in and out with integer
 * arithmetic only, so the whole program runs where there is no FPU. */
static void driver_fixed(G *g, int outn)
{
	const Module *m = g->m;
	FILE *f = g->f;
	fprintf(f, "\n#ifdef TG_MAIN\n#include <stdio.h>\n\n"
		   "static int tg_parse(const char *s, tg_t *v, int n)\n{\n"
		   "\tint k = 0;\n"
		   "\tfor (;;) {\n\t\tif (k == n) return 0;\n\t\ts = tg_fx_parse(s, &v[k++]);\n"
		   "\t\tif (!s) return 0;\n\t\tif (!*s) break;\n\t\tif (*s != ',') return 0;\n\t\ts++;\n\t}\n"
		   "\treturn k == n;\n}\n\n"
		   "int main(int argc, char **argv)\n{\n\tchar buf[24];\n");
	for (int i = 0; i < m->ninputs; i++) fprintf(f, "\tstatic tg_t in%d[%d];\n", i, shape_numel(&m->val[m->inputs[i]].sh));
	fprintf(f, "\tstatic tg_t out[%d];\n", outn);
	fprintf(f, "\tif (argc < 2 || (argc - 1) %% %d) { fputs(\"usage: unit inputs...\\n\", stderr); return 2; }\n", m->ninputs);
	fprintf(f, "\tfor (int s = 1; s < argc; s += %d) {\n", m->ninputs);
	for (int i = 0; i < m->ninputs; i++)
		fprintf(f, "\t\tif (!tg_parse(argv[s + %d], in%d, %d)) { fputs(\"bad input '%s'\\n\", stderr); return 2; }\n", i, i,
			shape_numel(&m->val[m->inputs[i]].sh), m->val[m->inputs[i]].name);
	fprintf(f, "\t\ttg_%s_run(", m->name);
	for (int i = 0; i < m->ninputs; i++) fprintf(f, "in%d, ", i);
	fputs("out);\n", f);
	fprintf(f, "\t\tfor (int i = 0; i < %d; i++) { tg_fx_format(out[i], buf); printf(\"%%s%%s\", i ? \" \" : \"\", buf); }\n\t\tputchar('\\n');\n", outn);
	if (m->nthink) fprintf(f, "\t\tfor (int i = 0; i < %d; i++) printf(\"steps %%d %%d\\n\", i, tg_%s_steps[i]);\n", m->nthink, m->name);
	fputs("\t}\n\treturn 0;\n}\n#endif\n", f);
}

void cgen_mode(const Module *m, FILE *f, int fixed)
{
	FX = fixed;
	cgen(m, f);
	FX = 0;
}

void cgen(const Module *m, FILE *f)
{
	G g = { m, f };
	const char *nm = m->name;
	int params = 0, states = 0;
	for (int v = 0; v < m->nval; v++) {
		if (m->val[v].kind == V_PARAM) params += shape_numel(&m->val[v].sh);
		if (m->val[v].kind == V_STATE) states += shape_numel(&m->val[v].sh);
	}
	int outn = shape_numel(&m->val[m->output].sh);

	fprintf(f, "/* Technograd generated unit: model '%s'. Do not edit.\n", nm);
	fprintf(f, " * arena %d bytes, parameters %d bytes, %d think loop(s).\n", m->arena * 4, params * 4, m->nthink);
	fputs(" * Entry: ", f);
	signature(&g);
	if (m->nupd)
		fprintf(f, "\n * Self-updating: %d byte(s) of state in RAM, rewritten at the end of every run.\n"
			   " * State arrays tg_%s_state_<name> are exported so firmware can save and restore\n"
			   " * what the model learned; tg_%s_reset() restores the initial values.",
			states * 4, nm, nm);
	if (FX) fputs("\n * Fixed point: every value is Q16.16 in tg_t (int32_t, 1.0 = 65536); no floating point.", f);
	fputs("\n */\n", f);
	const char *const *rt = FX ? tg_rtq_lines : tg_rt_lines;
	for (int i = 0; rt[i]; i++) fputs(rt[i], f);
	fputc('\n', f);

	for (int i = 0; i < m->ninputs; i++) {
		const Value *x = &m->val[m->inputs[i]];
		fprintf(f, "#define TG_%s_IN_%s %d\n", nm, x->name, shape_numel(&x->sh));
	}
	fprintf(f, "#define TG_%s_NIN %d\n#define TG_%s_OUT %d\n#define TG_%s_NTHINK %d\n#define TG_%s_ARENA_BYTES %d\n\n",
		nm, m->ninputs, nm, outn, nm, m->nthink, nm, m->arena * 4);

	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_PARAM || x->dead) continue;
		int n = shape_numel(&x->sh);
		if (x->qbits) { /* codes and per-channel scales */
			int nb = x->qbits == 8 ? n : (n + 1) / 2, nch = x->sh.dim[x->qaxis];
			fprintf(f, "static const signed char tgq_%s[%d] = { /* int%d codes */", x->name, nb, x->qbits);
			for (int i = 0; i < nb; i++) fprintf(f, "%s%d%s", i % 16 == 0 ? "\n\t" : "", x->q[i], i + 1 < nb ? ", " : "\n");
			fprintf(f, "};\nstatic const %s tgqs_%s[%d] = {", FX ? "int32_t" : "float", x->name, nch);
			for (int i = 0; i < nch; i++) {
				if (i % 6 == 0) fputs("\n\t", f);
				if (FX) fprintf(f, "%lld", to_fixed(x->qs[i], 24)); /* Q7.24 scales */
				else flt(&g, x->qs[i]);
				fputs(i + 1 < nch ? ", " : "\n", f);
			}
			fputs("};\n", f);
			if (!needs_f32(m, v)) continue;
		}
		fprintf(f, "static const %s tgp_%s[%d] = {", TY, x->name, n);
		for (int i = 0; i < n; i++) {
			if (i % 6 == 0) fputs("\n\t", f);
			flt(&g, x->data[i]);
			fputs(i + 1 < n ? ", " : "\n", f);
		}
		fputs("};\n", f);
	}
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_STATE) continue;
		int n = shape_numel(&x->sh);
		for (int pass = 0; pass < 2; pass++) { /* pristine const copy, then the RAM state */
			if (pass == 0) fprintf(f, "static const %s tgs0_%s[%d] = {", TY, x->name, n);
			else fprintf(f, "#define TG_%s_STATE_%s %d\n%s tg_%s_state_%s[%d] = {", nm, x->name, n, TY, nm, x->name, n);
			for (int i = 0; i < n; i++) {
				if (i % 6 == 0) fputs("\n\t", f);
				flt(&g, x->data[i]);
				fputs(i + 1 < n ? ", " : "\n", f);
			}
			fputs("};\n", f);
		}
	}
	fprintf(f, "\nstatic %s tg_%s_arena[%d];\nint tg_%s_steps[%d];\n\n", TY, nm, m->arena ? m->arena : 1, nm, m->nthink ? m->nthink : 1);
	if (m->nupd || states) {
		fprintf(f, "void tg_%s_reset(void);\nvoid tg_%s_reset(void)\n{\n", nm, nm);
		for (int v = 0; v < m->nval; v++)
			if (m->val[v].kind == V_STATE)
				fprintf(f, "\ttg_copy(tg_%s_state_%s, tgs0_%s, %d);\n", nm, m->val[v].name, m->val[v].name, shape_numel(&m->val[v].sh));
		fputs("}\n\n", f);
	}

	signature(&g);
	fprintf(f, ";\n");
	signature(&g);
	fprintf(f, "\n{\n\t%s *const A = tg_%s_arena;\n", TY, nm);
	if (m->nthink) fprintf(f, "\tint it[%d];\n", m->nthink);
	fprintf(f, "\t(void)A;\n");
	block(&g, &m->top, 1);
	char o[64];
	ref(&g, m->output, o, sizeof o);
	fprintf(f, "\ttg_copy(tg_out, %s, %d);\n", o, outn);
	if (m->nupd) {
		fputs("\t/* commit updates: stage all sources first, then write the states */\n", f);
		int off = m->stage;
		for (int i = 0; i < m->nupd; i++) {
			char s[64];
			int n = shape_numel(&m->val[m->upd_src[i]].sh);
			ref(&g, m->upd_src[i], s, sizeof s);
			fprintf(f, "\ttg_copy(A + %d, %s, %d);\n", off, s, n);
			off += (n + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
			if (m->upd_rows[i] >= 0) {
				int nr = shape_numel(&m->val[m->upd_rows[i]].sh);
				ref(&g, m->upd_rows[i], s, sizeof s);
				fprintf(f, "\ttg_copy(A + %d, %s, %d);\n", off, s, nr);
				off += (nr + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
			}
		}
		off = m->stage;
		for (int i = 0; i < m->nupd; i++) {
			char s[64];
			const Shape *st = &m->val[m->upd_state[i]].sh;
			int n = shape_numel(&m->val[m->upd_src[i]].sh);
			ref(&g, m->upd_state[i], s, sizeof s);
			if (m->upd_rows[i] < 0) {
				fprintf(f, "\ttg_copy(%s, A + %d, %d);\n", s, off, n);
				off += (n + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
			} else { /* row update: only the listed rows are written */
				int nr = shape_numel(&m->val[m->upd_rows[i]].sh);
				int ro = off + (n + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
				fprintf(f, "\ttg_put_rows(%s, A + %d, A + %d, %d, %d, %d);\n", s, off, ro, nr, st->dim[0], shape_numel(st) / st->dim[0]);
				off = ro + (nr + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
			}
		}
	}
	fputs("}\n", f);

	if (FX) {
		driver_fixed(&g, outn);
		return;
	}
	/* Optional test driver; output format matches `tgc run`. */
	fprintf(f, "\n#ifdef TG_MAIN\n#include <stdio.h>\n#include <stdlib.h>\n\n"
		   "static int tg_parse(const char *s, float *v, int n)\n{\n"
		   "\tint k = 0;\n\tchar *e;\n"
		   "\tfor (;;) {\n\t\tif (k == n) return 0;\n\t\tv[k++] = strtof(s, &e);\n"
		   "\t\tif (e == s) return 0;\n\t\tif (!*e) break;\n\t\tif (*e != ',') return 0;\n\t\ts = e + 1;\n\t}\n"
		   "\treturn k == n;\n}\n\n"
		   "int main(int argc, char **argv)\n{\n");
	for (int i = 0; i < m->ninputs; i++) {
		const Value *x = &m->val[m->inputs[i]];
		fprintf(f, "\tstatic float in%d[%d];\n", i, shape_numel(&x->sh));
	}
	fprintf(f, "\tstatic float out[%d];\n", outn);
	/* Each group of NIN arguments is one sample; samples run in order in one
	 * process, so a self-updating unit carries its state across them. */
	fprintf(f, "\tif (argc < 2 || (argc - 1) %% %d) { fprintf(stderr, \"usage: %%s", m->ninputs);
	for (int i = 0; i < m->ninputs; i++) fprintf(f, " %s", m->val[m->inputs[i]].name);
	fputs(" [more samples...]\\n\", argv[0]); return 2; }\n", f);
	fprintf(f, "\tfor (int s = 1; s < argc; s += %d) {\n", m->ninputs);
	for (int i = 0; i < m->ninputs; i++) {
		const Value *x = &m->val[m->inputs[i]];
		fprintf(f, "\t\tif (!tg_parse(argv[s + %d], in%d, %d)) { fprintf(stderr, \"input '%s' needs %d comma-separated values\\n\"); return 2; }\n",
			i, i, shape_numel(&x->sh), x->name, shape_numel(&x->sh));
	}
	fprintf(f, "\t\ttg_%s_run(", nm);
	for (int i = 0; i < m->ninputs; i++) fprintf(f, "in%d, ", i);
	fputs("out);\n", f);
	fprintf(f, "\t\tfor (int i = 0; i < %d; i++) printf(\"%%s%%.9g\", i ? \" \" : \"\", (double)out[i]);\n\t\tputchar('\\n');\n", outn);
	if (m->nthink)
		fprintf(f, "\t\tfor (int i = 0; i < %d; i++) printf(\"steps %%d %%d\\n\", i, tg_%s_steps[i]);\n", m->nthink, nm);
	fputs("\t}\n\treturn 0;\n}\n#endif\n", f);
}
