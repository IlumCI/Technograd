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

static void ind(G *g, int d)
{
	for (int i = 0; i < d; i++) fputc('\t', g->f);
}

static void flt(G *g, float v)
{
	if (!isfinite(v)) die(NULL, 0, "non-finite constant in model");
	fprintf(g->f, "%.9ef", (double)v);
}

static void block(G *g, const Block *b, int d)
{
	const Module *m = g->m;
	char o[64], a[64], c[64];
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		int n = shape_numel(&m->val[in->out].sh);
		ref(g, in->out, o, sizeof o);
		if (in->na > 0) ref(g, in->a[0], a, sizeof a);
		if (in->na > 1) ref(g, in->a[1], c, sizeof c);
		ind(g, d);
		switch (tg_ops[in->op].cls) {
		case CLS_CONST:
			fprintf(g->f, "%s[0] = ", o);
			flt(g, in->k);
			fputs(";\n", g->f);
			break;
		case CLS_BIN: {
			int sa = shape_numel(&m->val[in->a[0]].sh) == n;
			int sb = shape_numel(&m->val[in->a[1]].sh) == n;
			fprintf(g->f, "tg_%s(%s, %s, %d, %s, %d, %d);\n", tg_ops[in->op].name, o, a, sa, c, sb, n);
			break;
		}
		case CLS_UN:
			fprintf(g->f, "tg_%s(%s, %s, %d);\n", tg_ops[in->op].name, o, a, n);
			break;
		case CLS_MATMUL: {
			int mm, kk, nn;
			matmul_dims(&m->val[in->a[0]].sh, &m->val[in->a[1]].sh, &mm, &kk, &nn);
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
		case CLS_TRANS: {
			const Shape *s = &m->val[in->a[0]].sh;
			fprintf(g->f, "tg_transpose(%s, %s, %d, %d);\n", o, a, s->dim[0], s->dim[1]);
			break;
		}
		case CLS_THINK: {
			char init[64], y[64];
			ref(g, in->init, init, sizeof init);
			ref(g, in->yield, y, sizeof y);
			fprintf(g->f, "/* think: latent state %s, budget %d */\n", o, in->maxit);
			ind(g, d);
			fprintf(g->f, "tg_copy(%s, %s, %d);\n", o, init, n);
			ind(g, d);
			fprintf(g->f, "for (it[%d] = 0; it[%d] < %d;) {\n", in->tid, in->tid, in->maxit);
			block(g, in->body, d + 1);
			ind(g, d + 1);
			fprintf(g->f, "it[%d]++;\n", in->tid);
			ind(g, d + 1);
			fprintf(g->f, "float d = tg_delta(%s, %s, %d);\n", y, o, n);
			ind(g, d + 1);
			fprintf(g->f, "tg_copy(%s, %s, %d);\n", o, y, n);
			if (in->eps >= 0) {
				ind(g, d + 1);
				fputs("if (d <= ", g->f);
				flt(g, in->eps);
				fputs(") break;\n", g->f);
			} else {
				ind(g, d + 1);
				fputs("(void)d;\n", g->f);
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
		fprintf(g->f, "const float *tgi_%s, ", m->val[m->inputs[i]].name);
	fputs("float *tg_out)", g->f);
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
	fputs("\n */\n", f);
	for (int i = 0; tg_rt_lines[i]; i++) fputs(tg_rt_lines[i], f);
	fputc('\n', f);

	for (int i = 0; i < m->ninputs; i++) {
		const Value *x = &m->val[m->inputs[i]];
		fprintf(f, "#define TG_%s_IN_%s %d\n", nm, x->name, shape_numel(&x->sh));
	}
	fprintf(f, "#define TG_%s_NIN %d\n#define TG_%s_OUT %d\n#define TG_%s_NTHINK %d\n#define TG_%s_ARENA_BYTES %d\n\n",
		nm, m->ninputs, nm, outn, nm, m->nthink, nm, m->arena * 4);

	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_PARAM) continue;
		int n = shape_numel(&x->sh);
		fprintf(f, "static const float tgp_%s[%d] = {", x->name, n);
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
			if (pass == 0) fprintf(f, "static const float tgs0_%s[%d] = {", x->name, n);
			else fprintf(f, "#define TG_%s_STATE_%s %d\nfloat tg_%s_state_%s[%d] = {", nm, x->name, n, nm, x->name, n);
			for (int i = 0; i < n; i++) {
				if (i % 6 == 0) fputs("\n\t", f);
				flt(&g, x->data[i]);
				fputs(i + 1 < n ? ", " : "\n", f);
			}
			fputs("};\n", f);
		}
	}
	fprintf(f, "\nstatic float tg_%s_arena[%d];\nint tg_%s_steps[%d];\n\n", nm, m->arena ? m->arena : 1, nm, m->nthink ? m->nthink : 1);
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
	fprintf(f, "\n{\n\tfloat *const A = tg_%s_arena;\n", nm);
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
			int n = shape_numel(&m->val[m->upd_state[i]].sh);
			ref(&g, m->upd_src[i], s, sizeof s);
			fprintf(f, "\ttg_copy(A + %d, %s, %d);\n", off, s, n);
			off += (n + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
		}
		off = m->stage;
		for (int i = 0; i < m->nupd; i++) {
			char s[64];
			int n = shape_numel(&m->val[m->upd_state[i]].sh);
			ref(&g, m->upd_state[i], s, sizeof s);
			fprintf(f, "\ttg_copy(%s, A + %d, %d);\n", s, off, n);
			off += (n + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
		}
	}
	fputs("}\n", f);

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
