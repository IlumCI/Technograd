/* One-command training: `tgc train SOURCE`.
 *
 *   1. load SOURCE (any format data.c knows), infer column kinds, pick the
 *      target (or --target)
 *   2. seeded train/validation split; fit the featurization on training rows
 *   3. generate an editable Technograd training model that uses
 *      `train <loss> with <optimizer>`, and a frozen inference model; text
 *      features enter as sparse ELLPACK rows through `spmm`, so their cost
 *      scales with the words in a row, not with the hash dimension
 *   4. train in mini-batches for several epochs, evaluate each epoch on a
 *      single-row forward-only model, keep the best weights by validation
 *      loss, stop early
 *   5. write OUTDIR/{model.tg, infer.tg, infer.c, weights.*.bin,
 *      features.tgf, report.txt}
 *
 * `tgc predict OUTDIR SOURCE` featurizes new raw data with the saved spec and
 * runs the inference model; `tgc data inspect SOURCE` shows what was inferred;
 * `tgc data prep SOURCE -o out.csv` writes the numeric features. */
#include "tg.h"
#include "data.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {
	const char *source, *target, *out, *optimizer, *init, *column;
	int epochs, hidden, max_rows, seed, text_dim, batch, ssm, layers, mimo, steps;
	double lr, val;
} Opts;

static void usage_data(void)
{
	fputs("usage:\n"
	      "  tgc data inspect SOURCE [--target COL]\n"
	      "  tgc data prep SOURCE -o OUT.csv [--target COL]\n"
	      "  tgc train SOURCE [-o DIR] [--target COL] [--epochs N] [--hidden H] [--lr X]\n"
	      "                   [--optimizer sgd|momentum|adam|adamw|muon] [--val FRACTION] [--max-rows N]\n"
	      "                   [--batch N] [--text-dim N] [--model bow|ssm] [--layers N] [--mimo R] [--seed S]\n"
	      "                   [--init PRETRAINED_DIR]\n"
	      "  tgc pretrain SOURCE [-o DIR] [--column COL] [--epochs N] [--steps N] [--batch N] [--lr X]\n"
	      "                   [--layers N] [--mimo R] [--max-rows N] [--seed S]\n"
	      "  tgc predict DIR SOURCE [-o OUT.csv] [--model QUANTIZED.tgir] [--inputs ROWS.csv]\n"
	      "SOURCE: a .csv/.tsv/.json/.jsonl/.npy file, or hf:OWNER/NAME[/CONFIG[/SPLIT]]\n",
	      stderr);
	exit(2);
}

static void parse_opts(int argc, char **argv, int first, Opts *o)
{
	for (int i = first; i < argc; i++) {
		const char *a = argv[i];
		const char *v = i + 1 < argc ? argv[i + 1] : NULL;
		if (!v && strncmp(a, "--", 2) == 0) usage_data();
		if (!strcmp(a, "-o")) o->out = v;
		else if (!strcmp(a, "--target")) o->target = v;
		else if (!strcmp(a, "--epochs")) o->epochs = atoi(v);
		else if (!strcmp(a, "--hidden")) o->hidden = atoi(v);
		else if (!strcmp(a, "--lr")) o->lr = atof(v);
		else if (!strcmp(a, "--optimizer")) o->optimizer = v;
		else if (!strcmp(a, "--val")) o->val = atof(v);
		else if (!strcmp(a, "--max-rows")) o->max_rows = atoi(v);
		else if (!strcmp(a, "--seed")) o->seed = atoi(v);
		else if (!strcmp(a, "--text-dim")) o->text_dim = atoi(v);
		else if (!strcmp(a, "--batch")) o->batch = atoi(v);
		else if (!strcmp(a, "--layers")) o->layers = atoi(v);
		else if (!strcmp(a, "--mimo")) o->mimo = atoi(v);
		else if (!strcmp(a, "--init")) o->init = v;
		else if (!strcmp(a, "--column")) o->column = v;
		else if (!strcmp(a, "--steps")) o->steps = atoi(v);
		else if (!strcmp(a, "--model")) {
			if (strcmp(v, "bow") && strcmp(v, "ssm")) die(NULL, 0, "--model is 'bow' (bag of words) or 'ssm' (bag of words + selective state-space sequence layer)");
			o->ssm = !strcmp(v, "ssm");
		}
		else if (!o->source && a[0] != '-') { o->source = a; continue; }
		else die(NULL, 0, "unexpected argument '%s'", a);
		i++;
	}
	if (!o->source) usage_data();
}

static uint32_t rs;
static uint32_t rnd(void)
{
	rs ^= rs << 13;
	rs ^= rs >> 17;
	rs ^= rs << 5;
	return rs;
}

static void shuffle(int *v, int n)
{
	for (int i = n - 1; i > 0; i--) {
		int k = (int)(rnd() % (uint32_t)(i + 1)), t = v[i];
		v[i] = v[k];
		v[k] = t;
	}
}

static void model_name(const char *src, char *out, size_t n)
{
	const char *b = strrchr(src, '/');
	b = b ? b + 1 : src + (strncmp(src, "hf:", 3) ? 0 : 3);
	size_t k = 0;
	for (; *b && *b != '.' && k + 1 < n; b++) out[k++] = isalnum((unsigned char)*b) ? (char)tolower((unsigned char)*b) : '_';
	out[k] = 0;
	if (!k || isdigit((unsigned char)out[0])) snprintf(out, n, "m_%.40s", k ? out : "data");
}

static char *join(const char *dir, const char *file)
{
	char *p = xmalloc(strlen(dir) + strlen(file) + 2);
	sprintf(p, "%s/%s", dir, file);
	return p;
}

static void write_text(const char *path, const char *text)
{
	FILE *f = fopen(path, "w");
	if (!f) die(NULL, 0, "cannot write '%s'", path);
	fputs(text, f);
	if (fclose(f)) die(NULL, 0, "write failed '%s'", path);
}

enum { GEN_TRAIN, GEN_INFER, GEN_EVAL };

/* Feature layout: spec_apply writes one dense row of s->dim floats. Text
 * blocks (hashed, mostly zero) are moved into a sparse ELLPACK row of K
 * (index, value) pairs over a text space of ds buckets; everything else stays
 * in a dense row of dd floats. */
typedef struct {
	int dd, ds, k;
	int sf, T, V, hs; /* sequence path: text feature index (-1: none), steps, vocabulary, model width */
	int layers, mimo; /* Mamba-3 blocks, MIMO rank */
} Layout;

static Layout layout_of(const Spec *s)
{
	Layout l = { 0, 0, 0, -1, 0, 0, 0, 0, 0 };
	for (int i = 0; i < s->nf; i++) {
		if (s->f[i].kind == COL_TEXT) l.ds += s->f[i].dim;
		else l.dd += s->f[i].dim;
	}
	return l;
}

/* Split a featurized row. Writes up to l->k pairs (the largest |value|s when
 * the row has more), pads with (0, 0); returns the row's nonzero count. With
 * xd NULL and l->k 0 it only counts. */
static int split_row(const Spec *s, const Layout *l, const float *x, float *xd, float *ell)
{
	int o = 0, od = 0, ot = 0, nnz = 0;
	for (int i = 0; i < l->k; i++) ell[2 * i] = ell[2 * i + 1] = 0;
	for (int i = 0; i < s->nf; i++) {
		const Feat *f = &s->f[i];
		if (f->kind != COL_TEXT) {
			if (xd) memcpy(xd + od, x + o, (size_t)f->dim * sizeof *xd);
			od += f->dim;
		} else {
			for (int d = 0; d < f->dim; d++) {
				float v = x[o + d];
				if (v == 0) continue;
				nnz++;
				if (!l->k) continue;
				int at = nnz <= l->k ? nnz - 1 : -1;
				if (at < 0) { /* full: replace the smallest kept entry if this one is larger */
					for (int j = 0; j < l->k; j++)
						if (at < 0 || fabsf(ell[2 * j + 1]) < fabsf(ell[2 * at + 1])) at = j;
					if (fabsf(ell[2 * at + 1]) >= fabsf(v)) continue;
				}
				ell[2 * at] = (float)(ot + d);
				ell[2 * at + 1] = v;
			}
			ot += f->dim;
		}
		o += f->dim;
	}
	return nnz;
}

/* ---- the sequence path: stacked Mamba-3 blocks ------------------------------
 * Per block, with u = rmsnorm(e) * g1 (model width D, time-major rows TB):
 *   x = u Wx (heads Hh x P)   dt = softplus(u Wdt + bdt)   lam = sigmoid(u Wl + bl)
 *   B, C = BCNorm(u WB, u WC) + per-head biases (rank R x state N)
 *   rotation angles: cumulative sum over time of dt * (u Wth), applied to B and
 *   C (the complex state of Mamba-3 as data-dependent RoPE)
 *   h_t = al h_{t-1} + be B_{t-1} X_{t-1}^T + ga B_t X_t^T   (exponential-trapezoidal;
 *   al = exp(-dt A), be = (1 - lam) dt al, ga = lam dt), X = x scaled per rank (MIMO)
 *   y_t = C_t^T h_t, mixed over ranks, + dk x, gated by silu(u Wz), projected by Wo
 *   e += that; then e += SwiGLU(rmsnorm(e) * g2)
 * The products B X^T for all positions are computed before the scan; the scan
 * carries only the state and the previous product. */
#define SSM_HEADS 4
#define SSM_N 16
#define PW(...) (*o += (size_t)snprintf(b + *o, cap - *o, __VA_ARGS__))

static void ssm_w(char *b, size_t cap, size_t *o, int mode, const char *n, int l, const char *shape, const char *init)
{
	if (mode == GEN_INFER) PW("param %s_%d : %s = file(\"weights.%s_%d.bin\")\n", n, l, shape, n, l);
	else if (mode == GEN_EVAL) PW("param %s_%d : %s = zeros\n", n, l, shape);
	else PW("state %s_%d : %s = %s\n", n, l, shape, init);
}

static void gen_ssm_weights(char *b, size_t cap, size_t *o, const Layout *l, int Bm, int H, int mode)
{ /* H: width of the classifier the pooled sequence feeds (wp), 0 for none */
	int D = l->hs, Di = 2 * D, Hh = SSM_HEADS, P = Di / Hh, N = SSM_N, R = l->mimo, F = 2 * D, L = l->layers, T = l->T;
	char sh[64], in[64];
	if (mode == GEN_INFER) PW("param E : f32[%d, %d] = file(\"weights.E.bin\")\n", l->V, D);
	else if (mode == GEN_EVAL) PW("param E : f32[%d, %d] = zeros\n", l->V, D);
	else PW("state E : f32[%d, %d] = rand(31, 1)\n", l->V, D);
	for (int k = 1; k <= L; k++) {
		int sd = 100 * k;
		double a = sqrt(3.0 / D), ao = sqrt(3.0 / Di) / sqrt(2.0 * L), a2 = sqrt(3.0 / F) / sqrt(2.0 * L);
		struct { const char *n; int r, c; double s; } w[] = {
			{ "Wx", D, Di, a }, { "Wz", D, Di, a }, { "WB", D, R * N, a }, { "WC", D, R * N, a }, { "Wdt", D, Hh, a },
			{ "Wl", D, Hh, a }, { "Wth", D, Hh * N / 2, a }, { "Wo", Di, D, ao }, { "W1", D, F, a }, { "W3", D, F, a }, { "W2", F, D, a2 },
		};
		for (int i = 0; i < (int)(sizeof w / sizeof *w); i++) {
			snprintf(sh, sizeof sh, "f32[%d, %d]", w[i].r, w[i].c);
			snprintf(in, sizeof in, "rand(%d, %.6g)", sd + i, w[i].s);
			ssm_w(b, cap, o, mode, w[i].n, k, sh, in);
		}
		snprintf(sh, sizeof sh, "f32[%d]", D);
		ssm_w(b, cap, o, mode, "g1", k, sh, "ones");
		ssm_w(b, cap, o, mode, "g2", k, sh, "ones");
		snprintf(sh, sizeof sh, "f32[%d]", N);
		ssm_w(b, cap, o, mode, "gB", k, sh, "ones");
		ssm_w(b, cap, o, mode, "gC", k, sh, "ones");
		snprintf(sh, sizeof sh, "f32[%d, 1, %d]", Hh, N);
		ssm_w(b, cap, o, mode, "bB", k, sh, "ones");
		ssm_w(b, cap, o, mode, "bC", k, sh, "ones");
		snprintf(sh, sizeof sh, "f32[%d, %d, %d]", Hh, R, P);
		ssm_w(b, cap, o, mode, "Wxr", k, sh, "ones");
		{ /* output mix over ranks: 1/R */
			if (mode == GEN_TRAIN) PW("state Wyr_%d : %s = [", k, sh);
			else ssm_w(b, cap, o, mode, "Wyr", k, sh, "");
			if (mode == GEN_TRAIN) {
				for (int h = 0; h < Hh; h++) {
					PW("%s[", h ? ", " : "");
					for (int r = 0; r < R; r++) {
						PW("%s[", r ? ", " : "");
						for (int p = 0; p < P; p++) PW("%s%.6g", p ? ", " : "", 1.0 / R);
						PW("]");
					}
					PW("]");
				}
				PW("]\n");
			}
		}
		snprintf(sh, sizeof sh, "f32[%d]", Di);
		ssm_w(b, cap, o, mode, "dk", k, sh, "ones");
		snprintf(sh, sizeof sh, "f32[%d]", Hh);
		ssm_w(b, cap, o, mode, "bl", k, sh, "zeros");
		for (int which = 0; which < 2; which++) { /* log-spaced step sizes [0.001, 0.1] and decay rates [1, 16] per head, as in Mamba-2 */
			char v[512];
			size_t q = (size_t)snprintf(v, sizeof v, "[");
			for (int h = 0; h < Hh; h++) {
				double f = Hh > 1 ? (double)h / (Hh - 1) : 0;
				double x = which == 0 ? exp(log(0.001) + f * (log(0.1) - log(0.001))) : exp(f * log(16.0));
				q += (size_t)snprintf(v + q, sizeof v - q, "%s%.6g", h ? ", " : "", log(expm1(x))); /* softplus^-1 */
			}
			snprintf(v + q, sizeof v - q, "]");
			ssm_w(b, cap, o, mode, which == 0 ? "bdt" : "la", k, sh, v);
		}
	}
	snprintf(sh, sizeof sh, "f32[%d]", D);
	if (mode == GEN_INFER) PW("param gf : %s = file(\"weights.gf.bin\")\n", sh);
	else if (mode == GEN_EVAL) PW("param gf : %s = zeros\n", sh);
	else PW("state gf : %s = ones\n", sh);
	/* constants */
	PW("param sel2 : f32[2] = [0, 1]\nparam onesT : f32[%d] = ones\nparam Ltri : f32[%d, %d] = [", T, T, T);
	for (int i = 0; i < T; i++) {
		PW("%s[", i ? ", " : "");
		for (int j = 0; j < T; j++) PW("%s%d", j ? ", " : "", j <= i);
		PW("]");
	}
	PW("]\nparam dup : f32[%d, %d] = [", N / 2, N); /* one angle per channel pair */
	for (int i = 0; i < N / 2; i++) {
		PW("%s[", i ? ", " : "");
		for (int j = 0; j < N; j++) PW("%s%d", j ? ", " : "", j / 2 == i);
		PW("]");
	}
	PW("]\nparam J : f32[2, 2] = [[0, 1], [-1, 0]]\n"); /* (v0, v1) @ J = (-v1, v0): a quarter turn of each channel pair */
	PW("param h0 : f32[%d, %d, %d] = zeros\n", Bm * Hh, N, P);
	if (!H) return;
	snprintf(sh, sizeof sh, "f32[%d, %d]", D, H);
	snprintf(in, sizeof in, "rand(29, %.6g)", sqrt(6.0 / (D + H)));
	if (mode == GEN_INFER) PW("param wp : %s = file(\"weights.wp.bin\")\n", sh);
	else if (mode == GEN_EVAL) PW("param wp : %s = zeros\n", sh);
	else PW("state wp : %s = %s\n", sh, in);
}

static void gen_ssm_def(char *b, size_t cap, size_t *o, const Layout *l, int Bm, int pool)
{
	int D = l->hs, Di = 2 * D, Hh = SSM_HEADS, P = Di / Hh, N = SSM_N, R = l->mimo, T = l->T, TB = T * Bm, G = TB * Hh, BH = Bm * Hh;
	if (pool) PW("def ssm(tok: f32[%d, 1, 2]) -> f32[%d, %d]:\n", TB, Bm, D);
	else PW("def states(tok: f32[%d, 1, 2]) -> f32[%d, %d]:\n", TB, TB, D);
	PW("    m = reshape(reshape(tok, %d, 2) @ sel2, %d, 1)\n", TB, TB);
	PW("    e0 = spmm(tok, E)\n");
	for (int k = 1; k <= l->layers; k++) {
		int p = k - 1;
		PW("    # block %d: Mamba-3 mixer\n", k);
		PW("    u = rmsnorm(e%d) * g1_%d\n", p, k);
		PW("    x = u @ Wx_%d\n", k);
		PW("    dt = softplus(u @ Wdt_%d + bdt_%d) * m\n", k, k);
		PW("    lam = sigmoid(u @ Wl_%d + bl_%d)\n", k, k);
		PW("    al = exp(-(dt * softplus(la_%d)))\n", k);
		PW("    th = reshape(u @ Wth_%d, %d, %d, %d) * reshape(dt, %d, %d, 1)\n", k, TB, Hh, N / 2, TB, Hh);
		PW("    ph = reshape(reshape(Ltri @ reshape(th, %d, %d), %d, %d) @ dup, %d, 1, %d)\n", T, BH * N / 2, G, N / 2, G, N);
		PW("    co = cos(ph)\n    si = sin(ph)\n");
		PW("    Bg = reshape(reshape(rmsnorm(reshape(u @ WB_%d, %d, %d, %d)) * gB_%d, %d, 1, %d, %d) + bB_%d, %d, %d, %d)\n", k, TB, R, N, k, TB, R, N, k, G, R, N);
		PW("    Cg = reshape(reshape(rmsnorm(reshape(u @ WC_%d, %d, %d, %d)) * gC_%d, %d, 1, %d, %d) + bC_%d, %d, %d, %d)\n", k, TB, R, N, k, TB, R, N, k, G, R, N);
		PW("    Bt = Bg * co + reshape(reshape(Bg, %d, 2) @ J, %d, %d, %d) * si\n", G * R * N / 2, G, R, N);
		PW("    Ct = Cg * co + reshape(reshape(Cg, %d, 2) @ J, %d, %d, %d) * si\n", G * R * N / 2, G, R, N);
		PW("    X = reshape(reshape(x, %d, %d, 1, %d) * Wxr_%d, %d, %d, %d)\n", TB, Hh, P, k, G, R, P);
		PW("    U = reshape(transpose(Bt) @ X, %d, %d, %d, %d)\n", T, BH, N, P);
		PW("    Cs = reshape(Ct, %d, %d, %d, %d)\n", T, BH, R, N);
		PW("    As = reshape(al, %d, %d, 1, 1)\n", T, BH);
		PW("    Bs = reshape((1 - lam) * dt * al, %d, %d, 1, 1)\n", T, BH);
		PW("    Gs = reshape(lam * dt, %d, %d, 1, 1)\n", T, BH);
		PW("    h = h0\n    up = h0\n");
		PW("    scan h, up over a in As, be in Bs, ga in Gs, ut in U, ct in Cs:\n");
		PW("        h = a * h + be * up + ga * ut\n");
		PW("        up = ut\n");
		PW("        emit ys = ct @ h\n");
		PW("    xr = reshape(x, %d, %d, 1, %d)\n", TB, Hh, P);
		PW("    y = reshape(sum_to(reshape(ys, %d, %d, %d, %d) * Wyr_%d, xr), %d, %d) + x * dk_%d\n", TB, Hh, R, P, k, TB, Di, k);
		PW("    e%dm = e%d + (y * silu(u @ Wz_%d)) @ Wo_%d\n", k, p, k, k);
		PW("    # block %d: SwiGLU MLP\n", k);
		PW("    v = rmsnorm(e%dm) * g2_%d\n", k, k);
		PW("    e%d = e%dm + (silu(v @ W1_%d) * (v @ W3_%d)) @ W2_%d\n", k, k, k, k, k);
	}
	PW("    ef = rmsnorm(e%d) * gf * m\n", l->layers);
	if (!pool) {
		PW("    return ef\n\n");
		return;
	}
	PW("    tot = onesT @ reshape(ef, %d, %d)\n", T, Bm * D);
	PW("    cnt = onesT @ reshape(m, %d, %d)\n", T, Bm);
	PW("    return reshape(tot, %d, %d) / reshape(cnt + 0.000001, %d, 1)\n\n", Bm, D, Bm);
}
#undef PW

/* The generated models. Plain Technograd: the user can read and edit them.
 * Weights are stored (in, out), so `x @ w1` serves a (B, D) batch in training
 * and a single (D) row in inference with the same weight files. */
static char *gen_model(const char *name, const Spec *s, const Layout *l, int H, int B, const char *opt, double lr, const char *src,
			int mode)
{
	int D = s->dim, C = s->classify ? s->nclass : 1, Dd = l->dd, Ds = l->ds, K = l->k;
	double g1 = sqrt(6.0 / (D + H));
	size_t cap = (size_t)1 << 20;
	char *b = xmalloc(cap);
	size_t o = 0;
#define P(...) o += (size_t)snprintf(b + o, cap - o, __VA_ARGS__)
	P("# Generated by `tgc train %s`: %s of '%s' from %d features.\n", src, s->classify ? "classification" : "regression",
	  s->target, D);
	if (mode == GEN_INFER) P("# Frozen inference model: one row in, weights loaded from the files `tgc train` saved.\n");
	else if (mode == GEN_TRAIN)
		P("# Training model: weights are state; every run takes one optimizer step on a\n"
		  "# mini-batch of %d rows. Edit it freely; `tgc batch` with %d rows packed per\n"
		  "# line trains it further.\n", B, B);
	if (Ds)
		P("# Text enters as sparse rows: s holds %d (bucket, weight) pairs per row over\n"
		  "# %d hashed buckets; spmm(s, wt) costs O(%d x %d) instead of O(%d x %d).\n",
		  K, Ds, K, H, Ds, H);
	int Hs = l->hs, T = l->T, Bm = mode == GEN_TRAIN ? B : 1;
	if (l->sf >= 0)
		P("# Word order enters through %d stacked Mamba-3 blocks (ssm below) over up to\n"
		  "# %d tokens (arXiv:2603.15569): exponential-trapezoidal recurrence, complex\n"
		  "# state as data-dependent rotations of B and C, MIMO rank %d, BCNorm, each\n"
		  "# block followed by a SwiGLU MLP, pre-norm residual. Trained by\n"
		  "# backpropagation through time. tok holds one (word bucket, 1) pair per\n"
		  "# position, time-major; padding is (0, 0).\n", l->layers, T, l->mimo);
	P("model %s%s\n\n", name, mode == GEN_INFER ? "_infer" : mode == GEN_EVAL ? "_eval" : "");
	if (l->sf >= 0) gen_ssm_weights(b, cap, &o, l, Bm, H, mode);
	const char *wn[4] = { "w1", "wt", "b1", "w2" };
	int rows[4] = { Dd, Ds, 0, H }, cols[4] = { H, H, H, C };
	for (int i = 0; i < 5; i++) {
		const char *n = i < 4 ? wn[i] : "b2";
		int r = i < 4 ? rows[i] : 0, c = i < 4 ? cols[i] : C;
		if ((i == 0 && !Dd) || (i == 1 && !Ds)) continue;
		char shape[48];
		if (r) snprintf(shape, sizeof shape, "f32[%d, %d]", r, c);
		else snprintf(shape, sizeof shape, "f32[%d]", c);
		if (mode == GEN_INFER) P("param %s : %s = file(\"weights.%s.bin\")\n", n, shape, n);
		else if (mode == GEN_EVAL) P("param %s : %s = zeros\n", n, shape); /* data pointers are swapped to the training weights */
		else if (!r) P("state %s : %s = zeros\n", n, shape);
		else P("state %s : %s = rand(%d, %.6g)\n", n, shape, 11 + i, i == 3 ? sqrt(6.0 / (H + C)) : g1);
	}
	P("\n");
	if (l->sf >= 0) gen_ssm_def(b, cap, &o, l, Bm, 1);
	/* argument lists and the first layer, for a batch (train) or one row */
	char args[200] = "", call[32] = "", pre[160] = "";
	size_t ao = 0;
	int bt = mode == GEN_TRAIN;
	char bp[16] = "";
	if (bt) snprintf(bp, sizeof bp, "%d, ", B);
	if (Dd) ao += (size_t)snprintf(args + ao, sizeof args - ao, "x: f32[%s%d]", bp, Dd);
	if (Ds) ao += (size_t)snprintf(args + ao, sizeof args - ao, "%ss: f32[%s%d, 2]", Dd ? ", " : "", bp, K);
	snprintf(call, sizeof call, "%s%s%s", Dd ? "x" : "", Dd && Ds ? ", " : "", Ds ? "s" : "");
	snprintf(pre, sizeof pre, "%s%s%s", Dd ? "x @ w1" : "", Dd && Ds ? " + " : "", Ds ? "spmm(s, wt)" : "");
	if (l->sf >= 0) {
		size_t k = strlen(args), c = strlen(call), q = strlen(pre);
		snprintf(args + k, sizeof args - k, "%stok: f32[%d, 1, 2]", k ? ", " : "", T * Bm);
		snprintf(call + c, sizeof call - c, "%stok", c ? ", " : "");
		if (bt) snprintf(pre + q, sizeof pre - q, "%srmsnorm(ssm(tok)) @ wp", q ? " + " : "");
		else snprintf(pre + q, sizeof pre - q, "%srmsnorm(reshape(ssm(tok), %d)) @ wp", q ? " + " : "", Hs);
	}
	char outs[32];
	if (bt) snprintf(outs, sizeof outs, "f32[%d, %d]", B, C);
	else snprintf(outs, sizeof outs, "f32[%d]", C);
	P("def net(%s) -> %s:\n    h = gelu(%s + b1)\n    return h @ w2 + b2\n\n", args, outs, pre);
	if (bt) {
		if (s->classify)
			P("def forward(%s, target: f32[%d, %d]) -> %s:\n    logits = net(%s)\n"
			  "    train xent(logits, target) with %s(lr=%g)\n    return softmax(logits)\n", args, B, C, outs, call, opt, lr);
		else
			P("def forward(%s, target: f32[%d, 1]) -> %s:\n    y = net(%s)\n"
			  "    train mse(y, target) with %s(lr=%g)\n    return y\n", args, B, outs, call, opt, lr);
	} else {
		P("def forward(%s) -> %s:\n    return %snet(%s)%s\n", args, outs, s->classify ? "softmax(" : "", call, s->classify ? ")" : "");
	}
#undef P
	return b;
}

typedef struct {
	double loss, acc;
} Score;

/* Score rows one at a time with the single-row evaluation model. */
/* Featurized rows: dense part Xd[r, dd] and sparse part E[r, k, 2]. */
typedef struct {
	const Layout *l;
	float *Xd, *E, *Tok; /* Tok[r, T, 2]: (word bucket, 1) per position, (0, 0) padding */
} Rows;

static int row_inputs(const Rows *R, int r, const float **in)
{
	int n = 0;
	if (R->l->dd) in[n++] = R->Xd + (size_t)r * (size_t)R->l->dd;
	if (R->l->ds) in[n++] = R->E + (size_t)r * (size_t)R->l->k * 2;
	if (R->l->sf >= 0) in[n++] = R->Tok + (size_t)r * (size_t)R->l->T * 2;
	return n;
}

/* Token pairs of one row; returns the row's word count (may exceed T). */
static int row_tokens(const Spec *s, const Table *t, const Layout *l, int r, float *tok, int *ids)
{
	int n = spec_tokens(s, t, r, l->sf, ids, l->T);
	for (int i = 0; i < l->T; i++) {
		tok[2 * i] = i < n ? (float)ids[i] : 0;
		tok[2 * i + 1] = i < n ? 1 : 0;
	}
	return n;
}

static Score eval_rows(Module *em, const Spec *s, const Rows *R, const float *Y, const int *lab, const int *rows, int n,
		       float *arena, float *out, int *steps)
{
	int C = s->classify ? s->nclass : 1;
	Score sc = { 0, 0 };
	for (int i = 0; i < n; i++) {
		int r = rows[i];
		const float *in[3];
		row_inputs(R, r, in);
		vm_run_into(em, in, out, steps, arena, NULL);
		if (s->classify) {
			int best = 0;
			for (int c = 1; c < C; c++)
				if (out[c] > out[best]) best = c;
			sc.loss += -log(out[lab[r]] + 1e-12);
			sc.acc += best == lab[r];
		} else {
			double d = out[0] - Y[r];
			sc.loss += d * d;
		}
	}
	sc.loss /= n > 0 ? n : 1;
	sc.acc /= n > 0 ? n : 1;
	return sc;
}

/* One epoch of mini-batch steps: B rows are packed per run of the training
 * model, which takes one optimizer step. The last partial batch is dropped;
 * rows are reshuffled every epoch, so every row is used. Returns the mean
 * training loss, measured on the start-of-step predictions. */
static double train_epoch(Module *m, const Spec *s, int B, const Rows *R, const float *Y, const int *lab, const int *rows,
			  int n, float *xb, float *sb, float *tb, float *yb, float *outb, float *arena, int *steps)
{
	int C = s->classify ? s->nclass : 1, seen = 0, dd = R->l->dd, ke = R->l->k * 2;
	double loss = 0;
	for (int b0 = 0; b0 + B <= n; b0 += B) {
		for (int i = 0; i < B; i++) {
			int r = rows[b0 + i];
			if (dd) memcpy(xb + (size_t)i * (size_t)dd, R->Xd + (size_t)r * (size_t)dd, (size_t)dd * sizeof *xb);
			if (R->l->ds) memcpy(sb + (size_t)i * (size_t)ke, R->E + (size_t)r * (size_t)ke, (size_t)ke * sizeof *sb);
			memcpy(yb + (size_t)i * (size_t)C, Y + (size_t)r * (size_t)C, (size_t)C * sizeof *yb);
			if (R->l->sf >= 0) /* time-major: position p of row i at (p B + i) */
				for (int p = 0; p < R->l->T; p++) {
					const float *src = R->Tok + ((size_t)r * (size_t)R->l->T + (size_t)p) * 2;
					float *dst = tb + ((size_t)p * (size_t)B + (size_t)i) * 2;
					dst[0] = src[0];
					dst[1] = src[1];
				}
		}
		const float *in[4];
		int ni = 0;
		if (dd) in[ni++] = xb;
		if (R->l->ds) in[ni++] = sb;
		if (R->l->sf >= 0) in[ni++] = tb;
		in[ni] = yb;
		vm_run_into(m, in, outb, steps, arena, NULL);
		for (int i = 0; i < B; i++) {
			int r = rows[b0 + i];
			const float *out = outb + (size_t)i * (size_t)C;
			if (s->classify) loss += -log(out[lab[r]] + 1e-12);
			else loss += (out[0] - Y[r]) * (out[0] - Y[r]);
		}
		seen += B;
	}
	return seen ? loss / seen : 0;
}

static void prepare(Opts *o, Table **tp, int *target)
{
	Table *t = data_load(o->source, o->max_rows);
	data_analyze(t);
	*target = data_pick_target(t, o->target);
	data_drop_leaks(t, *target);
	*tp = t;
}

static void init_weights(Module *m, const char *dir);

static int cmd_train(Opts *o)
{
	Table *t;
	int target;
	prepare(o, &t, &target);
	data_describe(t, target, stderr);

	/* rows with a usable target, split train/validation */
	int *rows = xmalloc((size_t)t->nrows * sizeof *rows), n = 0;
	const Col *tc = &t->col[target];
	for (int r = 0; r < t->nrows; r++)
		if (tc->cell[r].t != CELL_NULL) rows[n++] = r;
	if (n < 2) die(NULL, 0, "fewer than 2 rows have a value for '%s'", tc->name);
	rs = (uint32_t)(o->seed ? o->seed : 1) * 2654435761u + 1;
	shuffle(rows, n);
	int nval = (int)(n * o->val);
	if (n >= 10 && nval < 1) nval = 1;
	if (n < 10) nval = 0;
	int ntr = n - nval;
	Spec *s = spec_fit(t, target, rows, ntr, o->text_dim);

	int D = s->dim, C = s->classify ? s->nclass : 1;
	float *X = xmalloc((size_t)D * sizeof *X), *Y = xmalloc((size_t)t->nrows * (size_t)C * sizeof *Y);
	int *lab = xmalloc((size_t)t->nrows * sizeof *lab), *keep = xmalloc((size_t)t->nrows * sizeof *keep);
	Layout L = layout_of(s);
	/* pass 1: targets, and the widest sparse row (K) */
	int k = 0, split = ntr; /* drop rows whose target class was unseen in training */
	Layout probe = L;
	for (int i = 0; i < n; i++) {
		int r = rows[i];
		keep[r] = spec_apply(s, t, r, X, Y + (size_t)r * (size_t)C, &lab[r]);
		if (keep[r]) {
			rows[k++] = r;
			if (L.ds) { int nz = split_row(s, &probe, X, NULL, NULL); if (nz > L.k) L.k = nz; }
		} else if (i < split) ntr--;
		else nval--;
	}
	if (L.ds && !L.k) L.k = 1;
	/* pass 2: dense and sparse parts */
	Rows R = { &L, xmalloc((size_t)t->nrows * (size_t)(L.dd ? L.dd : 1) * sizeof(float)),
		   xmalloc((size_t)t->nrows * (size_t)(L.k ? L.k : 1) * 2 * sizeof(float)), NULL };
	for (int i = 0; i < k; i++) {
		int r = rows[i];
		spec_apply(s, t, r, X, NULL, NULL);
		split_row(s, &L, X, R.Xd + (size_t)r * (size_t)L.dd, R.E + (size_t)r * (size_t)L.k * 2);
	}
	if (o->ssm) { /* sequence path over the first text column: T = longest training row, at most 64 words */
		for (int f = 0; f < s->nf && L.sf < 0; f++)
			if (s->f[f].kind == COL_TEXT) L.sf = f;
		if (L.sf < 0) die(NULL, 0, "--model ssm needs a text column");
		L.V = s->f[L.sf].dim;
		L.hs = 32;
		L.layers = o->layers > 0 ? o->layers : 2;
		L.mimo = o->mimo > 0 ? o->mimo : 4;
		int *ids = xmalloc(65 * sizeof *ids);
		for (int i = 0; i < ntr; i++) {
			int w = spec_tokens(s, t, rows[i], L.sf, ids, 64);
			if (w > L.T) L.T = w;
		}
		L.T = L.T < 1 ? 1 : L.T > 64 ? 64 : L.T;
		R.Tok = xmalloc((size_t)t->nrows * (size_t)L.T * 2 * sizeof(float));
		for (int i = 0; i < k; i++) row_tokens(s, t, &L, rows[i], R.Tok + (size_t)rows[i] * (size_t)L.T * 2, ids);
		xfree(ids);
	}
	int *tr = rows, *va = rows + ntr;
	if (!nval) va = tr, nval = ntr; /* tiny data: report training fit */

	int H = o->hidden ? o->hidden : D <= 8 ? 16 : D <= 64 || D > 512 ? 32 : 64; /* wide sparse inputs: narrow layer */
	const char *opt = o->optimizer ? o->optimizer : "adamw";
	double lr = o->lr > 0 ? o->lr : 0.01;
	int B = o->batch > 0 ? o->batch : 32;
	if (B > ntr) B = ntr;
	int epochs = o->epochs ? o->epochs : (int)fmin(500, fmax(20, ceil(3000.0 * B / ntr))); /* about 3000 optimizer steps */
	char name[64];
	model_name(o->source, name, sizeof name);
	char dflt[128];
	snprintf(dflt, sizeof dflt, "%s_model", name);
	const char *dir = o->out ? o->out : dflt;
	mkdir(dir, 0755);

	char *src = gen_model(name, s, &L, H, B, opt, lr, o->source, GEN_TRAIN);
	char *mpath = join(dir, "model.tg");
	write_text(mpath, src);
	Module *m = lower(surface_parse(src, mpath), mpath);
	plan(m);
	if (o->init) init_weights(m, o->init);
	/* evaluation: the single-row network reading the training weights in place */
	char *esrc = gen_model(name, s, &L, H, B, opt, lr, o->source, GEN_EVAL);
	Module *em = lower(surface_parse(esrc, "<eval>"), "<eval>");
	for (int v = 0; v < em->nval; v++)
		if (em->val[v].kind == V_PARAM)
			for (int w = 0; w < m->nval; w++)
				if (m->val[w].kind == V_STATE && !strcmp(m->val[w].name, em->val[v].name)) em->val[v].data = m->val[w].data;
	plan(em);

	fprintf(stderr, "\nmodel    %d -> %d (gelu) -> %d, %s(lr=%g), batch %d, %d training / %d validation rows, up to %d epochs\n",
		D, H, C, opt, lr, B, ntr, va == tr ? 0 : nval, epochs);
	fprintf(stderr, "\n%6s %12s %12s %10s\n", "epoch", "train loss", "val loss", s->classify ? "val acc" : "val rmse");

	float *arena = xmalloc((size_t)(m->arena ? m->arena : 1) * sizeof(float));
	float *earena = xmalloc((size_t)(em->arena ? em->arena : 1) * sizeof(float));
	float *out = xmalloc((size_t)C * sizeof(float));
	float *tb = xmalloc((size_t)B * (size_t)(L.T ? L.T : 1) * 2 * sizeof(float));
	float *xb = xmalloc((size_t)B * (size_t)(L.dd ? L.dd : 1) * sizeof(float)), *sb = xmalloc((size_t)B * (size_t)(L.k ? L.k : 1) * 2 * sizeof(float));
	float *yb = xmalloc((size_t)B * (size_t)C * sizeof(float));
	float *outb = xmalloc((size_t)B * (size_t)C * sizeof(float));
	int *steps = xmalloc((size_t)(m->nthink ? m->nthink : 1) * sizeof(int));
	/* best-weights snapshot over the user states */
	int nst = 0;
	for (int v = 0; v < m->nval; v++) nst += m->val[v].kind == V_STATE;
	float **snap = xmalloc((size_t)(nst ? nst : 1) * sizeof *snap);
	for (int v = 0, i = 0; v < m->nval; v++)
		if (m->val[v].kind == V_STATE) snap[i++] = xmalloc((size_t)shape_numel(&m->val[v].sh) * sizeof(float));
	double best = INFINITY;
	int best_ep = 0, patience = (int)fmax(10, epochs / 5);
	Score bv = { 0, 0 };
	double t0 = tr_now_ms();
	for (int ep = 1; ep <= epochs; ep++) {
		shuffle(tr, ntr);
		double st = train_epoch(m, s, B, &R, Y, lab, tr, ntr, xb, sb, tb, yb, outb, arena, steps);
		Score sv = eval_rows(em, s, &R, Y, lab, va, nval, earena, out, steps);
		int improved = sv.loss < best - 1e-9;
		if (improved) {
			best = sv.loss;
			best_ep = ep;
			bv = sv;
			for (int v = 0, i = 0; v < m->nval; v++)
				if (m->val[v].kind == V_STATE) memcpy(snap[i++], m->val[v].data, (size_t)shape_numel(&m->val[v].sh) * sizeof(float));
		}
		double metric = s->classify ? sv.acc * 100 : sqrt(sv.loss) * s->tstd;
		if (ep <= 3 || improved || ep % 10 == 0 || ep == epochs)
			fprintf(stderr, "%6d %12.5f %12.5f %9.2f%s%s\n", ep, st, sv.loss, metric, s->classify ? "%" : "", improved ? "  *" : "");
		if (ep - best_ep >= patience) {
			fprintf(stderr, "early stop: no validation improvement for %d epochs\n", patience);
			break;
		}
	}
	for (int v = 0, i = 0; v < m->nval; v++)
		if (m->val[v].kind == V_STATE) memcpy(m->val[v].data, snap[i++], (size_t)shape_numel(&m->val[v].sh) * sizeof(float));

	/* artifacts */
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_STATE || !strncmp(x->name, "__", 2)) continue;
		char fn[128];
		snprintf(fn, sizeof fn, "weights.%s.bin", x->name);
		char *p = join(dir, fn);
		FILE *f = fopen(p, "wb");
		if (!f) die(NULL, 0, "cannot write '%s'", p);
		io_write_row(f, 1, x->data, shape_numel(&x->sh), NULL, 0);
		if (fclose(f)) die(NULL, 0, "write failed '%s'", p);
	}
	char *spath = join(dir, "features.tgf");
	spec_save(s, spath);
	Layout Li = L; /* inference: headroom for rows longer than any seen in training */
	if (Li.ds) Li.k = (int)fmin(Li.ds, fmax(2 * L.k, 64));
	char *isrc = gen_model(name, s, &Li, H, B, opt, lr, o->source, GEN_INFER);
	char *ipath = join(dir, "infer.tg");
	write_text(ipath, isrc);
	Module *im = load_module(ipath);
	plan(im);
	char *cpath = join(dir, "infer.c");
	FILE *cf = fopen(cpath, "w");
	if (!cf) die(NULL, 0, "cannot write '%s'", cpath);
	cgen(im, cf);
	if (fclose(cf)) die(NULL, 0, "write failed '%s'", cpath);

	char rep[2048], ssm_note[160] = "";
	if (L.sf >= 0) snprintf(ssm_note, sizeof ssm_note, "\n            + %d Mamba-3 block(s) over '%s': %d steps, width %d, MIMO rank %d", L.layers, s->f[L.sf].name, L.T, L.hs, L.mimo);
	double metric = s->classify ? bv.acc * 100 : sqrt(bv.loss) * s->tstd;
	snprintf(rep, sizeof rep,
		 "source      %s (%s, %d rows)\ntask        %s of '%s'%s\nfeatures    %d (%d dense, %d sparse text buckets, <= %d active per row)\nmodel       %d -> %d -> %d, %s(lr=%g)%s\n"
		 "best epoch  %d (validation loss %.5f, %s %.2f%s)\ntime        %.1f s\n",
		 o->source, t->format, t->nrows, s->classify ? "classification" : "regression", s->target, "", D, L.dd, L.ds, L.k, D, H, C, opt, lr, ssm_note,
		 best_ep, bv.loss, s->classify ? "accuracy" : "rmse", metric, s->classify ? "%" : "", (tr_now_ms() - t0) / 1e3);
	char *rpath = join(dir, "report.txt");
	write_text(rpath, rep);
	fprintf(stderr, "\n%s\nwrote %s/: model.tg (training), infer.tg + weights.*.bin (inference), infer.c (C unit),\n"
			"      features.tgf (featurization), report.txt\nnext:  tgc predict %s NEW_DATA\n",
		rep, dir, dir);
	return 0;
}

static int cmd_predict(int argc, char **argv)
{
	if (argc < 4) usage_data();
	const char *dir = argv[2], *src = argv[3], *outp = NULL, *mpath = NULL, *ipath = NULL;
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "-o") && i + 1 < argc) outp = argv[++i];
		else if (!strcmp(argv[i], "--model") && i + 1 < argc) mpath = argv[++i]; /* e.g. a quantized infer model */
		else if (!strcmp(argv[i], "--inputs") && i + 1 < argc) ipath = argv[++i]; /* the model's input rows, batch format */
		else die(NULL, 0, "unexpected argument '%s'", argv[i]);
	}
	Spec *s = spec_load(join(dir, "features.tgf"));
	Module *m = load_module(mpath ? mpath : join(dir, "infer.tg"));
	plan(m);
	Table *t = data_load(src, 1 << 30);
	int D = s->dim, C = s->classify ? s->nclass : 1, cut = 0;
	Layout L = layout_of(s);
	for (int i = 0; i < m->ninputs; i++)
		if (!strcmp(m->val[m->inputs[i]].name, "s")) L.k = m->val[m->inputs[i]].sh.dim[0];
	if (L.ds && !L.k) die(NULL, 0, "infer.tg has no sparse input 's' for the text features in features.tgf");
	float *x = xmalloc((size_t)D * sizeof *x), *out = xmalloc((size_t)C * sizeof *out);
	for (int i = 0; i < m->ninputs; i++)
		if (!strcmp(m->val[m->inputs[i]].name, "tok")) L.T = m->val[m->inputs[i]].sh.dim[0];
	if (L.T)
		for (int f = 0; f < s->nf && L.sf < 0; f++)
			if (s->f[f].kind == COL_TEXT) L.sf = f;
	float *tok = xmalloc((size_t)(L.T ? L.T : 1) * 2 * sizeof *tok);
	int *ids = xmalloc((size_t)(L.T ? L.T : 1) * sizeof *ids), longer = 0;
	float *xd = xmalloc((size_t)(L.dd ? L.dd : 1) * sizeof *xd), *ell = xmalloc((size_t)(L.k ? L.k : 1) * 2 * sizeof *ell);
	int *steps = xmalloc((size_t)(m->nthink ? m->nthink : 1) * sizeof *steps);
	FILE *inf = ipath ? fopen(ipath, "w") : NULL;
	if (ipath && !inf) die(NULL, 0, "cannot write '%s'", ipath);
	FILE *f = outp ? fopen(outp, "w") : stdout;
	if (!f) die(NULL, 0, "cannot write '%s'", outp);
	fprintf(f, s->classify ? "prediction,confidence\n" : "prediction\n");
	int known = 0, correct = 0;
	double se = 0;
	for (int r = 0; r < t->nrows; r++) {
		int lab = -1;
		float y = 0;
		int has = spec_apply(s, t, r, x, s->classify ? NULL : &y, s->classify ? &lab : NULL);
		cut += split_row(s, &L, x, xd, ell) > L.k;
		if (L.sf >= 0) longer += row_tokens(s, t, &L, r, tok, ids) > L.T;
		const float *in[3];
		int ni = 0;
		if (L.dd) in[ni++] = xd;
		if (L.ds) in[ni++] = ell;
		if (L.sf >= 0) in[ni++] = tok;
		if (inf) { /* calibration data for tgc quantize --method gptq */
			for (int i = 0; i < ni; i++)
				for (int j = 0; j < shape_numel(&m->val[m->inputs[i]].sh); j++) fprintf(inf, "%s%.7g", i || j ? "," : "", (double)in[i][j]);
			fputc('\n', inf);
		}
		vm_run(m, in, out, steps);
		if (s->classify) {
			int b = 0;
			for (int c = 1; c < C; c++)
				if (out[c] > out[b]) b = c;
			fputc('"', f);
			for (const char *p = s->classes[b]; *p; p++) {
				if (*p == '"') fputc('"', f);
				fputc(*p, f);
			}
			fprintf(f, "\",%.4f\n", (double)out[b]);
			if (has) { known++; correct += b == lab; }
		} else {
			double v = out[0] * s->tstd + s->tmean;
			fprintf(f, "%.9g\n", v);
			if (has) { known++; se += (out[0] - y) * (out[0] - y); }
		}
	}
	if (inf && fclose(inf)) die(NULL, 0, "write failed '%s'", ipath);
	if (outp && fclose(f)) die(NULL, 0, "write failed '%s'", outp);
	if (longer) fprintf(stderr, "predict: %d row(s) had more than %d words; the sequence layer read the first %d\n", longer, L.T, L.T);
	if (cut) fprintf(stderr, "predict: %d row(s) had more than %d active text buckets; the largest %d were kept\n", cut, L.k, L.k);
	if (known && s->classify) fprintf(stderr, "predict: %d rows, accuracy %.2f%% on the %d rows that carry '%s'\n", t->nrows, 100.0 * correct / known, known, s->target);
	else if (known) fprintf(stderr, "predict: %d rows, rmse %.6g on the %d rows that carry '%s'\n", t->nrows, sqrt(se / known) * s->tstd, known, s->target);
	else fprintf(stderr, "predict: %d rows\n", t->nrows);
	return 0;
}

/* --init DIR: start every weight that DIR holds (weights.<name>.bin with the
 * same element count) from there; the rest keep their initialization. */
static void init_weights(Module *m, const char *dir)
{
	int n = 0, tot = 0;
	for (int v = 0; v < m->nval; v++) {
		Value *x = &m->val[v];
		if (x->kind != V_STATE || !strncmp(x->name, "__", 2)) continue;
		tot++;
		char fn[160];
		snprintf(fn, sizeof fn, "weights.%s.bin", x->name);
		char *p = join(dir, fn);
		FILE *f = fopen(p, "rb");
		xfree(p);
		if (!f) continue;
		size_t want = (size_t)shape_numel(&x->sh);
		fseek(f, 0, SEEK_END);
		long sz = ftell(f);
		fseek(f, 0, SEEK_SET);
		if (sz != (long)(want * sizeof(float)) || fread(x->data, sizeof(float), want, f) != want) {
			fclose(f);
			die(NULL, 0, "--init: %s/%s has %ld bytes, the model's '%s' needs %zu", dir, fn, sz, x->name, want * sizeof(float));
		}
		fclose(f);
		n++;
	}
	if (!n) die(NULL, 0, "--init: no weights of this model in '%s' (pretrain with the same --layers and --mimo)", dir);
	fprintf(stderr, "init     %d of %d weights from %s\n", n, tot, dir);
}

/* ---- tgc pretrain: next-token prediction on unlabeled text ----------------
 * Windows of T + 1 tokens; the model reads tokens 0..T-1 and predicts 1..T.
 * The softmax runs over the distinct target buckets of the batch (in-batch
 * sampled softmax), with the embedding table tied between input and output
 * and logits soft-capped at 15 (c tanh(z / c)) so exp cannot overflow. */
static char *gen_lm(const char *name, const Layout *l, int Bm, double lr, int mode, const char *src)
{
	size_t cap = (size_t)1 << 21;
	char *b = xmalloc(cap);
	size_t o = 0;
#define P(...) o += (size_t)snprintf(b + o, cap - o, __VA_ARGS__)
	int D = l->hs, TB = l->T * Bm;
	P("# Generated by `tgc pretrain %s`: next-token prediction with %d Mamba-3 block(s)\n"
	  "# over windows of %d tokens hashed into %d buckets, in batches of %d windows.\n"
	  "# Softmax over the distinct target buckets of the batch (in-batch sampled\n"
	  "# softmax), tied embeddings, logits soft-capped at 15.\n", src, l->layers, l->T, l->V, Bm);
	P("model %s_lm%s\n\n", name, mode == GEN_EVAL ? "_eval" : "");
	gen_ssm_weights(b, cap, &o, l, Bm, 0, mode);
	P("param onesD : f32[%d] = ones\nparam onesN : f32[%d] = ones\n\n", D, TB);
	gen_ssm_def(b, cap, &o, l, Bm, 0);
	P("def forward(tok: f32[%d, 1, 2], nxt: f32[%d, 1, 2]) -> f32:\n", TB, TB);
	P("    h = states(tok)\n");
	P("    r = active(nxt, E)\n");
	P("    L = 15 * tanh((h @ transpose(take(E, r))) / 15) - step(0 - r) * 10000\n");
	P("    pos = 15 * tanh(((h * spmm(nxt, E)) @ onesD) / 15)\n");
	P("    mk = reshape(nxt, %d, 2) @ sel2\n", TB);
	P("    loss = sum((log(exp(L) @ onesN) - pos) * mk) / (sum(mk) + 0.000001)\n");
	if (mode == GEN_TRAIN) P("    train loss with adamw(lr=%g, clip=1)\n", lr);
	P("    return loss\n");
#undef P
	return b;
}

static int cmd_pretrain(Opts *o)
{
	Table *t = data_load(o->source, o->max_rows);
	data_analyze(t);
	int col = -1;
	for (int i = 0; i < t->ncols && col < 0; i++)
		if (o->column ? !strcmp(t->col[i].name, o->column) : t->col[i].kind == COL_TEXT) col = i;
	if (col < 0) die(NULL, 0, o->column ? "pretrain: no column '%s'" : "pretrain: %s has no text column (name one with --column)", o->column ? o->column : o->source);
	Layout L = { 0, 0, 0, 0, 64, o->text_dim, 32, o->layers > 0 ? o->layers : 2, o->mimo > 0 ? o->mimo : 4 };
	int T = L.T, Bm = o->batch > 0 ? o->batch : 16;
	/* rows are packed into one token stream (as in GPT pretraining), cut into
	 * windows of T + 1 tokens that overlap by one */
	long cap = 1 << 16, ntok = 0;
	int *ids = xmalloc(4096 * sizeof *ids), *stream = xmalloc((size_t)cap * sizeof *stream);
	for (int r = 0; r < t->nrows; r++) {
		const Cell *c = &t->col[col].cell[r];
		if (c->t != CELL_STR) continue;
		int n = text_tokens(c->s, L.V, ids, 4096);
		if (n > 4096) n = 4096;
		if (ntok + n > cap) { while (ntok + n > cap) cap *= 2; stream = xrealloc(stream, (size_t)cap * sizeof *stream); }
		memcpy(stream + ntok, ids, (size_t)n * sizeof *ids);
		ntok += n;
	}
	int nw = ntok > T ? (int)((ntok - 1) / T) : 0;
	int *W = xmalloc((size_t)(nw ? nw : 1) * (size_t)(T + 1) * sizeof *W);
	for (int k = 0; k < nw; k++) memcpy(W + (size_t)k * (size_t)(T + 1), stream + (size_t)k * (size_t)T, (size_t)(T + 1) * sizeof *W);
	xfree(stream);
	if (nw < 2 * Bm) die(NULL, 0, "pretrain: %d windows of text; need at least %d", nw, 2 * Bm);
	rs = (uint32_t)(o->seed ? o->seed : 1) * 2654435761u + 1;
	int *ord = xmalloc((size_t)nw * sizeof *ord);
	for (int i = 0; i < nw; i++) ord[i] = i;
	shuffle(ord, nw);
	int nval = nw / 50 < Bm ? Bm : nw / 50 > 64 * Bm ? 64 * Bm : nw / 50, ntr = nw - nval;
	double lr = o->lr > 0 ? o->lr : 0.003;
	int epochs = o->epochs > 0 ? o->epochs : 1;
	char name[64], dflt[128];
	model_name(o->source, name, sizeof name);
	snprintf(dflt, sizeof dflt, "%s_lm", name);
	const char *dir = o->out ? o->out : dflt;
	mkdir(dir, 0755);
	char *src = gen_lm(name, &L, Bm, lr, GEN_TRAIN, o->source);
	char *mpath = join(dir, "lm.tg");
	write_text(mpath, src);
	Module *m = lower(surface_parse(src, mpath), mpath);
	plan(m);
	if (o->init) init_weights(m, o->init);
	char *esrc = gen_lm(name, &L, Bm, lr, GEN_EVAL, o->source);
	Module *em = lower(surface_parse(esrc, "<eval>"), "<eval>");
	for (int v = 0; v < em->nval; v++)
		if (em->val[v].kind == V_PARAM)
			for (int w = 0; w < m->nval; w++)
				if (m->val[w].kind == V_STATE && !strcmp(m->val[w].name, em->val[v].name)) em->val[v].data = m->val[w].data;
	plan(em);
	fprintf(stderr, "pretrain %ld words in %d rows of '%s' -> %d windows of %d tokens (%d training, %d validation)\n"
			"model    %d Mamba-3 block(s), width %d, MIMO rank %d, %d hashed buckets; batch %d, adamw(lr=%g, clip=1), %d epoch(s)%s\n\n",
		ntok, t->nrows, t->col[col].name, nw, T, ntr, nval, L.layers, L.hs, L.mimo, L.V, Bm, lr, epochs,
		o->steps ? ", step limit" : "");
	float *arena = xmalloc((size_t)(m->arena ? m->arena : 1) * sizeof(float));
	float *earena = xmalloc((size_t)(em->arena ? em->arena : 1) * sizeof(float));
	float *tok = xmalloc((size_t)T * (size_t)Bm * 2 * sizeof(float)), *nxt = xmalloc((size_t)T * (size_t)Bm * 2 * sizeof(float));
	int *steps = xmalloc(sizeof(int) * (size_t)(m->nthink + 1));
	double t0 = tr_now_ms();
	long step = 0;
#define FILL(first)                                                                                      \
	for (int i = 0; i < Bm; i++) {                                                                       \
		const int *w = W + (size_t)ord[(first) + i] * (size_t)(T + 1);                                   \
		for (int p = 0; p < T; p++) {                                                                    \
			float *a = tok + ((size_t)p * (size_t)Bm + (size_t)i) * 2, *z = nxt + ((size_t)p * (size_t)Bm + (size_t)i) * 2; \
			int ok = w[p] >= 0 && w[p + 1] >= 0;                                                         \
			a[0] = ok ? (float)w[p] : 0; a[1] = ok ? 1 : 0;                                              \
			z[0] = ok ? (float)w[p + 1] : 0; z[1] = ok ? 1 : 0;                                          \
		}                                                                                                \
	}
	double val0 = 0;
	for (int ep = 0; ep <= epochs; ep++) {
		double tl = 0;
		int nb = 0;
		if (ep > 0) {
			shuffle(ord, ntr);
			for (int b0 = 0; b0 + Bm <= ntr && (!o->steps || step < o->steps); b0 += Bm) {
				FILL(b0);
				const float *in[2] = { tok, nxt };
				float loss;
				vm_run_into(m, in, &loss, steps, arena, NULL);
				tl += loss;
				nb++;
				step++;
				if (step % 500 == 0) fprintf(stderr, "  step %ld: training loss %.4f (%.0f s)\n", step, tl / nb, (tr_now_ms() - t0) / 1e3);
			}
		}
		double vl = 0;
		int vb = 0;
		for (int b0 = ntr; b0 + Bm <= nw; b0 += Bm, vb++) {
			FILL(b0);
			const float *in[2] = { tok, nxt };
			float loss;
			vm_run_into(em, in, &loss, steps, earena, NULL);
			vl += loss;
		}
		vl /= vb ? vb : 1;
		if (ep == 0) val0 = vl;
		fprintf(stderr, "epoch %d: %s%.4f, validation loss %.4f (in-batch softmax over <= %d candidates)\n", ep, ep ? "training loss " : "initial, ",
			ep ? tl / (nb ? nb : 1) : vl, vl, T * Bm);
		if (o->steps && step >= o->steps) break;
	}
#undef FILL
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_STATE || !strncmp(x->name, "__", 2)) continue;
		char fn[128];
		snprintf(fn, sizeof fn, "weights.%s.bin", x->name);
		char *p = join(dir, fn);
		FILE *f = fopen(p, "wb");
		if (!f) die(NULL, 0, "cannot write '%s'", p);
		io_write_row(f, 1, x->data, shape_numel(&x->sh), NULL, 0);
		if (fclose(f)) die(NULL, 0, "write failed '%s'", p);
	}
	char rep[1024];
	snprintf(rep, sizeof rep, "source      %s (%d rows, column '%s', %ld words)\nmodel       %d Mamba-3 block(s), width %d, MIMO rank %d, %d buckets\n"
		 "steps       %ld (batch %d windows of %d tokens)\nvalidation  next-token loss %.4f -> last %s\ntime        %.1f s\n",
		 o->source, t->nrows, t->col[col].name, ntok, L.layers, L.hs, L.mimo, L.V, step, Bm, T, val0, "see log", (tr_now_ms() - t0) / 1e3);
	write_text(join(dir, "report.txt"), rep);
	fprintf(stderr, "\nwrote %s/: lm.tg, weights.*.bin, report.txt\nnext:  tgc train LABELED_DATA --model ssm --init %s\n", dir, dir);
	return 0;
}

int autotrain_main(int argc, char **argv)
{
	Opts o = { 0 };
	o.val = 0.2;
	o.max_rows = 20000;
	o.text_dim = 32768;
	const char *cmd = argv[1];
	if (!strcmp(cmd, "predict")) return cmd_predict(argc, argv);
	if (!strcmp(cmd, "pretrain")) {
		parse_opts(argc, argv, 2, &o);
		return cmd_pretrain(&o);
	}
	if (!strcmp(cmd, "train")) {
		parse_opts(argc, argv, 2, &o);
		if (o.val < 0 || o.val >= 1) die(NULL, 0, "--val must be in [0, 1)");
		return cmd_train(&o);
	}
	/* data inspect | data prep */
	if (argc < 4) usage_data();
	const char *sub = argv[2];
	parse_opts(argc, argv, 3, &o);
	Table *t;
	int target;
	prepare(&o, &t, &target);
	if (!strcmp(sub, "inspect")) {
		data_describe(t, target, stdout);
		return 0;
	}
	if (!strcmp(sub, "prep")) {
		if (!o.out) die(NULL, 0, "data prep needs -o OUT.csv");
		int *rows = xmalloc((size_t)t->nrows * sizeof *rows);
		for (int r = 0; r < t->nrows; r++) rows[r] = r;
		Spec *s = spec_fit(t, target, rows, t->nrows, o.text_dim);
		int D = s->dim, C = s->classify ? s->nclass : 1, kept = 0;
		float *x = xmalloc((size_t)D * sizeof *x), *y = xmalloc((size_t)C * sizeof *y);
		FILE *f = fopen(o.out, "w");
		if (!f) die(NULL, 0, "cannot write '%s'", o.out);
		fprintf(f, "# %d features then %d target value(s) per row; featurization in %s.tgf\n", D, C, o.out);
		for (int r = 0; r < t->nrows; r++) {
			if (!spec_apply(s, t, r, x, y, NULL)) continue;
			for (int i = 0; i < D; i++) fprintf(f, "%.6g,", (double)x[i]);
			for (int i = 0; i < C; i++) fprintf(f, "%.6g%s", (double)y[i], i + 1 < C ? "," : "\n");
			kept++;
		}
		if (fclose(f)) die(NULL, 0, "write failed '%s'", o.out);
		char *sp = xmalloc(strlen(o.out) + 8);
		sprintf(sp, "%s.tgf", o.out);
		spec_save(s, sp);
		fprintf(stderr, "prep: %d rows x (%d features + %d target) -> %s\n", kept, D, C, o.out);
		return 0;
	}
	usage_data();
	return 2;
}
