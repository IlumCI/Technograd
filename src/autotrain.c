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
	const char *source, *target, *out, *optimizer;
	int epochs, hidden, max_rows, seed, text_dim, batch;
	double lr, val;
} Opts;

static void usage_data(void)
{
	fputs("usage:\n"
	      "  tgc data inspect SOURCE [--target COL]\n"
	      "  tgc data prep SOURCE -o OUT.csv [--target COL]\n"
	      "  tgc train SOURCE [-o DIR] [--target COL] [--epochs N] [--hidden H] [--lr X]\n"
	      "                   [--optimizer sgd|momentum|adam|adamw|muon] [--val FRACTION] [--max-rows N]\n"
	      "                   [--batch N] [--text-dim N] [--seed S]\n"
	      "  tgc predict DIR SOURCE [-o OUT.csv]\n"
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
} Layout;

static Layout layout_of(const Spec *s)
{
	Layout l = { 0, 0, 0 };
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

/* The generated models. Plain Technograd: the user can read and edit them.
 * Weights are stored (in, out), so `x @ w1` serves a (B, D) batch in training
 * and a single (D) row in inference with the same weight files. */
static char *gen_model(const char *name, const Spec *s, const Layout *l, int H, int B, const char *opt, double lr, const char *src,
			int mode)
{
	int D = s->dim, C = s->classify ? s->nclass : 1, Dd = l->dd, Ds = l->ds, K = l->k;
	double g1 = sqrt(6.0 / (D + H));
	size_t cap = 8192;
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
	P("model %s%s\n\n", name, mode == GEN_INFER ? "_infer" : mode == GEN_EVAL ? "_eval" : "");
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
	/* argument lists and the first layer, for a batch (train) or one row */
	char args[160] = "", call[32] = "", pre[96] = "";
	size_t ao = 0;
	int bt = mode == GEN_TRAIN;
	char bp[16] = "";
	if (bt) snprintf(bp, sizeof bp, "%d, ", B);
	if (Dd) ao += (size_t)snprintf(args + ao, sizeof args - ao, "x: f32[%s%d]", bp, Dd);
	if (Ds) ao += (size_t)snprintf(args + ao, sizeof args - ao, "%ss: f32[%s%d, 2]", Dd ? ", " : "", bp, K);
	snprintf(call, sizeof call, "%s%s%s", Dd ? "x" : "", Dd && Ds ? ", " : "", Ds ? "s" : "");
	snprintf(pre, sizeof pre, "%s%s%s", Dd ? "x @ w1" : "", Dd && Ds ? " + " : "", Ds ? "spmm(s, wt)" : "");
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
	float *Xd, *E;
} Rows;

static int row_inputs(const Rows *R, int r, const float **in)
{
	int n = 0;
	if (R->l->dd) in[n++] = R->Xd + (size_t)r * (size_t)R->l->dd;
	if (R->l->ds) in[n++] = R->E + (size_t)r * (size_t)R->l->k * 2;
	return n;
}

static Score eval_rows(Module *em, const Spec *s, const Rows *R, const float *Y, const int *lab, const int *rows, int n,
		       float *arena, float *out, int *steps)
{
	int C = s->classify ? s->nclass : 1;
	Score sc = { 0, 0 };
	for (int i = 0; i < n; i++) {
		int r = rows[i];
		const float *in[2];
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
			  int n, float *xb, float *sb, float *yb, float *outb, float *arena, int *steps)
{
	int C = s->classify ? s->nclass : 1, seen = 0, dd = R->l->dd, ke = R->l->k * 2;
	double loss = 0;
	for (int b0 = 0; b0 + B <= n; b0 += B) {
		for (int i = 0; i < B; i++) {
			int r = rows[b0 + i];
			if (dd) memcpy(xb + (size_t)i * (size_t)dd, R->Xd + (size_t)r * (size_t)dd, (size_t)dd * sizeof *xb);
			if (R->l->ds) memcpy(sb + (size_t)i * (size_t)ke, R->E + (size_t)r * (size_t)ke, (size_t)ke * sizeof *sb);
			memcpy(yb + (size_t)i * (size_t)C, Y + (size_t)r * (size_t)C, (size_t)C * sizeof *yb);
		}
		const float *in[3];
		int ni = 0;
		if (dd) in[ni++] = xb;
		if (R->l->ds) in[ni++] = sb;
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
		   xmalloc((size_t)t->nrows * (size_t)(L.k ? L.k : 1) * 2 * sizeof(float)) };
	for (int i = 0; i < k; i++) {
		int r = rows[i];
		spec_apply(s, t, r, X, NULL, NULL);
		split_row(s, &L, X, R.Xd + (size_t)r * (size_t)L.dd, R.E + (size_t)r * (size_t)L.k * 2);
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
		double st = train_epoch(m, s, B, &R, Y, lab, tr, ntr, xb, sb, yb, outb, arena, steps);
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

	char rep[2048];
	double metric = s->classify ? bv.acc * 100 : sqrt(bv.loss) * s->tstd;
	snprintf(rep, sizeof rep,
		 "source      %s (%s, %d rows)\ntask        %s of '%s'%s\nfeatures    %d (%d dense, %d sparse text buckets, <= %d active per row)\nmodel       %d -> %d -> %d, %s(lr=%g)\n"
		 "best epoch  %d (validation loss %.5f, %s %.2f%s)\ntime        %.1f s\n",
		 o->source, t->format, t->nrows, s->classify ? "classification" : "regression", s->target, "", D, L.dd, L.ds, L.k, D, H, C, opt, lr,
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
	const char *dir = argv[2], *src = argv[3], *outp = NULL;
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "-o") && i + 1 < argc) outp = argv[++i];
		else die(NULL, 0, "unexpected argument '%s'", argv[i]);
	}
	Spec *s = spec_load(join(dir, "features.tgf"));
	Module *m = load_module(join(dir, "infer.tg"));
	plan(m);
	Table *t = data_load(src, 1 << 30);
	int D = s->dim, C = s->classify ? s->nclass : 1, cut = 0;
	Layout L = layout_of(s);
	for (int i = 0; i < m->ninputs; i++)
		if (!strcmp(m->val[m->inputs[i]].name, "s")) L.k = m->val[m->inputs[i]].sh.dim[0];
	if (L.ds && !L.k) die(NULL, 0, "infer.tg has no sparse input 's' for the text features in features.tgf");
	float *x = xmalloc((size_t)D * sizeof *x), *out = xmalloc((size_t)C * sizeof *out);
	float *xd = xmalloc((size_t)(L.dd ? L.dd : 1) * sizeof *xd), *ell = xmalloc((size_t)(L.k ? L.k : 1) * 2 * sizeof *ell);
	int *steps = xmalloc((size_t)(m->nthink ? m->nthink : 1) * sizeof *steps);
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
		const float *in[2];
		int ni = 0;
		if (L.dd) in[ni++] = xd;
		if (L.ds) in[ni++] = ell;
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
	if (outp && fclose(f)) die(NULL, 0, "write failed '%s'", outp);
	if (cut) fprintf(stderr, "predict: %d row(s) had more than %d active text buckets; the largest %d were kept\n", cut, L.k, L.k);
	if (known && s->classify) fprintf(stderr, "predict: %d rows, accuracy %.2f%% on the %d rows that carry '%s'\n", t->nrows, 100.0 * correct / known, known, s->target);
	else if (known) fprintf(stderr, "predict: %d rows, rmse %.6g on the %d rows that carry '%s'\n", t->nrows, sqrt(se / known) * s->tstd, known, s->target);
	else fprintf(stderr, "predict: %d rows\n", t->nrows);
	return 0;
}

int autotrain_main(int argc, char **argv)
{
	Opts o = { 0 };
	o.val = 0.2;
	o.max_rows = 20000;
	o.text_dim = 8192;
	const char *cmd = argv[1];
	if (!strcmp(cmd, "predict")) return cmd_predict(argc, argv);
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
