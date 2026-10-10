/* Post-training weight quantization: `tgc quantize`.
 *
 *   tgc quantize MODEL [-o OUT.tgir] [--bits 8|4] [--method rtn|gptq]
 *                [--calib DATA] [--min-size N]
 *
 * Weights become int8 or int4 codes with one f32 scale per output channel
 * (rows of W in W @ x and of spmm tables, columns of W in x @ W), and the
 * result is a self-contained TGIR file the VM, `tgc c` and `tgc predict` run
 * directly. Activations stay f32 (weight-only quantization: the memory and
 * flash cost of a model is its weights; this cuts them 4x or 8x).
 *
 * Eligible: rank-2 params with at least --min-size (default 256) elements
 * whose every use is a matmul or spmm operand that reads codes on one axis.
 *
 * Methods:
 *   rtn   round to nearest with per-channel scales chosen by MSE-optimal
 *         clipping (a grid search over 1.0 .. 0.55 of each channel's absmax).
 *   gptq  GPTQ (Frantar et al., arXiv:2210.17323): channels are quantized
 *         one input dimension at a time and the rounding error is pushed
 *         onto the not-yet-quantized dimensions through the inverse Hessian
 *         H = X X^T of the layer's calibration inputs (Cholesky form,
 *         1% dampening). Needs --calib rows (the `tgc batch` format); spmm
 *         tables and layers wider than 2048 inputs fall back to rtn. */
#include "tg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	int v, axis, C, K; /* param, axis, channels, inputs per channel */
	double *H;         /* K x K calibration Hessian, or NULL */
	long nsamp;
} QP;

typedef struct {
	QP *p;
	int n;
} QSet;

/* -1: not eligible; else the one axis every use agrees on. */
static int use_axis(const Module *m, int v, const Block *b, int *axis)
{
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		if (in->op == OP_THINK) {
			if (in->init == v || in->yield == v || !use_axis(m, v, in->body, axis)) return 0;
		} else if (in->op == OP_SCAN) {
			const Scan *s = in->sc;
			for (int k = 0; k < s->nc; k++)
				if (s->init[k] == v || s->next[k] == v) return 0;
			for (int j = 0; j < s->nx; j++)
				if (s->x[j] == v) return 0;
			for (int y = 0; y < s->ny; y++)
				if (s->y[y] == v) return 0;
			if (!use_axis(m, v, in->body, axis)) return 0;
		} else {
			for (int j = 0; j < in->na; j++) {
				if (in->a[j] != v) continue;
				int a = q_axis_for(m, v, in);
				if (a < 0 || (*axis >= 0 && *axis != a)) return 0;
				*axis = a;
			}
		}
	}
	return 1;
}

static QSet eligible(const Module *m, int min_size)
{
	QSet s = { NULL, 0 };
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_PARAM || x->dead || x->qbits || x->sh.rank != 2 || shape_numel(&x->sh) < min_size) continue;
		if (x->name && !strncmp(x->name, "__", 2)) continue;
		if (m->output == v) continue;
		int axis = -1, ok = use_axis(m, v, &m->top, &axis);
		for (int i = 0; i < m->nupd; i++) ok &= m->upd_src[i] != v && m->upd_rows[i] != v;
		if (!ok || axis < 0) continue;
		s.p = xrealloc(s.p, (size_t)(s.n + 1) * sizeof *s.p);
		QP *q = &s.p[s.n++];
		memset(q, 0, sizeof *q);
		q->v = v;
		q->axis = axis;
		q->C = x->sh.dim[axis];
		q->K = x->sh.dim[1 - axis];
	}
	return s;
}

/* Calibration: accumulate X X^T of each eligible matmul weight's other operand. */
static void hook(void *ctx, const Module *m, const Ins *in, const float *a, const float *b)
{
	QSet *s = ctx;
	int mm, kk, nn;
	matmul_dims(&m->val[in->a[0]].sh, &m->val[in->a[1]].sh, &mm, &kk, &nn);
	for (int i = 0; i < s->n; i++) {
		QP *q = &s->p[i];
		if (!q->H) continue;
		if (in->a[0] == q->v && q->axis == 0) { /* W @ X: the columns of X are inputs */
			for (int j = 0; j < nn; j++, q->nsamp++)
				for (int r = 0; r < kk; r++) {
					double xr = b[r * nn + j];
					if (xr == 0) continue;
					for (int c = 0; c < kk; c++) q->H[r * kk + c] += xr * b[c * nn + j];
				}
		} else if (in->a[1] == q->v && q->axis == 1) { /* X @ W: the rows of X are inputs */
			for (int r0 = 0; r0 < mm; r0++, q->nsamp++) {
				const float *x = a + (size_t)r0 * (size_t)kk;
				for (int r = 0; r < kk; r++) {
					double xr = x[r];
					if (xr == 0) continue;
					for (int c = 0; c < kk; c++) q->H[r * kk + c] += xr * x[c];
				}
			}
		}
	}
}

/* Run the calibration rows; with s, collect Hessians; returns all outputs. */
static float *calibrate(Module *m, QSet *s, const char *data, int *nrows)
{
	int per = 0, rows = 0;
	for (int i = 0; i < m->ninputs; i++) per += shape_numel(&m->val[m->inputs[i]].sh);
	char *raw = read_file(data, NULL);
	float *row = xmalloc((size_t)per * sizeof *row), *out = xmalloc((size_t)shape_numel(&m->val[m->output].sh) * sizeof(float));
	int *steps = xmalloc((size_t)(m->nthink ? m->nthink : 1) * sizeof *steps);
	const float **in = xmalloc((size_t)(m->ninputs ? m->ninputs : 1) * sizeof *in);
	vm_matmul_hook = s ? hook : NULL;
	vm_hook_ctx = s;
	int line = 1, no = shape_numel(&m->val[m->output].sh);
	float *all = NULL;
	for (char *p = raw; *p; line++) {
		char *nl = strchr(p, '\n');
		if (nl) *nl = 0;
		const char *bad;
		int k = io_parse_row(p, row, per, &bad);
		if (k < 0) die(data, line, "bad number near '%.20s'", bad);
		if (k > 0) {
			if (k != per) die(data, line, "row has %d values, model %s needs %d", k, m->name, per);
			for (int i = 0, o = 0; i < m->ninputs; i++) {
				in[i] = row + o;
				o += shape_numel(&m->val[m->inputs[i]].sh);
			}
			vm_run(m, in, out, steps);
			all = xrealloc(all, (size_t)(rows + 1) * (size_t)no * sizeof *all);
			memcpy(all + (size_t)rows * (size_t)no, out, (size_t)no * sizeof *out);
			rows++;
		}
		if (!nl) break;
		p = nl + 1;
	}
	vm_matmul_hook = NULL;
	if (!rows) die(data, 0, "no calibration rows");
	*nrows = rows;
	return all;
}

static double weight(const Value *x, const QP *q, int c, int j)
{
	return q->axis == 0 ? x->data[c * q->K + j] : x->data[j * q->C + c];
}

static void set_code(Value *x, const QP *q, int c, int j, int code)
{
	int i = q->axis == 0 ? c * q->K + j : j * q->C + c;
	if (x->qbits == 8) {
		x->q[i] = (signed char)code;
		return;
	}
	unsigned char b = (unsigned char)x->q[i / 2];
	b = i % 2 ? (unsigned char)((b & 0x0f) | (unsigned char)(code << 4)) : (unsigned char)((b & 0xf0) | (code & 0x0f));
	x->q[i / 2] = (signed char)b;
}

static int code_of(double w, double s, int lim)
{
	if (s <= 0) return 0;
	double r = nearbyint(w / s);
	return (int)(r > lim ? lim : r < -lim ? -lim : r);
}

/* Per-channel scale: the clipping of absmax that minimizes the (H-weighted) rounding error. */
static double best_scale(const double *w, int K, const double *hd, int lim)
{
	double amax = 0;
	for (int j = 0; j < K; j++)
		if (fabs(w[j]) > amax) amax = fabs(w[j]);
	if (amax == 0) return 1e-8;
	double best = INFINITY, bs = amax / lim;
	for (int g = 0; g < 10; g++) {
		double s = amax * (1.0 - 0.05 * g) / lim, e = 0;
		for (int j = 0; j < K; j++) {
			double d = w[j] - s * code_of(w[j], s, lim);
			e += d * d * (hd ? hd[j] : 1.0);
		}
		if (e < best) { best = e; bs = s; }
	}
	return bs;
}

/* Lower Cholesky of a symmetric positive-definite n x n matrix, in place. */
static int cholesky(double *A, int n)
{
	for (int j = 0; j < n; j++) {
		double d = A[j * n + j];
		for (int k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
		if (d <= 0) return 0;
		d = sqrt(d);
		A[j * n + j] = d;
		for (int i = j + 1; i < n; i++) {
			double s = A[i * n + j];
			for (int k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
			A[i * n + j] = s / d;
		}
		for (int k = j + 1; k < n; k++) A[j * n + k] = 0;
	}
	return 1;
}

/* GPTQ needs U, the upper Cholesky factor of H^-1: invert H through its
 * Cholesky factor L (H^-1 = L^-T L^-1), then factor H^-1 = U^T U. */
static int inv_chol_upper(const double *H, int n, double *U)
{
	double *L = xmalloc((size_t)n * (size_t)n * sizeof *L), *Li = xmalloc((size_t)n * (size_t)n * sizeof *Li);
	memcpy(L, H, (size_t)n * (size_t)n * sizeof *L);
	int ok = cholesky(L, n);
	if (ok) {
		for (int i = 0; i < n * n; i++) Li[i] = 0;
		for (int c = 0; c < n; c++) /* Li = L^-1, column by column (forward substitution) */
			for (int i = c; i < n; i++) {
				double s = i == c ? 1.0 : 0.0;
				for (int k = c; k < i; k++) s -= L[i * n + k] * Li[k * n + c];
				Li[i * n + c] = s / L[i * n + i];
			}
		for (int i = 0; i < n; i++) /* Hinv = Li^T Li, symmetric */
			for (int j = i; j < n; j++) {
				double s = 0;
				for (int k = j; k < n; k++) s += Li[k * n + i] * Li[k * n + j];
				U[i * n + j] = U[j * n + i] = s;
			}
		ok = cholesky(U, n); /* U now holds lower L2 with Hinv = L2 L2^T; we want the upper factor L2^T */
		if (ok)
			for (int i = 0; i < n; i++)
				for (int j = i + 1; j < n; j++) {
					U[i * n + j] = U[j * n + i];
					U[j * n + i] = 0;
				}
	}
	xfree(L);
	xfree(Li);
	return ok;
}

/* Note on the factor: GPTQ uses the upper Cholesky factor of H^-1 such that
 * H^-1 = U^T U. With Hinv = L2 L2^T (lower L2), U = L2^T satisfies it. */

static void quantize_param(Module *m, QP *q, int bits, int gptq)
{
	Value *x = &m->val[q->v];
	int lim = bits == 8 ? 127 : 7, C = q->C, K = q->K, n = C * K;
	x->qbits = bits;
	x->qaxis = q->axis;
	x->q = xmalloc((size_t)(bits == 8 ? n : (n + 1) / 2));
	x->qs = xmalloc((size_t)C * sizeof(float));
	double *W = xmalloc((size_t)n * sizeof *W), *hd = NULL, *U = NULL;
	for (int c = 0; c < C; c++)
		for (int j = 0; j < K; j++) W[c * K + j] = weight(x, q, c, j);
	if (gptq && q->H && q->nsamp) {
		double *H = q->H, mean = 0;
		hd = xmalloc((size_t)K * sizeof *hd);
		for (int j = 0; j < K; j++) mean += H[j * K + j];
		mean /= K;
		for (int j = 0; j < K; j++) {
			if (H[j * K + j] == 0) { /* an input never active: its weights do not matter */
				H[j * K + j] = 1;
				for (int c = 0; c < C; c++) W[c * K + j] = 0;
			}
			H[j * K + j] += 0.01 * mean; /* dampening */
			hd[j] = H[j * K + j];
		}
		U = xmalloc((size_t)K * (size_t)K * sizeof *U);
		if (!inv_chol_upper(H, K, U)) {
			fprintf(stderr, "quantize: %s: Hessian not positive definite, using rtn\n", x->name);
			xfree(U);
			U = NULL;
		}
	}
	double num = 0, den = 0;
	for (int c = 0; c < C; c++) {
		double *w = W + (size_t)c * (size_t)K;
		double s = best_scale(w, K, hd, lim);
		x->qs[c] = (float)s;
		for (int j = 0; j < K; j++) {
			int code = code_of(w[j], s, lim);
			set_code(x, q, c, j, code);
			if (U) { /* push this column's error onto the remaining inputs */
				double err = (w[j] - s * code) / U[j * K + j];
				for (int jj = j + 1; jj < K; jj++) w[jj] -= err * U[j * K + jj];
			}
		}
	}
	for (int c = 0; c < C; c++) /* report against the original weights; keep data = dequantized */
		for (int j = 0; j < K; j++) {
			int i = q->axis == 0 ? c * K + j : j * C + c;
			double w0 = x->data[i], wq = (double)tg_qget_host(x->q, bits, i) * x->qs[c];
			num += (w0 - wq) * (w0 - wq);
			den += w0 * w0;
			x->data[i] = (float)wq;
		}
	fprintf(stderr, "  %-12s (f32 %d %d) -> int%d, %d %s scales, %s, relative weight error %.4f\n", x->name, x->sh.dim[0], x->sh.dim[1],
		bits, C, q->axis == 0 ? "row" : "column", U ? "gptq" : "rtn", den > 0 ? sqrt(num / den) : 0.0);
	xfree(W);
	xfree(hd);
	xfree(U);
}

int quantize_main(int argc, char **argv)
{
	const char *src = NULL, *out = NULL, *calib = NULL, *method = "rtn";
	int bits = 8, min_size = 256;
	for (int i = 2; i < argc; i++) {
		const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
		if (!strcmp(a, "-o") && v) out = argv[++i];
		else if (!strcmp(a, "--bits") && v) bits = atoi(argv[++i]);
		else if (!strcmp(a, "--method") && v) method = argv[++i];
		else if (!strcmp(a, "--calib") && v) calib = argv[++i];
		else if (!strcmp(a, "--min-size") && v) min_size = atoi(argv[++i]);
		else if (!src && a[0] != '-') src = a;
		else die(NULL, 0, "usage: tgc quantize MODEL [-o OUT.tgir] [--bits 8|4] [--method rtn|gptq] [--calib DATA] [--min-size N]");
	}
	if (!src) die(NULL, 0, "usage: tgc quantize MODEL [-o OUT.tgir] [--bits 8|4] [--method rtn|gptq] [--calib DATA] [--min-size N]");
	if (bits != 8 && bits != 4) die(NULL, 0, "--bits is 8 or 4");
	int gptq = !strcmp(method, "gptq");
	if (!gptq && strcmp(method, "rtn")) die(NULL, 0, "--method is rtn or gptq");
	if (gptq && !calib) die(NULL, 0, "--method gptq needs --calib DATA (input rows in the tgc batch format)");
	Module *m = load_module(src);
	plan(m);
	QSet s = eligible(m, min_size);
	if (!s.n) die(NULL, 0, "%s: no param is eligible (rank-2, >= %d elements, used only as a matmul or spmm operand)", src, min_size);
	float *ref = NULL;
	int nref = 0;
	if (calib && !gptq) ref = calibrate(m, NULL, calib, &nref);
	if (gptq) {
		for (int i = 0; i < s.n; i++) /* spmm tables never reach the hook and stay rtn */
			if (s.p[i].K <= 2048) s.p[i].H = xmalloc((size_t)s.p[i].K * (size_t)s.p[i].K * sizeof(double));
		ref = calibrate(m, &s, calib, &nref);
	}
	long before = 0, after = 0;
	for (int v = 0; v < m->nval; v++)
		if (m->val[v].kind == V_PARAM && !m->val[v].dead) before += 4L * shape_numel(&m->val[v].sh);
	fprintf(stderr, "quantize: %d param(s) of %s\n", s.n, m->name);
	for (int i = 0; i < s.n; i++) quantize_param(m, &s.p[i], bits, gptq);
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_PARAM || x->dead) continue;
		int n = shape_numel(&x->sh);
		after += x->qbits ? (x->qbits == 8 ? n : (n + 1) / 2) + 4L * x->sh.dim[x->qaxis] : 4L * n;
	}
	if (ref) { /* what quantization does to the model's outputs on the calibration rows */
		int dummy, no = shape_numel(&m->val[m->output].sh);
		float *got = calibrate(m, NULL, calib, &dummy);
		double num = 0, den = 0, mx = 0;
		for (long i = 0; i < (long)nref * no; i++) {
			double d = (double)got[i] - ref[i];
			num += d * d;
			den += (double)ref[i] * ref[i];
			if (fabs(d) > mx) mx = fabs(d);
		}
		fprintf(stderr, "quantize: on %d calibration row(s): relative output error %.5f, max |output change| %.5f\n", nref,
			den > 0 ? sqrt(num / den) : 0.0, mx);
	}
	char dflt[1024];
	if (!out) {
		snprintf(dflt, sizeof dflt, "%.*s.q%d.tgir", (int)(strrchr(src, '.') ? strrchr(src, '.') - src : (long)strlen(src)), src, bits);
		out = dflt;
	}
	FILE *f = fopen(out, "w");
	if (!f) die(NULL, 0, "cannot write '%s'", out);
	ir_write(m, f);
	if (fclose(f)) die(NULL, 0, "write failed '%s'", out);
	fprintf(stderr, "quantize: params %ld -> %ld bytes (%.1fx smaller); wrote %s\n", before, after, after ? (double)before / after : 0.0, out);
	return 0;
}
