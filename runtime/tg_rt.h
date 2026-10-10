/* Technograd runtime kernels.
 *
 * Shared verbatim by the reference VM and by every generated C unit, so the
 * interpreter and the compiled black box execute bit-identical arithmetic.
 * Freestanding apart from <math.h>; no allocation, no I/O. */
#ifndef TG_RT_H
#define TG_RT_H

#include <math.h>

#ifndef TG_FN /* tgc asm --support defines it empty: extern kernels for native code */
#if defined(__GNUC__) || defined(__clang__)
#define TG_FN static inline __attribute__((unused))
#else
#define TG_FN static inline
#endif
#endif

TG_FN void tg_copy(float *o, const float *a, int n)
{
	if (o == a) return;
	for (int i = 0; i < n; i++) o[i] = a[i];
}

/* Binary elementwise. sa/sb are 1 for a full operand, 0 for a broadcast scalar. */
TG_FN void tg_add(float *o, const float *a, int sa, const float *b, int sb, int n)
{ for (int i = 0; i < n; i++) o[i] = a[i * sa] + b[i * sb]; }
TG_FN void tg_sub(float *o, const float *a, int sa, const float *b, int sb, int n)
{ for (int i = 0; i < n; i++) o[i] = a[i * sa] - b[i * sb]; }
TG_FN void tg_mul(float *o, const float *a, int sa, const float *b, int sb, int n)
{ for (int i = 0; i < n; i++) o[i] = a[i * sa] * b[i * sb]; }
TG_FN void tg_div(float *o, const float *a, int sa, const float *b, int sb, int n)
{ for (int i = 0; i < n; i++) o[i] = a[i * sa] / b[i * sb]; }
TG_FN void tg_max(float *o, const float *a, int sa, const float *b, int sb, int n)
{ for (int i = 0; i < n; i++) { float x = a[i * sa], y = b[i * sb]; o[i] = x > y ? x : y; } }
TG_FN void tg_min(float *o, const float *a, int sa, const float *b, int sb, int n)
{ for (int i = 0; i < n; i++) { float x = a[i * sa], y = b[i * sb]; o[i] = x < y ? x : y; } }

/* Unary elementwise. */
TG_FN void tg_neg(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = -a[i]; }
TG_FN void tg_tanh(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = tanhf(a[i]); }
TG_FN void tg_relu(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = a[i] > 0.0f ? a[i] : 0.0f; }
TG_FN void tg_sigmoid(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = 1.0f / (1.0f + expf(-a[i])); }
TG_FN void tg_exp(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = expf(a[i]); }
TG_FN void tg_sqrt(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = sqrtf(a[i]); }
TG_FN void tg_gelu(float *o, const float *a, int n)
{
	for (int i = 0; i < n; i++) {
		float x = a[i];
		o[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
	}
}
TG_FN void tg_silu(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = a[i] / (1.0f + expf(-a[i])); }

/* Overflow-free softplus: log(1 + e^x) = max(x, 0) + log1p(e^-|x|). */
TG_FN void tg_softplus(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = (a[i] > 0.0f ? a[i] : 0.0f) + log1pf(expf(-fabsf(a[i]))); }
TG_FN void tg_log(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = logf(a[i]); }
TG_FN void tg_sin(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = sinf(a[i]); }
TG_FN void tg_cos(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = cosf(a[i]); }
TG_FN void tg_floor(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = floorf(a[i]); }
/* Heaviside step, x > 0. The derivative of relu, max and min. */
TG_FN void tg_step(float *o, const float *a, int n)
{ for (int i = 0; i < n; i++) o[i] = a[i] > 0.0f ? 1.0f : 0.0f; }

#ifndef TG_ASM_MATMUL /* a native backend supplies its own */
/* o[m,n] = a[m,k] @ b[k,n]. o must not alias a or b. Each output sums its k
 * products in order p = 0..k-1; rows of b are streamed (i-p-j order), so the
 * inner loop is contiguous and vectorizes. */
TG_FN void tg_matmul(float *o, const float *a, const float *b, int m, int k, int n)
{
	if (n == 1) {
		for (int i = 0; i < m; i++) {
			float s = 0.0f;
			for (int p = 0; p < k; p++) s += a[i * k + p] * b[p];
			o[i] = s;
		}
		return;
	}
#define TG_MM_NARROW(W) /* n == W: the row of outputs stays in registers */ \
	if (n == W) { \
		for (int i = 0; i < m; i++) { \
			float s[W] = { 0.0f }; \
			for (int p = 0; p < k; p++) { \
				const float aip = a[(long)i * k + p]; \
				for (int j = 0; j < W; j++) s[j] += aip * b[(long)p * W + j]; \
			} \
			for (int j = 0; j < W; j++) o[(long)i * W + j] = s[j]; \
		} \
		return; \
	}
	TG_MM_NARROW(2)
	TG_MM_NARROW(3)
	TG_MM_NARROW(4)
	TG_MM_NARROW(8)
#undef TG_MM_NARROW
	for (int i = 0; i < m; i++) {
		float *restrict oi = o + (long)i * n;
		for (int j = 0; j < n; j++) oi[j] = 0.0f;
		for (int p = 0; p < k; p++) {
			const float aip = a[(long)i * k + p], *restrict bp = b + (long)p * n;
			for (int j = 0; j < n; j++) oi[j] += aip * bp[j];
		}
	}
}
#else
TG_FN void tg_matmul(float *o, const float *a, const float *b, int m, int k, int n);
#endif

/* Quantized weights: signed int8 codes, or int4 codes packed two per byte
 * (element i in byte i/2, low nibble first), times one f32 scale per output
 * channel. Accumulation is in f32: weight-only quantization (W8A32/W4A32). */
TG_FN int tg_qget(const signed char *q, int bits, int i)
{
	if (bits == 8) return q[i];
	int b = q[i >> 1];
	return (i & 1) ? (b >> 4) : (int)(signed char)(unsigned char)((unsigned)b << 4) >> 4;
}

/* o[m,n] = diag(s) Q[m,k] @ b[k,n]: quantized left operand, a scale per row. */
TG_FN void tg_matmul_qa(float *o, const signed char *q, int bits, const float *s, const float *b, int m, int k, int n)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			float acc = 0.0f;
			for (int p = 0; p < k; p++) acc += (float)tg_qget(q, bits, i * k + p) * b[p * n + j];
			o[i * n + j] = s[i] * acc;
		}
}

/* o[m,n] = a[m,k] @ Q[k,n] diag(s): quantized right operand, a scale per column. */
TG_FN void tg_matmul_qb(float *o, const float *a, const signed char *q, int bits, const float *s, int m, int k, int n)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			float acc = 0.0f;
			for (int p = 0; p < k; p++) acc += a[i * k + p] * (float)tg_qget(q, bits, p * n + j);
			o[i * n + j] = s[j] * acc;
		}
}

/* Row-wise ops over the last axis. */
TG_FN void tg_softmax(float *o, const float *a, int rows, int cols)
{
	for (int r = 0; r < rows; r++) {
		const float *x = a + r * cols;
		float *y = o + r * cols;
		float mx = x[0], s = 0.0f;
		for (int i = 1; i < cols; i++) if (x[i] > mx) mx = x[i];
		for (int i = 0; i < cols; i++) { y[i] = expf(x[i] - mx); s += y[i]; }
		for (int i = 0; i < cols; i++) y[i] /= s;
	}
}
TG_FN void tg_rmsnorm(float *o, const float *a, int rows, int cols)
{
	for (int r = 0; r < rows; r++) {
		const float *x = a + r * cols;
		float *y = o + r * cols;
		float s = 0.0f;
		for (int i = 0; i < cols; i++) s += x[i] * x[i];
		float inv = 1.0f / sqrtf(s / (float)cols + 1e-6f);
		for (int i = 0; i < cols; i++) y[i] = x[i] * inv;
	}
}

/* Reductions to a scalar. */
TG_FN void tg_sum(float *o, const float *a, int n)
{ float s = 0.0f; for (int i = 0; i < n; i++) s += a[i]; o[0] = s; }
TG_FN void tg_mean(float *o, const float *a, int n)
{ float s = 0.0f; for (int i = 0; i < n; i++) s += a[i]; o[0] = s / (float)n; }

TG_FN void tg_transpose(float *o, const float *a, int r, int c)
{
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++) o[j * r + i] = a[i * c + j];
}

/* o[m,n] = a[m] b[n]^T, the rank-1 write of delta-rule and Hebbian updates. */
TG_FN void tg_outer(float *o, const float *a, const float *b, int m, int n)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) o[i * n + j] = a[i] * b[j];
}

/* Sparse rows in ELLPACK form: x[r, K, 2] holds K (index, value) pairs per
 * row; padding has value 0. An index is rounded to the nearest integer and
 * entries outside [0, d) are skipped, so a malformed row cannot address out
 * of bounds. Cost is O(r K h), independent of d (embedding bag, DLRM-style). */
TG_FN int tg_ell_idx(float v, int d)
{
	float r = floorf(v + 0.5f);
	return r >= 0.0f && r < (float)d ? (int)r : -1;
}

/* o[r, h] = sum_k x[r,k].value * w[x[r,k].index, :] */
TG_FN void tg_spmm(float *o, const float *x, const float *w, int r, int k, int d, int h)
{
	for (int i = 0; i < r; i++) {
		float *y = o + i * h;
		for (int j = 0; j < h; j++) y[j] = 0.0f;
		for (int p = 0; p < k; p++) {
			const float *e = x + (i * k + p) * 2;
			int t = tg_ell_idx(e[0], d);
			if (t < 0 || e[1] == 0.0f) continue;
			for (int j = 0; j < h; j++) y[j] += e[1] * w[t * h + j];
		}
	}
}

/* spmm with a quantized table: a scale per table row. */
TG_FN void tg_spmm_q(float *o, const float *x, const signed char *q, int bits, const float *s, int r, int k, int d, int h)
{
	for (int i = 0; i < r; i++) {
		float *y = o + i * h;
		for (int j = 0; j < h; j++) y[j] = 0.0f;
		for (int p = 0; p < k; p++) {
			const float *e = x + (i * k + p) * 2;
			int t = tg_ell_idx(e[0], d);
			if (t < 0 || e[1] == 0.0f) continue;
			float v = e[1] * s[t];
			for (int j = 0; j < h; j++) y[j] += v * (float)tg_qget(q, bits, t * h + j);
		}
	}
}

/* Gradient with respect to w: o[d, h] = x^T g, a scatter-add of value * g row. */
TG_FN void tg_spmm_t(float *o, const float *x, const float *g, int r, int k, int d, int h)
{
	for (int i = 0; i < d * h; i++) o[i] = 0.0f;
	for (int i = 0; i < r; i++)
		for (int p = 0; p < k; p++) {
			const float *e = x + (i * k + p) * 2;
			int t = tg_ell_idx(e[0], d);
			if (t < 0 || e[1] == 0.0f) continue;
			for (int j = 0; j < h; j++) o[t * h + j] += e[1] * g[i * h + j];
		}
}

/* Gradient with respect to x: 0 for indices, w[index] . g[row] for values. */
TG_FN void tg_spmm_dx(float *o, const float *x, const float *w, const float *g, int r, int k, int d, int h)
{
	for (int i = 0; i < r; i++)
		for (int p = 0; p < k; p++) {
			const float *e = x + (i * k + p) * 2;
			float *q = o + (i * k + p) * 2;
			int t = tg_ell_idx(e[0], d);
			float s = 0.0f;
			if (t >= 0)
				for (int j = 0; j < h; j++) s += w[t * h + j] * g[i * h + j];
			q[0] = 0.0f;
			q[1] = s;
		}
}

/* ---- row-sparse steps ----------------------------------------------------
 * A row set r[n] lists distinct row indices in ascending order, padded with
 * -1. It lets an optimizer touch only the rows a batch used (lazy updates):
 * cost O(n H) instead of O(D H). */

TG_FN void tg_sift(float *o, int root, int end)
{
	for (;;) {
		int ch = 2 * root + 1;
		if (ch >= end) return;
		if (ch + 1 < end && o[ch + 1] > o[ch]) ch++;
		if (o[root] >= o[ch]) return;
		float t = o[root]; o[root] = o[ch]; o[ch] = t;
		root = ch;
	}
}

/* Distinct valid indices of sparse rows x[rows, k, 2] that carry a nonzero
 * value, ascending (heapsort, no allocation), then -1 padding. */
TG_FN void tg_active(float *o, const float *x, int rows, int k, int d)
{
	int n = rows * k, c = 0;
	for (int i = 0; i < n; i++) {
		int t = tg_ell_idx(x[2 * i], d);
		if (t >= 0 && x[2 * i + 1] != 0.0f) o[c++] = (float)t;
	}
	for (int s = c / 2 - 1; s >= 0; s--) tg_sift(o, s, c);
	for (int end = c - 1; end > 0; end--) {
		float t = o[0]; o[0] = o[end]; o[end] = t;
		tg_sift(o, 0, end);
	}
	int u = 0;
	for (int i = 0; i < c; i++)
		if (u == 0 || o[i] != o[u - 1]) o[u++] = o[i];
	for (int i = u; i < n; i++) o[i] = -1.0f;
}

/* Position of row t in the row set, or -1. */
TG_FN int tg_row_slot(const float *r, int n, int t)
{
	int lo = 0, hi = n;
	while (lo < hi) { /* -1 padding sorts last */
		int mid = (lo + hi) / 2;
		if (r[mid] >= 0.0f && r[mid] < (float)t) lo = mid + 1;
		else hi = mid;
	}
	return lo < n && r[lo] == (float)t ? lo : -1;
}

/* o[n, h] = rows r of w[d, h]; zero for padding. */
TG_FN void tg_take(float *o, const float *w, const float *r, int n, int d, int h)
{
	for (int i = 0; i < n; i++) {
		int t = tg_ell_idx(r[i], d);
		for (int j = 0; j < h; j++) o[i * h + j] = t >= 0 ? w[t * h + j] : 0.0f;
	}
}

/* Gradient of take: o[d, h] = 0, then g's rows added at rows r. */
TG_FN void tg_take_t(float *o, const float *r, const float *g, int n, int d, int h)
{
	for (int i = 0; i < d * h; i++) o[i] = 0.0f;
	for (int i = 0; i < n; i++) {
		int t = tg_ell_idx(r[i], d);
		if (t >= 0)
			for (int j = 0; j < h; j++) o[t * h + j] += g[i * h + j];
	}
}

/* Compact gradient: o[n, h] = (x^T g) restricted to the rows in r. */
TG_FN void tg_spmm_tc(float *o, const float *x, const float *g, const float *r, int rows, int k, int n, int h)
{
	for (int i = 0; i < n * h; i++) o[i] = 0.0f;
	for (int i = 0; i < rows; i++)
		for (int p = 0; p < k; p++) {
			const float *e = x + (i * k + p) * 2;
			if (e[1] == 0.0f || !(e[0] > -0.5f && e[0] < 16777216.0f)) continue; /* also rejects NaN */
			int s = tg_row_slot(r, n, (int)floorf(e[0] + 0.5f));
			if (s < 0) continue;
			for (int j = 0; j < h; j++) o[s * h + j] += e[1] * g[i * h + j];
		}
}

/* Row-scatter commit: rows r of state w[d, h] = src[n, h]; padding and
 * out-of-range indices are skipped, a repeated index takes the later row. */
TG_FN void tg_put_rows(float *w, const float *src, const float *r, int n, int d, int h)
{
	for (int i = 0; i < n; i++) {
		int t = tg_ell_idx(r[i], d);
		if (t >= 0)
			for (int j = 0; j < h; j++) w[t * h + j] = src[i * h + j];
	}
}

/* Halting criterion for latent loops: max |a - b|. */
TG_FN float tg_delta(const float *a, const float *b, int n)
{
	float d = 0.0f;
	for (int i = 0; i < n; i++) { float e = fabsf(a[i] - b[i]); if (e > d) d = e; }
	return d;
}

/* ---- loop control and broadcasting for native backends (tgc asm) ---------
 * Generated assembly never touches floating-point registers: think-loop
 * halting and row broadcasting are these calls. */
TG_FN int tg_k_think(float *o, const float *y, int n, const float *eps, const float *p, float *sv, const float *thr)
{
	float d = tg_delta(y, o, n);
	tg_copy(o, y, n);
	if (eps && d <= *eps) return 1;
	if (p) {
		float q = *p < 0.0f ? 0.0f : *p > 1.0f ? 1.0f : *p;
		*sv *= 1.0f - q;
		if (1.0f - *sv > *thr) return 1;
	}
	return 0;
}
TG_FN void tg_k_bin(int op, float *o, const float *a, int na, const float *b, int nb, int n)
{
	void (*k)(float *, const float *, int, const float *, int, int) = op == 0 ? tg_add : op == 1 ? tg_sub : op == 2 ? tg_mul : op == 3 ? tg_div : op == 4 ? tg_max : tg_min;
	if ((na == n || na == 1) && (nb == n || nb == 1)) {
		k(o, a, na == n, b, nb == n, n);
		return;
	}
	int w = na < n ? na : nb;
	for (int r = 0; r < n / w; r++) k(o + r * w, na < n ? a : a + r * w, 1, nb < n ? b : b + r * w, 1, w);
}

/* ---- broadcasting, reduction to a broadcast operand, batched products -----
 * d: rank r, the r dimensions of the iteration space, then the element
 * strides of a and b (0 along broadcast axes); the innermost axis has
 * stride 0 or 1 in each operand (see bcast_desc in the compiler). */
TG_FN void tg_bin_bc(int op, float *o, const float *a, const float *b, const int *d)
{
	void (*k)(float *, const float *, int, const float *, int, int) = op == 0 ? tg_add : op == 1 ? tg_sub : op == 2 ? tg_mul : op == 3 ? tg_div : op == 4 ? tg_max : tg_min;
	int r = d[0], idx[8] = { 0 };
	const int *dim = d + 1, *sa = d + 1 + r, *sb = d + 1 + 2 * r;
	int w = dim[r - 1];
	long n = 1, ia = 0, ib = 0;
	for (int i = 0; i < r - 1; i++) n *= dim[i];
	for (long i = 0; i < n; i++) {
		k(o + i * w, a + ia, sa[r - 1] != 0, b + ib, sb[r - 1] != 0, w);
		for (int x = r - 2; x >= 0; x--) {
			ia += sa[x];
			ib += sb[x];
			if (++idx[x] < dim[x]) break;
			ia -= (long)sa[x] * dim[x];
			ib -= (long)sb[x] * dim[x];
			idx[x] = 0;
		}
	}
}
/* o (no elements) += x over the iteration space of x; d's first strides are o's */
TG_FN void tg_sum_to(float *o, const float *x, int no, const int *d)
{
	int r = d[0], idx[8] = { 0 };
	const int *dim = d + 1, *so = d + 1 + r;
	int w = dim[r - 1];
	long n = 1, io = 0;
	for (int i = 0; i < no; i++) o[i] = 0;
	for (int i = 0; i < r - 1; i++) n *= dim[i];
	for (long i = 0; i < n; i++) {
		const float *p = x + i * w;
		if (so[r - 1]) for (int j = 0; j < w; j++) o[io + j] += p[j];
		else { float s = 0.0f; for (int j = 0; j < w; j++) s += p[j]; o[io] += s; }
		for (int q = r - 2; q >= 0; q--) {
			io += so[q];
			if (++idx[q] < dim[q]) break;
			io -= (long)so[q] * dim[q];
			idx[q] = 0;
		}
	}
}
TG_FN void tg_bmm(float *o, const float *a, const float *b, int g, int m, int k, int n)
{
	for (int i = 0; i < g; i++) tg_matmul(o + (long)i * m * n, a + (long)i * m * k, b + (long)i * k * n, m, k, n);
}
TG_FN void tg_btranspose(float *o, const float *a, int g, int r, int c)
{
	for (int i = 0; i < g; i++) tg_transpose(o + (long)i * r * c, a + (long)i * r * c, r, c);
}

#endif
