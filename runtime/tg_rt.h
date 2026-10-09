/* Technograd runtime kernels.
 *
 * Shared verbatim by the reference VM and by every generated C unit, so the
 * interpreter and the compiled black box execute bit-identical arithmetic.
 * Freestanding apart from <math.h>; no allocation, no I/O. */
#ifndef TG_RT_H
#define TG_RT_H

#include <math.h>

#if defined(__GNUC__) || defined(__clang__)
#define TG_FN static inline __attribute__((unused))
#else
#define TG_FN static inline
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

/* o[m,n] = a[m,k] @ b[k,n]. o must not alias a or b. */
TG_FN void tg_matmul(float *o, const float *a, const float *b, int m, int k, int n)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			float s = 0.0f;
			for (int p = 0; p < k; p++) s += a[i * k + p] * b[p * n + j];
			o[i * n + j] = s;
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

#endif
