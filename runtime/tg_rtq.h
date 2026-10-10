/* Technograd fixed-point runtime: Q16.16 kernels for targets without an FPU.
 *
 * Embedded by `tgc c --fixed` in place of tg_rt.h. Every value is a tg_t
 * (int32_t) with 16 fractional bits: range [-32768, 32768), resolution
 * 2^-16 = 1.5e-5. Products accumulate in 64 bits and round once; results
 * saturate instead of wrapping. Transcendentals use range reduction and
 * polynomials evaluated in Q30 (exp, log, sin) or are built from them (tanh,
 * sigmoid, softplus, gelu, silu); sqrt is an exact integer square root.
 * Quantized weight scales are Q7.24 (int32). No float or double appears
 * anywhere, so the unit links on a Cortex-M0/M3 with no soft-float library.
 * The kernels have the same names and argument order as tg_rt.h. */
#ifndef TG_RTQ_H
#define TG_RTQ_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define TG_FN static inline __attribute__((unused))
#else
#define TG_FN static inline
#endif

typedef int32_t tg_t;
#define TG_ONE 65536
#define TG_MAXV 2147483647
#define TG_MINV (-2147483647 - 1)

TG_FN tg_t tg_sat(int64_t v) { return v > TG_MAXV ? TG_MAXV : v < TG_MINV ? TG_MINV : (tg_t)v; }
TG_FN tg_t tg_fmul(tg_t a, tg_t b) { return tg_sat(((int64_t)a * b + 32768) >> 16); }
TG_FN tg_t tg_fdiv(tg_t a, tg_t b)
{
	if (b == 0) return a > 0 ? TG_MAXV : a < 0 ? TG_MINV : 0;
	int64_t n = (int64_t)a * 65536, h = b > 0 ? b / 2 : -(b / 2);
	return tg_sat(((n >= 0) == (b > 0) ? n + h : n - h) / b); /* rounded */
}
TG_FN tg_t tg_absq(tg_t a) { return a < 0 ? (a == TG_MINV ? TG_MAXV : -a) : a; }

TG_FN void tg_copy(tg_t *o, const tg_t *a, int n)
{
	if (o == a) return;
	for (int i = 0; i < n; i++) o[i] = a[i];
}

/* ---- elementwise ---------------------------------------------------------- */
TG_FN void tg_add(tg_t *o, const tg_t *a, int sa, const tg_t *b, int sb, int n)
{ for (int i = 0; i < n; i++) o[i] = tg_sat((int64_t)a[i * sa] + b[i * sb]); }
TG_FN void tg_sub(tg_t *o, const tg_t *a, int sa, const tg_t *b, int sb, int n)
{ for (int i = 0; i < n; i++) o[i] = tg_sat((int64_t)a[i * sa] - b[i * sb]); }
TG_FN void tg_mul(tg_t *o, const tg_t *a, int sa, const tg_t *b, int sb, int n)
{ for (int i = 0; i < n; i++) o[i] = tg_fmul(a[i * sa], b[i * sb]); }
TG_FN void tg_div(tg_t *o, const tg_t *a, int sa, const tg_t *b, int sb, int n)
{ for (int i = 0; i < n; i++) o[i] = tg_fdiv(a[i * sa], b[i * sb]); }
TG_FN void tg_max(tg_t *o, const tg_t *a, int sa, const tg_t *b, int sb, int n)
{ for (int i = 0; i < n; i++) { tg_t x = a[i * sa], y = b[i * sb]; o[i] = x > y ? x : y; } }
TG_FN void tg_min(tg_t *o, const tg_t *a, int sa, const tg_t *b, int sb, int n)
{ for (int i = 0; i < n; i++) { tg_t x = a[i * sa], y = b[i * sb]; o[i] = x < y ? x : y; } }

/* exp: x = k ln2 + r, 0 <= r < ln2; exp(r) by a degree-8 polynomial in Q30. */
TG_FN tg_t tg_expq(tg_t x)
{
	if (x > 681390) return TG_MAXV;   /* exp(10.397) = 32768 */
	if (x < -786432) return 0;        /* exp(-12) < 2^-16 */
	const int64_t ln2 = 45426;        /* ln 2 in Q16 */
	int64_t k = x >= 0 ? x / ln2 : -((-(int64_t)x + ln2 - 1) / ln2);
	int64_t r = ((int64_t)x - k * ln2) << 14; /* Q30, [0, ln2) */
	static const int64_t c[9] = { 1073741824, 1073741824, 536870912, 178956971, 44739243, 8947849, 1491308, 213044, 26631 };
	int64_t p = c[8];
	for (int i = 7; i >= 0; i--) p = c[i] + ((p * r) >> 30);
	int sh = (int)k - 14; /* Q30 * 2^k -> Q16 */
	int64_t v = sh >= 0 ? p << sh : (p + ((int64_t)1 << (-sh - 1))) >> -sh;
	return tg_sat(v);
}

/* log: x = m 2^e with m in [1, 2); log m = 2 atanh((m-1)/(m+1)), Q30 series. */
TG_FN tg_t tg_logq(tg_t x)
{
	if (x <= 0) return TG_MINV;
	int e = 0;
	uint32_t m = (uint32_t)x;
	while (m >= 131072u) { m >>= 1; e++; }
	while (m < 65536u) { m <<= 1; e--; }
	int64_t s = ((int64_t)(m - 65536) << 30) / (m + 65536), s2 = (s * s) >> 30, t = s, acc = 0;
	for (int k = 1; k <= 13; k += 2) {
		acc += t / k;
		t = (t * s2) >> 30;
	}
	return tg_sat(((2 * acc) >> 14) + (int64_t)e * 45426);
}

TG_FN uint64_t tg_isqrt64(uint64_t v)
{
	uint64_t r = 0, bit = (uint64_t)1 << 62;
	while (bit > v) bit >>= 2;
	while (bit) {
		if (v >= r + bit) { v -= r + bit; r = (r >> 1) + bit; }
		else r >>= 1;
		bit >>= 2;
	}
	return r;
}

/* sin: reduce to [-pi, pi], fold to [-pi/2, pi/2], Taylor to x^13 in Q30. */
TG_FN tg_t tg_sinq(tg_t x)
{
	const int64_t pi = 205887, two_pi = 411775, half_pi = 102944; /* Q16 */
	int64_t r = (int64_t)x % two_pi;
	if (r > pi) r -= two_pi;
	if (r < -pi) r += two_pi;
	if (r > half_pi) r = pi - r;
	if (r < -half_pi) r = -pi - r;
	int64_t z = r << 14, z2 = (z * z) >> 30, t = z, acc = 0; /* Q30 */
	for (int k = 1; k <= 13; k += 2) {
		acc += t;
		t = -((t * z2) >> 30) / ((k + 1) * (k + 2));
	}
	return tg_sat((acc + 8192) >> 14);
}

TG_FN void tg_neg(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = tg_sat(-(int64_t)a[i]); }
TG_FN void tg_relu(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = a[i] > 0 ? a[i] : 0; }
TG_FN void tg_step(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = a[i] > 0 ? TG_ONE : 0; }
TG_FN void tg_floor(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = (tg_t)((uint32_t)a[i] & 0xffff0000u); }
TG_FN void tg_exp(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = tg_expq(a[i]); }
TG_FN void tg_log(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = tg_logq(a[i]); }
TG_FN void tg_sqrt(tg_t *o, const tg_t *a, int n)
{ for (int i = 0; i < n; i++) o[i] = a[i] > 0 ? (tg_t)tg_isqrt64((uint64_t)a[i] << 16) : 0; }
TG_FN void tg_sin(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = tg_sinq(a[i]); }
TG_FN void tg_cos(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = tg_sinq(tg_sat((int64_t)a[i] + 102944)); }

TG_FN tg_t tg_tanhq(tg_t x)
{
	tg_t t = tg_absq(x);
	if (t > 9 * TG_ONE) return x < 0 ? -TG_ONE : TG_ONE;
	tg_t e = tg_expq(-2 * t); /* (0, 1] */
	tg_t y = tg_fdiv(TG_ONE - e, TG_ONE + e);
	return x < 0 ? -y : y;
}
TG_FN tg_t tg_sigmoidq(tg_t x)
{
	tg_t e = tg_expq(-tg_absq(x)); /* (0, 1] */
	return x >= 0 ? tg_fdiv(TG_ONE, TG_ONE + e) : tg_fdiv(e, TG_ONE + e);
}
TG_FN void tg_tanh(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = tg_tanhq(a[i]); }
TG_FN void tg_sigmoid(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = tg_sigmoidq(a[i]); }
TG_FN void tg_silu(tg_t *o, const tg_t *a, int n) { for (int i = 0; i < n; i++) o[i] = tg_fmul(a[i], tg_sigmoidq(a[i])); }
TG_FN void tg_softplus(tg_t *o, const tg_t *a, int n)
{ for (int i = 0; i < n; i++) o[i] = tg_sat((int64_t)(a[i] > 0 ? a[i] : 0) + tg_logq(TG_ONE + tg_expq(-tg_absq(a[i])))); }
TG_FN void tg_gelu(tg_t *o, const tg_t *a, int n)
{
	for (int i = 0; i < n; i++) { /* 0.5 x (1 + tanh(0.7978845608 (x + 0.044715 x^3))) */
		int64_t x = a[i], x3 = (((x * x) >> 16) * x) >> 16;
		int64_t u = ((x + ((x3 * 2930) >> 16)) * 52290) >> 16;
		tg_t t = tg_tanhq(tg_sat(u));
		o[i] = tg_sat((x * ((int64_t)TG_ONE + t)) >> 17);
	}
}

/* ---- linear algebra --------------------------------------------------------- */
TG_FN void tg_matmul(tg_t *o, const tg_t *a, const tg_t *b, int m, int k, int n)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			int64_t s = 0;
			for (int p = 0; p < k; p++) s += (int64_t)a[i * k + p] * b[p * n + j];
			o[i * n + j] = tg_sat((s + 32768) >> 16);
		}
}

TG_FN int tg_qget(const signed char *q, int bits, int i)
{
	if (bits == 8) return q[i];
	int b = q[i >> 1];
	return (i & 1) ? (b >> 4) : (int)(signed char)(unsigned char)((unsigned)b << 4) >> 4;
}

/* Quantized weights: integer codes times Q7.24 scales, 64-bit accumulation. */
TG_FN tg_t tg_qscale(int64_t acc, int32_t s24) { return tg_sat((acc * s24 + ((int64_t)1 << 23)) >> 24); }
TG_FN void tg_matmul_qa(tg_t *o, const signed char *q, int bits, const int32_t *s, const tg_t *b, int m, int k, int n)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			int64_t acc = 0;
			for (int p = 0; p < k; p++) acc += (int64_t)tg_qget(q, bits, i * k + p) * b[p * n + j];
			o[i * n + j] = tg_qscale(acc, s[i]);
		}
}
TG_FN void tg_matmul_qb(tg_t *o, const tg_t *a, const signed char *q, int bits, const int32_t *s, int m, int k, int n)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			int64_t acc = 0;
			for (int p = 0; p < k; p++) acc += (int64_t)a[i * k + p] * tg_qget(q, bits, p * n + j);
			o[i * n + j] = tg_qscale(acc, s[j]);
		}
}

TG_FN void tg_softmax(tg_t *o, const tg_t *a, int rows, int cols)
{
	for (int r = 0; r < rows; r++) {
		const tg_t *x = a + r * cols;
		tg_t *y = o + r * cols, mx = x[0];
		int64_t s = 0;
		for (int i = 1; i < cols; i++) if (x[i] > mx) mx = x[i];
		for (int i = 0; i < cols; i++) { y[i] = tg_expq(tg_sat((int64_t)x[i] - mx)); s += y[i]; }
		for (int i = 0; i < cols; i++) y[i] = (tg_t)(((int64_t)y[i] * 65536 + s / 2) / s);
	}
}
TG_FN void tg_rmsnorm(tg_t *o, const tg_t *a, int rows, int cols)
{
	for (int r = 0; r < rows; r++) {
		const tg_t *x = a + r * cols;
		tg_t *y = o + r * cols;
		uint64_t ss = 0;
		for (int i = 0; i < cols; i++) ss += (uint64_t)((int64_t)x[i] * x[i]); /* Q32 */
		uint64_t ms = ss / (uint64_t)cols + 4295; /* + 1e-6 */
		tg_t rms = (tg_t)tg_isqrt64(ms); /* Q16 */
		for (int i = 0; i < cols; i++) y[i] = tg_fdiv(x[i], rms > 0 ? rms : 1);
	}
}
TG_FN void tg_sum(tg_t *o, const tg_t *a, int n) { int64_t s = 0; for (int i = 0; i < n; i++) s += a[i]; o[0] = tg_sat(s); }
TG_FN void tg_mean(tg_t *o, const tg_t *a, int n) { int64_t s = 0; for (int i = 0; i < n; i++) s += a[i]; o[0] = tg_sat(s / n); }
TG_FN void tg_transpose(tg_t *o, const tg_t *a, int r, int c)
{
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++) o[j * r + i] = a[i * c + j];
}
TG_FN void tg_outer(tg_t *o, const tg_t *a, const tg_t *b, int m, int n)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) o[i * n + j] = tg_fmul(a[i], b[j]);
}
TG_FN tg_t tg_delta(const tg_t *a, const tg_t *b, int n)
{
	tg_t d = 0;
	for (int i = 0; i < n; i++) { tg_t e = tg_absq(tg_sat((int64_t)a[i] - b[i])); if (e > d) d = e; }
	return d;
}

/* ---- sparse rows (indices are Q16.16 too) ------------------------------------ */
TG_FN int tg_ell_idx(tg_t v, int d)
{
	int64_t r = ((int64_t)v + 32768) >> 16;
	return r >= 0 && r < d ? (int)r : -1;
}
TG_FN void tg_spmm(tg_t *o, const tg_t *x, const tg_t *w, int r, int k, int d, int h)
{
	for (int i = 0; i < r; i++) {
		tg_t *y = o + i * h;
		for (int j = 0; j < h; j++) y[j] = 0;
		for (int p = 0; p < k; p++) {
			const tg_t *e = x + (i * k + p) * 2;
			int t = tg_ell_idx(e[0], d);
			if (t < 0 || e[1] == 0) continue;
			for (int j = 0; j < h; j++) y[j] = tg_sat((int64_t)y[j] + tg_fmul(e[1], w[t * h + j]));
		}
	}
}
TG_FN void tg_spmm_q(tg_t *o, const tg_t *x, const signed char *q, int bits, const int32_t *s, int r, int k, int d, int h)
{
	for (int i = 0; i < r; i++) {
		tg_t *y = o + i * h;
		for (int j = 0; j < h; j++) y[j] = 0;
		for (int p = 0; p < k; p++) {
			const tg_t *e = x + (i * k + p) * 2;
			int t = tg_ell_idx(e[0], d);
			if (t < 0 || e[1] == 0) continue;
			for (int j = 0; j < h; j++) y[j] = tg_sat((int64_t)y[j] + tg_qscale((int64_t)e[1] * tg_qget(q, bits, t * h + j), s[t]));
		}
	}
}
TG_FN void tg_spmm_t(tg_t *o, const tg_t *x, const tg_t *g, int r, int k, int d, int h)
{
	for (int i = 0; i < d * h; i++) o[i] = 0;
	for (int i = 0; i < r; i++)
		for (int p = 0; p < k; p++) {
			const tg_t *e = x + (i * k + p) * 2;
			int t = tg_ell_idx(e[0], d);
			if (t < 0 || e[1] == 0) continue;
			for (int j = 0; j < h; j++) o[t * h + j] = tg_sat((int64_t)o[t * h + j] + tg_fmul(e[1], g[i * h + j]));
		}
}
TG_FN void tg_spmm_dx(tg_t *o, const tg_t *x, const tg_t *w, const tg_t *g, int r, int k, int d, int h)
{
	for (int i = 0; i < r; i++)
		for (int p = 0; p < k; p++) {
			const tg_t *e = x + (i * k + p) * 2;
			tg_t *q = o + (i * k + p) * 2;
			int t = tg_ell_idx(e[0], d);
			int64_t s = 0;
			if (t >= 0)
				for (int j = 0; j < h; j++) s += (int64_t)w[t * h + j] * g[i * h + j];
			q[0] = 0;
			q[1] = tg_sat((s + 32768) >> 16);
		}
}
TG_FN void tg_sift(tg_t *o, int root, int end)
{
	for (;;) {
		int ch = 2 * root + 1;
		if (ch >= end) return;
		if (ch + 1 < end && o[ch + 1] > o[ch]) ch++;
		if (o[root] >= o[ch]) return;
		tg_t t = o[root]; o[root] = o[ch]; o[ch] = t;
		root = ch;
	}
}
TG_FN void tg_active(tg_t *o, const tg_t *x, int rows, int k, int d)
{
	int n = rows * k, c = 0;
	for (int i = 0; i < n; i++) {
		int t = tg_ell_idx(x[2 * i], d);
		if (t >= 0 && x[2 * i + 1] != 0) o[c++] = (tg_t)t * TG_ONE;
	}
	for (int s = c / 2 - 1; s >= 0; s--) tg_sift(o, s, c);
	for (int end = c - 1; end > 0; end--) { tg_t t = o[0]; o[0] = o[end]; o[end] = t; tg_sift(o, 0, end); }
	int u = 0;
	for (int i = 0; i < c; i++) if (u == 0 || o[i] != o[u - 1]) o[u++] = o[i];
	for (int i = u; i < n; i++) o[i] = -TG_ONE;
}
TG_FN int tg_row_slot(const tg_t *r, int n, int t)
{
	int lo = 0, hi = n;
	tg_t want = (tg_t)t * TG_ONE;
	while (lo < hi) {
		int mid = (lo + hi) / 2;
		if (r[mid] >= 0 && r[mid] < want) lo = mid + 1;
		else hi = mid;
	}
	return lo < n && r[lo] == want ? lo : -1;
}
TG_FN void tg_take(tg_t *o, const tg_t *w, const tg_t *r, int n, int d, int h)
{
	for (int i = 0; i < n; i++) {
		int t = tg_ell_idx(r[i], d);
		for (int j = 0; j < h; j++) o[i * h + j] = t >= 0 ? w[t * h + j] : 0;
	}
}
TG_FN void tg_take_t(tg_t *o, const tg_t *r, const tg_t *g, int n, int d, int h)
{
	for (int i = 0; i < d * h; i++) o[i] = 0;
	for (int i = 0; i < n; i++) {
		int t = tg_ell_idx(r[i], d);
		if (t >= 0)
			for (int j = 0; j < h; j++) o[t * h + j] = tg_sat((int64_t)o[t * h + j] + g[i * h + j]);
	}
}
TG_FN void tg_spmm_tc(tg_t *o, const tg_t *x, const tg_t *g, const tg_t *r, int rows, int k, int n, int h)
{
	for (int i = 0; i < n * h; i++) o[i] = 0;
	for (int i = 0; i < rows; i++)
		for (int p = 0; p < k; p++) {
			const tg_t *e = x + (i * k + p) * 2;
			if (e[1] == 0 || e[0] < -32768) continue;
			int s = tg_row_slot(r, n, (int)(((int64_t)e[0] + 32768) >> 16));
			if (s < 0) continue;
			for (int j = 0; j < h; j++) o[s * h + j] = tg_sat((int64_t)o[s * h + j] + tg_fmul(e[1], g[i * h + j]));
		}
}
TG_FN void tg_put_rows(tg_t *w, const tg_t *src, const tg_t *r, int n, int d, int h)
{
	for (int i = 0; i < n; i++) {
		int t = tg_ell_idx(r[i], d);
		if (t >= 0)
			for (int j = 0; j < h; j++) w[t * h + j] = src[i * h + j];
	}
}

/* ---- decimal text <-> Q16.16 without floating point (test drivers) ---------- */
TG_FN const char *tg_fx_parse(const char *s, tg_t *v)
{
	int neg = 0;
	if (*s == '-' || *s == '+') neg = *s++ == '-';
	int64_t ip = 0, fn = 0, fd = 1;
	int any = 0;
	while (*s >= '0' && *s <= '9') { if (ip < 100000) ip = ip * 10 + (*s - '0'); s++; any = 1; }
	if (*s == '.') {
		s++;
		while (*s >= '0' && *s <= '9') { if (fd < 1000000000000LL) { fn = fn * 10 + (*s - '0'); fd *= 10; } s++; any = 1; }
	}
	if (!any) return 0;
	int ex = 0;
	if (*s == 'e' || *s == 'E') {
		int en = 0;
		s++;
		if (*s == '-' || *s == '+') en = *s++ == '-';
		while (*s >= '0' && *s <= '9') ex = ex * 10 + (*s++ - '0');
		if (en) ex = -ex;
	}
	int64_t num = ip * fd + fn, den = fd; /* value = num / den * 10^ex */
	for (; ex > 0; ex--) { if (num > ((int64_t)1 << 47)) break; num *= 10; }
	for (; ex < 0; ex++) den *= 10;
	int64_t q = (num * 65536 + den / 2) / den;
	*v = tg_sat(neg ? -q : q);
	return s;
}
/* writes v with 6 decimals into buf (>= 16 bytes) */
TG_FN void tg_fx_format(tg_t v, char *buf)
{
	int64_t x = v, neg = x < 0;
	if (neg) x = -x;
	int64_t ip = x >> 16, fr = ((x & 0xffff) * 1000000 + 32768) >> 16;
	if (fr >= 1000000) { ip++; fr -= 1000000; }
	char t[24];
	int n = 0;
	for (int i = 0; i < 6; i++) { t[n++] = (char)('0' + fr % 10); fr /= 10; }
	t[n++] = '.';
	do { t[n++] = (char)('0' + ip % 10); ip /= 10; } while (ip);
	if (neg) t[n++] = '-';
	int k = 0;
	while (n) buf[k++] = t[--n];
	buf[k] = 0;
}

#endif
