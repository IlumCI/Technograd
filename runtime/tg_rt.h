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

/* Halting criterion for latent loops: max |a - b|. */
TG_FN float tg_delta(const float *a, const float *b, int n)
{
	float d = 0.0f;
	for (int i = 0; i < n; i++) { float e = fabsf(a[i] - b[i]); if (e > d) d = e; }
	return d;
}

#endif
