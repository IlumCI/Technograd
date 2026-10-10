/* Operation table and shape inference. Single source of truth for both the
 * surface lowering and the TGIR verifier. */
#include "tg.h"

#include <string.h>

const OpInfo tg_ops[OP_COUNT] = {
	[OP_CONST]     = { "const",     0, CLS_CONST },
	[OP_ADD]       = { "add",       2, CLS_BIN },
	[OP_SUB]       = { "sub",       2, CLS_BIN },
	[OP_MUL]       = { "mul",       2, CLS_BIN },
	[OP_DIV]       = { "div",       2, CLS_BIN },
	[OP_MAX]       = { "max",       2, CLS_BIN },
	[OP_MIN]       = { "min",       2, CLS_BIN },
	[OP_NEG]       = { "neg",       1, CLS_UN },
	[OP_TANH]      = { "tanh",      1, CLS_UN },
	[OP_RELU]      = { "relu",      1, CLS_UN },
	[OP_SIGMOID]   = { "sigmoid",   1, CLS_UN },
	[OP_EXP]       = { "exp",       1, CLS_UN },
	[OP_SQRT]      = { "sqrt",      1, CLS_UN },
	[OP_GELU]      = { "gelu",      1, CLS_UN },
	[OP_SILU]      = { "silu",      1, CLS_UN },
	[OP_SOFTPLUS]  = { "softplus",  1, CLS_UN },
	[OP_LOG]       = { "log",       1, CLS_UN },
	[OP_MATMUL]    = { "matmul",    2, CLS_MATMUL },
	[OP_SOFTMAX]   = { "softmax",   1, CLS_ROW },
	[OP_RMSNORM]   = { "rmsnorm",   1, CLS_ROW },
	[OP_SUM]       = { "sum",       1, CLS_RED },
	[OP_MEAN]      = { "mean",      1, CLS_RED },
	[OP_TRANSPOSE] = { "transpose", 1, CLS_TRANS },
	[OP_OUTER]     = { "outer",     2, CLS_OUTER },
	[OP_STEP]      = { "step",      1, CLS_UN },
	[OP_SPMM]      = { "spmm",      2, CLS_SPMM },
	[OP_SPMM_T]    = { "spmm_t",    3, CLS_SPMM },
	[OP_SPMM_DX]   = { "spmm_dx",   3, CLS_SPMM },
	[OP_ACTIVE]    = { "active",    2, CLS_ROWS },
	[OP_TAKE]      = { "take",      2, CLS_ROWS },
	[OP_TAKE_T]    = { "take_t",    3, CLS_ROWS },
	[OP_SPMM_TC]   = { "spmm_tc",   3, CLS_ROWS },
	[OP_SIN]       = { "sin",       1, CLS_UN },
	[OP_COS]       = { "cos",       1, CLS_UN },
	[OP_RESHAPE]   = { "reshape",   1, CLS_RESHAPE },
	[OP_FLOOR]     = { "floor",     1, CLS_UN },
	[OP_SUM_TO]    = { "sum_to",    2, CLS_SUMTO },
	[OP_THINK]     = { "think",     0, CLS_THINK },
	[OP_SCAN]      = { "scan",      0, CLS_SCAN },
};

int op_lookup(const char *name)
{
	for (int i = 0; i < OP_COUNT; i++)
		if (i != OP_CONST && i != OP_THINK && i != OP_SCAN && strcmp(tg_ops[i].name, name) == 0) return i;
	return -1;
}

/* Per-group dimensions. (G, M, K) @ (G, K, N) is G products of M x K by
 * K x N; (G, M, K) @ (K, N) or @ (K) is one product with G * M rows. */
void matmul_dims(const Shape *a, const Shape *b, int *m, int *k, int *n)
{
	int g = matmul_groups(a, b);
	*m = a->rank == 3 ? (g > 1 || b->rank == 3 ? a->dim[1] : a->dim[0] * a->dim[1]) : a->rank == 2 ? a->dim[0] : 1;
	*k = a->dim[a->rank - 1];
	*n = b->rank >= 2 ? b->dim[b->rank - 1] : 1;
}

int matmul_groups(const Shape *a, const Shape *b)
{
	return a->rank == 3 && b->rank == 3 ? a->dim[0] : 1;
}

int bin_simple(const Shape *out, const Shape *a, const Shape *b)
{
	int n = shape_numel(out), na = shape_numel(a), nb = shape_numel(b);
	if ((na == n || na == 1) && (nb == n || nb == 1)) return 1;
	return (shape_eq(a, out) && shape_suffix(b, out)) || (shape_eq(b, out) && shape_suffix(a, out));
}

void spmm_dims(const Shape *x, const Shape *w, int *rows, int *k, int *d, int *h)
{
	*rows = x->rank == 3 ? x->dim[0] : 1;
	*k = x->dim[x->rank - 2];
	*d = w->dim[0];
	*h = w->dim[1];
}

void row_dims(const Shape *s, int *rows, int *cols)
{
	*cols = s->rank ? s->dim[s->rank - 1] : 1;
	*rows = shape_numel(s) / *cols;
}

int op_infer(Op op, const Shape *a, int na, Shape *out, char *err, size_t errn)
{
	const OpInfo *oi = &tg_ops[op];
	char s0[64], s1[64];
	if (na != oi->arity) {
		snprintf(err, errn, "'%s' takes %d operand(s), got %d", oi->name, oi->arity, na);
		return 0;
	}
	memset(out, 0, sizeof *out);
	switch (oi->cls) {
	case CLS_CONST:
		return 1;
	case CLS_BIN:
		if (shape_broadcast(&a[0], &a[1], out)) return 1;
		shape_str(&a[0], s0, sizeof s0);
		shape_str(&a[1], s1, sizeof s1);
		snprintf(err, errn, "'%s' shape mismatch %s vs %s (dimensions are compared from the right; each pair must be equal or one of them 1)", oi->name, s0, s1);
		return 0;
	case CLS_SUMTO: { /* sum_to(x, like): x summed over the axes like was broadcast along */
		Shape b;
		if (!shape_broadcast(&a[0], &a[1], &b) || !shape_eq(&b, &a[0])) {
			shape_str(&a[0], s0, sizeof s0);
			shape_str(&a[1], s1, sizeof s1);
			snprintf(err, errn, "'sum_to' cannot reduce %s to %s (%s must broadcast to %s)", s0, s1, s1, s0);
			return 0;
		}
		*out = a[1];
		return 1;
	}
	case CLS_UN:
		*out = a[0];
		return 1;
	case CLS_MATMUL: {
		/* (M,K)/(K) @ (K,N)/(K); (G,M,K) @ (K,N)/(K) with a shared right
		 * operand; (G,M,K) @ (G,K,N) batched */
		int ra = a[0].rank, rb = a[1].rank;
		int ka = ra >= 1 ? a[0].dim[ra - 1] : -1;
		int kb = rb == 3 ? a[1].dim[1] : rb >= 1 ? a[1].dim[0] : -1;
		if (ra < 1 || ra > 3 || rb < 1 || rb > 3 || ka != kb || (rb == 3 && (ra != 3 || a[0].dim[0] != a[1].dim[0]))) {
			shape_str(&a[0], s0, sizeof s0);
			shape_str(&a[1], s1, sizeof s1);
			snprintf(err, errn, "'matmul' incompatible operands %s @ %s", s0, s1);
			return 0;
		}
		for (int i = 0; i < ra - 1; i++) out->dim[out->rank++] = a[0].dim[i];
		if (rb >= 2) out->dim[out->rank++] = a[1].dim[rb - 1];
		return 1;
	}
	case CLS_ROW:
		if (a[0].rank == 0) {
			snprintf(err, errn, "'%s' needs rank >= 1", oi->name);
			return 0;
		}
		*out = a[0];
		return 1;
	case CLS_RED:
		return 1; /* scalar */
	case CLS_TRANS:
		if (a[0].rank != 2 && a[0].rank != 3) {
			snprintf(err, errn, "'transpose' needs rank 2, or rank 3 (the last two axes swap)");
			return 0;
		}
		*out = a[0];
		out->dim[out->rank - 2] = a[0].dim[a[0].rank - 1];
		out->dim[out->rank - 1] = a[0].dim[a[0].rank - 2];
		return 1;
	case CLS_OUTER:
		if (a[0].rank != 1 || a[1].rank != 1) {
			shape_str(&a[0], s0, sizeof s0);
			shape_str(&a[1], s1, sizeof s1);
			snprintf(err, errn, "'outer' needs two vectors, got %s and %s", s0, s1);
			return 0;
		}
		out->rank = 2;
		out->dim[0] = a[0].dim[0];
		out->dim[1] = a[1].dim[0];
		return 1;
	case CLS_SPMM: {
		/* spmm(x, w); spmm_t(x, g, w) -> shape of w; spmm_dx(x, w, g) -> shape of x */
		const Shape *x = &a[0], *w = &a[op == OP_SPMM_T ? 2 : 1];
		if ((x->rank != 2 && x->rank != 3) || x->dim[x->rank - 1] != 2 || w->rank != 2) {
			shape_str(x, s0, sizeof s0);
			shape_str(w, s1, sizeof s1);
			snprintf(err, errn, "'%s' needs sparse rows (K, 2) or (B, K, 2) of (index, value) pairs and a matrix (D, H), got %s and %s",
				 oi->name, s0, s1);
			return 0;
		}
		Shape y = { 0 };
		if (x->rank == 3) y.dim[y.rank++] = x->dim[0];
		y.dim[y.rank++] = w->dim[1];
		if (op == OP_SPMM) { *out = y; return 1; }
		const Shape *g = &a[op == OP_SPMM_T ? 1 : 2];
		if (!shape_eq(g, &y)) {
			shape_str(g, s0, sizeof s0);
			shape_str(&y, s1, sizeof s1);
			snprintf(err, errn, "'%s' gradient %s does not match the product shape %s", oi->name, s0, s1);
			return 0;
		}
		*out = op == OP_SPMM_T ? *w : *x;
		return 1;
	}
	case CLS_ROWS: {
		/* active(s, w) -> (n): the distinct rows of w that s touches, ascending, padded with -1
		 * take(w, r) -> (n, ...): rows r of w, zero for -1
		 * take_t(r, g, w) -> shape of w: rows of g added at r (the gradient of take)
		 * spmm_tc(s, g, r) -> (n, H): s^T g restricted to rows r (compact gradient) */
		if (op == OP_TAKE_T) {
			Shape want = a[2];
			int ok = a[0].rank == 1 && want.rank >= 1;
			if (ok) want.dim[0] = a[0].dim[0];
			if (!ok || !shape_eq(&a[1], &want)) {
				snprintf(err, errn, "'take_t' needs rows (n), a gradient (n, ...) and the table it scatters into");
				return 0;
			}
			*out = a[2];
			return 1;
		}
		if (op == OP_TAKE) {
			if (a[0].rank < 1 || a[1].rank != 1) {
				shape_str(&a[0], s0, sizeof s0);
				shape_str(&a[1], s1, sizeof s1);
				snprintf(err, errn, "'take' needs a tensor (D, ...) and row indices (n), got %s and %s", s0, s1);
				return 0;
			}
			*out = a[0];
			out->dim[0] = a[1].dim[0];
			return 1;
		}
		const Shape *x = &a[0];
		if ((x->rank != 2 && x->rank != 3) || x->dim[x->rank - 1] != 2) {
			shape_str(x, s0, sizeof s0);
			snprintf(err, errn, "'%s' needs sparse rows (K, 2) or (B, K, 2), got %s", oi->name, s0);
			return 0;
		}
		int n = x->rank == 3 ? x->dim[0] * x->dim[1] : x->dim[0];
		if (op == OP_ACTIVE) {
			if (a[1].rank < 1) {
				snprintf(err, errn, "'active' needs the table (D, ...) the rows index");
				return 0;
			}
			out->rank = 1;
			out->dim[0] = n;
			return 1;
		}
		const Shape *g = &a[1], *r = &a[2];
		int gr = x->rank == 3 ? 2 : 1;
		if (g->rank != gr || (gr == 2 && g->dim[0] != x->dim[0]) || r->rank != 1 || r->dim[0] != n) {
			shape_str(g, s0, sizeof s0);
			shape_str(r, s1, sizeof s1);
			snprintf(err, errn, "'spmm_tc' needs a product gradient with one row per sparse row and rows (%d), got %s and %s", n, s0, s1);
			return 0;
		}
		out->rank = 2;
		out->dim[0] = n;
		out->dim[1] = g->dim[g->rank - 1];
		return 1;
	}
	case CLS_RESHAPE: /* the target shape is not a function of the operand: see ir_reshape */
		snprintf(err, errn, "'reshape' takes a tensor and its new dimensions, e.g. reshape(x, 4, 8)");
		return 0;
	case CLS_THINK:
	case CLS_SCAN:
		break;
	}
	snprintf(err, errn, "'%s' is not a value operation", oi->name);
	return 0;
}

/* The quantization axis a param needs as an operand of instruction in:
 * matmul left (W @ x) and spmm table -> 0 (per row), matmul right
 * (x @ W) -> 1 (per column); -1 if this use cannot read quantized codes. */
int q_axis_for(const Module *m, int param, const Ins *in)
{
	if (m->val[param].sh.rank != 2) return -1;
	if (in->op == OP_MATMUL) {
		if (in->a[0] == param && in->a[1] != param) return 0;
		if (in->a[1] == param && in->a[0] != param) return 1;
		return -1;
	}
	if (in->op == OP_SPMM && in->a[1] == param && in->a[0] != param) return 0;
	return -1;
}
