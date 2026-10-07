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
	[OP_THINK]     = { "think",     0, CLS_THINK },
};

int op_lookup(const char *name)
{
	for (int i = 0; i < OP_COUNT; i++)
		if (i != OP_CONST && i != OP_THINK && strcmp(tg_ops[i].name, name) == 0) return i;
	return -1;
}

void matmul_dims(const Shape *a, const Shape *b, int *m, int *k, int *n)
{
	*m = a->rank == 2 ? a->dim[0] : 1;
	*k = a->rank == 2 ? a->dim[1] : a->dim[0];
	*n = b->rank == 2 ? b->dim[1] : 1;
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
		if (shape_eq(&a[0], &a[1]) || a[1].rank == 0) { *out = a[0]; return 1; }
		if (a[0].rank == 0) { *out = a[1]; return 1; }
		shape_str(&a[0], s0, sizeof s0);
		shape_str(&a[1], s1, sizeof s1);
		snprintf(err, errn, "'%s' shape mismatch %s vs %s (only equal shapes or scalar broadcast)", oi->name, s0, s1);
		return 0;
	case CLS_UN:
		*out = a[0];
		return 1;
	case CLS_MATMUL: {
		int ra = a[0].rank, rb = a[1].rank;
		int ka = ra == 2 ? a[0].dim[1] : ra == 1 ? a[0].dim[0] : -1;
		int kb = rb >= 1 ? a[1].dim[0] : -1;
		if (ra < 1 || ra > 2 || rb < 1 || rb > 2 || ka != kb) {
			shape_str(&a[0], s0, sizeof s0);
			shape_str(&a[1], s1, sizeof s1);
			snprintf(err, errn, "'matmul' incompatible operands %s @ %s", s0, s1);
			return 0;
		}
		if (ra == 2) out->dim[out->rank++] = a[0].dim[0];
		if (rb == 2) out->dim[out->rank++] = a[1].dim[1];
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
		if (a[0].rank != 2) {
			snprintf(err, errn, "'transpose' needs rank 2");
			return 0;
		}
		out->rank = 2;
		out->dim[0] = a[0].dim[1];
		out->dim[1] = a[0].dim[0];
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
	case CLS_THINK:
		break;
	}
	snprintf(err, errn, "'%s' is not a value operation", oi->name);
	return 0;
}
