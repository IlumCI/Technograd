/* Reference interpreter. Executes the planned IR over a single arena using the
 * same kernels (runtime/tg_rt.h) that generated C code links against. */
#include "tg.h"
#include "../runtime/tg_rt.h"

#include <stdlib.h>

typedef struct {
	const Module *m;
	float *arena;
	const float **in;
	int *steps;
} VM;

static float *ptr(VM *vm, int v)
{
	const Value *x = &vm->m->val[v];
	switch (x->kind) {
	case V_PARAM: return x->data;
	case V_INPUT: return (float *)vm->in[x->index];
	case V_TMP: break;
	}
	return vm->arena + x->off;
}

static void exec(VM *vm, const Block *b)
{
	const Module *m = vm->m;
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		const Shape *os = &m->val[in->out].sh;
		float *o = ptr(vm, in->out);
		int n = shape_numel(os);
		const float *a = in->na > 0 ? ptr(vm, in->a[0]) : NULL;
		const float *c = in->na > 1 ? ptr(vm, in->a[1]) : NULL;
		switch (tg_ops[in->op].cls) {
		case CLS_CONST:
			o[0] = in->k;
			break;
		case CLS_BIN: {
			int sa = shape_numel(&m->val[in->a[0]].sh) == n ? 1 : 0;
			int sb = shape_numel(&m->val[in->a[1]].sh) == n ? 1 : 0;
			switch (in->op) {
			case OP_ADD: tg_add(o, a, sa, c, sb, n); break;
			case OP_SUB: tg_sub(o, a, sa, c, sb, n); break;
			case OP_MUL: tg_mul(o, a, sa, c, sb, n); break;
			case OP_DIV: tg_div(o, a, sa, c, sb, n); break;
			case OP_MAX: tg_max(o, a, sa, c, sb, n); break;
			case OP_MIN: tg_min(o, a, sa, c, sb, n); break;
			default: abort();
			}
			break;
		}
		case CLS_UN:
			switch (in->op) {
			case OP_NEG: tg_neg(o, a, n); break;
			case OP_TANH: tg_tanh(o, a, n); break;
			case OP_RELU: tg_relu(o, a, n); break;
			case OP_SIGMOID: tg_sigmoid(o, a, n); break;
			case OP_EXP: tg_exp(o, a, n); break;
			case OP_SQRT: tg_sqrt(o, a, n); break;
			case OP_GELU: tg_gelu(o, a, n); break;
			case OP_SILU: tg_silu(o, a, n); break;
			default: abort();
			}
			break;
		case CLS_MATMUL: {
			int mm, kk, nn;
			matmul_dims(&m->val[in->a[0]].sh, &m->val[in->a[1]].sh, &mm, &kk, &nn);
			tg_matmul(o, a, c, mm, kk, nn);
			break;
		}
		case CLS_ROW: {
			int rows, cols;
			row_dims(&m->val[in->a[0]].sh, &rows, &cols);
			if (in->op == OP_SOFTMAX) tg_softmax(o, a, rows, cols);
			else tg_rmsnorm(o, a, rows, cols);
			break;
		}
		case CLS_RED: {
			int k = shape_numel(&m->val[in->a[0]].sh);
			if (in->op == OP_SUM) tg_sum(o, a, k);
			else tg_mean(o, a, k);
			break;
		}
		case CLS_TRANS: {
			const Shape *s = &m->val[in->a[0]].sh;
			tg_transpose(o, a, s->dim[0], s->dim[1]);
			break;
		}
		case CLS_THINK: {
			tg_copy(o, ptr(vm, in->init), n);
			const float *y = NULL;
			int it = 0;
			while (it < in->maxit) {
				exec(vm, in->body);
				it++;
				y = ptr(vm, in->yield);
				float d = tg_delta(y, o, n);
				tg_copy(o, y, n);
				if (in->eps >= 0 && d <= in->eps) break;
			}
			vm->steps[in->tid] = it;
			break;
		}
		}
	}
}

void vm_run(const Module *m, const float **in, float *out, int *steps)
{
	VM vm = { m, NULL, in, steps };
	vm.arena = xmalloc((size_t)(m->arena ? m->arena : 1) * sizeof(float));
	exec(&vm, &m->top);
	tg_copy(out, ptr(&vm, m->output), shape_numel(&m->val[m->output].sh));
	free(vm.arena);
}
