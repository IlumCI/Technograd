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
	int dbg; /* per-instruction and per-iteration debug events */
} VM;

/* Final halting delta of each think loop in the last vm_run (single run at a time). */
static float *g_delta;
static int g_ndelta;

float vm_last_delta(int tid) { return tid < g_ndelta ? g_delta[tid] : 0.0f; }

static void trace_ins(VM *vm, const Ins *in, const float *o, int n)
{
	const Module *m = vm->m;
	char on[64], a0[64] = "", a1[64] = "", sh[64];
	value_name(m, in->out, on, sizeof on);
	if (in->na > 0) value_name(m, in->a[0], a0, sizeof a0);
	if (in->na > 1) value_name(m, in->a[1], a1, sizeof a1);
	if (in->op == OP_CONST) snprintf(a0, sizeof a0, "%.6g", (double)in->k);
	shape_str(&m->val[in->out].sh, sh, sizeof sh);
	double mn = o[0], mx = o[0], sum = 0;
	int bad = 0;
	for (int i = 0; i < n; i++) {
		double x = o[i];
		if (x != x || x > 3.4e38 || x < -3.4e38) { bad++; continue; }
		if (x < mn) mn = x;
		if (x > mx) mx = x;
		sum += x;
	}
	tr_begin(2, "vm");
	tr_str("value", on);
	tr_str("op", tg_ops[in->op].name);
	tr_str("shape", sh);
	tr_num("min", mn);
	tr_num("max", mx);
	tr_num("mean", n > bad ? sum / (n - bad) : 0);
	tr_num("nonfinite", bad);
	tr_end("%s = %s(%s%s%s) %s  min %.4g max %.4g mean %.4g%s", on, tg_ops[in->op].name, a0, in->na > 1 ? ", " : "", a1, sh,
	       mn, mx, n > bad ? sum / (n - bad) : 0.0, bad ? "  NONFINITE" : "");
}

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
			case OP_SOFTPLUS: tg_softplus(o, a, n); break;
			case OP_LOG: tg_log(o, a, n); break;
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
				if (vm->dbg) {
					tr_begin(2, "think");
					tr_num("loop", in->tid);
					tr_num("iteration", it);
					tr_num("delta", d);
					tr_end("think[%d] iteration %d delta %.6g", in->tid, it, (double)d);
				}
				if (in->tid < g_ndelta) g_delta[in->tid] = d;
				tg_copy(o, y, n);
				if (in->eps >= 0 && d <= in->eps) break;
			}
			vm->steps[in->tid] = it;
			break;
		}
		}
		if (vm->dbg && in->op != OP_THINK) trace_ins(vm, in, o, n);
	}
}

void vm_run(const Module *m, const float **in, float *out, int *steps)
{
	VM vm = { m, NULL, in, steps, m->trace && tr_on(2) && tg_verbose >= 2 };
	if (g_ndelta < m->nthink) {
		g_delta = alloc_keep(xrealloc(g_delta, (size_t)m->nthink * sizeof *g_delta)); /* outlives scoped releases */
		g_ndelta = m->nthink;
	}
	vm.arena = xmalloc((size_t)(m->arena ? m->arena : 1) * sizeof(float));
	exec(&vm, &m->top);
	tg_copy(out, ptr(&vm, m->output), shape_numel(&m->val[m->output].sh));
	xfree(vm.arena);
}
