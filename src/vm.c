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
	int dbg;     /* per-instruction and per-iteration debug events */
	float *delta; /* final halting delta per think loop, or NULL */
} VM;

/* Row-split matmul for large products. Each output element is computed by the
 * same sequence of operations as the serial kernel, so results are
 * bit-identical at any thread count. */
#define PAR_MATMUL_MIN (1L << 16) /* multiply-adds below which threading costs more than it saves */

typedef struct {
	float *o;
	const float *a, *b;
	int k, n;
} MMJob;

static void mm_rows(void *ctx, int lo, int hi, int tid)
{
	(void)tid;
	MMJob *j = ctx;
	tg_matmul(j->o + (size_t)lo * (size_t)j->n, j->a + (size_t)lo * (size_t)j->k, j->b, hi - lo, j->k, j->n);
}

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
	case V_STATE: return x->data;
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
		const float *e = in->na > 2 ? ptr(vm, in->a[2]) : NULL;
		switch (tg_ops[in->op].cls) {
		case CLS_CONST:
			o[0] = in->k;
			break;
		case CLS_BIN: {
			void (*k)(float *, const float *, int, const float *, int, int) =
				in->op == OP_ADD ? tg_add : in->op == OP_SUB ? tg_sub : in->op == OP_MUL ? tg_mul
				: in->op == OP_DIV ? tg_div : in->op == OP_MAX ? tg_max : tg_min;
			int na = shape_numel(&m->val[in->a[0]].sh), nb = shape_numel(&m->val[in->a[1]].sh);
			if ((na == n || na == 1) && (nb == n || nb == 1)) {
				k(o, a, na == n, c, nb == n, n);
			} else { /* row broadcast: the smaller operand repeats over the rows */
				int w = na < n ? na : nb;
				for (int r = 0; r < n / w; r++)
					k(o + r * w, na < n ? a : a + r * w, 1, nb < n ? c : c + r * w, 1, w);
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
			case OP_STEP: tg_step(o, a, n); break;
			default: abort();
			}
			break;
		case CLS_MATMUL: {
			int mm, kk, nn;
			matmul_dims(&m->val[in->a[0]].sh, &m->val[in->a[1]].sh, &mm, &kk, &nn);
			if (par_threads() > 1 && mm >= 2 && (long)mm * kk * nn >= PAR_MATMUL_MIN) {
				MMJob j = { o, a, c, kk, nn };
				int grain = mm / (4 * par_threads());
				par_for(mm, grain > 0 ? grain : 1, mm_rows, &j);
			} else {
				tg_matmul(o, a, c, mm, kk, nn);
			}
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
		case CLS_OUTER:
			tg_outer(o, a, c, shape_numel(&m->val[in->a[0]].sh), shape_numel(&m->val[in->a[1]].sh));
			break;
		case CLS_SPMM: {
			int r, k, d, h;
			spmm_dims(&m->val[in->a[0]].sh, &m->val[in->a[in->op == OP_SPMM_T ? 2 : 1]].sh, &r, &k, &d, &h);
			if (in->op == OP_SPMM) tg_spmm(o, a, c, r, k, d, h);
			else if (in->op == OP_SPMM_T) tg_spmm_t(o, a, c, r, k, d, h);
			else tg_spmm_dx(o, a, c, e, r, k, d, h);
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
				if (vm->delta) vm->delta[in->tid] = d;
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

/* Apply `update`s after the run: copy every source into the staging area,
 * then into its state. Two phases, so a source that reads another state sees
 * the start-of-run value whatever the update order. */
static void commit(VM *vm, int trace)
{
	const Module *m = vm->m;
	int off = m->stage;
	for (int i = 0; i < m->nupd; i++) {
		int n = shape_numel(&m->val[m->upd_state[i]].sh);
		tg_copy(vm->arena + off, ptr(vm, m->upd_src[i]), n);
		off += (n + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
	}
	off = m->stage;
	for (int i = 0; i < m->nupd; i++) {
		const Value *s = &m->val[m->upd_state[i]];
		int n = shape_numel(&s->sh);
		if (trace) {
			float d = tg_delta(vm->arena + off, s->data, n);
			tr_begin(1, "update");
			tr_str("state", s->name);
			tr_num("max_change", d);
			tr_end("state %s updated, max |change| %.4g", s->name, (double)d);
		}
		tg_copy(s->data, vm->arena + off, n);
		off += (n + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
	}
}

/* Thread-safe entry: no allocation, no tracing, no shared state. `arena` must
 * hold m->arena floats and be private to the caller; `delta` may be NULL. */
void vm_run_into(const Module *m, const float **in, float *out, int *steps, float *arena, float *delta)
{
	VM vm = { m, arena, in, steps, 0, delta };
	exec(&vm, &m->top);
	tg_copy(out, ptr(&vm, m->output), shape_numel(&m->val[m->output].sh));
	commit(&vm, 0);
}

void vm_run(const Module *m, const float **in, float *out, int *steps)
{
	if (g_ndelta < m->nthink) {
		g_delta = alloc_keep(xrealloc(g_delta, (size_t)m->nthink * sizeof *g_delta)); /* outlives scoped releases */
		g_ndelta = m->nthink;
	}
	VM vm = { m, NULL, in, steps, m->trace && tr_on(2) && tg_verbose >= 2, g_delta };
	vm.arena = xmalloc((size_t)(m->arena ? m->arena : 1) * sizeof(float));
	exec(&vm, &m->top);
	tg_copy(out, ptr(&vm, m->output), shape_numel(&m->val[m->output].sh));
	commit(&vm, m->trace && tr_on(1));
	xfree(vm.arena);
}
