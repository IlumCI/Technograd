/* Native backends: `tgc asm MODEL --target T [-o model.s]` and
 * `tgc asm MODEL --target T --support [-o support.c]`.
 *
 * The assembly implements tg_<model>_run from the planned IR: every value's
 * address is resolved at compile time (arena offsets, parameter and state
 * symbols, the input pointers kept in the frame), loops (think, scan,
 * learned halting) and the update commit are native code, and all parameter
 * data is emitted as .rodata. Each instruction is a call into a kernel. The
 * hot kernel, matmul, is hand-written for each target:
 *
 *   cortex-m3   ARMv7-M Thumb-2, no FPU: Q16.16 fixed point (as tgc c
 *               --fixed); matmul accumulates with SMLAL (32 x 32 -> 64).
 *   cortex-m4f  ARMv7E-M with FPv4-SP, hard-float ABI: matmul with VFMA.
 *   rv64gcv     RISC-V RV64GC + the V extension (vector-length agnostic):
 *               matmul with vfmacc.vf over output columns, or vfmacc.vv and
 *               vfredusum for matrix-vector products.
 *
 * The other kernels come from the support file (`--support`): the runtime
 * of runtime/tg_rt.h or tg_rtq.h compiled as extern functions, plus the
 * TG_MAIN test driver. The generated assembly never uses floating-point
 * registers itself; loop halting goes through tg_k_think. */
#include "tg.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef enum { T_M3, T_M4F, T_RV64V } Target;

typedef struct {
	const Module *m;
	FILE *f;
	Target t;
	int fixed;
	int W;      /* pointer size in the frame */
	int slots;  /* loop counter slots used */
	int nslots; /* loop counter slots needed */
	int label;
	int in_off; /* frame offset of the input pointer slots */
	int lp_off; /* frame offset of the loop slots */
	int frame;
	int *desc, ndesc; /* broadcast descriptors, TG_BCAST_DESC ints each, emitted as .rodata */
} A;

enum { ARG_VAL, ARG_SYM, ARG_IMM, ARG_NULL, ARG_VALT };

typedef struct {
	int kind;
	int v;          /* ARG_VAL / ARG_VALT */
	long imm;       /* ARG_IMM; ARG_VALT: bytes per step of t */
	char sym[96];   /* ARG_SYM */
	int tslot;      /* ARG_VALT: frame slot holding t */
} Arg;

static Arg V(int v) { Arg a = { ARG_VAL, v, 0, "", 0 }; return a; }
static Arg I(long x) { Arg a = { ARG_IMM, -1, x, "", 0 }; return a; }
static Arg NUL(void) { Arg a = { ARG_NULL, -1, 0, "", 0 }; return a; }
static Arg S(const char *fmt, const char *x, int k)
{
	Arg a = { ARG_SYM, -1, 0, "", 0 };
	snprintf(a.sym, sizeof a.sym, fmt, x, k);
	return a;
}
static Arg VT(int v, long bytes, int tslot) { Arg a = { ARG_VALT, v, bytes, "", tslot }; return a; }

static int is_arm(const A *a) { return a->t != T_RV64V; }

static void count_loops(const Block *b, int *n)
{
	for (int i = 0; i < b->len; i++)
		if (b->v[i].op == OP_THINK || b->v[i].op == OP_SCAN) {
			*n += 2;
			count_loops(b->v[i].body, n);
		}
}

/* ---- loading operands -------------------------------------------------------- */

static void sym_of(const A *a, int v, char *buf, size_t n, long *off)
{
	const Value *x = &a->m->val[v];
	*off = 0;
	switch (x->kind) {
	case V_TMP: snprintf(buf, n, "tg_%s_arena", a->m->name); *off = 4L * x->off; break;
	case V_PARAM: snprintf(buf, n, "tgp_%s", x->name); break;
	case V_STATE: snprintf(buf, n, "tg_%s_state_%s", a->m->name, x->name); break;
	case V_INPUT: buf[0] = 0; break;
	}
}

static void load_addr(A *a, const char *r, const char *sym, long off)
{
	if (is_arm(a)) {
		fprintf(a->f, "\tmovw %s, #:lower16:(%s+%ld)\n\tmovt %s, #:upper16:(%s+%ld)\n", r, sym, off, r, sym, off);
	} else {
		fprintf(a->f, "\tlla %s, %s+%ld\n", r, sym, off);
	}
}

static void load_imm(A *a, const char *r, long x)
{
	if (is_arm(a)) {
		uint32_t u = (uint32_t)x;
		fprintf(a->f, "\tmovw %s, #%u\n", r, u & 0xffffu);
		if (u >> 16) fprintf(a->f, "\tmovt %s, #%u\n", r, u >> 16);
	} else {
		fprintf(a->f, "\tli %s, %ld\n", r, x);
	}
}

static void load_slot(A *a, const char *r, int off)
{
	if (is_arm(a)) fprintf(a->f, "\tldr %s, [sp, #%d]\n", r, off);
	else fprintf(a->f, "\tld %s, %d(sp)\n", r, off);
}

static void store_slot(A *a, const char *r, int off)
{
	if (is_arm(a)) fprintf(a->f, "\tstr %s, [sp, #%d]\n", r, off);
	else fprintf(a->f, "\tsd %s, %d(sp)\n", r, off);
}

/* Put argument x in register r; s1, s2 are free scratch registers. */
static void load_arg(A *a, const Arg *x, const char *r, const char *s1, const char *s2)
{
	char sym[160];
	long off;
	switch (x->kind) {
	case ARG_IMM: load_imm(a, r, x->imm); return;
	case ARG_NULL: load_imm(a, r, 0); return;
	case ARG_SYM: load_addr(a, r, x->sym, 0); return;
	case ARG_VAL:
	case ARG_VALT:
		if (a->m->val[x->v].kind == V_INPUT) load_slot(a, r, a->in_off + a->W * a->m->val[x->v].index);
		else {
			sym_of(a, x->v, sym, sizeof sym, &off);
			load_addr(a, r, sym, off);
		}
		if (x->kind == ARG_VALT) { /* + t * bytes */
			load_slot(a, s1, x->tslot);
			load_imm(a, s2, x->imm);
			if (is_arm(a)) fprintf(a->f, "\tmla %s, %s, %s, %s\n", r, s1, s2, r);
			else fprintf(a->f, "\tmul %s, %s, %s\n\tadd %s, %s, %s\n", s1, s1, s2, r, r, s1);
		}
		return;
	}
}

static void call(A *a, const char *fn, const Arg *x, int n)
{
	static const char *armr[4] = { "r0", "r1", "r2", "r3" };
	static const char *rvr[8] = { "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7" };
	int nreg = is_arm(a) ? 4 : 8;
	for (int i = nreg; i < n; i++) { /* stack arguments first: scratch registers only */
		if (is_arm(a)) {
			load_arg(a, &x[i], "ip", "lr", "r0");
			fprintf(a->f, "\tstr ip, [sp, #%d]\n", 4 * (i - nreg));
		} else {
			load_arg(a, &x[i], "t0", "t1", "t2");
			fprintf(a->f, "\tsd t0, %d(sp)\n", 8 * (i - nreg));
		}
	}
	for (int i = 0; i < n && i < nreg; i++) {
		if (is_arm(a)) load_arg(a, &x[i], armr[i], "ip", "lr");
		else load_arg(a, &x[i], rvr[i], "t1", "t2");
	}
	fprintf(a->f, is_arm(a) ? "\tbl %s\n" : "\tcall %s\n", fn);
}

static int label(A *a) { return a->label++; }

/* Symbol of a new broadcast descriptor. */
static Arg desc(A *a, const Shape *out, const Shape *x, const Shape *y)
{
	a->desc = xrealloc(a->desc, (size_t)(a->ndesc + 1) * TG_BCAST_DESC * sizeof(int));
	int *d = a->desc + (size_t)a->ndesc * TG_BCAST_DESC;
	memset(d, 0, TG_BCAST_DESC * sizeof(int));
	bcast_desc(out, x, y, d);
	return S("tg_%s_d%d", a->m->name, a->ndesc++);
}

/* ---- instructions ------------------------------------------------------------- */

static int numel(const A *a, int v) { return shape_numel(&a->m->val[v].sh); }

static void block(A *a, const Block *b);

static void ins_think(A *a, const Ins *in)
{
	const Module *m = a->m;
	int n = numel(a, in->out), cs = a->lp_off + a->W * a->slots, l0 = label(a), l1 = label(a), tid = in->tid;
	a->slots += 2;
	Arg c[3] = { V(in->out), V(in->init), I(n) };
	call(a, "tg_copy", c, 3);
	const char *z = is_arm(a) ? "r0" : "t0";
	char svs[96]; /* survival probability, reset to 1 */
	snprintf(svs, sizeof svs, "tg_%s_sv%d", m->name, tid);
	if (in->halt >= 0) {
		if (is_arm(a)) {
			load_addr(a, "r1", svs, 0);
			load_imm(a, "r0", a->fixed ? 65536 : 0x3f800000);
			fprintf(a->f, "\tstr r0, [r1]\n");
		} else {
			fprintf(a->f, "\tlla t1, %s\n", svs);
			load_imm(a, "t0", 0x3f800000);
			fprintf(a->f, "\tsw t0, 0(t1)\n");
		}
	}
	load_imm(a, z, 0);
	store_slot(a, z, cs);
	fprintf(a->f, ".Ltg%d:\n", l0);
	block(a, in->body);
	load_slot(a, z, cs);
	fprintf(a->f, is_arm(a) ? "\tadds r0, r0, #1\n" : "\taddi t0, t0, 1\n");
	store_slot(a, z, cs);
	Arg k[7] = { V(in->out), V(in->yield), I(n), NUL(), NUL(), NUL(), NUL() };
	if (in->eps >= 0) k[3] = S("tg_%s_eps%d", m->name, tid);
	if (in->halt >= 0) {
		k[4] = V(in->halt);
		k[5] = S("tg_%s_sv%d", m->name, tid);
		k[6] = S("tg_%s_thr%d", m->name, tid);
	}
	call(a, "tg_k_think", k, 7);
	if (is_arm(a)) {
		fprintf(a->f, "\tcmp r0, #0\n\tbne.w .Ltg%d\n", l1);
		load_slot(a, "r0", cs);
		load_imm(a, "r1", in->maxit);
		fprintf(a->f, "\tcmp r0, r1\n\tblt.w .Ltg%d\n", l0);
	} else {
		fprintf(a->f, "\tbnez a0, .Ltg%d\n", l1);
		load_slot(a, "t0", cs);
		load_imm(a, "t1", in->maxit);
		fprintf(a->f, "\tblt t0, t1, .Ltg%d\n", l0);
	}
	fprintf(a->f, ".Ltg%d:\n", l1);
	char st[96];
	snprintf(st, sizeof st, "tg_%s_steps", m->name);
	if (is_arm(a)) {
		load_slot(a, "r0", cs);
		load_addr(a, "r1", st, 4L * tid);
		fprintf(a->f, "\tstr r0, [r1]\n");
	} else {
		load_slot(a, "t0", cs);
		fprintf(a->f, "\tlla t1, %s+%d\n\tsw t0, 0(t1)\n", st, 4 * tid);
	}
}

static void ins_scan(A *a, const Ins *in)
{
	const Module *m = a->m;
	const Scan *s = in->sc;
	int ss = a->lp_off + a->W * a->slots, ts = ss + a->W, l0 = label(a);
	a->slots += 2;
	for (int k = 0; k < s->nc; k++) {
		Arg c[3] = { V(s->c[k]), V(s->init[k]), I(numel(a, s->c[k])) };
		call(a, "tg_copy", c, 3);
	}
	const char *z = is_arm(a) ? "r0" : "t0", *y = is_arm(a) ? "r1" : "t1";
	load_imm(a, z, 0);
	store_slot(a, z, ss);
	fprintf(a->f, ".Ltg%d:\n", l0);
	load_slot(a, z, ss); /* t = s, or T - 1 - s */
	if (s->reverse) {
		load_imm(a, y, s->T - 1);
		fprintf(a->f, is_arm(a) ? "\tsubs r0, r1, r0\n" : "\tsub t0, t1, t0\n");
	}
	store_slot(a, z, ts);
	for (int j = 0; j < s->nx; j++) {
		int w = numel(a, s->xt[j]);
		Arg c[3] = { V(s->xt[j]), VT(s->x[j], 4L * w, ts), I(w) };
		call(a, "tg_copy", c, 3);
	}
	block(a, in->body);
	for (int k = 0; k < s->ny; k++) {
		int w = numel(a, s->y[k]);
		Arg c[3] = { VT(s->ys[k], 4L * w, ts), V(s->y[k]), I(w) };
		call(a, "tg_copy", c, 3);
	}
	for (int k = 0; k < s->nc; k++) {
		Arg c[3] = { V(s->c[k]), V(s->next[k]), I(numel(a, s->c[k])) };
		call(a, "tg_copy", c, 3);
	}
	load_slot(a, z, ss);
	if (is_arm(a)) {
		fprintf(a->f, "\tadds r0, r0, #1\n");
		store_slot(a, "r0", ss);
		load_imm(a, "r1", s->T);
		fprintf(a->f, "\tcmp r0, r1\n\tblt.w .Ltg%d\n", l0);
	} else {
		fprintf(a->f, "\taddi t0, t0, 1\n");
		store_slot(a, "t0", ss);
		load_imm(a, "t1", s->T);
		fprintf(a->f, "\tblt t0, t1, .Ltg%d\n", l0);
	}
	if (s->tid >= 0) {
		char st[96];
		snprintf(st, sizeof st, "tg_%s_steps", m->name);
		if (is_arm(a)) {
			load_imm(a, "r0", s->T);
			load_addr(a, "r1", st, 4L * s->tid);
			fprintf(a->f, "\tstr r0, [r1]\n");
		} else {
			load_imm(a, "t0", s->T);
			fprintf(a->f, "\tlla t1, %s+%d\n\tsw t0, 0(t1)\n", st, 4 * s->tid);
		}
	}
}

static void ins(A *a, const Ins *in)
{
	const Module *m = a->m;
	int n = numel(a, in->out);
	const Shape *s0 = in->na > 0 ? &m->val[in->a[0]].sh : NULL;
	fprintf(a->f, "\t%s %s\n", is_arm(a) ? "@" : "#", tg_ops[in->op].name);
	switch (tg_ops[in->op].cls) {
	case CLS_CONST: {
		long bits = 0;
		if (a->fixed) {
			double v = (double)in->k * 65536.0;
			bits = v >= 2147483647.0 ? 2147483647L : v <= -2147483648.0 ? -2147483647L - 1 : lround(v);
		} else {
			uint32_t u;
			memcpy(&u, &in->k, 4);
			bits = (long)u;
		}
		char sym[160];
		long off;
		sym_of(a, in->out, sym, sizeof sym, &off);
		if (is_arm(a)) {
			load_addr(a, "r1", sym, off);
			load_imm(a, "r0", bits);
			fprintf(a->f, "\tstr r0, [r1]\n");
		} else {
			fprintf(a->f, "\tlla t1, %s+%ld\n", sym, off);
			load_imm(a, "t0", (long)(int32_t)bits);
			fprintf(a->f, "\tsw t0, 0(t1)\n");
		}
		return;
	}
	case CLS_BIN: {
		int op = in->op - OP_ADD;
		if (!bin_simple(&m->val[in->out].sh, s0, &m->val[in->a[1]].sh)) {
			Arg c[5] = { I(op), V(in->out), V(in->a[0]), V(in->a[1]), desc(a, &m->val[in->out].sh, s0, &m->val[in->a[1]].sh) };
			call(a, "tg_bin_bc", c, 5);
			return;
		}
		Arg c[7] = { I(op), V(in->out), V(in->a[0]), I(numel(a, in->a[0])), V(in->a[1]), I(numel(a, in->a[1])), I(n) };
		call(a, "tg_k_bin", c, 7);
		return;
	}
	case CLS_UN:
	case CLS_RESHAPE: {
		char fn[64];
		snprintf(fn, sizeof fn, "tg_%s", in->op == OP_RESHAPE ? "copy" : tg_ops[in->op].name);
		Arg c[3] = { V(in->out), V(in->a[0]), I(n) };
		call(a, fn, c, 3);
		return;
	}
	case CLS_MATMUL: {
		int mm, kk, nn;
		matmul_dims(s0, &m->val[in->a[1]].sh, &mm, &kk, &nn);
		const Value *qa = &m->val[in->a[0]], *qb = &m->val[in->a[1]];
		int gg = matmul_groups(s0, &m->val[in->a[1]].sh);
		if (gg > 1) {
			Arg c[7] = { V(in->out), V(in->a[0]), V(in->a[1]), I(gg), I(mm), I(kk), I(nn) };
			call(a, "tg_bmm", c, 7);
		} else if (qa->qbits && q_axis_for(m, in->a[0], in) == qa->qaxis) {
			Arg c[8] = { V(in->out), S("tgq_%s", qa->name, 0), I(qa->qbits), S("tgqs_%s", qa->name, 0), V(in->a[1]), I(mm), I(kk), I(nn) };
			call(a, "tg_matmul_qa", c, 8);
		} else if (qb->qbits && q_axis_for(m, in->a[1], in) == qb->qaxis) {
			Arg c[8] = { V(in->out), V(in->a[0]), S("tgq_%s", qb->name, 0), I(qb->qbits), S("tgqs_%s", qb->name, 0), I(mm), I(kk), I(nn) };
			call(a, "tg_matmul_qb", c, 8);
		} else {
			Arg c[6] = { V(in->out), V(in->a[0]), V(in->a[1]), I(mm), I(kk), I(nn) };
			call(a, "tg_matmul", c, 6);
		}
		return;
	}
	case CLS_ROW: {
		int rows, cols;
		row_dims(s0, &rows, &cols);
		Arg c[4] = { V(in->out), V(in->a[0]), I(rows), I(cols) };
		call(a, in->op == OP_SOFTMAX ? "tg_softmax" : "tg_rmsnorm", c, 4);
		return;
	}
	case CLS_RED: {
		Arg c[3] = { V(in->out), V(in->a[0]), I(numel(a, in->a[0])) };
		call(a, in->op == OP_SUM ? "tg_sum" : "tg_mean", c, 3);
		return;
	}
	case CLS_TRANS: {
		if (s0->rank == 3) {
			Arg c[5] = { V(in->out), V(in->a[0]), I(s0->dim[0]), I(s0->dim[1]), I(s0->dim[2]) };
			call(a, "tg_btranspose", c, 5);
		} else {
			Arg c[4] = { V(in->out), V(in->a[0]), I(s0->dim[0]), I(s0->dim[1]) };
			call(a, "tg_transpose", c, 4);
		}
		return;
	}
	case CLS_OUTER: {
		Arg c[5] = { V(in->out), V(in->a[0]), V(in->a[1]), I(numel(a, in->a[0])), I(numel(a, in->a[1])) };
		call(a, "tg_outer", c, 5);
		return;
	}
	case CLS_SPMM: {
		int r, k, d, h;
		spmm_dims(s0, &m->val[in->a[in->op == OP_SPMM_T ? 2 : 1]].sh, &r, &k, &d, &h);
		const Value *qt = &m->val[in->a[1]];
		if (in->op == OP_SPMM_DX) {
			Arg c[8] = { V(in->out), V(in->a[0]), V(in->a[1]), V(in->a[2]), I(r), I(k), I(d), I(h) };
			call(a, "tg_spmm_dx", c, 8);
		} else if (in->op == OP_SPMM && qt->qbits && qt->qaxis == 0) {
			Arg c[9] = { V(in->out), V(in->a[0]), S("tgq_%s", qt->name, 0), I(qt->qbits), S("tgqs_%s", qt->name, 0), I(r), I(k), I(d), I(h) };
			call(a, "tg_spmm_q", c, 9);
		} else {
			Arg c[7] = { V(in->out), V(in->a[0]), V(in->a[1]), I(r), I(k), I(d), I(h) };
			call(a, in->op == OP_SPMM ? "tg_spmm" : "tg_spmm_t", c, 7);
		}
		return;
	}
	case CLS_ROWS: {
		if (in->op == OP_TAKE) {
			Arg c[6] = { V(in->out), V(in->a[0]), V(in->a[1]), I(m->val[in->a[1]].sh.dim[0]), I(s0->dim[0]), I(numel(a, in->a[0]) / s0->dim[0]) };
			call(a, "tg_take", c, 6);
		} else if (in->op == OP_TAKE_T) {
			const Shape *w = &m->val[in->a[2]].sh;
			Arg c[6] = { V(in->out), V(in->a[0]), V(in->a[1]), I(s0->dim[0]), I(w->dim[0]), I(shape_numel(w) / w->dim[0]) };
			call(a, "tg_take_t", c, 6);
		} else {
			int r = s0->rank == 3 ? s0->dim[0] : 1, k = s0->dim[s0->rank - 2];
			if (in->op == OP_ACTIVE) {
				Arg c[5] = { V(in->out), V(in->a[0]), I(r), I(k), I(m->val[in->a[1]].sh.dim[0]) };
				call(a, "tg_active", c, 5);
			} else {
				Arg c[8] = { V(in->out), V(in->a[0]), V(in->a[1]), V(in->a[2]), I(r), I(k), I(r * k), I(m->val[in->out].sh.dim[1]) };
				call(a, "tg_spmm_tc", c, 8);
			}
		}
		return;
	}
	case CLS_SUMTO: {
		Arg c[4] = { V(in->out), V(in->a[0]), I(n), desc(a, s0, &m->val[in->a[1]].sh, NULL) };
		call(a, "tg_sum_to", c, 4);
		return;
	}
	case CLS_THINK:
		ins_think(a, in);
		return;
	case CLS_SCAN:
		ins_scan(a, in);
		return;
	}
}

static void block(A *a, const Block *b)
{
	for (int i = 0; i < b->len; i++) ins(a, &b->v[i]);
}

/* ---- data ------------------------------------------------------------------ */

static long value_bits(const A *a, float v)
{
	if (a->fixed) {
		double s = (double)v * 65536.0;
		return s >= 2147483647.0 ? 2147483647L : s <= -2147483648.0 ? -2147483647L - 1 : lround(s);
	}
	uint32_t u;
	memcpy(&u, &v, 4);
	return (long)(int32_t)u;
}

static void words(A *a, const char *name, const float *v, int n, int global)
{
	if (global) fprintf(a->f, "\t.global %s\n", name);
	fprintf(a->f, "\t.align 2\n%s:\n", name);
	for (int i = 0; i < n; i++) fprintf(a->f, "%s%ld%s", i % 8 == 0 ? "\t.word " : "", value_bits(a, v[i]), i % 8 == 7 || i + 1 == n ? "\n" : ", ");
}

static void data(A *a)
{
	const Module *m = a->m;
	char nm[192];
	fputs("\n\t.section .rodata\n", a->f);
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_PARAM || x->dead) continue;
		int n = shape_numel(&x->sh);
		snprintf(nm, sizeof nm, "tgp_%s", x->name);
		if (cgen_needs_f32(m, v)) words(a, nm, x->data, n, 0);
		if (x->qbits) {
			int nb = x->qbits == 8 ? n : (n + 1) / 2, nch = x->sh.dim[x->qaxis];
			fprintf(a->f, "tgq_%s:\n", x->name);
			for (int i = 0; i < nb; i++) fprintf(a->f, "%s%d%s", i % 16 == 0 ? "\t.byte " : "", x->q[i], i % 16 == 15 || i + 1 == nb ? "\n" : ", ");
			fprintf(a->f, "\t.align 2\ntgqs_%s:\n", x->name);
			for (int i = 0; i < nch; i++) {
				long w;
				if (a->fixed) w = lround((double)x->qs[i] * 16777216.0); /* Q7.24 */
				else w = value_bits(a, x->qs[i]);
				fprintf(a->f, "\t.word %ld\n", w);
			}
		}
	}
	fputs("\n\t.data\n", a->f);
	for (int v = 0; v < m->nval; v++) {
		const Value *x = &m->val[v];
		if (x->kind != V_STATE) continue;
		snprintf(nm, sizeof nm, "tg_%s_state_%s", m->name, x->name);
		words(a, nm, x->data, shape_numel(&x->sh), 1);
	}
	fprintf(a->f, "\n\t.bss\n\t.align 3\ntg_%s_arena:\n\t.space %d\n\t.global tg_%s_steps\n\t.align 2\ntg_%s_steps:\n\t.space %d\n", m->name,
		4 * (m->arena ? m->arena : 1), m->name, m->name, 4 * (m->nthink ? m->nthink : 1));
}

static void loop_consts(A *a, const Block *b)
{
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		if (in->op == OP_THINK) {
			if (in->eps >= 0) {
				float e = in->eps;
				long bits = value_bits(a, e);
				if (a->fixed && bits == 0 && e > 0) bits = 1;
				fprintf(a->f, "\t.align 2\ntg_%s_eps%d:\n\t.word %ld\n", a->m->name, in->tid, bits);
			}
			if (in->halt >= 0)
				fprintf(a->f, "\t.align 2\ntg_%s_thr%d:\n\t.word %ld\n", a->m->name, in->tid, value_bits(a, in->hthr));
		}
		if (in->op == OP_THINK || in->op == OP_SCAN) loop_consts(a, in->body);
	}
}

static void loop_vars(A *a, const Block *b)
{
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		if (in->op == OP_THINK && in->halt >= 0) fprintf(a->f, "\t.align 2\ntg_%s_sv%d:\n\t.space 4\n", a->m->name, in->tid);
		if (in->op == OP_THINK || in->op == OP_SCAN) loop_vars(a, in->body);
	}
}

/* ---- hand-written kernels ---------------------------------------------------- */

static const char *k_m4f =
	"@ tg_matmul(o, a, b, m, k, n): o[m,n] = a[m,k] @ b[k,n], f32 with VFMA\n"
	"\t.text\n\t.align 2\n\t.global tg_matmul\n\t.type tg_matmul, %function\n\t.thumb_func\n"
	"tg_matmul:\n"
	"\tpush {r4-r11, lr}\n"
	"\tldr r4, [sp, #36]\n"   /* k */
	"\tldr r5, [sp, #40]\n"   /* n */
	"\tlsl r10, r5, #2\n"     /* row stride of b in bytes */
	"\tmovs r6, #0\n"         /* i */
	"1:\tcmp r6, r3\n\tbge 6f\n"
	"\tmovs r7, #0\n"         /* j */
	"2:\tcmp r7, r5\n\tbge 5f\n"
	"\tmul r8, r6, r4\n\tadd r8, r1, r8, lsl #2\n" /* &a[i*k] */
	"\tadd r9, r2, r7, lsl #2\n"                   /* &b[j] */
	"\tmovs r12, #0\n\tvmov s0, r12\n"
	"\tmovs r11, r4\n\tbeq 4f\n"
	"3:\tvldmia r8!, {s1}\n\tvldr s2, [r9]\n\tadd r9, r9, r10\n\tvfma.f32 s0, s1, s2\n\tsubs r11, r11, #1\n\tbne 3b\n"
	"4:\tmla r12, r6, r5, r7\n\tadd r12, r0, r12, lsl #2\n\tvstr s0, [r12]\n"
	"\tadds r7, r7, #1\n\tb 2b\n"
	"5:\tadds r6, r6, #1\n\tb 1b\n"
	"6:\tpop {r4-r11, pc}\n"
	"\t.size tg_matmul, .-tg_matmul\n";

static const char *k_m3 =
	"@ tg_matmul(o, a, b, m, k, n): Q16.16 with SMLAL (64-bit accumulation), rounded once, saturated\n"
	"\t.text\n\t.align 2\n\t.global tg_matmul\n\t.type tg_matmul, %function\n\t.thumb_func\n"
	"tg_matmul:\n"
	"\tpush {r4-r11, lr}\n"
	"\tsub sp, sp, #8\n"
	"\tstr r0, [sp, #0]\n"    /* o */
	"\tstr r3, [sp, #4]\n"    /* m */
	"\tldr r4, [sp, #44]\n"   /* k */
	"\tldr r5, [sp, #48]\n"   /* n */
	"\tlsl r10, r5, #2\n"
	"\tmovs r6, #0\n"
	"1:\tldr r3, [sp, #4]\n\tcmp r6, r3\n\tbge 7f\n"
	"\tmovs r7, #0\n"
	"2:\tcmp r7, r5\n\tbge 6f\n"
	"\tmul r8, r6, r4\n\tadd r8, r1, r8, lsl #2\n"
	"\tadd r9, r2, r7, lsl #2\n"
	"\tmovs r3, #0\n\tmovs r0, #0\n"            /* acc = r0:r3 (hi:lo) */
	"\tmovs r11, r4\n\tbeq 4f\n"
	"3:\tldr r12, [r8], #4\n\tldr lr, [r9]\n\tadd r9, r9, r10\n\tsmlal r3, r0, r12, lr\n\tsubs r11, r11, #1\n\tbne 3b\n"
	"4:\tadds r3, r3, #32768\n\tadc r0, r0, #0\n"  /* round */
	"\tasr r12, r0, #15\n\tadds r12, r12, #1\n\tcmp r12, #1\n\tbls 5f\n" /* hi:lo >> 16 fits in 32 bits? */
	"\tcmp r0, #0\n\tite lt\n\tmovlt r3, #0x80000000\n\tmvnge r3, #0x80000000\n\tb 8f\n"
	"5:\tlsr r3, r3, #16\n\torr r3, r3, r0, lsl #16\n"
	"8:\tmla r12, r6, r5, r7\n\tldr r0, [sp, #0]\n\tadd r12, r0, r12, lsl #2\n\tstr r3, [r12]\n"
	"\tadds r7, r7, #1\n\tb 2b\n"
	"6:\tadds r6, r6, #1\n\tb 1b\n"
	"7:\tadd sp, sp, #8\n\tpop {r4-r11, pc}\n"
	"\t.size tg_matmul, .-tg_matmul\n";

static const char *k_rvv =
	"# tg_matmul(o, a, b, m, k, n): f32 with the V extension, vector-length agnostic.\n"
	"# n == 1: each row is a dot product, vfmacc.vv over k then vfredusum.\n"
	"# n > 1: vfmacc.vf over a strip of output columns, for every p.\n"
	"\t.text\n\t.align 2\n\t.global tg_matmul\n\t.type tg_matmul, @function\n"
	"tg_matmul:\n"
	"\tli t6, 1\n\tbne a5, t6, 5f\n"
	"\tli t0, 0\n"                                    /* i */
	"1:\tbge t0, a3, 9f\n"
	"\tmul t1, t0, a4\n\tslli t1, t1, 2\n\tadd t1, a1, t1\n" /* &a[i*k] */
	"\tmv t2, a2\n\tmv t3, a4\n"
	"\tvsetvli t4, zero, e32, m4, ta, ma\n\tvmv.v.i v8, 0\n"
	"2:\tbeqz t3, 3f\n"
	"\tvsetvli t4, t3, e32, m4, tu, ma\n"
	"\tvle32.v v0, (t1)\n\tvle32.v v4, (t2)\n\tvfmacc.vv v8, v0, v4\n"
	"\tslli t5, t4, 2\n\tadd t1, t1, t5\n\tadd t2, t2, t5\n\tsub t3, t3, t4\n\tj 2b\n"
	"3:\tvsetvli t4, zero, e32, m4, ta, ma\n\tvmv.s.x v12, zero\n\tvfredusum.vs v12, v8, v12\n\tvfmv.f.s ft0, v12\n"
	"\tslli t5, t0, 2\n\tadd t5, a0, t5\n\tfsw ft0, 0(t5)\n"
	"\taddi t0, t0, 1\n\tj 1b\n"
	"5:\tli t0, 0\n"                                  /* i */
	"6:\tbge t0, a3, 9f\n"
	"\tli t1, 0\n"                                    /* j */
	"7:\tbge t1, a5, 8f\n"
	"\tsub t2, a5, t1\n\tvsetvli t2, t2, e32, m4, ta, ma\n\tvmv.v.i v8, 0\n"
	"\tmul t3, t0, a4\n\tslli t3, t3, 2\n\tadd t3, a1, t3\n" /* &a[i*k] */
	"\tslli t4, t1, 2\n\tadd t4, a2, t4\n"                    /* &b[j] */
	"\tslli t6, a5, 2\n\tmv t5, a4\n"
	"10:\tbeqz t5, 11f\n\tflw ft0, 0(t3)\n\tvle32.v v4, (t4)\n\tvfmacc.vf v8, ft0, v4\n"
	"\taddi t3, t3, 4\n\tadd t4, t4, t6\n\taddi t5, t5, -1\n\tj 10b\n"
	"11:\tmul t3, t0, a5\n\tadd t3, t3, t1\n\tslli t3, t3, 2\n\tadd t3, a0, t3\n\tvse32.v v8, (t3)\n"
	"\tadd t1, t1, t2\n\tj 7b\n"
	"8:\taddi t0, t0, 1\n\tj 6b\n"
	"9:\tret\n"
	"\t.size tg_matmul, .-tg_matmul\n";

/* ---- entry ------------------------------------------------------------------ */

static Target parse_target(const char *s)
{
	if (!strcmp(s, "cortex-m3")) return T_M3;
	if (!strcmp(s, "cortex-m4f")) return T_M4F;
	if (!strcmp(s, "rv64gcv")) return T_RV64V;
	die(NULL, 0, "--target is cortex-m3 (fixed point, no FPU), cortex-m4f (FPv4-SP) or rv64gcv (RISC-V vector)");
	return T_M3;
}

int asm_target_fixed(const char *target) { return parse_target(target) == T_M3; }

void asmgen(const Module *m, FILE *f, const char *target)
{
	A a = { m, f, parse_target(target), 0, 4, 0, 0, 0, 0, 0, 0, NULL, 0 };
	a.fixed = a.t == T_M3;
	a.W = is_arm(&a) ? 4 : 8;
	count_loops(&m->top, &a.nslots);
	int nargs = m->ninputs + 1, outstk = is_arm(&a) ? 32 : 64;
	a.in_off = outstk;
	a.lp_off = a.in_off + a.W * nargs;
	a.frame = (a.lp_off + a.W * (a.nslots ? a.nslots : 1) + 15) / 16 * 16;
	const char *c = is_arm(&a) ? "@" : "#";
	fprintf(f, "%s Technograd native unit: model '%s', target %s%s. Do not edit.\n", c, m->name, target,
		a.fixed ? " (Q16.16 fixed point)" : "");
	fprintf(f, "%s Entry: void tg_%s_run(%s...inputs, %s *out); kernels other than tg_matmul\n%s come from `tgc asm %s --target %s --support`.\n",
		c, m->name, a.fixed ? "const tg_t *" : "const float *", a.fixed ? "tg_t" : "float", c, "MODEL", target);
	if (is_arm(&a)) fprintf(f, "\t.syntax unified\n\t.thumb\n%s", a.t == T_M4F ? "\t.fpu fpv4-sp-d16\n" : "");
	else fputs("\t.option arch, +v\n", f);
	fprintf(f, "\t.text\n\t.align 2\n\t.global tg_%s_run\n\t.type tg_%s_run, %s\n%s", m->name, m->name, is_arm(&a) ? "%function" : "@function",
		is_arm(&a) ? "\t.thumb_func\n" : "");
	fprintf(f, "tg_%s_run:\n", m->name);
	if (is_arm(&a)) {
		fprintf(f, "\tpush {r4, lr}\n\tsub sp, sp, #%d\n", a.frame);
		for (int i = 0; i < nargs; i++) {
			if (i < 4) fprintf(f, "\tstr r%d, [sp, #%d]\n", i, a.in_off + 4 * i);
			else fprintf(f, "\tldr ip, [sp, #%d]\n\tstr ip, [sp, #%d]\n", a.frame + 8 + 4 * (i - 4), a.in_off + 4 * i);
		}
	} else {
		fprintf(f, "\taddi sp, sp, -%d\n\tsd ra, %d(sp)\n", a.frame + 16, a.frame);
		for (int i = 0; i < nargs; i++) {
			if (i < 8) fprintf(f, "\tsd a%d, %d(sp)\n", i, a.in_off + 8 * i);
			else fprintf(f, "\tld t0, %d(sp)\n\tsd t0, %d(sp)\n", a.frame + 16 + 8 * (i - 8), a.in_off + 8 * i);
		}
	}
	block(&a, &m->top);
	/* output, then the two-phase update commit */
	{
		Arg src = V(m->output), cnt = I(shape_numel(&m->val[m->output].sh));
		if (is_arm(&a)) {
			load_slot(&a, "r0", a.in_off + a.W * m->ninputs);
			load_arg(&a, &src, "r1", "ip", "lr");
			load_arg(&a, &cnt, "r2", "ip", "lr");
			fputs("\tbl tg_copy\n", f);
		} else {
			load_slot(&a, "a0", a.in_off + a.W * m->ninputs);
			load_arg(&a, &src, "a1", "t1", "t2");
			load_arg(&a, &cnt, "a2", "t1", "t2");
			fputs("\tcall tg_copy\n", f);
		}
	}
	char stage[96];
	snprintf(stage, sizeof stage, "tg_%s_arena", m->name);
	int off = m->stage;
	for (int i = 0; i < m->nupd; i++) {
		int nsrc = numel(&a, m->upd_src[i]);
		Arg c3[3] = { S("%s+%d", stage, 4 * off), V(m->upd_src[i]), I(nsrc) };
		call(&a, "tg_copy", c3, 3);
		off += (nsrc + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
		if (m->upd_rows[i] >= 0) {
			int nr = numel(&a, m->upd_rows[i]);
			Arg r3[3] = { S("%s+%d", stage, 4 * off), V(m->upd_rows[i]), I(nr) };
			call(&a, "tg_copy", r3, 3);
			off += (nr + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
		}
	}
	off = m->stage;
	for (int i = 0; i < m->nupd; i++) {
		const Shape *st = &m->val[m->upd_state[i]].sh;
		int nsrc = numel(&a, m->upd_src[i]);
		Arg dst = V(m->upd_state[i]), src = S("%s+%d", stage, 4 * off);
		if (m->upd_rows[i] < 0) {
			Arg c3[3] = { dst, src, I(nsrc) };
			call(&a, "tg_copy", c3, 3);
			off += (nsrc + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
		} else {
			int nr = numel(&a, m->upd_rows[i]), ro = off + (nsrc + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
			Arg rows = S("%s+%d", stage, 4 * ro);
			Arg c6[6] = { dst, src, rows, I(nr), I(st->dim[0]), I(shape_numel(st) / st->dim[0]) };
			call(&a, "tg_put_rows", c6, 6);
			off = ro + (nr + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
		}
	}
	if (is_arm(&a)) fprintf(f, "\tadd sp, sp, #%d\n\tpop {r4, pc}\n", a.frame);
	else fprintf(f, "\tld ra, %d(sp)\n\taddi sp, sp, %d\n\tret\n", a.frame, a.frame + 16);
	fprintf(f, "\t.size tg_%s_run, .-tg_%s_run\n\n", m->name, m->name);
	fputs(a.t == T_M3 ? k_m3 : a.t == T_M4F ? k_m4f : k_rvv, f);
	data(&a);
	fputs("\n\t.section .rodata\n", f);
	loop_consts(&a, &m->top);
	for (int i = 0; i < a.ndesc; i++) {
		const int *d = a.desc + (size_t)i * TG_BCAST_DESC;
		fprintf(f, "\t.align 2\ntg_%s_d%d:\n\t.word ", m->name, i);
		for (int j = 0; j < 1 + 3 * d[0]; j++) fprintf(f, "%s%d", j ? ", " : "", d[j]);
		fputc('\n', f);
	}
	xfree(a.desc);
	fputs("\n\t.bss\n", f);
	loop_vars(&a, &m->top);
}
