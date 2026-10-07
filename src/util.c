#include "tg.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Every allocation carries a header linking it into one list in allocation
 * order. alloc_release(mark) frees everything allocated since alloc_mark(),
 * which lets the auto-fixer run thousands of trial compilations without
 * leaking their ASTs and modules. */
typedef union Hdr {
	struct {
		union Hdr *prev, *next;
		size_t seq;
		int linked;
	} h;
	max_align_t align;
} Hdr;

static Hdr alloc_head = { { &alloc_head, &alloc_head, 0, 1 } };
static size_t alloc_seq;

static void link_tail(Hdr *b)
{
	b->h.prev = alloc_head.h.prev;
	b->h.next = &alloc_head;
	alloc_head.h.prev->h.next = b;
	alloc_head.h.prev = b;
	b->h.linked = 1;
}

static void unlink_hdr(Hdr *b)
{
	if (!b->h.linked) return;
	b->h.prev->h.next = b->h.next;
	b->h.next->h.prev = b->h.prev;
	b->h.linked = 0;
}

void *xmalloc(size_t n)
{
	Hdr *b = calloc(1, sizeof(Hdr) + n);
	if (!b) die(NULL, 0, "out of memory");
	b->h.seq = ++alloc_seq;
	link_tail(b);
	return b + 1;
}

void *xrealloc(void *p, size_t n)
{
	if (!p) return xmalloc(n);
	Hdr *b = (Hdr *)p - 1;
	int linked = b->h.linked;
	unlink_hdr(b);
	Hdr *nb = realloc(b, sizeof(Hdr) + n);
	if (!nb) die(NULL, 0, "out of memory");
	if (linked) {
		/* re-insert in seq order so release-from-tail stays correct */
		Hdr *at = alloc_head.h.prev;
		while (at != &alloc_head && at->h.seq > nb->h.seq) at = at->h.prev;
		nb->h.prev = at;
		nb->h.next = at->h.next;
		at->h.next->h.prev = nb;
		at->h.next = nb;
		nb->h.linked = 1;
	}
	return nb + 1;
}

void xfree(void *p)
{
	if (!p) return;
	Hdr *b = (Hdr *)p - 1;
	unlink_hdr(b);
	free(b);
}

size_t alloc_mark(void) { return alloc_seq + 1; }

void alloc_release(size_t mark)
{
	while (alloc_head.h.prev != &alloc_head && alloc_head.h.prev->h.seq >= mark) {
		Hdr *b = alloc_head.h.prev;
		unlink_hdr(b);
		free(b);
	}
}

void *alloc_keep(void *p)
{
	if (p) unlink_hdr((Hdr *)p - 1);
	return p;
}

char *xstrdup(const char *s)
{
	size_t n = strlen(s) + 1;
	return memcpy(xmalloc(n), s, n);
}

char *read_file(const char *path, size_t *len)
{
	FILE *f = fopen(path, "rb");
	if (!f) die(NULL, 0, "cannot open '%s'", path);
	size_t cap = 4096, n = 0;
	char *buf = xmalloc(cap);
	for (;;) {
		if (n + 1 >= cap) buf = xrealloc(buf, cap *= 2);
		size_t r = fread(buf + n, 1, cap - n - 1, f);
		if (r == 0) break;
		n += r;
	}
	fclose(f);
	buf[n] = 0;
	if (len) *len = n;
	return buf;
}

jmp_buf *tg_trap;
Diag tg_diag;

void die(const char *file, int line, const char *fmt, ...)
{
	va_list ap;
	snprintf(tg_diag.file, sizeof tg_diag.file, "%s", file ? file : "");
	tg_diag.line = file ? line : 0;
	va_start(ap, fmt);
	vsnprintf(tg_diag.msg, sizeof tg_diag.msg, fmt, ap);
	va_end(ap);
	if (tg_trap) longjmp(*tg_trap, 1);
	tr_error(file, file ? line : 0, tg_diag.msg);
	if (file) fprintf(stderr, "%s:%d: ", file, line);
	fprintf(stderr, "error: %s\n", tg_diag.msg);
	exit(1);
}

int shape_numel(const Shape *s)
{
	int n = 1;
	for (int i = 0; i < s->rank; i++) n *= s->dim[i];
	return n;
}

/* small is a strict suffix of big: (H) of (B,H), (C,H) of (A,C,H). Row broadcasting. */
int shape_suffix(const Shape *small, const Shape *big)
{
	if (small->rank < 1 || small->rank >= big->rank) return 0;
	for (int i = 1; i <= small->rank; i++)
		if (small->dim[small->rank - i] != big->dim[big->rank - i]) return 0;
	return 1;
}

int shape_eq(const Shape *a, const Shape *b)
{
	if (a->rank != b->rank) return 0;
	for (int i = 0; i < a->rank; i++)
		if (a->dim[i] != b->dim[i]) return 0;
	return 1;
}

void shape_str(const Shape *s, char *buf, size_t n)
{
	size_t o = (size_t)snprintf(buf, n, "(f32");
	for (int i = 0; i < s->rank && o < n; i++)
		o += (size_t)snprintf(buf + o, n - o, " %d", s->dim[i]);
	if (o < n) snprintf(buf + o, n - o, ")");
}

/* ---- modules ------------------------------------------------------------ */

Module *mod_new(const char *name)
{
	Module *m = xmalloc(sizeof *m);
	m->name = xstrdup(name);
	m->output = -1;
	return m;
}

int mod_value(Module *m, VKind k, const Shape *sh, const char *name)
{
	if (m->nval == m->capval) {
		m->capval = m->capval ? m->capval * 2 : 64;
		m->val = xrealloc(m->val, (size_t)m->capval * sizeof *m->val);
	}
	Value *v = &m->val[m->nval];
	memset(v, 0, sizeof *v);
	v->kind = k;
	v->sh = *sh;
	v->name = name ? xstrdup(name) : NULL;
	v->def = v->last = v->off = -1;
	if (k == V_INPUT) {
		v->index = m->ninputs;
		m->inputs = xrealloc(m->inputs, (size_t)(m->ninputs + 1) * sizeof *m->inputs);
		m->inputs[m->ninputs++] = m->nval;
	}
	return m->nval++;
}

void mod_note_def(Module *m, int v, Block *b, int idx)
{
	if (v >= m->ndef) {
		int n = m->nval > v + 1 ? m->nval : v + 1;
		m->def_blk = xrealloc(m->def_blk, (size_t)n * sizeof *m->def_blk);
		m->def_idx = xrealloc(m->def_idx, (size_t)n * sizeof *m->def_idx);
		for (int i = m->ndef; i < n; i++) m->def_blk[i] = NULL;
		m->ndef = n;
	}
	m->def_blk[v] = b;
	m->def_idx[v] = idx;
}

void mod_update(Module *m, int state, int src) { mod_update_rows(m, state, src, -1); }

void mod_update_rows(Module *m, int state, int src, int rows)
{
	m->upd_rows = xrealloc(m->upd_rows, (size_t)(m->nupd + 1) * sizeof *m->upd_rows);
	m->upd_rows[m->nupd] = rows;
	m->upd_state = xrealloc(m->upd_state, (size_t)(m->nupd + 1) * sizeof *m->upd_state);
	m->upd_src = xrealloc(m->upd_src, (size_t)(m->nupd + 1) * sizeof *m->upd_src);
	m->upd_state[m->nupd] = state;
	m->upd_src[m->nupd] = src;
	m->nupd++;
}

Ins *block_push(Block *b)
{
	if (b->len == b->cap) {
		b->cap = b->cap ? b->cap * 2 : 16;
		b->v = xrealloc(b->v, (size_t)b->cap * sizeof *b->v);
	}
	Ins *i = &b->v[b->len++];
	memset(i, 0, sizeof *i);
	i->out = i->init = i->yield = -1;
	for (int k = 0; k < TG_MAXARGS; k++) i->a[k] = -1;
	return i;
}

void value_name(const Module *m, int v, char *buf, size_t n)
{
	if (m->val[v].name) snprintf(buf, n, "%s", m->val[v].name);
	else snprintf(buf, n, "%%%d", v);
}

int block_count(const Block *b)
{
	int n = 0;
	for (int i = 0; i < b->len; i++) n += 1 + (b->v[i].op == OP_THINK ? block_count(b->v[i].body) : 0);
	return n;
}

static void trace_module(const Module *m, const char *path, double ms)
{
	int params = 0, pvals = 0;
	for (int v = 0; v < m->nval; v++)
		if (m->val[v].kind == V_PARAM) { pvals++; params += shape_numel(&m->val[v].sh); }
	int nins = block_count(&m->top);
	tr_begin(1, "lower");
	tr_str("file", path);
	tr_str("model", m->name);
	tr_num("inputs", m->ninputs);
	tr_num("params", pvals);
	tr_num("param_floats", params);
	tr_num("values", m->nval);
	tr_num("instructions", nins);
	tr_num("think_loops", m->nthink);
	tr_num("ms", ms);
	tr_end("model %s: %d input(s), %d param(s) (%d floats), %d instruction(s), %d think loop(s) in %.2f ms",
	       m->name, m->ninputs, pvals, params, nins, m->nthink, ms);
}

Module *load_module(const char *path)
{
	char *src = read_file(path, NULL);
	const char *p = src;
	while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
	int is_ir = (*p == '(' || *p == ';');
	Module *m;
	double t0 = tr_now_ms();
	if (is_ir) {
		m = ir_read(sx_read(src, path), path); /* TGIR is already flat: no imports */
		xfree(src);
		if (tr_on(1)) {
			tr_begin(1, "parse");
			tr_str("file", path);
			tr_str("format", "tgir");
			tr_end("read TGIR %s", path);
			trace_module(m, path, tr_now_ms() - t0);
		}
		return m;
	}
	xfree(src);
	Sx *ast = surface_load(path);
	double t1 = tr_now_ms();
	if (tr_on(1)) {
		tr_begin(1, "parse");
		tr_str("file", path);
		tr_str("format", "surface");
		tr_num("decls", ast->len);
		tr_num("ms", t1 - t0);
		tr_end("parsed %s and its imports: %d declaration(s) in %.2f ms", path, ast->len, t1 - t0);
	}
	m = lower(ast, path);
	if (tr_on(1)) trace_module(m, path, tr_now_ms() - t1);
	return m;
}

/* A full update needs the state's shape. A row update `state[rows] = src`
 * needs a rank >= 1 state (D, ...), rows (n) and src (n, ...). */
int upd_check(const Module *m, int state, int src, int rows, char *err, size_t n)
{
	const Shape *st = &m->val[state].sh, *sr = &m->val[src].sh;
	char s0[64], s1[64];
	shape_str(st, s0, sizeof s0);
	shape_str(sr, s1, sizeof s1);
	if (rows < 0) {
		if (shape_eq(st, sr)) return 1;
		snprintf(err, n, "update of state '%s' %s with a value of shape %s", m->val[state].name, s0, s1);
		return 0;
	}
	const Shape *ri = &m->val[rows].sh;
	int ok = st->rank >= 1 && ri->rank == 1 && sr->rank == st->rank && sr->dim[0] == ri->dim[0];
	for (int i = 1; ok && i < st->rank; i++) ok = sr->dim[i] == st->dim[i];
	if (ok) return 1;
	char s2[64];
	shape_str(ri, s2, sizeof s2);
	snprintf(err, n, "row update of state '%s' %s needs rows (n) and a value (n, ...) matching its trailing dimensions; got rows %s and value %s",
		 m->val[state].name, s0, s2, s1);
	return 0;
}
