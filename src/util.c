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

Ins *block_push(Block *b)
{
	if (b->len == b->cap) {
		b->cap = b->cap ? b->cap * 2 : 16;
		b->v = xrealloc(b->v, (size_t)b->cap * sizeof *b->v);
	}
	Ins *i = &b->v[b->len++];
	memset(i, 0, sizeof *i);
	i->out = i->init = i->yield = -1;
	i->a[0] = i->a[1] = -1;
	return i;
}

void value_name(const Module *m, int v, char *buf, size_t n)
{
	if (m->val[v].name) snprintf(buf, n, "%s", m->val[v].name);
	else snprintf(buf, n, "%%%d", v);
}

Module *load_module(const char *path)
{
	char *src = read_file(path, NULL);
	const char *p = src;
	while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
	int is_ir = (*p == '(' || *p == ';');
	Module *m;
	if (is_ir) {
		m = ir_read(sx_read(src, path), path); /* TGIR is already flat: no imports */
	} else {
		xfree(src);
		m = lower(surface_load(path), path);
		return m;
	}
	xfree(src);
	return m;
}
