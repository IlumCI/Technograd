#include "tg.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

void *xmalloc(size_t n)
{
	void *p = calloc(1, n ? n : 1);
	if (!p) die(NULL, 0, "out of memory");
	return p;
}

void *xrealloc(void *p, size_t n)
{
	p = realloc(p, n ? n : 1);
	if (!p) die(NULL, 0, "out of memory");
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

void die(const char *file, int line, const char *fmt, ...)
{
	va_list ap;
	if (file) fprintf(stderr, "%s:%d: ", file, line);
	fputs("error: ", stderr);
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
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
	Module *m;
	if (*p == '(' || *p == ';') m = ir_read(sx_read(src, path), path);
	else m = lower(surface_parse(src, path), path);
	free(src);
	return m;
}
