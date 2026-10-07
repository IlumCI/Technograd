/* S-expression reader. TGIR is plain S-expressions:
 *   atom   := symbol | number | "string"
 *   list   := '(' { atom | list } ')'
 *   ;      comments run to end of line */
#include "tg.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

Sx *sx_new(SxKind k, int line)
{
	Sx *x = xmalloc(sizeof *x);
	x->k = k;
	x->line = line;
	return x;
}

Sx *sx_sym(const char *s, int line)
{
	Sx *x = sx_new(SX_SYM, line);
	x->s = xstrdup(s);
	return x;
}

Sx *sx_num(double n, int line)
{
	Sx *x = sx_new(SX_NUM, line);
	x->n = n;
	return x;
}

Sx *sx_str(const char *s, int line)
{
	Sx *x = sx_new(SX_STR, line);
	x->s = xstrdup(s);
	return x;
}

void sx_push(Sx *l, Sx *e)
{
	if (l->len == l->cap) {
		l->cap = l->cap ? l->cap * 2 : 4;
		l->v = xrealloc(l->v, (size_t)l->cap * sizeof *l->v);
	}
	l->v[l->len++] = e;
}

Sx *sx_list(int line, int n, ...)
{
	Sx *l = sx_new(SX_LIST, line);
	va_list ap;
	va_start(ap, n);
	for (int i = 0; i < n; i++) sx_push(l, va_arg(ap, Sx *));
	va_end(ap);
	return l;
}

int sx_issym(const Sx *x, const char *s)
{
	return x && x->k == SX_SYM && strcmp(x->s, s) == 0;
}

typedef struct {
	const char *p;
	const char *file;
	int line;
} Reader;

static void skip(Reader *r)
{
	for (;;) {
		while (isspace((unsigned char)*r->p)) {
			if (*r->p == '\n') r->line++;
			r->p++;
		}
		if (*r->p != ';') return;
		while (*r->p && *r->p != '\n') r->p++;
	}
}

static Sx *read_form(Reader *r)
{
	skip(r);
	int line = r->line;
	char c = *r->p;
	if (!c) die(r->file, line, "unexpected end of input");
	if (c == ')') die(r->file, line, "unexpected ')'");
	if (c == '(') {
		r->p++;
		Sx *l = sx_new(SX_LIST, line);
		for (;;) {
			skip(r);
			if (!*r->p) die(r->file, line, "unterminated list");
			if (*r->p == ')') { r->p++; return l; }
			sx_push(l, read_form(r));
		}
	}
	if (c == '"') {
		const char *s = ++r->p;
		while (*r->p && *r->p != '"' && *r->p != '\n') r->p++;
		if (*r->p != '"') die(r->file, line, "unterminated string");
		size_t n = (size_t)(r->p - s);
		char *buf = xmalloc(n + 1);
		memcpy(buf, s, n);
		r->p++;
		Sx *x = sx_new(SX_STR, line);
		x->s = buf;
		return x;
	}
	const char *s = r->p;
	while (*r->p && !isspace((unsigned char)*r->p) && *r->p != '(' && *r->p != ')' && *r->p != ';' && *r->p != '"')
		r->p++;
	size_t n = (size_t)(r->p - s);
	char *buf = xmalloc(n + 1);
	memcpy(buf, s, n);
	char *end;
	double d = strtod(buf, &end);
	if (*end == 0 && (isdigit((unsigned char)buf[0]) || ((buf[0] == '-' || buf[0] == '+' || buf[0] == '.') && n > 1))) {
		xfree(buf);
		return sx_num(d, line);
	}
	Sx *x = sx_new(SX_SYM, line);
	x->s = buf;
	return x;
}

Sx *sx_read(const char *src, const char *file)
{
	Reader r = { src, file, 1 };
	Sx *top = sx_new(SX_LIST, 1);
	for (;;) {
		skip(&r);
		if (!*r.p) return top;
		sx_push(top, read_form(&r));
	}
}
