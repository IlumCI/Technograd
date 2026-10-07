/* Surface syntax: indentation-structured, Python-like. Produces an S-expression
 * AST consumed by lower.c.
 *
 *   program  := { NL | decl }
 *   decl     := 'model' NAME NL
 *             | 'param' NAME ':' type '=' init NL
 *             | 'def' NAME '(' [arg {',' arg}] ')' '->' type ':' block
 *   arg      := NAME ':' type
 *   type     := 'f32' [ '[' INT {',' INT} ']' ]
 *   init     := 'zeros' | 'ones' | 'fill' '(' num ')' | 'rand' '(' num ',' num ')'
 *             | 'file' '(' STR ')' | literal
 *   literal  := num | '[' literal {',' literal} ']'
 *   block    := NL INDENT stmt {stmt} DEDENT
 *   stmt     := NAME '=' expr NL | 'return' expr NL
 *             | 'think' NAME 'for' INT ['until' num] ':' block
 *   expr     := term {('+'|'-') term}
 *   term     := unary {('*'|'/'|'@') unary}
 *   unary    := '-' unary | primary
 *   primary  := NUM | NAME | NAME '(' [expr {',' expr}] ')' | '(' expr ')'
 *
 * AST forms:
 *   (model N) (param N TYPE INIT) (def N ((a TYPE)...) TYPE (stmts...))
 *   (set N E) (return E) (think N MAX EPS (stmts...))
 *   (num V) (ref N) (call F E...)
 *   TYPE = (f32 d...)   INIT = (zeros) (ones) (fill v) (rand s k) (file "p") (data v...) */
#include "tg.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef enum { T_NAME, T_NUM, T_STR, T_OP, T_NL, T_INDENT, T_DEDENT, T_EOF } TokKind;

typedef struct {
	TokKind k;
	char *s;
	double n;
	int line;
} Tok;

typedef struct {
	Tok *t;
	int n, cap, i;
	const char *file;
} P;

static void emit(P *p, TokKind k, const char *s, size_t len, double n, int line)
{
	if (p->n == p->cap) {
		p->cap = p->cap ? p->cap * 2 : 256;
		p->t = xrealloc(p->t, (size_t)p->cap * sizeof *p->t);
	}
	Tok *t = &p->t[p->n++];
	t->k = k;
	t->s = xmalloc(len + 1);
	memcpy(t->s, s, len);
	t->n = n;
	t->line = line;
}

static void lex(P *p, const char *src)
{
	int stack[64], sp = 0, line = 1, depth = 0;
	const char *c = src;
	stack[0] = 0;
	int bol = 1;
	while (*c) {
		if (bol && depth == 0) {
			int col = 0;
			while (*c == ' ') { col++; c++; }
			if (*c == '\t') die(p->file, line, "tabs are not allowed for indentation");
			if (*c == '\n' || *c == '#' || *c == '\r' || !*c) { /* blank or comment line */
				while (*c && *c != '\n') c++;
				if (*c) { c++; line++; }
				continue;
			}
			bol = 0;
			if (col > stack[sp]) {
				if (sp == 63) die(p->file, line, "nesting too deep");
				stack[++sp] = col;
				emit(p, T_INDENT, "", 0, 0, line);
			} else {
				while (col < stack[sp]) { sp--; emit(p, T_DEDENT, "", 0, 0, line); }
				if (col != stack[sp]) die(p->file, line, "inconsistent dedent");
			}
		}
		if (*c == '\n') {
			if (depth == 0) { emit(p, T_NL, "", 0, 0, line); bol = 1; }
			c++;
			line++;
			continue;
		}
		if (*c == ' ' || *c == '\r' || *c == '\t') { c++; continue; }
		if (*c == '#') { while (*c && *c != '\n') c++; continue; }
		if (isalpha((unsigned char)*c) || *c == '_') {
			const char *s = c;
			while (isalnum((unsigned char)*c) || *c == '_') c++;
			emit(p, T_NAME, s, (size_t)(c - s), 0, line);
			continue;
		}
		if (isdigit((unsigned char)*c) || (*c == '.' && isdigit((unsigned char)c[1]))) {
			char *end;
			double v = strtod(c, &end);
			emit(p, T_NUM, c, (size_t)(end - c), v, line);
			c = end;
			continue;
		}
		if (*c == '"') {
			const char *s = ++c;
			while (*c && *c != '"' && *c != '\n') c++;
			if (*c != '"') die(p->file, line, "unterminated string");
			emit(p, T_STR, s, (size_t)(c - s), 0, line);
			c++;
			continue;
		}
		if (c[0] == '-' && c[1] == '>') { emit(p, T_OP, c, 2, 0, line); c += 2; continue; }
		if (strchr("+-*/@()[],:=", *c)) {
			if (*c == '(' || *c == '[') depth++;
			if ((*c == ')' || *c == ']') && depth > 0) depth--;
			emit(p, T_OP, c, 1, 0, line);
			c++;
			continue;
		}
		die(p->file, line, "unexpected character '%c'", *c);
	}
	if (p->n && p->t[p->n - 1].k != T_NL) emit(p, T_NL, "", 0, 0, line);
	while (sp > 0) { sp--; emit(p, T_DEDENT, "", 0, 0, line); }
	emit(p, T_EOF, "", 0, 0, line);
}

/* ---- parser ------------------------------------------------------------- */

static Tok *peek(P *p) { return &p->t[p->i]; }
static Tok *next(P *p) { Tok *t = &p->t[p->i]; if (t->k != T_EOF) p->i++; return t; }

static const char *tokdesc(Tok *t)
{
	switch (t->k) {
	case T_NL: return "end of line";
	case T_INDENT: return "indent";
	case T_DEDENT: return "dedent";
	case T_EOF: return "end of file";
	default: return t->s;
	}
}

static int isop(Tok *t, const char *s) { return t->k == T_OP && strcmp(t->s, s) == 0; }
static int iskw(Tok *t, const char *s) { return t->k == T_NAME && strcmp(t->s, s) == 0; }

static Tok *expect_op(P *p, const char *s)
{
	Tok *t = next(p);
	if (!isop(t, s)) die(p->file, t->line, "expected '%s', got '%s'", s, tokdesc(t));
	return t;
}

static Tok *expect(P *p, TokKind k, const char *what)
{
	Tok *t = next(p);
	if (t->k != k) die(p->file, t->line, "expected %s, got '%s'", what, tokdesc(t));
	return t;
}

static const char *kw[] = { "model", "param", "def", "return", "think", "for", "until", "f32", "import", NULL };

static Tok *expect_name(P *p)
{
	Tok *t = expect(p, T_NAME, "name");
	for (int i = 0; kw[i]; i++)
		if (strcmp(t->s, kw[i]) == 0) die(p->file, t->line, "'%s' is a reserved word", t->s);
	return t;
}

static double signed_num(P *p)
{
	int neg = 0;
	if (isop(peek(p), "-")) { next(p); neg = 1; }
	Tok *t = expect(p, T_NUM, "number");
	return neg ? -t->n : t->n;
}

static int posint(P *p)
{
	Tok *t = expect(p, T_NUM, "integer");
	if (t->n < 1 || t->n != (double)(int)t->n || t->n > 1 << 24) die(p->file, t->line, "expected positive integer, got '%s'", t->s);
	return (int)t->n;
}

static Sx *type(P *p)
{
	Tok *t = next(p);
	if (!iskw(t, "f32")) die(p->file, t->line, "expected type 'f32', got '%s'", tokdesc(t));
	Sx *ty = sx_list(t->line, 1, sx_sym("f32", t->line));
	if (isop(peek(p), "[")) {
		next(p);
		for (;;) {
			if (ty->len > TG_MAXRANK) die(p->file, t->line, "rank exceeds %d", TG_MAXRANK);
			sx_push(ty, sx_num(posint(p), t->line));
			if (isop(peek(p), "]")) { next(p); break; }
			expect_op(p, ",");
		}
	}
	return ty;
}

static void literal(P *p, Sx *data)
{
	if (isop(peek(p), "[")) {
		next(p);
		for (;;) {
			literal(p, data);
			if (isop(peek(p), "]")) { next(p); return; }
			expect_op(p, ",");
		}
	}
	sx_push(data, sx_num(signed_num(p), peek(p)->line));
}

static Sx *init(P *p)
{
	Tok *t = peek(p);
	int line = t->line;
	if (iskw(t, "zeros") || iskw(t, "ones")) {
		next(p);
		return sx_list(line, 1, sx_sym(t->s, line));
	}
	if (iskw(t, "fill")) {
		next(p);
		expect_op(p, "(");
		Sx *v = sx_num(signed_num(p), line);
		expect_op(p, ")");
		return sx_list(line, 2, sx_sym("fill", line), v);
	}
	if (iskw(t, "rand")) {
		next(p);
		expect_op(p, "(");
		Sx *seed = sx_num(signed_num(p), line);
		expect_op(p, ",");
		Sx *scale = sx_num(signed_num(p), line);
		expect_op(p, ")");
		return sx_list(line, 3, sx_sym("rand", line), seed, scale);
	}
	if (iskw(t, "file")) {
		next(p);
		expect_op(p, "(");
		Tok *s = expect(p, T_STR, "string");
		expect_op(p, ")");
		return sx_list(line, 2, sx_sym("file", line), sx_str(s->s, line));
	}
	Sx *data = sx_list(line, 1, sx_sym("data", line));
	literal(p, data);
	return data;
}

static Sx *expr(P *p);

static Sx *primary(P *p)
{
	Tok *t = next(p);
	if (t->k == T_NUM) return sx_list(t->line, 2, sx_sym("num", t->line), sx_num(t->n, t->line));
	if (isop(t, "(")) {
		Sx *e = expr(p);
		expect_op(p, ")");
		return e;
	}
	if (t->k == T_NAME) {
		for (int i = 0; kw[i]; i++)
			if (strcmp(t->s, kw[i]) == 0) die(p->file, t->line, "unexpected keyword '%s' in expression", t->s);
		if (isop(peek(p), "(")) {
			next(p);
			Sx *c = sx_list(t->line, 2, sx_sym("call", t->line), sx_sym(t->s, t->line));
			if (isop(peek(p), ")")) { next(p); return c; }
			for (;;) {
				sx_push(c, expr(p));
				if (isop(peek(p), ")")) { next(p); return c; }
				expect_op(p, ",");
			}
		}
		return sx_list(t->line, 2, sx_sym("ref", t->line), sx_sym(t->s, t->line));
	}
	die(p->file, t->line, "expected expression, got '%s'", tokdesc(t));
	return NULL;
}

static Sx *call2(const char *op, Sx *a, Sx *b, int line)
{
	Sx *c = sx_list(line, 3, sx_sym("call", line), sx_sym(op, line), a);
	if (b) sx_push(c, b);
	return c;
}

static Sx *unary(P *p)
{
	if (isop(peek(p), "-")) {
		int line = next(p)->line;
		Tok *t = peek(p);
		if (t->k == T_NUM) { /* fold negative literals */
			next(p);
			return sx_list(line, 2, sx_sym("num", line), sx_num(-t->n, line));
		}
		return call2("neg", unary(p), NULL, line);
	}
	return primary(p);
}

static Sx *term(P *p)
{
	Sx *e = unary(p);
	for (;;) {
		Tok *t = peek(p);
		const char *op = isop(t, "*") ? "mul" : isop(t, "/") ? "div" : isop(t, "@") ? "matmul" : NULL;
		if (!op) return e;
		next(p);
		e = call2(op, e, unary(p), t->line);
	}
}

static Sx *expr(P *p)
{
	Sx *e = term(p);
	for (;;) {
		Tok *t = peek(p);
		const char *op = isop(t, "+") ? "add" : isop(t, "-") ? "sub" : NULL;
		if (!op) return e;
		next(p);
		e = call2(op, e, term(p), t->line);
	}
}

static Sx *block(P *p);

static Sx *stmt(P *p)
{
	Tok *t = peek(p);
	int line = t->line;
	if (iskw(t, "return")) {
		next(p);
		Sx *e = expr(p);
		expect(p, T_NL, "end of line");
		return sx_list(line, 2, sx_sym("return", line), e);
	}
	if (iskw(t, "think")) {
		next(p);
		Tok *n = expect_name(p);
		Tok *f = next(p);
		if (!iskw(f, "for")) die(p->file, f->line, "expected 'for' after think state, got '%s'", tokdesc(f));
		int max = posint(p);
		double eps = -1;
		if (iskw(peek(p), "until")) {
			next(p);
			eps = signed_num(p);
			if (eps < 0) die(p->file, line, "halting threshold must be >= 0");
		}
		expect_op(p, ":");
		Sx *b = block(p);
		return sx_list(line, 5, sx_sym("think", line), sx_sym(n->s, line), sx_num(max, line), sx_num(eps, line), b);
	}
	Tok *n = expect_name(p);
	expect_op(p, "=");
	Sx *e = expr(p);
	expect(p, T_NL, "end of line");
	return sx_list(line, 3, sx_sym("set", line), sx_sym(n->s, line), e);
}

static Sx *block(P *p)
{
	expect(p, T_NL, "end of line");
	Tok *t = expect(p, T_INDENT, "indented block");
	Sx *b = sx_new(SX_LIST, t->line);
	while (peek(p)->k != T_DEDENT && peek(p)->k != T_EOF) sx_push(b, stmt(p));
	expect(p, T_DEDENT, "dedent");
	return b;
}

static Sx *decl(P *p)
{
	Tok *t = next(p);
	int line = t->line;
	if (iskw(t, "import")) {
		Tok *s = expect(p, T_STR, "import path string");
		expect(p, T_NL, "end of line");
		return sx_list(line, 2, sx_sym("import", line), sx_str(s->s, line));
	}
	if (iskw(t, "model")) {
		Tok *n = expect_name(p);
		expect(p, T_NL, "end of line");
		return sx_list(line, 2, sx_sym("model", line), sx_sym(n->s, line));
	}
	if (iskw(t, "param")) {
		Tok *n = expect_name(p);
		expect_op(p, ":");
		Sx *ty = type(p);
		expect_op(p, "=");
		Sx *in = init(p);
		expect(p, T_NL, "end of line");
		return sx_list(line, 4, sx_sym("param", line), sx_sym(n->s, line), ty, in);
	}
	if (iskw(t, "def")) {
		Tok *n = expect_name(p);
		expect_op(p, "(");
		Sx *args = sx_new(SX_LIST, line);
		if (!isop(peek(p), ")")) {
			for (;;) {
				Tok *a = expect_name(p);
				expect_op(p, ":");
				sx_push(args, sx_list(a->line, 2, sx_sym(a->s, a->line), type(p)));
				if (isop(peek(p), ")")) break;
				expect_op(p, ",");
			}
		}
		expect_op(p, ")");
		expect_op(p, "->");
		Sx *rt = type(p);
		expect_op(p, ":");
		Sx *b = block(p);
		return sx_list(line, 5, sx_sym("def", line), sx_sym(n->s, line), args, rt, b);
	}
	die(p->file, line, "expected 'import', 'model', 'param' or 'def', got '%s'", tokdesc(t));
	return NULL;
}

Sx *surface_parse(const char *src, const char *file)
{
	P p = { 0 };
	p.file = file;
	lex(&p, src);
	Sx *top = sx_new(SX_LIST, 1);
	for (;;) {
		Tok *t = peek(&p);
		if (t->k == T_EOF) break;
		if (t->k == T_NL) { next(&p); continue; }
		if (t->k == T_INDENT) die(file, t->line, "unexpected indent");
		sx_push(top, decl(&p));
	}
	return top;
}
