/* Auto-fixer: diagnostic-driven program repair ranked by a neural decision
 * forest.
 *
 * Loop (per compiler error):
 *   1. Classify the diagnostic.
 *   2. Generate candidate edits with repair operators chosen by that class
 *      (rename, re-indent, operand swap/transpose, token insert/delete,
 *      make-return, line delete, bad-character fix).
 *   3. Trial-compile every candidate under a trap. The outcome (compiles, the
 *      error moved, the error class changed) becomes part of a 35-dimensional
 *      feature vector together with the error class, the operator, the edit
 *      locality and size, name similarity and usage, natural indentation, and
 *      two set-level
 *      features: how many candidates compile (ambiguity) and the similarity
 *      margin over the other compiling candidates.
 *   4. Score each vector with a forest of soft, differentiable decision trees
 *      (Kontschieder et al., ICCV 2015). The forest is itself a Technograd
 *      program (fixer/forest.tg), compiled by tgc and run on its VM.
 *   5. Apply the best candidate if its score is >= 0.95 and it makes progress,
 *      then repeat.
 *
 * Training is self-supervised, in the style of DrRepair (arXiv:2005.10636)
 * and Break-It-Fix-It (arXiv:2106.06600): valid programs are corrupted by
 * breakers, and a candidate is labelled positive iff it compiles to IR
 * identical to the original program's. No labelled data is needed. */
#include "tg.h"
#include "forest_embed.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NF 35        /* features */
#define NT 16        /* trees */
#define ND 3         /* tree depth */
#define NN 7         /* internal nodes = 2^ND - 1 */
#define NL 8         /* leaves = 2^ND */
#define NSUB 25      /* features visible to each tree (random subspace) */
#define MAX_ITERS 8
#define THRESHOLD 0.95f

static float threshold(void)
{
	const char *e = getenv("TG_FIXER_THRESHOLD");
	return e ? (float)atof(e) : THRESHOLD;
}

enum { E_UNDEF, E_UNKFN, E_SHAPE, E_INDENT, E_EXPECT, E_RETURN, E_LEX, E_OTHER, E_N };
enum { R_RENAME, R_INDENT, R_SWAP, R_TRANSPOSE, R_INSERT, R_DELETE, R_RETURN, R_DELLINE, R_CHAR, R_N };

static const char *rname[R_N] = {
	"rename", "re-indent", "swap-operands", "transpose-operand", "insert-token",
	"delete-token", "make-return", "delete-line", "fix-character",
};

static const char *keywords[] = { "model", "param", "def", "return", "think", "for", "until", "f32", "import", "state", "update",
				  "zeros", "ones", "fill", "rand", "file", NULL };

/* ---- small helpers ------------------------------------------------------ */

static uint32_t rng_s = 0x2545f491u;
static uint32_t rnd(void)
{
	uint32_t x = rng_s;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return rng_s = x;
}
static int rndn(int n) { return n > 0 ? (int)(rnd() % (uint32_t)n) : 0; }
static double rndu(void) { return (rnd() >> 8) / 16777216.0; }

static float clip1(double v) { return (float)(v < -1 ? -1 : v > 1 ? 1 : v); }

static int is_keyword(const char *s, size_t n)
{
	for (int i = 0; keywords[i]; i++)
		if (strlen(keywords[i]) == n && strncmp(keywords[i], s, n) == 0) return 1;
	return 0;
}

/* Optimal string alignment distance: Levenshtein plus adjacent transposition. */
static int lev(const char *a, const char *b)
{
	int la = (int)strlen(a), lb = (int)strlen(b);
	if (la > 255 || lb > 255) return la > lb ? la : lb;
	int pp[256], prev[256], cur[256];
	for (int j = 0; j <= lb; j++) prev[j] = pp[j] = j;
	for (int i = 1; i <= la; i++) {
		cur[0] = i;
		for (int j = 1; j <= lb; j++) {
			int c = prev[j - 1] + (a[i - 1] != b[j - 1]);
			if (prev[j] + 1 < c) c = prev[j] + 1;
			if (cur[j - 1] + 1 < c) c = cur[j - 1] + 1;
			if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] && pp[j - 2] + 1 < c) c = pp[j - 2] + 1;
			cur[j] = c;
		}
		memcpy(pp, prev, (size_t)(lb + 1) * sizeof *pp);
		memcpy(prev, cur, (size_t)(lb + 1) * sizeof *prev);
	}
	return prev[lb];
}

static float similarity(const char *a, const char *b)
{
	int la = (int)strlen(a), lb = (int)strlen(b), m = la > lb ? la : lb;
	return m ? 1.0f - (float)lev(a, b) / (float)m : 1.0f;
}

/* Text of the k-th '...' quoted segment of msg, or 0. */
static int quoted(const char *msg, int k, char *buf, size_t n)
{
	const char *p = msg;
	for (;;) {
		const char *a = strchr(p, '\'');
		if (!a) return 0;
		const char *b = strchr(a + 1, '\'');
		if (!b) return 0;
		if (k-- == 0) {
			size_t len = (size_t)(b - a - 1);
			if (len >= n) return 0;
			memcpy(buf, a + 1, len);
			buf[len] = 0;
			return 1;
		}
		p = b + 1;
	}
}

/* Quoted text following a given prefix, e.g. "expected '" or "got '". */
static int after(const char *msg, const char *prefix, char *buf, size_t n)
{
	const char *a = strstr(msg, prefix);
	if (!a) return 0;
	a += strlen(prefix);
	const char *b = strchr(a, '\'');
	if (!b || (size_t)(b - a) >= n) return 0;
	memcpy(buf, a, (size_t)(b - a));
	buf[b - a] = 0;
	return 1;
}

/* ---- source lines ------------------------------------------------------- */

typedef struct {
	char **l;
	int n;
} Lines;

static Lines split(const char *src)
{
	Lines L = { 0 };
	int cap = 16;
	L.l = xmalloc((size_t)cap * sizeof *L.l);
	const char *s = src;
	for (;;) {
		const char *e = strchr(s, '\n');
		size_t len = e ? (size_t)(e - s) : strlen(s);
		if (L.n == cap) L.l = xrealloc(L.l, (size_t)(cap *= 2) * sizeof *L.l);
		L.l[L.n] = xmalloc(len + 1);
		memcpy(L.l[L.n++], s, len);
		if (!e) return L;
		s = e + 1;
	}
}

static void lines_free(Lines *L)
{
	for (int i = 0; i < L->n; i++) xfree(L->l[i]);
	xfree(L->l);
}

/* Join with one line replaced (repl != NULL), deleted (del), or inserted before (ins). */
static char *join_edit(const Lines *L, int at, const char *repl, int del, const char *ins)
{
	size_t n = 1;
	for (int i = 0; i < L->n; i++) n += strlen(L->l[i]) + 1;
	n += (repl ? strlen(repl) : 0) + (ins ? strlen(ins) + 1 : 0);
	char *out = xmalloc(n), *p = out;
	int first = 1;
	for (int i = 0; i <= L->n; i++) {
		const char *parts[2] = { NULL, NULL };
		if (i == at && ins) parts[0] = ins;
		if (i < L->n) parts[1] = (i == at && repl) ? repl : (i == at && del) ? NULL : L->l[i];
		for (int k = 0; k < 2; k++) {
			if (!parts[k]) continue;
			if (!first) *p++ = '\n';
			first = 0;
			size_t len = strlen(parts[k]);
			memcpy(p, parts[k], len);
			p += len;
		}
	}
	*p = 0;
	return out;
}

static int indent_of(const char *s)
{
	int n = 0;
	while (s[n] == ' ') n++;
	return n;
}

static int blank(const char *s)
{
	while (*s == ' ' || *s == '\t' || *s == '\r') s++;
	return *s == 0 || *s == '#';
}

/* End of code on a line: before a comment, trailing spaces trimmed. */
static int code_end(const char *s)
{
	int n = 0, in_str = 0, end = 0;
	for (; s[n]; n++) {
		if (s[n] == '"') in_str = !in_str;
		if (s[n] == '#' && !in_str) break;
		if (s[n] != ' ' && s[n] != '\t' && s[n] != '\r') end = n + 1;
	}
	return end;
}

static char *with_indent(const char *s, int ind)
{
	const char *b = s;
	while (*b == ' ' || *b == '\t') b++;
	char *o = xmalloc((size_t)ind + strlen(b) + 1);
	memset(o, ' ', (size_t)ind);
	strcpy(o + ind, b);
	return o;
}

static char *splice(const char *s, int at, int dellen, const char *ins)
{
	size_t n = strlen(s), k = strlen(ins);
	char *o = xmalloc(n - (size_t)dellen + k + 1);
	memcpy(o, s, (size_t)at);
	memcpy(o + at, ins, k);
	strcpy(o + at + k, s + at + dellen);
	return o;
}

typedef struct {
	int col, len;
} Span;

/* Identifier spans on a line, skipping strings, numbers and comments. */
static int idents(const char *s, Span *out, int max)
{
	int n = 0;
	for (int i = 0; s[i] && s[i] != '#';) {
		if (s[i] == '"') {
			i++;
			while (s[i] && s[i] != '"') i++;
			if (s[i]) i++;
		} else if (isdigit((unsigned char)s[i]) || (s[i] == '.' && isdigit((unsigned char)s[i + 1]))) {
			while (isalnum((unsigned char)s[i]) || s[i] == '.' ||
			       ((s[i] == '+' || s[i] == '-') && (s[i - 1] == 'e' || s[i - 1] == 'E')))
				i++;
		} else if (isalpha((unsigned char)s[i]) || s[i] == '_') {
			int st = i;
			while (isalnum((unsigned char)s[i]) || s[i] == '_') i++;
			if (n < max) out[n++] = (Span){ st, i - st };
		} else {
			i++;
		}
	}
	return n;
}

/* Find a token (identifier: whole word; otherwise literal) in code part of s. */
static int find_tok(const char *s, const char *tok)
{
	int end = code_end(s), k = (int)strlen(tok);
	int word = isalpha((unsigned char)tok[0]) || tok[0] == '_';
	for (int i = 0; i + k <= end; i++) {
		if (strncmp(s + i, tok, (size_t)k) != 0) continue;
		if (word && ((i > 0 && (isalnum((unsigned char)s[i - 1]) || s[i - 1] == '_')) ||
			     isalnum((unsigned char)s[i + k]) || s[i + k] == '_'))
			continue;
		return i;
	}
	return -1;
}

static char *replace_word_all(const char *s, const char *from, const char *to)
{
	char *cur = xstrdup(s);
	Span sp[128];
	for (;;) {
		int n = idents(cur, sp, 128), hit = -1;
		for (int i = 0; i < n; i++)
			if ((size_t)sp[i].len == strlen(from) && strncmp(cur + sp[i].col, from, (size_t)sp[i].len) == 0) { hit = i; break; }
		if (hit < 0) return cur;
		char *nx = splice(cur, sp[hit].col, sp[hit].len, to);
		xfree(cur);
		cur = nx;
	}
}

/* ---- trial compilation -------------------------------------------------- */

typedef struct {
	int ok;
	Diag d;
	Module *m;
} Trial;

static Trial g_trial;

static void trial(const char *src, const char *path)
{
	jmp_buf jb;
	jmp_buf *prev = tg_trap;
	g_trial.ok = 0;
	g_trial.m = NULL;
	tg_trap = &jb;
	if (setjmp(jb) == 0) {
		g_trial.m = lower(surface_parse(src, path), path);
		g_trial.ok = 1;
	} else {
		g_trial.d = tg_diag;
	}
	tg_trap = prev;
}

/* IR text with param and input names replaced by positional names, so that
 * repairs which only rename a declaration still compare equal. */
static char *ir_canon(const Module *cm)
{
	Module *m = (Module *)cm;
	char **saved = xmalloc((size_t)m->nval * sizeof *saved), buf[32];
	for (int v = 0; v < m->nval; v++) {
		saved[v] = m->val[v].name;
		if (m->val[v].kind != V_TMP) {
			snprintf(buf, sizeof buf, "@%d", v);
			m->val[v].name = xstrdup(buf);
		}
	}
	FILE *f = tmpfile();
	if (!f) die(NULL, 0, "tmpfile failed");
	ir_write(m, f);
	long n = ftell(f);
	rewind(f);
	char *s = xmalloc((size_t)n + 1);
	if (fread(s, 1, (size_t)n, f) != (size_t)n) die(NULL, 0, "tmpfile read failed");
	s[n] = 0;
	fclose(f);
	for (int v = 0; v < m->nval; v++) {
		if (m->val[v].kind != V_TMP) xfree(m->val[v].name);
		m->val[v].name = saved[v];
	}
	xfree(saved);
	return s;
}

static int classify(const char *msg)
{
	if (strstr(msg, "undefined name") || strstr(msg, "undefined value")) return E_UNDEF;
	if (strstr(msg, "unknown function") || strstr(msg, "unknown op")) return E_UNKFN;
	if (strstr(msg, "shape") || strstr(msg, "incompatible") || strstr(msg, "declared") ||
	    strstr(msg, "argument") || strstr(msg, "needs rank"))
		return E_SHAPE;
	if (strstr(msg, "indent") || strstr(msg, "dedent") || strstr(msg, "tabs")) return E_INDENT;
	if (strstr(msg, "unexpected character") || strstr(msg, "unterminated")) return E_LEX;
	if (strstr(msg, "expected")) return E_EXPECT;
	if (strstr(msg, "return") || strstr(msg, "unreachable")) return E_RETURN;
	return E_OTHER;
}

/* ---- candidates --------------------------------------------------------- */

typedef struct {
	char *src;
	int op, line, rank;
	float sim, size;
	float f_unused, f_badonce, f_builtin, f_single, f_natural;
	char *old, *new; /* changed line for display; NULL = inserted / deleted */
	float x[NF];
	float score;
	int label;
} Cand;

typedef struct {
	Cand *v;
	int n, cap;
	int per_op[R_N];
} Cands;

static void cand_add(Cands *C, const Lines *L, int at, const char *repl, int del, const char *ins, int op, float sim)
{
	if (at < 0 || at > L->n || (at == L->n && !ins)) return;
	if (repl && strcmp(repl, L->l[at]) == 0) return;
	char *src = join_edit(L, at, repl, del, ins);
	for (int i = 0; i < C->n; i++)
		if (strcmp(C->v[i].src, src) == 0) { xfree(src); return; }
	if (C->n == C->cap) {
		C->cap = C->cap ? C->cap * 2 : 16;
		C->v = xrealloc(C->v, (size_t)C->cap * sizeof *C->v);
	}
	Cand *c = &C->v[C->n++];
	memset(c, 0, sizeof *c);
	c->src = src;
	c->op = op;
	c->line = at;
	c->rank = C->per_op[op]++;
	c->sim = sim;
	const char *o = at < L->n && !ins ? L->l[at] : "";
	const char *nw = repl ? repl : ins ? ins : "";
	c->size = (float)lev(o, nw);
	c->old = (!ins && at < L->n) ? xstrdup(L->l[at]) : NULL;
	c->new = repl ? xstrdup(repl) : ins ? xstrdup(ins) : NULL;
}

static void cands_free(Cands *C)
{
	for (int i = 0; i < C->n; i++) {
		xfree(C->v[i].src);
		xfree(C->v[i].old);
		xfree(C->v[i].new);
	}
	xfree(C->v);
}

typedef struct {
	char *s;
	float sim;
	int count, builtin;
} Named;

static int by_sim(const void *a, const void *b)
{
	float d = ((const Named *)b)->sim - ((const Named *)a)->sim;
	return d > 0 ? 1 : d < 0 ? -1 : strcmp(((const Named *)a)->s, ((const Named *)b)->s);
}

/* Whole-word occurrences of w in the program. */
static int word_count(const Lines *L, const char *w)
{
	Span sp[128];
	int c = 0;
	size_t k = strlen(w);
	for (int i = 0; i < L->n; i++) {
		int n = idents(L->l[i], sp, 128);
		for (int j = 0; j < n; j++) c += (size_t)sp[j].len == k && !strncmp(L->l[i] + sp[j].col, w, k);
	}
	return c;
}

static void add_name(Named **names, int *nn, int *cap, const char *s, size_t len, const char *bad, int builtin)
{
	if (is_keyword(s, len) || (strlen(bad) == len && !strncmp(s, bad, len))) return;
	for (int q = 0; q < *nn; q++)
		if (strlen((*names)[q].s) == len && !strncmp((*names)[q].s, s, len)) return;
	if (*nn == *cap) *names = xrealloc(*names, (size_t)(*cap = *cap ? *cap * 2 : 32) * sizeof **names);
	Named *n = &(*names)[(*nn)++];
	n->s = xmalloc(len + 1);
	memcpy(n->s, s, len);
	n->sim = similarity(bad, n->s);
	n->builtin = builtin || op_lookup(n->s) >= 0 || !strcmp(n->s, "dot");
}

/* Replace `bad` on (or near) the error line by known names. Candidates: the
 * most similar names, names that are defined but never used (typo at the
 * definition), and, for one-letter names, other one-letter names. */
static void gen_rename(Cands *C, const Lines *L, int el, const char *bad, int want_fn)
{
	int line = -1;
	for (int d = 0; d < 3 && line < 0; d++) {
		if (el - d >= 0 && el - d < L->n && find_tok(L->l[el - d], bad) >= 0) line = el - d;
		else if (el + d < L->n && find_tok(L->l[el + d], bad) >= 0) line = el + d;
	}
	if (line < 0) return;

	Named *names = NULL;
	int nn = 0, cap = 0;
	Span sp[128];
	for (int i = 0; i < L->n; i++) {
		int n = idents(L->l[i], sp, 128);
		for (int j = 0; j < n; j++) add_name(&names, &nn, &cap, L->l[i] + sp[j].col, (size_t)sp[j].len, bad, 0);
	}
	for (int op = 0; op < OP_COUNT; op++)
		if (op != OP_CONST && op != OP_THINK) add_name(&names, &nn, &cap, tg_ops[op].name, strlen(tg_ops[op].name), bad, 1);
	add_name(&names, &nn, &cap, "dot", 3, bad, 1);
	for (int i = 0; i < nn; i++) {
		names[i].count = names[i].builtin ? 0 : word_count(L, names[i].s);
		if (want_fn != names[i].builtin) names[i].sim *= 0.5f;
	}
	qsort(names, (size_t)nn, sizeof *names, by_sim);

	int badonce = word_count(L, bad) == 1, single = strlen(bad) == 1;
	int extra_unused = 0, extra_single = 0;
	for (int i = 0; i < nn; i++) {
		int unused = names[i].count == 1, one = single && strlen(names[i].s) == 1;
		int take = i < 6 || (unused && extra_unused < 4) || (one && extra_single < 4);
		if (!take) continue;
		if (i >= 6) { extra_unused += unused; extra_single += one; }
		int before = C->n;
		char *r = replace_word_all(L->l[line], bad, names[i].s);
		cand_add(C, L, line, r, 0, NULL, R_RENAME, names[i].sim);
		xfree(r);
		if (C->n > before) {
			Cand *c = &C->v[C->n - 1];
			c->f_unused = unused;
			c->f_badonce = badonce;
			c->f_builtin = names[i].builtin;
			c->f_single = one;
		}
	}
	for (int i = 0; i < nn; i++) xfree(names[i].s);
	xfree(names);
}

/* "missing entry point": rename a def header to forward. */
static void gen_entry(Cands *C, const Lines *L)
{
	for (int i = 0; i < L->n; i++) {
		const char *t = L->l[i] + indent_of(L->l[i]);
		if (strncmp(t, "def ", 4) != 0) continue;
		Span sp[4];
		if (idents(L->l[i], sp, 4) < 2) continue;
		char name[128];
		if (sp[1].len >= (int)sizeof name) continue;
		memcpy(name, L->l[i] + sp[1].col, (size_t)sp[1].len);
		name[sp[1].len] = 0;
		char *r = splice(L->l[i], sp[1].col, sp[1].len, "forward");
		cand_add(C, L, i, r, 0, NULL, R_RENAME, similarity(name, "forward"));
		xfree(r);
	}
}

static void gen_indent(Cands *C, const Lines *L, int el)
{
	if (el < 0 || el >= L->n) return;
	int levels[64], nl = 0, prev = -1, opener = 0;
	levels[nl++] = 0;
	for (int i = 0; i < el; i++) {
		if (blank(L->l[i])) continue;
		int d = indent_of(L->l[i]), dup = 0;
		for (int j = 0; j < nl; j++) dup |= levels[j] == d;
		if (!dup && nl < 63) levels[nl++] = d;
		prev = d;
		int e = code_end(L->l[i]);
		opener = e > 0 && L->l[i][e - 1] == ':';
	}
	/* the indentation the line would have if it continued the code above */
	int natural = prev < 0 ? 0 : opener ? prev + 4 : prev;
	if (prev >= 0) {
		int dup = 0;
		for (int j = 0; j < nl; j++) dup |= levels[j] == prev + 4;
		if (!dup) levels[nl++] = prev + 4;
	}
	for (int j = 0; j < nl; j++) {
		int before = C->n;
		char *r = with_indent(L->l[el], levels[j]);
		cand_add(C, L, el, r, 0, NULL, R_INDENT, 0);
		if (C->n > before) C->v[C->n - 1].f_natural = levels[j] == natural;
		xfree(r);
	}
	if (strchr(L->l[el], '\t')) {
		char *r = xstrdup(L->l[el]);
		char *o = xmalloc(strlen(r) * 4 + 1), *p = o;
		const char *s = r;
		while (*s == ' ' || *s == '\t') {
			if (*s == '\t') { memcpy(p, "    ", 4); p += 4; }
			else *p++ = ' ';
			s++;
		}
		strcpy(p, s);
		int before = C->n;
		cand_add(C, L, el, o, 0, NULL, R_CHAR, 0);
		if (C->n > before) C->v[C->n - 1].f_natural = indent_of(o) == natural;
		xfree(r);
		xfree(o);
	}
}

/* Operand spans around an '@' at position at. */
static int operand_left(const char *s, int at, int *st)
{
	int i = at - 1;
	while (i >= 0 && s[i] == ' ') i--;
	int end = i + 1;
	if (i < 0) return 0;
	if (s[i] == ')') {
		int d = 0;
		for (; i >= 0; i--) {
			if (s[i] == ')') d++;
			if (s[i] == '(' && --d == 0) break;
		}
		if (i < 0) return 0;
		i--;
	}
	while (i >= 0 && (isalnum((unsigned char)s[i]) || s[i] == '_')) i--;
	*st = i + 1;
	return end - *st;
}

static int operand_right(const char *s, int at, int *st)
{
	int i = at + 1;
	while (s[i] == ' ') i++;
	*st = i;
	while (isalnum((unsigned char)s[i]) || s[i] == '_') i++;
	if (s[i] == '(') {
		int d = 0;
		for (; s[i]; i++) {
			if (s[i] == '(') d++;
			if (s[i] == ')' && --d == 0) { i++; break; }
		}
		if (d) return 0;
	}
	return i - *st;
}

static void gen_matmul(Cands *C, const Lines *L, int el)
{
	if (el < 0 || el >= L->n) return;
	const char *s = L->l[el];
	int end = code_end(s);
	for (int at = 0; at < end; at++) {
		if (s[at] != '@') continue;
		int ls, rs;
		int ll = operand_left(s, at, &ls), rl = operand_right(s, at, &rs);
		if (ll <= 0 || rl <= 0) continue;
		char *A = xmalloc((size_t)ll + 1), *B = xmalloc((size_t)rl + 1);
		memcpy(A, s + ls, (size_t)ll);
		memcpy(B, s + rs, (size_t)rl);
		size_t cap = strlen(s) + (size_t)ll + (size_t)rl + 32;
		char *mid = xmalloc(cap);
		snprintf(mid, cap, "%s @ %s", B, A);
		char *r = splice(s, ls, rs + rl - ls, mid);
		cand_add(C, L, el, r, 0, NULL, R_SWAP, 0);
		xfree(r);
		snprintf(mid, cap, "transpose(%s) @ %s", A, B);
		r = splice(s, ls, rs + rl - ls, mid);
		cand_add(C, L, el, r, 0, NULL, R_TRANSPOSE, 0);
		xfree(r);
		snprintf(mid, cap, "%s @ transpose(%s)", A, B);
		r = splice(s, ls, rs + rl - ls, mid);
		cand_add(C, L, el, r, 0, NULL, R_TRANSPOSE, 0);
		xfree(r);
		xfree(mid);
		xfree(A);
		xfree(B);
	}
}

static void gen_expect(Cands *C, const Lines *L, int el, const char *msg)
{
	char X[64], Y[64];
	int hx = after(msg, "expected '", X, sizeof X);
	int hy = after(msg, "got '", Y, sizeof Y);
	if (hy && (!strcmp(Y, "end of line") || !strcmp(Y, "indent") || !strcmp(Y, "dedent") || !strcmp(Y, "end of file")))
		hy = 0;
	if (hx) {
		for (int li = el; li >= el - 2 && li >= 0; li--) {
			if (li >= L->n || blank(L->l[li])) continue;
			int e = code_end(L->l[li]);
			char *r = splice(L->l[li], e, 0, X);
			cand_add(C, L, li, r, 0, NULL, R_INSERT, 0);
			xfree(r);
		}
	}
	if (hy && el >= 0 && el < L->n) {
		int p = find_tok(L->l[el], Y);
		if (p >= 0) {
			char *r = splice(L->l[el], p, (int)strlen(Y), "");
			cand_add(C, L, el, r, 0, NULL, R_DELETE, 0);
			xfree(r);
			if (hx) {
				r = splice(L->l[el], p, 0, X);
				cand_add(C, L, el, r, 0, NULL, R_INSERT, 0);
				xfree(r);
			}
		}
	}
}

/* Close a bracket left open on the error line or up to three lines above it
 * (newlines inside brackets are not significant, so the error surfaces
 * later). The closer is proposed at every token boundary after the last
 * unmatched opener; when several placements compile the repair is ambiguous,
 * which the ambiguity feature exposes to the ranker. */
static void gen_balance(Cands *C, const Lines *L, int el)
{
	for (int li = el; li >= 0 && li >= el - 3; li--) {
		if (li >= L->n || blank(L->l[li])) continue;
		const char *s = L->l[li];
		int e = code_end(s), st[64], sp = 0, open = -1;
		char closer = 0;
		for (int i = 0; i < e; i++) {
			if ((s[i] == '(' || s[i] == '[') && sp < 64) st[sp++] = i;
			if ((s[i] == ')' || s[i] == ']') && sp > 0) sp--;
		}
		if (sp == 0) continue;
		open = st[sp - 1];
		closer = s[open] == '(' ? ')' : ']';
		char cl[2] = { closer, 0 };
		int added = 0;
		for (int q = e; q > open + 1 && added < 12; q--) {
			char a = s[q - 1];
			if (!(isalnum((unsigned char)a) || a == '_' || a == ')' || a == ']' || a == '.')) continue;
			if (q < e && (isalnum((unsigned char)s[q]) || s[q] == '_' || s[q] == '.')) continue;
			char *r = splice(s, q, 0, cl);
			cand_add(C, L, li, r, 0, NULL, R_INSERT, 0);
			xfree(r);
			added++;
		}
	}
}

/* A name defined once and never read is likely a typo of `bad` at its
 * definition: rename the definition instead of every use. */
static void gen_rename_def(Cands *C, const Lines *L, int el, const char *bad)
{
	Span sp[128];
	for (int i = 0; i < L->n && i <= el; i++) {
		int n = idents(L->l[i], sp, 128);
		for (int j = 0; j < n; j++) {
			char w[128];
			if (sp[j].len >= (int)sizeof w || is_keyword(L->l[i] + sp[j].col, (size_t)sp[j].len)) continue;
			memcpy(w, L->l[i] + sp[j].col, (size_t)sp[j].len);
			w[sp[j].len] = 0;
			if (!strcmp(w, bad) || op_lookup(w) >= 0 || word_count(L, w) != 1) continue;
			int before = C->n;
			char *r = splice(L->l[i], sp[j].col, sp[j].len, bad);
			cand_add(C, L, i, r, 0, NULL, R_RENAME, similarity(bad, w));
			xfree(r);
			if (C->n > before) {
				Cand *c = &C->v[C->n - 1];
				c->f_unused = 1;
				c->f_badonce = word_count(L, bad) == 1;
				c->f_single = strlen(bad) == 1 && strlen(w) == 1;
			}
		}
	}
}

static int assign_line(const char *s, char *name, size_t nn, const char **rhs)
{
	const char *p = s;
	while (*p == ' ') p++;
	const char *n0 = p;
	while (isalnum((unsigned char)*p) || *p == '_') p++;
	size_t len = (size_t)(p - n0);
	if (!len || len >= nn || is_keyword(n0, len)) return 0;
	while (*p == ' ') p++;
	if (*p != '=') return 0;
	p++;
	while (*p == ' ') p++;
	memcpy(name, n0, len);
	name[len] = 0;
	*rhs = p;
	return 1;
}

static void gen_return(Cands *C, const Lines *L, int el, const char *msg)
{
	if (el < 0 || el >= L->n) return;
	if (strstr(msg, "unreachable")) {
		cand_add(C, L, el, NULL, 1, NULL, R_DELLINE, 0);
		return;
	}
	if (strstr(msg, "inside think")) {
		int ind = indent_of(L->l[el]);
		for (int i = el - 1; i >= 0; i--) {
			if (blank(L->l[i]) || indent_of(L->l[i]) >= ind) continue;
			const char *t = L->l[i] + indent_of(L->l[i]);
			if (strncmp(t, "think ", 6) != 0) break;
			char st[64];
			int k = 0;
			t += 6;
			while (*t == ' ') t++;
			while ((isalnum((unsigned char)*t) || *t == '_') && k < 63) st[k++] = *t++;
			st[k] = 0;
			const char *r = L->l[el] + ind;
			if (strncmp(r, "return", 6) != 0) break;
			r += 6;
			while (*r == ' ') r++;
			size_t cap = strlen(r) + (size_t)ind + 72;
			char *o = xmalloc(cap);
			snprintf(o, cap, "%*s%s = %s", ind, "", st, r);
			cand_add(C, L, el, o, 0, NULL, R_RETURN, 0);
			xfree(o);
			break;
		}
		return;
	}
	if (strstr(msg, "no 'return'")) {
		int ind = indent_of(L->l[el]), last = -1, end = L->n;
		for (int i = el; i < L->n; i++) {
			if (blank(L->l[i])) continue;
			int d = indent_of(L->l[i]);
			if (d < ind) { end = i; break; }
			if (d == ind) last = i;
		}
		if (last < 0) return;
		char name[64];
		const char *rhs;
		if (!assign_line(L->l[last], name, sizeof name, &rhs)) return;
		size_t cap = strlen(rhs) + (size_t)ind + 72;
		char *o = xmalloc(cap);
		snprintf(o, cap, "%*sreturn %s", ind, "", rhs);
		cand_add(C, L, last, o, 0, NULL, R_RETURN, 0);
		snprintf(o, cap, "%*sreturn %s", ind, "", name);
		int at = end;
		while (at > last + 1 && blank(L->l[at - 1])) at--;
		cand_add(C, L, at, NULL, 0, o, R_RETURN, 0);
		xfree(o);
	}
}

static void gen_lex(Cands *C, const Lines *L, int el, const char *msg)
{
	char ch[8];
	if (el < 0 || el >= L->n) return;
	if (strstr(msg, "unexpected character") && quoted(msg, 0, ch, sizeof ch)) {
		const char *p = strchr(L->l[el], ch[0]);
		if (p) {
			char *r = splice(L->l[el], (int)(p - L->l[el]), 1, "");
			cand_add(C, L, el, r, 0, NULL, R_CHAR, 0);
			xfree(r);
		}
	}
	if (strstr(msg, "unterminated string")) {
		char *r = splice(L->l[el], code_end(L->l[el]), 0, "\"");
		cand_add(C, L, el, r, 0, NULL, R_INSERT, 0);
		xfree(r);
	}
}

static int is_header(const char *s)
{
	while (*s == ' ') s++;
	return !strncmp(s, "def ", 4) || !strncmp(s, "think ", 6) || !strncmp(s, "param ", 6) || !strncmp(s, "model ", 6);
}

static int generate(Cands *C, const Lines *L, const Diag *d)
{
	int cls = classify(d->msg), el = d->line - 1;
	char q[128];
	switch (cls) {
	case E_UNDEF:
	case E_UNKFN:
		if (quoted(d->msg, 0, q, sizeof q)) {
			gen_rename(C, L, el, q, cls == E_UNKFN);
			if (cls == E_UNDEF) gen_rename_def(C, L, el, q);
		}
		break;
	case E_SHAPE:
		if (strstr(d->msg, "'matmul'")) gen_matmul(C, L, el);
		break;
	case E_INDENT:
		gen_indent(C, L, el);
		break;
	case E_EXPECT:
		gen_expect(C, L, el, d->msg);
		gen_balance(C, L, el);
		if (strstr(d->msg, "'model', 'param'")) gen_indent(C, L, el); /* top-level declaration expected */
		if (strstr(d->msg, "indented block")) gen_indent(C, L, el);
		break;
	case E_RETURN:
		gen_return(C, L, el, d->msg);
		break;
	case E_LEX:
		gen_lex(C, L, el, d->msg);
		break;
	case E_OTHER:
		if (strstr(d->msg, "missing entry point")) gen_entry(C, L);
		if (strstr(d->msg, "think state") && quoted(d->msg, 0, q, sizeof q)) {
			gen_rename(C, L, el, q, 0);
			gen_rename_def(C, L, el, q);
		}
		if (strstr(d->msg, "reserved word")) gen_indent(C, L, el);
		break;
	}
	return cls;
}

static void features(Cand *c, const Lines *L, const Diag *d0, int cls0, const Trial *t)
{
	float *x = c->x;
	memset(x, 0, NF * sizeof *x);
	x[cls0] = 1;
	x[E_N + c->op] = 1;
	int b = E_N + R_N; /* 17 */
	if (t->ok) {
		x[b] = 1;
	} else {
		x[b + 1] = classify(t->d.msg) == cls0;
		x[b + 2] = strcmp(t->d.msg, d0->msg) == 0 && t->d.line == d0->line;
		x[b + 3] = t->d.line > d0->line;
		x[b + 4] = clip1((t->d.line - d0->line) / 8.0);
	}
	x[b + 5] = clip1(abs(c->line - (d0->line - 1)) / 4.0);
	x[b + 6] = c->sim;
	x[b + 7] = clip1(c->size / 8.0);
	x[b + 8] = clip1(c->rank / 8.0);
	x[b + 9] = c->line < L->n && is_header(L->l[c->line]);
	x[b + 10] = c->f_unused;
	x[b + 11] = c->f_badonce;
	x[b + 12] = c->f_builtin;
	x[b + 13] = c->f_single;
	x[b + 14] = 0; /* ambiguity, set by expand() once all trials are known */
	x[b + 15] = 0; /* similarity margin over the other compiling candidates, set by expand() */
	x[b + 16] = c->f_natural;
	x[b + 17] = 1; /* constant; must stay last (always visible to every tree) */
}
_Static_assert(E_N + R_N + 17 == NF - 1, "feature layout and NF disagree");

/* Generate, trial and featurize all candidates for one diagnostic. */
static void expand(Cands *C, const char *src, const char *path, const Diag *d0, const char *ref_ir)
{
	Lines L = split(src);
	int cls = generate(C, &L, d0);
	for (int i = 0; i < C->n; i++) {
		size_t mark = alloc_mark();
		trial(C->v[i].src, path);
		Trial t = g_trial;
		features(&C->v[i], &L, d0, cls, &t);
		if (ref_ir && t.ok) C->v[i].label = strcmp(ir_canon(t.m), ref_ir) == 0;
		alloc_release(mark);
	}
	int nok = 0;
	for (int i = 0; i < C->n; i++) nok += C->v[i].x[E_N + R_N] == 1;
	for (int i = 0; i < C->n; i++) C->v[i].x[E_N + R_N + 14] = clip1((nok - 1) / 4.0);
	for (int i = 0; i < C->n; i++) {
		float other = 0;
		for (int j = 0; j < C->n; j++)
			if (j != i && C->v[j].x[E_N + R_N] == 1 && C->v[j].sim > other) other = C->v[j].sim;
		C->v[i].x[E_N + R_N + 15] = clip1(C->v[i].sim - other);
	}
	lines_free(&L);
}

/* ---- the forest --------------------------------------------------------- */

/* Path matrices: A[l][n] = 1 if leaf l routes left at node n, B for right. */
static void paths(double A[NL][NN], double B[NL][NN])
{
	memset(A, 0, sizeof(double) * NL * NN);
	memset(B, 0, sizeof(double) * NL * NN);
	for (int l = 0; l < NL; l++) {
		int node = 0;
		for (int d = ND - 1; d >= 0; d--) {
			int right = (l >> d) & 1;
			if (right) B[l][node] = 1;
			else A[l][node] = 1;
			node = 2 * node + 1 + right;
		}
	}
}

typedef struct {
	double w[NN][NF], b[NN], q[NL], m[NF];
} Tree;

static double sp(double z) { return (z > 0 ? z : 0) + log1p(exp(-fabs(z))); }
static double sg(double z) { return 1.0 / (1.0 + exp(-z)); }

/* p(correct) for one tree; fills mu and z for backprop. */
static double tree_fwd(const Tree *t, const float *x, double A[NL][NN], double B[NL][NN], double *z, double *mu)
{
	for (int n = 0; n < NN; n++) {
		double s = t->b[n];
		for (int j = 0; j < NF; j++) s += t->w[n][j] * x[j] * t->m[j];
		z[n] = s;
	}
	double p = 0;
	for (int l = 0; l < NL; l++) {
		double lm = 0;
		for (int n = 0; n < NN; n++) lm -= A[l][n] * sp(-z[n]) + B[l][n] * sp(z[n]);
		mu[l] = exp(lm);
		p += sg(t->q[l]) * mu[l];
	}
	return p;
}

static double logit(double p)
{
	if (p < 1e-6) p = 1e-6;
	if (p > 1 - 1e-6) p = 1 - 1e-6;
	return log(p / (1 - p));
}

/* Calibrated ensemble output; must match the program write_forest() emits. */
static double forest_fwd(const Tree *T, double ca, double cb, const float *x, double A[NL][NN], double B[NL][NN])
{
	double z[NN], mu[NL], p = 0;
	for (int k = 0; k < NT; k++) p += tree_fwd(&T[k], x, A, B, z, mu);
	return sg(ca * logit(p / NT) + cb);
}

static void train_tree(Tree *t, float (*X)[NF], const int *Y, int n, double A[NL][NN], double B[NL][NN], unsigned char *inbag)
{
	memset(t, 0, sizeof *t);
	/* random subspace, the constant feature is always visible */
	int idx[NF];
	for (int j = 0; j < NF; j++) idx[j] = j;
	for (int j = NF - 1; j > 0; j--) { int k = rndn(j + 1), s = idx[j]; idx[j] = idx[k]; idx[k] = s; }
	for (int j = 0; j < NSUB; j++) t->m[idx[j]] = 1;
	t->m[NF - 1] = 1;
	for (int i = 0; i < NN; i++)
		for (int j = 0; j < NF; j++) t->w[i][j] = (rndu() * 2 - 1) * 0.5;

	/* bootstrap sample */
	int *bs = xmalloc((size_t)n * sizeof *bs), pos = 0;
	memset(inbag, 0, (size_t)n);
	for (int i = 0; i < n; i++) { bs[i] = rndn(n); pos += Y[bs[i]]; inbag[bs[i]] = 1; }
	double wpos = pos ? (double)(n - pos) / pos : 1.0;
	if (wpos < 1) wpos = 1;

	double z[NN], mu[NL];
	for (int ep = 0; ep < 60; ep++) {
		double lr = 0.05 / (1 + ep * 0.05);
		for (int i = n - 1; i > 0; i--) { int k = rndn(i + 1), s = bs[i]; bs[i] = bs[k]; bs[k] = s; }
		for (int s = 0; s < n; s++) {
			const float *x = X[bs[s]];
			int y = Y[bs[s]];
			double p = tree_fwd(t, x, A, B, z, mu);
			if (p < 1e-6) p = 1e-6;
			if (p > 1 - 1e-6) p = 1 - 1e-6;
			double wt = y ? wpos : 1.0;
			double gp = wt * (p - y) / (p * (1 - p)); /* dL/dp, weighted BCE */
			double gz[NN] = { 0 };
			for (int l = 0; l < NL; l++) {
				double pi = sg(t->q[l]);
				double glm = gp * pi * mu[l];
				t->q[l] -= lr * gp * mu[l] * pi * (1 - pi);
				for (int nn = 0; nn < NN; nn++)
					gz[nn] += glm * (A[l][nn] * sg(-z[nn]) - B[l][nn] * sg(z[nn]));
			}
			for (int nn = 0; nn < NN; nn++) {
				t->b[nn] -= lr * gz[nn];
				for (int j = 0; j < NF; j++)
					if (t->m[j] != 0) t->w[nn][j] -= lr * gz[nn] * x[j];
			}
		}
	}
	xfree(bs);
}

static void put_mat(FILE *f, const char *name, int r, int c, const double *v)
{
	if (r > 1) fprintf(f, "param %s : f32[%d, %d] = [", name, r, c);
	else fprintf(f, "param %s : f32[%d] = [", name, c);
	for (int i = 0; i < r * c; i++) {
		if (i) fputs(i % 8 ? ", " : ",\n    ", f);
		fprintf(f, "%.9g", v[i]);
	}
	fputs("]\n", f);
}

static void write_forest(FILE *f, Tree *T, double A[NL][NN], double B[NL][NN], int ns, int np, double ca, double cb)
{
	fprintf(f, "# Auto-fix ranker: a forest of %d soft decision trees of depth %d\n"
		   "# (deep neural decision forest, Kontschieder et al. ICCV 2015).\n"
		   "# Generated by `tgc fixer-train` from %d self-supervised samples (%d positive).\n"
		   "# Input: %d features describing (diagnostic, candidate edit, trial outcome).\n"
		   "# Output: calibrated probability that the edit is the correct repair.\n"
		   "# Do not edit by hand; regenerate with `make fixer`.\n"
		   "model fixer_forest\n\n",
		NT, ND, ns, np, NF);
	put_mat(f, "path_l", NL, NN, &A[0][0]);
	put_mat(f, "path_r", NL, NN, &B[0][0]);
	fprintf(f, "\n# Platt calibration fitted on out-of-bag scores.\nparam cal_a : f32 = %.9g\nparam cal_b : f32 = %.9g\n", ca, cb);
	char nm[32];
	for (int k = 0; k < NT; k++) {
		fputc('\n', f);
		snprintf(nm, sizeof nm, "m%d", k);
		put_mat(f, nm, 1, NF, T[k].m);
		snprintf(nm, sizeof nm, "w%d", k);
		put_mat(f, nm, NN, NF, &T[k].w[0][0]);
		snprintf(nm, sizeof nm, "b%d", k);
		put_mat(f, nm, 1, NN, T[k].b);
		snprintf(nm, sizeof nm, "q%d", k);
		put_mat(f, nm, 1, NL, T[k].q);
	}
	fprintf(f, "\n# Soft routing: d = sigmoid(z); log mu = -(path_l @ softplus(-z) + path_r @ softplus(z)).\n"
		   "def tree(x: f32[%d], m: f32[%d], w: f32[%d, %d], b: f32[%d], q: f32[%d]) -> f32:\n"
		   "    z = w @ (x * m) + b\n"
		   "    mu = exp(-(path_l @ softplus(-z) + path_r @ softplus(z)))\n"
		   "    return dot(sigmoid(q), mu)\n\n"
		   "def forward(x: f32[%d]) -> f32:\n"
		   "    s = tree(x, m0, w0, b0, q0)\n",
		NF, NF, NN, NF, NN, NL, NF);
	for (int k = 1; k < NT; k++) fprintf(f, "    s = s + tree(x, m%d, w%d, b%d, q%d)\n", k, k, k, k);
	fprintf(f, "    p = min(max(s / %d, 0.000001), 0.999999)\n"
		   "    return sigmoid(cal_a * log(p / (1 - p)) + cal_b)\n", NT);
}

static Module *forest(void)
{
	static Module *m;
	static int tried;
	if (tried) return m;
	tried = 1;
	size_t n = 1;
	for (int i = 0; tg_forest_lines[i]; i++) n += strlen(tg_forest_lines[i]);
	if (n == 1) return NULL;
	char *src = xmalloc(n), *p = src;
	for (int i = 0; tg_forest_lines[i]; i++) {
		size_t k = strlen(tg_forest_lines[i]);
		memcpy(p, tg_forest_lines[i], k);
		p += k;
	}
	*p = 0;
	m = lower(surface_parse(src, "<fixer/forest.tg>"), "<fixer/forest.tg>");
	plan(m);
	xfree(src);
	return m;
}

static float score(const float *x)
{
	Module *m = forest();
	const float *in[1] = { x };
	float out;
	int steps[1];
	vm_run(m, in, &out, steps);
	return out;
}

/* ---- the fix loop ------------------------------------------------------- */

static void show(const Cand *c, const Diag *d, const char *path)
{
	fprintf(stderr, "%s:%d: error: %s\n", path, d->line, d->msg);
	fprintf(stderr, "  autofix: %s at line %d (confidence %.2f)\n", rname[c->op], c->line + 1, (double)c->score);
	if (c->old) fprintf(stderr, "  - %s\n", c->old);
	if (c->new) fprintf(stderr, "  + %s\n", c->new);
	tr_begin(1, "autofix");
	tr_quiet();
	tr_str("file", path);
	tr_num("line", d->line);
	tr_str("diagnostic", d->msg);
	tr_str("decision", "applied");
	tr_str("operator", rname[c->op]);
	tr_num("edit_line", c->line + 1);
	tr_num("confidence", c->score);
	tr_str("old", c->old ? c->old : "");
	tr_str("new", c->new ? c->new : "");
	tr_end("%s at line %d (confidence %.2f)", rname[c->op], c->line + 1, (double)c->score);
}

char *autofix(const char *src, const char *path, int verbose, int *nfixed)
{
	*nfixed = 0;
	if (!forest()) {
		if (verbose) fprintf(stderr, "autofix: no forest model built in (run `make fixer`)\n");
		return NULL;
	}
	char *cur = xstrdup(src);
	char **seen = NULL;
	int nseen = 0;
	for (int it = 0; it <= MAX_ITERS; it++) {
		size_t mark = alloc_mark();
		trial(cur, path);
		int ok = g_trial.ok;
		Diag d0 = g_trial.d;
		alloc_release(mark);
		if (ok) break;
		if (it == MAX_ITERS) goto fail;
		Cands C = { 0 };
		expand(&C, cur, path, &d0, NULL);
		int best = -1;
		for (int i = 0; i < C.n; i++) {
			Cand *c = &C.v[i];
			c->score = score(c->x);
			int progress = c->x[E_N + R_N] == 1 || c->x[E_N + R_N + 2] == 0;
			int cycle = 0;
			for (int k = 0; k < nseen; k++) cycle |= strcmp(seen[k], c->src) == 0;
			if (progress && !cycle && c->score >= threshold() && (best < 0 || c->score > C.v[best].score)) best = i;
		}
		if (best < 0) {
			if (verbose) {
				fprintf(stderr, "%s:%d: error: %s\n  autofix: no confident repair among %d candidate(s)\n", path, d0.line, d0.msg, C.n);
				tr_begin(1, "autofix");
				tr_quiet();
				tr_str("file", path);
				tr_num("line", d0.line);
				tr_str("diagnostic", d0.msg);
				tr_str("decision", "abstained");
				tr_num("candidates", C.n);
				tr_end("no confident repair among %d candidate(s)", C.n);
			}
			cands_free(&C);
			goto fail;
		}
		if (verbose) show(&C.v[best], &d0, path);
		seen = xrealloc(seen, (size_t)(nseen + 1) * sizeof *seen);
		seen[nseen++] = cur;
		cur = xstrdup(C.v[best].src);
		(*nfixed)++;
		cands_free(&C);
	}
	for (int k = 0; k < nseen; k++) xfree(seen[k]);
	xfree(seen);
	return cur;
fail:
	for (int k = 0; k < nseen; k++) xfree(seen[k]);
	xfree(seen);
	xfree(cur);
	return NULL;
}

/* ---- breakers (self-supervised corruption) ------------------------------ */

enum { B_TYPO, B_FNTYPO, B_INDENT, B_COLON, B_PAREN, B_SWAP, B_RETURN, B_CHAR, B_TAB, B_N };
static const char *bname[B_N] = { "typo", "fn-typo", "indent", "colon", "paren", "swap", "return", "char", "tab" };

/* Typo targets: any non-keyword identifier, or (fn-typo) only call names. */
static int typo_target(const char *s, Span w, int kind)
{
	if (is_keyword(s + w.col, (size_t)w.len)) return 0;
	return kind == B_TYPO || s[w.col + w.len] == '(';
}

static int code_line(const char *s) { return !blank(s); }

/* Returns a corrupted copy, or NULL if the breaker does not apply. */
static char *breaker(const char *src, int kind)
{
	Lines L = split(src);
	char *out = NULL, *r = NULL;
	int cand[4096], nc = 0;
	for (int i = 0; i < L.n && nc < 4096; i++) {
		const char *s = L.l[i];
		if (!code_line(s)) continue;
		const char *t = s + indent_of(s);
		int e = code_end(s);
		switch (kind) {
		case B_TYPO:
		case B_FNTYPO: {
			Span sp[64];
			int n = idents(s, sp, 64), ok = 0;
			for (int j = 0; j < n; j++) ok |= typo_target(s, sp[j], kind);
			if (ok && strncmp(t, "model", 5) != 0) cand[nc++] = i;
			break;
		}
		case B_INDENT: if (i > 0) cand[nc++] = i; break;
		case B_COLON: if (e > 0 && s[e - 1] == ':') cand[nc++] = i; break;
		case B_PAREN: if (strchr(s, ')') && strncmp(t, "param", 5) != 0) cand[nc++] = i; break;
		case B_SWAP: if (strchr(s, '@')) cand[nc++] = i; break;
		case B_RETURN: if (!strncmp(t, "return ", 7)) cand[nc++] = i; break;
		case B_CHAR: cand[nc++] = i; break;
		case B_TAB: if (indent_of(s) > 0) cand[nc++] = i; break;
		}
	}
	if (!nc) goto done;
	int li = cand[rndn(nc)];
	const char *s = L.l[li];
	int ind = indent_of(s), e = code_end(s);
	switch (kind) {
	case B_TYPO:
	case B_FNTYPO: {
		Span sp[64];
		int n = idents(s, sp, 64), pick[64], np = 0;
		for (int j = 0; j < n; j++)
			if (typo_target(s, sp[j], kind)) pick[np++] = j;
		Span w = sp[pick[rndn(np)]];
		char word[80];
		if (w.len >= 70) goto done;
		memcpy(word, s + w.col, (size_t)w.len);
		word[w.len] = 0;
		int len = w.len, at = rndn(len), how = rndn(4);
		char c = (char)('a' + rndn(26));
		if (how == 1 && len > 1) memmove(word + at, word + at + 1, (size_t)(len - at)); /* delete */
		else if (how == 2) { memmove(word + at + 1, word + at, (size_t)(len - at + 1)); word[at] = c; } /* insert */
		else if (how == 3 && len > 1 && at + 1 < len) { char x = word[at]; word[at] = word[at + 1]; word[at + 1] = x; }
		else word[at] = c; /* substitute */
		r = splice(s, w.col, w.len, word);
		break;
	}
	case B_INDENT: {
		int delta = (rndn(2) ? 1 : -1) * (1 + rndn(4));
		if (ind + delta < 0) delta = -delta;
		r = with_indent(s, ind + delta);
		break;
	}
	case B_COLON: r = splice(s, e - 1, 1, ""); break;
	case B_PAREN: {
		int ps[64], n = 0;
		for (int j = 0; j < e && n < 64; j++) if (s[j] == ')') ps[n++] = j;
		r = splice(s, ps[rndn(n)], 1, "");
		break;
	}
	case B_SWAP: {
		const char *a = strchr(s, '@');
		int ls, rs;
		int ll = operand_left(s, (int)(a - s), &ls), rl = operand_right(s, (int)(a - s), &rs);
		if (ll <= 0 || rl <= 0) goto done;
		char mid[512];
		if (ll + rl + 4 >= (int)sizeof mid) goto done;
		snprintf(mid, sizeof mid, "%.*s @ %.*s", rl, s + rs, ll, s + ls);
		r = splice(s, ls, rs + rl - ls, mid);
		break;
	}
	case B_RETURN: r = splice(s, ind, 6, "r_ ="); break;
	case B_CHAR: {
		static const char junk[] = "$?!`;{";
		char j[2] = { junk[rndn(6)], 0 };
		r = splice(s, ind + rndn(e - ind + 1), 0, j);
		break;
	}
	case B_TAB: r = splice(s, 0, ind, "\t"); break;
	}
	out = join_edit(&L, li, r, 0, NULL);
done:
	xfree(r);
	lines_free(&L);
	return out;
}

typedef struct {
	char *path, *src, *ir;
} Prog;

static int load_corpus(char **files, int n, Prog *P)
{
	int k = 0;
	for (int i = 0; i < n; i++) {
		char *src = read_file(files[i], NULL);
		size_t mark = alloc_mark();
		trial(src, files[i]);
		if (!g_trial.ok) {
			fprintf(stderr, "skipping %s: does not compile (%s)\n", files[i], g_trial.d.msg);
			alloc_release(mark);
			xfree(src);
			continue;
		}
		P[k].path = files[i];
		P[k].src = src;
		P[k].ir = alloc_keep(ir_canon(g_trial.m));
		alloc_release(mark);
		k++;
	}
	return k;
}

/* One broken program per call; returns NULL if the corruption still compiles. */
static char *break_one(const Prog *p, int *kind, Diag *d)
{
	for (int tries = 0; tries < 16; tries++) {
		*kind = rndn(B_N);
		char *b = breaker(p->src, *kind);
		if (!b) continue;
		size_t mark = alloc_mark();
		trial(b, p->path);
		int ok = g_trial.ok;
		*d = g_trial.d;
		alloc_release(mark);
		if (!ok) return b;
		xfree(b);
	}
	return NULL;
}

#define ROUNDS 160

int fixer_train(const char *out, char **files, int nfiles)
{
	Prog *P = xmalloc((size_t)nfiles * sizeof *P);
	int np = load_corpus(files, nfiles, P);
	if (!np) die(NULL, 0, "empty training corpus");
	rng_s = 0x9e3779b9u;

	float (*X)[NF] = NULL;
	int *Y = NULL, n = 0, cap = 0, pos = 0, solvable = 0, total = 0;
	for (int i = 0; i < np; i++) {
		for (int r = 0; r < ROUNDS; r++) {
			int kind;
			Diag d0;
			char *b = break_one(&P[i], &kind, &d0);
			if (!b) continue;
			Cands C = { 0 };
			expand(&C, b, P[i].path, &d0, P[i].ir);
			int any = 0;
			for (int c = 0; c < C.n; c++) {
				if (n == cap) {
					cap = cap ? cap * 2 : 1024;
					X = xrealloc(X, (size_t)cap * sizeof *X);
					Y = xrealloc(Y, (size_t)cap * sizeof *Y);
				}
				memcpy(X[n], C.v[c].x, sizeof X[n]);
				Y[n] = C.v[c].label;
				pos += Y[n];
				any |= Y[n];
				n++;
			}
			solvable += any;
			total++;
			cands_free(&C);
			xfree(b);
		}
	}
	fprintf(stderr, "fixer-train: %d programs, %d corruptions (%d repairable by one operator), %d samples, %d positive\n",
		np, total, solvable, n, pos);

	double A[NL][NN], B[NL][NN];
	paths(A, B);
	Tree *T = xmalloc(NT * sizeof *T);
	unsigned char *inbag = xmalloc((size_t)NT * (size_t)n);
	for (int k = 0; k < NT; k++) train_tree(&T[k], X, Y, n, A, B, inbag + (size_t)k * (size_t)n);

	/* Platt calibration on out-of-bag scores: each sample is scored only by
	 * trees that did not see it, then sigmoid(a * logit(p) + b) is fitted by
	 * Newton's method on the log loss. Averaged random-subspace members are
	 * under-confident; this maps their output to calibrated probabilities. */
	double z[NN], mu[NL];
	double *oob = xmalloc((size_t)n * sizeof *oob);
	int noob = 0;
	for (int i = 0; i < n; i++) {
		double p = 0;
		int c = 0;
		for (int k = 0; k < NT; k++)
			if (!inbag[(size_t)k * (size_t)n + (size_t)i]) { p += tree_fwd(&T[k], X[i], A, B, z, mu); c++; }
		oob[i] = c ? logit(p / c) : NAN;
		noob += c > 0;
	}
	double ca = 1, cb = 0;
	for (int it = 0; it < 50; it++) {
		double ga = 0, gb = 0, haa = 1e-9, hab = 0, hbb = 1e-9;
		for (int i = 0; i < n; i++) {
			if (isnan(oob[i])) continue;
			double q = sg(ca * oob[i] + cb), r = q - Y[i], w = q * (1 - q);
			ga += r * oob[i];
			gb += r;
			haa += w * oob[i] * oob[i];
			hab += w * oob[i];
			hbb += w;
		}
		double det = haa * hbb - hab * hab;
		if (fabs(det) < 1e-12) break;
		ca -= (hbb * ga - hab * gb) / det;
		cb -= (haa * gb - hab * ga) / det;
	}
	xfree(oob);
	xfree(inbag);
	fprintf(stderr, "fixer-train: %d out-of-bag samples, calibration a=%.4f b=%.4f\n", noob, ca, cb);

	/* training-set quality of the calibrated ensemble */
	int tp = 0, fp = 0, fn = 0;
	for (int i = 0; i < n; i++) {
		int pred = forest_fwd(T, ca, cb, X[i], A, B) >= THRESHOLD;
		tp += pred && Y[i];
		fp += pred && !Y[i];
		fn += !pred && Y[i];
	}
	fprintf(stderr, "fixer-train: train precision %.3f recall %.3f at threshold %.2f\n",
		tp + fp ? (double)tp / (tp + fp) : 0.0, tp + fn ? (double)tp / (tp + fn) : 0.0, (double)THRESHOLD);

	FILE *f = fopen(out, "w");
	if (!f) die(NULL, 0, "cannot write '%s'", out);
	write_forest(f, T, A, B, n, pos, ca, cb);
	if (fclose(f)) die(NULL, 0, "write failed '%s'", out);

	/* The emitted program must compile and agree with the trainer. */
	trial(read_file(out, NULL), out);
	if (!g_trial.ok) die(NULL, 0, "generated forest does not compile: %s", g_trial.d.msg);
	Module *m = g_trial.m;
	plan(m);
	double maxerr = 0;
	for (int i = 0; i < n && i < 2000; i++) {
		const float *in[1] = { X[i] };
		float o;
		int steps[1];
		vm_run(m, in, &o, steps);
		double e = fabs(o - forest_fwd(T, ca, cb, X[i], A, B));
		if (e > maxerr) maxerr = e;
	}
	fprintf(stderr, "fixer-train: wrote %s; max |VM - trainer| = %.2g\n", out, maxerr);
	if (maxerr > 1e-3) die(NULL, 0, "forest VM output disagrees with trainer");
	xfree(X);
	xfree(Y);
	xfree(T);
	return 0;
}

int fixer_eval(char **files, int nfiles)
{
	Prog *P = xmalloc((size_t)nfiles * sizeof *P);
	int np = load_corpus(files, nfiles, P);
	if (!np) die(NULL, 0, "empty evaluation corpus");
	if (!forest()) die(NULL, 0, "no forest model built in (run `make fixer`)");
	rng_s = 0x51f15e5du; /* disjoint from the training stream */
	int stat[B_N][4] = { { 0 } }; /* broken, exact, wrong (compiles, different IR), unrepaired */
	for (int i = 0; i < np; i++) {
		for (int r = 0; r < ROUNDS; r++) {
			int kind, nfix;
			Diag d0;
			char *b = break_one(&P[i], &kind, &d0);
			if (!b) continue;
			stat[kind][0]++;
			char *fixed = autofix(b, P[i].path, 0, &nfix);
			if (!fixed && getenv("TG_FIXER_DEBUG")) fprintf(stderr, "UNFIXED %s %s:%d %s\n", bname[kind], P[i].path, d0.line, d0.msg);
			if (!fixed) stat[kind][3]++;
			else {
				size_t mark = alloc_mark();
				trial(fixed, P[i].path);
				int good = strcmp(ir_canon(g_trial.m), P[i].ir) == 0;
				stat[kind][good ? 1 : 2]++;
				if (!good && getenv("TG_FIXER_DEBUG")) fprintf(stderr, "WRONG %s [%s] %s:%d %s\n", bname[kind], P[i].path, P[i].path, d0.line, d0.msg);
				alloc_release(mark);
				xfree(fixed);
			}
			xfree(b);
		}
	}
	int t[4] = { 0 };
	printf("%-8s %7s %8s %8s %10s\n", "breaker", "broken", "exact", "wrong", "unrepaired");
	for (int k = 0; k < B_N; k++) {
		printf("%-8s %7d %7.1f%% %7.1f%% %9.1f%%\n", bname[k], stat[k][0],
		       stat[k][0] ? 100.0 * stat[k][1] / stat[k][0] : 0.0,
		       stat[k][0] ? 100.0 * stat[k][2] / stat[k][0] : 0.0,
		       stat[k][0] ? 100.0 * stat[k][3] / stat[k][0] : 0.0);
		for (int j = 0; j < 4; j++) t[j] += stat[k][j];
	}
	printf("%-8s %7d %7.1f%% %7.1f%% %9.1f%%\n", "total", t[0], 100.0 * t[1] / t[0], 100.0 * t[2] / t[0], 100.0 * t[3] / t[0]);
	return 0;
}
