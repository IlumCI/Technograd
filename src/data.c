/* Datasets: load anything tabular, infer what each column is, featurize.
 *
 * Sources
 *   *.csv *.tsv *.txt     delimiter sniffed (, ; tab |), header detected, RFC 4180 quotes
 *   *.jsonl *.ndjson      one JSON object per line
 *   *.json                array of objects, or an object holding one ("rows",
 *                         "data", ...); Hugging Face API pages are recognized
 *   *.npy                 NumPy arrays (f4/f8/i1-i8/u1-u8, 1-D or 2-D, C order)
 *   hf:OWNER/NAME[/CONFIG[/SPLIT]]
 *                         Hugging Face datasets-server rows API, fetched with
 *                         curl (spawned directly, no shell), cached on disk;
 *                         HF_TOKEN is sent when set (gated datasets)
 *   anything else         sniffed from content: JSON if it starts with { or [,
 *                         otherwise delimited text
 *
 * Column kinds are inferred: numeric, categorical (few distinct values),
 * text (hashed character trigrams), vector (fixed-length numeric lists),
 * or dropped (identifiers, nested objects, inconsistent lists), each with a
 * stated reason. The featurization fitted on training rows is saved as a
 * plain-text spec so new data is transformed identically. */
#define _POSIX_C_SOURCE 200809L
#include "tg.h"
#include "data.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <spawn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

/* ---- JSON --------------------------------------------------------------- */

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } JKind;

typedef struct J {
	JKind k;
	double n;
	char *s;
	struct J **v;
	char **keys;
	int len, cap;
} J;

typedef struct {
	const char *p;
	const char *file;
	int line;
} JP;

static void jws(JP *j)
{
	while (*j->p == ' ' || *j->p == '\t' || *j->p == '\r' || *j->p == '\n') {
		if (*j->p == '\n') j->line++;
		j->p++;
	}
}

static void jpush(J *a, J *v, char *key)
{
	if (a->len == a->cap) {
		a->cap = a->cap ? a->cap * 2 : 8;
		a->v = xrealloc(a->v, (size_t)a->cap * sizeof *a->v);
		if (a->k == J_OBJ) a->keys = xrealloc(a->keys, (size_t)a->cap * sizeof *a->keys);
	}
	if (a->k == J_OBJ) a->keys[a->len] = key;
	a->v[a->len++] = v;
}

static void put_utf8(char **o, unsigned c)
{
	if (c < 0x80) *(*o)++ = (char)c;
	else if (c < 0x800) { *(*o)++ = (char)(0xC0 | c >> 6); *(*o)++ = (char)(0x80 | (c & 63)); }
	else { *(*o)++ = (char)(0xE0 | c >> 12); *(*o)++ = (char)(0x80 | ((c >> 6) & 63)); *(*o)++ = (char)(0x80 | (c & 63)); }
}

static char *jstring(JP *j)
{
	const char *s = ++j->p;
	size_t raw = 0; /* raw length to the closing quote; decoding never lengthens */
	while (s[raw] && s[raw] != '"') raw += s[raw] == '\\' && s[raw + 1] ? 2 : 1;
	char *out = xmalloc(raw + 4), *o = out;
	while (*j->p && *j->p != '"') {
		char c = *j->p++;
		if (c != '\\') { *o++ = c; continue; }
		c = *j->p++;
		switch (c) {
		case 'n': *o++ = '\n'; break;
		case 't': *o++ = '\t'; break;
		case 'r': *o++ = '\r'; break;
		case 'b': *o++ = '\b'; break;
		case 'f': *o++ = '\f'; break;
		case 'u': {
			unsigned u = 0;
			for (int i = 0; i < 4 && isxdigit((unsigned char)*j->p); i++, j->p++)
				u = u * 16 + (unsigned)(isdigit((unsigned char)*j->p) ? *j->p - '0' : (tolower((unsigned char)*j->p) - 'a' + 10));
			put_utf8(&o, u);
			break;
		}
		default: *o++ = c;
		}
	}
	if (*j->p != '"') die(j->file, j->line, "unterminated JSON string");
	j->p++;
	*o = 0;
	return out;
}

static J *jvalue(JP *j, int depth)
{
	if (depth > 64) die(j->file, j->line, "JSON nested too deeply");
	jws(j);
	J *v = xmalloc(sizeof *v);
	char c = *j->p;
	if (c == '{' || c == '[') {
		v->k = c == '{' ? J_OBJ : J_ARR;
		j->p++;
		jws(j);
		if (*j->p == (c == '{' ? '}' : ']')) { j->p++; return v; }
		for (;;) {
			char *key = NULL;
			jws(j);
			if (v->k == J_OBJ) {
				if (*j->p != '"') die(j->file, j->line, "expected a JSON object key");
				key = jstring(j);
				jws(j);
				if (*j->p++ != ':') die(j->file, j->line, "expected ':' in JSON object");
			}
			jpush(v, jvalue(j, depth + 1), key);
			jws(j);
			if (*j->p == ',') { j->p++; continue; }
			if (*j->p == (c == '{' ? '}' : ']')) { j->p++; return v; }
			die(j->file, j->line, "malformed JSON near '%.20s'", j->p);
		}
	}
	if (c == '"') { v->k = J_STR; v->s = jstring(j); return v; }
	if (!strncmp(j->p, "true", 4)) { v->k = J_BOOL; v->n = 1; j->p += 4; return v; }
	if (!strncmp(j->p, "false", 5)) { v->k = J_BOOL; j->p += 5; return v; }
	if (!strncmp(j->p, "null", 4)) { v->k = J_NULL; j->p += 4; return v; }
	if (!strncmp(j->p, "NaN", 3)) { v->k = J_NULL; j->p += 3; return v; }
	char *e;
	v->n = strtod(j->p, &e);
	if (e == j->p) die(j->file, j->line, "malformed JSON near '%.20s'", j->p);
	v->k = J_NUM;
	j->p = e;
	return v;
}

static J *jget(const J *o, const char *key)
{
	if (!o || o->k != J_OBJ) return NULL;
	for (int i = 0; i < o->len; i++)
		if (!strcmp(o->keys[i], key)) return o->v[i];
	return NULL;
}

/* ---- tables ------------------------------------------------------------- */

static int col_index(Table *t, const char *name)
{
	for (int i = 0; i < t->ncols; i++)
		if (!strcmp(t->col[i].name, name)) return i;
	if (t->ncols == 4096) die(NULL, 0, "more than 4096 columns");
	t->col = xrealloc(t->col, (size_t)(t->ncols + 1) * sizeof *t->col);
	Col *c = &t->col[t->ncols];
	memset(c, 0, sizeof *c);
	c->name = xstrdup(name);
	c->cell = xmalloc((size_t)(t->caprows ? t->caprows : 1) * sizeof *c->cell); /* earlier rows: NULL cells */
	return t->ncols++;
}

static void new_row(Table *t)
{
	if (t->nrows == t->caprows) {
		int cap = t->caprows ? t->caprows * 2 : 64;
		for (int i = 0; i < t->ncols; i++) {
			t->col[i].cell = xrealloc(t->col[i].cell, (size_t)cap * sizeof *t->col[i].cell);
			memset(t->col[i].cell + t->caprows, 0, (size_t)(cap - t->caprows) * sizeof *t->col[i].cell);
		}
		t->caprows = cap;
	}
	t->nrows++;
}

static Cell *cell(Table *t, int col) { return &t->col[col].cell[t->nrows - 1]; }

static void set_json(Cell *c, const J *v)
{
	switch (v->k) {
	case J_NULL: c->t = CELL_NULL; break;
	case J_BOOL:
	case J_NUM: c->t = CELL_NUM; c->n = v->n; break;
	case J_STR: c->t = CELL_STR; c->s = v->s; break;
	case J_ARR: {
		int ok = v->len > 0;
		for (int i = 0; i < v->len && ok; i++) ok = v->v[i]->k == J_NUM || v->v[i]->k == J_BOOL;
		if (!ok) { c->t = CELL_OTHER; break; }
		c->t = CELL_LIST;
		c->nl = v->len;
		c->l = xmalloc((size_t)v->len * sizeof *c->l);
		for (int i = 0; i < v->len; i++) c->l[i] = (float)v->v[i]->n;
		break;
	}
	case J_OBJ: c->t = CELL_OTHER; break;
	}
}

static void add_object(Table *t, const J *o)
{
	const J *row = jget(o, "row"); /* Hugging Face rows API: {"row_idx":..., "row":{...}} */
	if (row && row->k == J_OBJ && jget(o, "row_idx")) o = row;
	if (o->k != J_OBJ) die(NULL, 0, "expected JSON objects (one per row)");
	new_row(t);
	for (int i = 0; i < o->len; i++) set_json(cell(t, col_index(t, o->keys[i])), o->v[i]);
}

/* ---- readers ------------------------------------------------------------ */

static int ends_with(const char *s, const char *suf)
{
	size_t a = strlen(s), b = strlen(suf);
	return a >= b && !strcmp(s + a - b, suf);
}

static void load_json_text(Table *t, const char *text, const char *name, int lines)
{
	JP j = { text, name, 1 };
	if (lines) {
		for (;;) {
			jws(&j);
			if (!*j.p) break;
			add_object(t, jvalue(&j, 0));
		}
		return;
	}
	J *v = jvalue(&j, 0);
	if (v->k == J_OBJ) { /* look for the array of rows */
		static const char *keys[] = { "rows", "data", "records", "items", "examples", "train", NULL };
		J *a = NULL;
		for (int i = 0; keys[i] && !a; i++) a = jget(v, keys[i]);
		for (int i = 0; i < v->len && !a; i++)
			if (v->v[i]->k == J_ARR) a = v->v[i];
		if (!a || a->k != J_ARR) die(NULL, 0, "%s: JSON object holds no array of rows", name);
		v = a;
	}
	if (v->k != J_ARR) die(NULL, 0, "%s: expected a JSON array of rows", name);
	for (int i = 0; i < v->len; i++) {
		J *r = v->v[i];
		if (r->k == J_OBJ) add_object(t, r);
		else if (r->k == J_ARR) { /* array of arrays: positional columns */
			new_row(t);
			for (int c = 0; c < r->len; c++) {
				char nm[32];
				snprintf(nm, sizeof nm, "col%d", c);
				set_json(cell(t, col_index(t, nm)), r->v[c]);
			}
		} else die(NULL, 0, "%s: rows must be objects or arrays", name);
	}
}

static int is_number(const char *s, double *v)
{
	while (isspace((unsigned char)*s)) s++;
	if (!*s) return 0;
	char *e;
	double d = strtod(s, &e);
	while (isspace((unsigned char)*e)) e++;
	if (*e) return 0;
	if (v) *v = d;
	return 1;
}

/* Split one delimited record starting at *p; advances *p past the line. */
static int split_record(char **p, char d, char **f, int maxf)
{
	int n = 0;
	char *s = *p;
	if (!*s) return -1;
	for (;;) {
		char *start = s, *o = s;
		if (*s == '"') { /* quoted field, "" escapes a quote */
			s++;
			for (;;) {
				if (!*s) break;
				if (*s == '"' && s[1] == '"') { *o++ = '"'; s += 2; continue; }
				if (*s == '"') { s++; break; }
				*o++ = *s++;
			}
			while (*s && *s != d && *s != '\n' && *s != '\r') s++;
		} else {
			while (*s && *s != d && *s != '\n' && *s != '\r') *o++ = *s++;
		}
		char end = *s; /* read before terminating: the terminator may overwrite it */
		*o = 0;
		if (n < maxf) f[n++] = start;
		if (end == d) { s++; continue; }
		if (end == '\r') { s++; if (*s == '\n') s++; }
		else if (end == '\n') s++;
		break;
	}
	*p = s;
	return n;
}

static void load_delimited(Table *t, char *text, const char *name)
{
	/* delimiter: the candidate with the most consistent nonzero count over the first lines */
	const char cand[] = { ',', '\t', ';', '|' };
	char d = ',';
	int best = -1;
	for (int k = 0; k < 4; k++) {
		int lines = 0, cnt0 = -1, consistent = 1;
		for (const char *s = text; *s && lines < 20; lines++) {
			int c = 0, q = 0;
			for (; *s && *s != '\n'; s++) {
				if (*s == '"') q = !q;
				if (!q && *s == cand[k]) c++;
			}
			if (*s) s++;
			if (cnt0 < 0) cnt0 = c;
			else if (c != cnt0) consistent = 0;
		}
		int score = cnt0 > 0 ? cnt0 * 2 + consistent * 1000 : -1;
		if (score > best) { best = score; d = cand[k]; }
	}
	char **f = xmalloc(4096 * sizeof *f);
	char *first = xstrdup(text), *p = first; /* sniff on a copy: splitting writes into the text */
	int nh = split_record(&p, d, f, 4096);
	if (nh <= 0) die(NULL, 0, "%s: empty file", name);
	/* header: first row has a non-numeric field where the second row is numeric */
	char *save = text + (p - first);
	char **g = xmalloc(4096 * sizeof *g);
	char *q = xstrdup(p), *qq = q;
	int ng = split_record(&qq, d, g, 4096);
	int header = 0;
	for (int i = 0; i < nh && i < ng; i++)
		if (!is_number(f[i], NULL) && *f[i] && is_number(g[i], NULL)) header = 1;
	if (ng <= 0) for (int i = 0; i < nh; i++) header |= !is_number(f[i], NULL) && *f[i]; /* single-row file */
	int *ci = xmalloc((size_t)nh * sizeof *ci);
	for (int i = 0; i < nh; i++) {
		char nm[300];
		if (header && *f[i]) {
			snprintf(nm, sizeof nm, "%.250s", f[i]);
			for (int k = 2, j = 0; j < t->ncols; j++) /* duplicate names stay separate columns */
				if (!strcmp(t->col[j].name, nm)) { snprintf(nm, sizeof nm, "%.250s_%d", f[i], k++); j = -1; }
			ci[i] = col_index(t, nm);
		} else {
			snprintf(nm, sizeof nm, "col%d", i);
			ci[i] = col_index(t, nm);
		}
	}
	if (!header) p = text; /* the first row is data */
	else p = save;
	for (;;) {
		int n = split_record(&p, d, f, 4096);
		if (n < 0) break;
		if (n == 1 && !*f[0]) continue; /* blank line */
		new_row(t);
		for (int i = 0; i < n && i < nh; i++) {
			Cell *c = cell(t, ci[i]);
			if (!*f[i]) c->t = CELL_NULL;
			else { c->t = CELL_STR; c->s = xstrdup(f[i]); }
		}
	}
}

static void load_npy(Table *t, const char *path)
{
	size_t len;
	char *raw = read_file(path, &len);
	const unsigned char *b = (const unsigned char *)raw;
	if (len < 10 || memcmp(b, "\x93NUMPY", 6)) die(NULL, 0, "%s: not a .npy file", path);
	size_t hl = b[6] == 1 ? (size_t)(b[8] | b[9] << 8) : (size_t)(b[8] | b[9] << 8 | b[10] << 16 | (size_t)b[11] << 24);
	size_t off = b[6] == 1 ? 10 : 12;
	if (off + hl > len) die(NULL, 0, "%s: truncated header", path);
	char *h = xmalloc(hl + 1);
	memcpy(h, raw + off, hl);
	char *ds = strstr(h, "'descr':"), *fo = strstr(h, "'fortran_order':"), *sh = strstr(h, "'shape':");
	if (!ds || !fo || !sh) die(NULL, 0, "%s: unsupported .npy header", path);
	char code[8] = { 0 };
	sscanf(strchr(ds + 8, '\'') + 1, "%7[^']", code);
	if (strstr(fo, "True")) die(NULL, 0, "%s: Fortran-order arrays are not supported", path);
	long d0 = 0, d1 = 1;
	char *ps = strchr(sh, '(');
	if (sscanf(ps, "(%ld, %ld)", &d0, &d1) < 1 && sscanf(ps, "(%ld,)", &d0) < 1) die(NULL, 0, "%s: unsupported shape", path);
	char kind = code[1];
	int size = atoi(code + 2);
	if (code[0] == '>' && size > 1) die(NULL, 0, "%s: big-endian arrays are not supported", path);
	if (!strchr("fiub", kind) || !(size == 1 || size == 2 || size == 4 || size == 8) || (kind == 'f' && size < 4))
		die(NULL, 0, "%s: unsupported dtype '%s'", path, code);
	const unsigned char *data = b + off + hl;
	if ((size_t)(d0 * d1 * size) > len - off - hl) die(NULL, 0, "%s: truncated data", path);
	int *ci = xmalloc((size_t)d1 * sizeof *ci);
	for (long c = 0; c < d1; c++) {
		char nm[32];
		snprintf(nm, sizeof nm, "col%ld", c);
		ci[c] = col_index(t, nm);
	}
	for (long r = 0; r < d0; r++) {
		new_row(t);
		for (long c = 0; c < d1; c++) {
			const unsigned char *e = data + (r * d1 + c) * size;
			uint64_t u = 0;
			for (int k = 0; k < size; k++) u |= (uint64_t)e[k] << (8 * k);
			double v;
			if (kind == 'f' && size == 4) { uint32_t w = (uint32_t)u; float x; memcpy(&x, &w, 4); v = x; }
			else if (kind == 'f') { double x; memcpy(&x, &u, 8); v = x; }
			else if (kind == 'i') { int64_t x = (int64_t)(u << (64 - 8 * size)) >> (64 - 8 * size); v = (double)x; }
			else v = (double)u;
			Cell *cc = cell(t, ci[c]);
			cc->t = CELL_NUM;
			cc->n = v;
		}
	}
}

/* ---- Hugging Face ------------------------------------------------------- */

/* Run curl without a shell; returns the response body or dies with a hint. */
static char *http_get(const char *url, const char *what)
{
	int fd[2];
	if (pipe(fd)) die(NULL, 0, "pipe failed");
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, fd[1], 1);
	posix_spawn_file_actions_addclose(&fa, fd[0]);
	char auth[600];
	const char *tok = getenv("HF_TOKEN");
	char *argv[16];
	int a = 0;
	argv[a++] = "curl";
	argv[a++] = "-sS";
	argv[a++] = "-f";
	argv[a++] = "-L";
	argv[a++] = "-m";
	argv[a++] = "120";
	argv[a++] = "--retry"; /* transient errors and rate limits (429, 5xx), with backoff */
	argv[a++] = "4";
	if (tok && *tok && strlen(tok) < 500) {
		snprintf(auth, sizeof auth, "Authorization: Bearer %s", tok);
		argv[a++] = "-H";
		argv[a++] = auth;
	}
	argv[a++] = (char *)url;
	argv[a] = NULL;
	pid_t pid;
	int rc = posix_spawnp(&pid, "curl", &fa, NULL, argv, environ);
	posix_spawn_file_actions_destroy(&fa);
	close(fd[1]);
	if (rc) die(NULL, 0, "cannot run curl (needed for hf: datasets): %s", strerror(rc));
	size_t cap = 1 << 16, n = 0;
	char *buf = xmalloc(cap);
	for (;;) {
		if (n + 1 >= cap) buf = xrealloc(buf, cap *= 2);
		ssize_t r = read(fd[0], buf + n, cap - n - 1);
		if (r < 0 && errno == EINTR) continue;
		if (r <= 0) break;
		n += (size_t)r;
	}
	buf[n] = 0;
	close(fd[0]);
	int st;
	waitpid(pid, &st, 0);
	if (!WIFEXITED(st) || WEXITSTATUS(st))
		die(NULL, 0, "fetching %s failed (curl exit %d); check the dataset id, network access%s", what,
		    WIFEXITED(st) ? WEXITSTATUS(st) : -1, tok ? "" : ", or set HF_TOKEN for gated datasets");
	return buf;
}

static void url_part(char *o, size_t n, const char *s)
{
	size_t k = 0;
	for (; *s && k + 4 < n; s++) {
		if (isalnum((unsigned char)*s) || strchr("-._", *s)) o[k++] = *s;
		else k += (size_t)snprintf(o + k, n - k, "%%%02X", (unsigned char)*s);
	}
	o[k] = 0;
}

static void load_hf(Table *t, const char *spec, int max_rows)
{
	char id[256] = "", config[128] = "", split[128] = "";
	const char *s = spec + 3;
	for (const char *c = s; *c; c++)
		if (!isalnum((unsigned char)*c) && !strchr("-._/", *c)) die(NULL, 0, "invalid character in dataset id '%s'", spec);
	/* OWNER/NAME[/CONFIG[/SPLIT]] (single-name datasets: NAME[/CONFIG[/SPLIT]] is ambiguous; owners are required) */
	char parts[4][128] = { "", "", "", "" };
	int np = 0;
	for (const char *c = s; *c && np < 4;) {
		const char *e = strchr(c, '/');
		size_t l = e ? (size_t)(e - c) : strlen(c);
		if (l >= sizeof parts[0]) die(NULL, 0, "dataset id too long");
		memcpy(parts[np], c, l);
		parts[np++][l] = 0;
		if (!e) break;
		c = e + 1;
	}
	if (np < 2) die(NULL, 0, "use hf:OWNER/NAME[/CONFIG[/SPLIT]], e.g. hf:scikit-learn/iris");
	snprintf(id, sizeof id, "%s/%s", parts[0], parts[1]);
	snprintf(config, sizeof config, "%s", parts[2]);
	snprintf(split, sizeof split, "%s", parts[3]);

	char eid[600], url[2048];
	url_part(eid, sizeof eid, id);
	if (!*config || !*split) { /* choose config (first) and split (train if present) */
		snprintf(url, sizeof url, "https://datasets-server.huggingface.co/splits?dataset=%s", eid);
		J *v;
		JP j = { http_get(url, id), "splits", 1 };
		v = jvalue(&j, 0);
		J *sp = jget(v, "splits");
		if (!sp || sp->k != J_ARR || !sp->len) die(NULL, 0, "%s: no splits available (the dataset may need conversion on the Hub)", id);
		for (int i = 0; i < sp->len; i++) {
			J *c = jget(sp->v[i], "config"), *p = jget(sp->v[i], "split");
			if (!c || !p) continue;
			if (*config && strcmp(c->s, config)) continue;
			if (!*config) snprintf(config, sizeof config, "%s", c->s);
			if (!strcmp(c->s, config) && (!*split || !strcmp(p->s, "train"))) {
				if (*split && strcmp(split, "train") && !strcmp(p->s, "train")) continue;
				snprintf(split, sizeof split, "%s", p->s);
			}
		}
		if (!*split) die(NULL, 0, "%s: config '%s' not found", id, config);
	}

	/* cache: one page response per line */
	const char *home = getenv("HOME");
	char dir[1024] = "", cache[2048] = "";
	const char *xdg = getenv("XDG_CACHE_HOME");
	if (xdg && *xdg) snprintf(dir, sizeof dir, "%s/technograd", xdg);
	else if (home && *home) snprintf(dir, sizeof dir, "%s/.cache/technograd", home);
	if (*dir) {
		mkdir(dir, 0755);
		char flat[600]; /* id 256 + config 128 + split 128 + separators and the row cap */
		snprintf(flat, sizeof flat, "%s__%s__%s__%d", id, config, split, max_rows);
		for (char *c = flat; *c; c++) if (*c == '/') *c = '_';
		snprintf(cache, sizeof cache, "%s/hf_%s.jsonl", dir, flat);
		FILE *f = fopen(cache, "rb");
		if (f) {
			fclose(f);
			char *text = read_file(cache, NULL);
			JP j = { text, cache, 1 };
			for (;;) {
				jws(&j);
				if (!*j.p) break;
				J *page = jvalue(&j, 0);
				J *rows = jget(page, "rows");
				for (int i = 0; rows && i < rows->len; i++) add_object(t, rows->v[i]);
			}
			fprintf(stderr, "data: %s/%s/%s: %d rows from cache %s\n", id, config, split, t->nrows, cache);
			return;
		}
	}
	char ec[300], es[300];
	url_part(ec, sizeof ec, config);
	url_part(es, sizeof es, split);
	char part[2100];
	snprintf(part, sizeof part, "%s.part", cache); /* renamed into place only when complete */
	FILE *out = *cache ? fopen(part, "wb") : NULL;
	long total = -1;
	for (long off = 0; off < max_rows && (total < 0 || off < total); off += 100) {
		long want = max_rows - off < 100 ? max_rows - off : 100;
		snprintf(url, sizeof url, "https://datasets-server.huggingface.co/rows?dataset=%s&config=%s&split=%s&offset=%ld&length=%ld",
			 eid, ec, es, off, want);
		char *body = http_get(url, id);
		JP j = { body, "rows", 1 };
		J *page = jvalue(&j, 0);
		J *rows = jget(page, "rows"), *tot = jget(page, "num_rows_total");
		J *err = jget(page, "error");
		if (err && err->k == J_STR) die(NULL, 0, "%s: the Hugging Face API says: %s", id, err->s);
		if (!rows || rows->k != J_ARR) die(NULL, 0, "%s: unexpected response from the rows API", id);
		if (tot && tot->k == J_NUM) total = (long)tot->n;
		for (int i = 0; i < rows->len; i++) add_object(t, rows->v[i]);
		if (out) { fputs(body, out); fputc('\n', out); }
		fprintf(stderr, "\rdata: fetching %s/%s/%s: %d of %ld rows", id, config, split, t->nrows,
			total < 0 ? (long)max_rows : (total < max_rows ? total : (long)max_rows));
		if (!rows->len) break;
	}
	fputc('\n', stderr);
	if (out && (fclose(out) || rename(part, cache))) remove(part);
}

/* ---- loading ------------------------------------------------------------ */

Table *data_load(const char *src, int max_rows)
{
	Table *t = xmalloc(sizeof *t);
	t->source = xstrdup(src);
	if (!strncmp(src, "hf:", 3)) {
		t->format = "huggingface";
		load_hf(t, src, max_rows);
	} else if (ends_with(src, ".npy")) {
		t->format = "npy";
		load_npy(t, src);
	} else if (ends_with(src, ".bin") || ends_with(src, ".f32")) {
		die(NULL, 0, "%s: raw f32 has no columns; convert it with numpy (.npy) or use tgc batch directly", src);
	} else {
		size_t len;
		char *text = read_file(src, &len);
		const char *p = text;
		while (isspace((unsigned char)*p)) p++;
		int jl = ends_with(src, ".jsonl") || ends_with(src, ".ndjson");
		if (!jl && *p == '{') { /* one object per line, or a single object holding rows */
			const char *nl = strchr(p, '\n');
			while (nl && isspace((unsigned char)*nl)) nl++;
			jl = nl && *nl == '{';
		}
		if (jl) { t->format = "jsonl"; load_json_text(t, text, src, 1); }
		else if (*p == '{' || *p == '[') { t->format = "json"; load_json_text(t, text, src, 0); }
		else { t->format = "delimited text"; load_delimited(t, text, src); }
	}
	if (t->nrows > max_rows) t->nrows = max_rows;
	if (!t->nrows) die(NULL, 0, "%s: no rows", src);
	return t;
}

/* ---- column analysis ---------------------------------------------------- */

static int cell_num(const Cell *c, double *v)
{
	if (c->t == CELL_NUM) { *v = c->n; return 1; }
	if (c->t == CELL_STR) return is_number(c->s, v);
	return 0;
}

static const char *cell_str(const Cell *c, char *buf, size_t n)
{
	if (c->t == CELL_STR) return c->s;
	if (c->t == CELL_NUM) { snprintf(buf, n, "%.10g", c->n); return buf; }
	return "";
}

static int levels_add(Col *c, const char *s)
{
	for (int i = 0; i < c->nlevels; i++)
		if (!strcmp(c->levels[i], s)) { c->counts[i]++; return i; }
	if (c->nlevels >= 100000) return -1;
	c->levels = xrealloc(c->levels, (size_t)(c->nlevels + 1) * sizeof *c->levels);
	c->counts = xrealloc(c->counts, (size_t)(c->nlevels + 1) * sizeof *c->counts);
	c->levels[c->nlevels] = xstrdup(s);
	c->counts[c->nlevels] = 1;
	return c->nlevels++;
}

static int is_id_name(const char *n)
{
	char l[64];
	size_t i = 0;
	for (; n[i] && i < sizeof l - 1; i++) l[i] = (char)tolower((unsigned char)n[i]);
	l[i] = 0;
	return !strcmp(l, "id") || !strcmp(l, "idx") || !strcmp(l, "index") || !strcmp(l, "row_idx") || !strcmp(l, "uuid") ||
	       !strcmp(l, "unnamed: 0") || (i > 3 && !strcmp(l + i - 3, "_id"));
}

void data_analyze(Table *t)
{
	for (int ci = 0; ci < t->ncols; ci++) {
		Col *c = &t->col[ci];
		int nnull = 0, nnum = 0, nstr = 0, nlist = 0, nother = 0, listlen = -1, listok = 1, allint = 1, increasing = 1;
		double prev = -INFINITY, v;
		size_t textlen = 0;
		char buf[64];
		for (int r = 0; r < t->nrows; r++) {
			Cell *x = &c->cell[r];
			if (x->t == CELL_NULL) { nnull++; continue; }
			if (x->t == CELL_OTHER) { nother++; continue; }
			if (x->t == CELL_LIST) {
				nlist++;
				if (listlen < 0) listlen = x->nl;
				else if (x->nl != listlen) listok = 0;
				continue;
			}
			if (cell_num(x, &v)) {
				nnum++;
				if (v != floor(v)) allint = 0;
				if (v <= prev) increasing = 0;
				prev = v;
			} else {
				nstr++;
				textlen += strlen(x->s);
			}
		}
		int nval = t->nrows - nnull;
		c->missing = nnull;
		if (nval == 0) { c->kind = COL_DROP; c->why = "empty"; continue; }
		if (nother) { c->kind = COL_DROP; c->why = "nested objects (images, audio, ...)"; continue; }
		if (nlist) {
			if (nlist != nval || !listok || listlen > 4096) { c->kind = COL_DROP; c->why = "lists of varying length"; continue; }
			c->kind = COL_VEC;
			c->veclen = listlen;
			continue;
		}
		for (int r = 0; r < t->nrows; r++)
			if (c->cell[r].t != CELL_NULL) levels_add(c, cell_str(&c->cell[r], buf, sizeof buf));
		if (nnum == nval) {
			if (is_id_name(c->name) || (allint && increasing && c->nlevels == nval && nval > 20)) {
				c->kind = COL_DROP;
				c->why = "identifier";
				continue;
			}
			c->kind = COL_NUM;
			c->intlike = allint && c->nlevels <= 20;
			continue;
		}
		double avg = (double)textlen / (nstr ? nstr : 1);
		if (c->nlevels <= 64 || (c->nlevels <= 0.05 * nval && avg < 40)) {
			c->kind = COL_CAT;
		} else if (avg >= 15 || c->nlevels > 0.5 * nval) {
			if (is_id_name(c->name) || (c->nlevels == nval && avg < 15)) { c->kind = COL_DROP; c->why = "identifier"; continue; }
			c->kind = COL_TEXT;
		} else {
			c->kind = COL_CAT;
		}
	}
}

int data_pick_target(const Table *t, const char *want)
{
	if (want) {
		for (int i = 0; i < t->ncols; i++)
			if (!strcmp(t->col[i].name, want)) return i;
		die(NULL, 0, "no column named '%s'", want);
	}
	static const char *names[] = { "label", "labels", "target", "class", "y", "species", "category", "outcome",
				       "output", "sentiment", "answer", "price", "score", NULL };
	for (int k = 0; names[k]; k++)
		for (int i = 0; i < t->ncols; i++) {
			const char *n = t->col[i].name;
			size_t j = 0;
			while (n[j] && names[k][j] && tolower((unsigned char)n[j]) == names[k][j]) j++;
			if (!n[j] && !names[k][j] && t->col[i].kind != COL_DROP && t->col[i].kind != COL_TEXT) return i;
		}
	for (int i = t->ncols - 1; i >= 0; i--)
		if (t->col[i].kind == COL_NUM || t->col[i].kind == COL_CAT) return i;
	die(NULL, 0, "no usable target column; name one with --target");
	return -1;
}

static const char *kind_name(ColKind k)
{
	static const char *n[] = { "drop", "numeric", "categorical", "text", "vector" };
	return n[k];
}

void data_describe(const Table *t, int target, FILE *f)
{
	char buf[64];
	fprintf(f, "source   %s\nformat   %s\nrows     %d\ncolumns  %d\n\n", t->source, t->format, t->nrows, t->ncols);
	fprintf(f, "%-24s %-12s %8s  %s\n", "column", "kind", "distinct", "examples / notes");
	for (int i = 0; i < t->ncols; i++) {
		const Col *c = &t->col[i];
		char ex[160] = "";
		size_t o = 0;
		if (c->kind == COL_DROP) snprintf(ex, sizeof ex, "dropped: %s", c->why);
		else if (c->kind == COL_VEC) snprintf(ex, sizeof ex, "%d numbers per row", c->veclen);
		else
			for (int r = 0, k = 0; r < t->nrows && k < 3 && o < sizeof ex - 40; r++) {
				const Cell *x = &c->cell[r];
				if (x->t == CELL_NULL) continue;
				o += (size_t)snprintf(ex + o, sizeof ex - o, "%s%.24s", k++ ? " | " : "", cell_str(x, buf, sizeof buf));
			}
		fprintf(f, "%-24.24s %-12s %8d  %s%s%s\n", c->name, kind_name(c->kind), c->nlevels, ex,
			c->missing ? "  (has missing)" : "", i == target ? "   <- target" : "");
	}
	if (target >= 0) {
		const Col *c = &t->col[target];
		int cls = c->kind == COL_CAT || (c->kind == COL_NUM && c->intlike);
		fprintf(f, "\ntask     %s on '%s'", cls ? "classification" : "regression", c->name);
		if (cls) fprintf(f, " (%d classes)", c->nlevels);
		fputc('\n', f);
	}
}

/* ---- featurization ------------------------------------------------------ */

#define MAX_ONEHOT 32

static void pct(FILE *f, const char *s) /* level names are percent-escaped in the spec */
{
	for (; *s; s++) {
		if (*s == '%' || *s == '\t' || *s == '\n' || *s == '\r' || *s == ' ') fprintf(f, "%%%02X", (unsigned char)*s);
		else fputc(*s, f);
	}
}

static char *unpct(const char *s)
{
	char *o = xmalloc(strlen(s) + 1), *p = o;
	for (; *s; s++) {
		if (*s == '%' && isxdigit((unsigned char)s[1]) && isxdigit((unsigned char)s[2])) {
			char h[3] = { s[1], s[2], 0 };
			*p++ = (char)strtol(h, NULL, 16);
			s += 2;
		} else *p++ = *s;
	}
	*p = 0;
	return o;
}

/* Fit a spec on the given rows (the training split). */
Spec *spec_fit(const Table *t, int target, const int *rows, int nrows, int text_dim)
{
	Spec *s = xmalloc(sizeof *s);
	s->target = xstrdup(t->col[target].name);
	const Col *tc = &t->col[target];
	double v;
	s->classify = tc->kind == COL_CAT || (tc->kind == COL_NUM && tc->intlike);
	if (s->classify) {
		if (tc->nlevels < 2) die(NULL, 0, "target '%s' has a single value; nothing to learn", tc->name);
		if (tc->nlevels > 1000) die(NULL, 0, "target '%s' has %d classes; pick another column with --target", tc->name, tc->nlevels);
		s->nclass = tc->nlevels;
		s->classes = xmalloc((size_t)s->nclass * sizeof *s->classes);
		for (int i = 0; i < s->nclass; i++) s->classes[i] = xstrdup(tc->levels[i]);
	} else {
		double sum = 0, sq = 0;
		int n = 0;
		for (int i = 0; i < nrows; i++)
			if (cell_num(&tc->cell[rows[i]], &v)) { sum += v; sq += v * v; n++; }
		s->tmean = n ? sum / n : 0;
		s->tstd = n ? sqrt(fmax(sq / n - s->tmean * s->tmean, 0)) : 1;
		if (s->tstd < 1e-12) s->tstd = 1;
	}
	for (int ci = 0; ci < t->ncols; ci++) {
		const Col *c = &t->col[ci];
		if (ci == target || c->kind == COL_DROP) continue;
		s->f = xrealloc(s->f, (size_t)(s->nf + 1) * sizeof *s->f);
		Feat *f = &s->f[s->nf++];
		memset(f, 0, sizeof *f);
		f->name = xstrdup(c->name);
		f->kind = c->kind;
		int w = c->kind == COL_VEC ? c->veclen : 1;
		if (c->kind == COL_NUM || c->kind == COL_VEC) {
			f->dim = w;
			if (c->kind == COL_NUM && c->missing) { f->flag = 1; f->dim = w + 1; } /* missing indicator */
			f->mean = xmalloc((size_t)w * sizeof *f->mean);
			f->std = xmalloc((size_t)w * sizeof *f->std);
			for (int k = 0; k < w; k++) {
				double sum = 0, sq = 0;
				int n = 0;
				for (int i = 0; i < nrows; i++) {
					const Cell *x = &c->cell[rows[i]];
					if (c->kind == COL_VEC ? x->t == CELL_LIST && (v = x->l[k], 1) : cell_num(x, &v)) { sum += v; sq += v * v; n++; }
				}
				f->mean[k] = n ? sum / n : 0;
				f->std[k] = n ? sqrt(fmax(sq / n - f->mean[k] * f->mean[k], 0)) : 1;
				if (f->std[k] < 1e-12) f->std[k] = 1;
			}
		} else if (c->kind == COL_CAT) { /* most frequent levels; the rest share all-zero */
			int *idx = xmalloc((size_t)c->nlevels * sizeof *idx);
			for (int i = 0; i < c->nlevels; i++) idx[i] = i;
			for (int i = 1; i < c->nlevels; i++) /* insertion sort by count, stable */
				for (int j = i; j > 0 && c->counts[idx[j]] > c->counts[idx[j - 1]]; j--) { int x = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = x; }
			f->nlev = c->nlevels < MAX_ONEHOT ? c->nlevels : MAX_ONEHOT;
			f->levels = xmalloc((size_t)f->nlev * sizeof *f->levels);
			for (int i = 0; i < f->nlev; i++) f->levels[i] = xstrdup(c->levels[idx[i]]);
			f->dim = f->nlev;
			xfree(idx);
		} else { /* text: signed hashed word buckets */
			f->dim = text_dim;
		}
		s->dim += f->dim;
	}
	if (!s->dim) die(NULL, 0, "no usable feature columns besides the target");
	return s;
}

static uint32_t fnv(const unsigned char *s, int n)
{
	uint32_t h = 2166136261u;
	for (int i = 0; i < n; i++) h = (h ^ s[i]) * 16777619u;
	return h;
}

/* Featurize one row; x has s->dim floats. Returns 0 if a target is required
 * but missing/unknown, otherwise 1; y (if non-NULL) gets the target encoding. */
int spec_apply(const Spec *s, const Table *t, int row, float *x, float *y, int *label)
{
	char buf[64];
	double v;
	int o = 0;
	for (int k = 0; k < s->nf; k++) {
		const Feat *f = &s->f[k];
		int ci = -1;
		for (int i = 0; i < t->ncols; i++)
			if (!strcmp(t->col[i].name, f->name)) ci = i;
		const Cell *c = ci >= 0 ? &t->col[ci].cell[row] : NULL;
		for (int d = 0; d < f->dim; d++) x[o + d] = 0;
		if (f->flag && (!c || c->t == CELL_NULL || !cell_num(c, &v))) x[o + 1] = 1;
		if (c && c->t != CELL_NULL) {
			if (f->kind == COL_NUM) {
				if (cell_num(c, &v)) x[o] = (float)((v - f->mean[0]) / f->std[0]);
			} else if (f->kind == COL_VEC) {
				if (c->t == CELL_LIST && c->nl == f->dim)
					for (int d = 0; d < f->dim; d++) x[o + d] = (float)((c->l[d] - f->mean[d]) / f->std[d]);
			} else if (f->kind == COL_CAT) {
				const char *str = cell_str(c, buf, sizeof buf);
				for (int i = 0; i < f->nlev; i++)
					if (!strcmp(f->levels[i], str)) x[o + i] = 1;
			} else if (f->kind == COL_TEXT && c->t == CELL_STR) {
				/* words (lowercase alphanumerics and apostrophes) are hashed into dim
				 * buckets with a hash-derived sign (feature hashing), then L2-normalized.
				 * Unigrams only: bigrams measured worse at these data sizes. */
				for (const char *p = c->s; *p;) {
					while (*p && !isalnum((unsigned char)*p)) p++;
					if (!*p) break;
					unsigned char w[64];
					int n = 0;
					while (*p && (isalnum((unsigned char)*p) || *p == '\'')) {
						if (n < 64) w[n++] = (unsigned char)tolower((unsigned char)*p);
						p++;
					}
					uint32_t h = fnv(w, n);
					x[o + h % (uint32_t)f->dim] += (h >> 31) ? -1.0f : 1.0f;
				}
				double norm = 0;
				for (int d = 0; d < f->dim; d++) norm += x[o + d] * x[o + d];
				if (norm > 0)
					for (int d = 0; d < f->dim; d++) x[o + d] = (float)(x[o + d] / sqrt(norm));
			}
		}
		o += f->dim;
	}
	if (!y && !label) return 1;
	int ti = -1;
	for (int i = 0; i < t->ncols; i++)
		if (!strcmp(t->col[i].name, s->target)) ti = i;
	if (ti < 0 || t->col[ti].cell[row].t == CELL_NULL) return 0;
	const Cell *c = &t->col[ti].cell[row];
	if (s->classify) {
		const char *str = cell_str(c, buf, sizeof buf);
		int cls = -1;
		for (int i = 0; i < s->nclass; i++)
			if (!strcmp(s->classes[i], str)) cls = i;
		if (cls < 0) return 0;
		if (y) for (int i = 0; i < s->nclass; i++) y[i] = i == cls ? 1.0f : 0.0f;
		if (label) *label = cls;
		return 1;
	}
	if (!cell_num(c, &v)) return 0;
	if (y) y[0] = (float)((v - s->tmean) / s->tstd);
	return 1;
}

void spec_save(const Spec *s, const char *path)
{
	FILE *f = fopen(path, "w");
	if (!f) die(NULL, 0, "cannot write '%s'", path);
	fprintf(f, "technograd-features 1\ntarget ");
	pct(f, s->target);
	if (s->classify) {
		fprintf(f, " classification %d", s->nclass);
		for (int i = 0; i < s->nclass; i++) { fputc(' ', f); pct(f, s->classes[i]); }
	} else {
		fprintf(f, " regression %.17g %.17g", s->tmean, s->tstd);
	}
	fputc('\n', f);
	for (int k = 0; k < s->nf; k++) {
		const Feat *x = &s->f[k];
		fprintf(f, "feature ");
		pct(f, x->name);
		fprintf(f, " %s %d", kind_name(x->kind), x->dim);
		if (x->kind == COL_NUM && x->flag) fprintf(f, " %.17g %.17g missing", x->mean[0], x->std[0]);
		else if (x->kind == COL_NUM || x->kind == COL_VEC)
			for (int d = 0; d < x->dim; d++) fprintf(f, " %.17g %.17g", x->mean[d], x->std[d]);
		if (x->kind == COL_CAT)
			for (int i = 0; i < x->nlev; i++) { fputc(' ', f); pct(f, x->levels[i]); }
		fputc('\n', f);
	}
	if (fclose(f)) die(NULL, 0, "write failed '%s'", path);
}

Spec *spec_load(const char *path)
{
	char *text = read_file(path, NULL), *save;
	Spec *s = xmalloc(sizeof *s);
	int line = 0;
	for (char *ln = strtok_r(text, "\n", &save); ln; ln = strtok_r(NULL, "\n", &save)) {
		line++;
		char *tok[8192], *s2;
		int n = 0;
		for (char *w = strtok_r(ln, " ", &s2); w && n < 8192; w = strtok_r(NULL, " ", &s2)) tok[n++] = w;
		if (n == 0) continue; /* blank or whitespace-only line */
		if (line == 1) {
			if (n < 2 || strcmp(tok[0], "technograd-features")) die(path, 1, "not a feature spec");
			continue;
		}
		if (!strcmp(tok[0], "target") && n >= 3) {
			s->target = unpct(tok[1]);
			s->classify = !strcmp(tok[2], "classification");
			if (s->classify) {
				s->nclass = atoi(tok[3]);
				if (n != 4 + s->nclass) die(path, line, "malformed target line");
				s->classes = xmalloc((size_t)s->nclass * sizeof *s->classes);
				for (int i = 0; i < s->nclass; i++) s->classes[i] = unpct(tok[4 + i]);
			} else {
				s->tmean = atof(tok[3]);
				s->tstd = atof(tok[4]);
			}
		} else if (!strcmp(tok[0], "feature") && n >= 4) {
			s->f = xrealloc(s->f, (size_t)(s->nf + 1) * sizeof *s->f);
			Feat *f = &s->f[s->nf++];
			memset(f, 0, sizeof *f);
			f->name = unpct(tok[1]);
			for (int k = 0; k < 5; k++)
				if (!strcmp(tok[2], kind_name((ColKind)k))) f->kind = (ColKind)k;
			f->dim = atoi(tok[3]);
			if (f->kind == COL_NUM || f->kind == COL_VEC) {
				if (f->kind == COL_NUM && n == 7 && !strcmp(tok[6], "missing")) {
					f->flag = 1;
					f->mean = xmalloc(sizeof *f->mean);
					f->std = xmalloc(sizeof *f->std);
					f->mean[0] = atof(tok[4]);
					f->std[0] = atof(tok[5]);
					s->dim += f->dim;
					continue;
				}
				if (n != 4 + 2 * f->dim) die(path, line, "malformed feature line");
				f->mean = xmalloc((size_t)f->dim * sizeof *f->mean);
				f->std = xmalloc((size_t)f->dim * sizeof *f->std);
				for (int d = 0; d < f->dim; d++) { f->mean[d] = atof(tok[4 + 2 * d]); f->std[d] = atof(tok[5 + 2 * d]); }
			} else if (f->kind == COL_CAT) {
				f->nlev = n - 4;
				f->levels = xmalloc((size_t)(f->nlev ? f->nlev : 1) * sizeof *f->levels);
				for (int i = 0; i < f->nlev; i++) f->levels[i] = unpct(tok[4 + i]);
			}
			s->dim += f->dim;
		} else {
			die(path, line, "malformed feature spec line");
		}
	}
	if (!s->target || !s->dim) die(path, 1, "incomplete feature spec");
	return s;
}

/* Drop feature columns that give the answer away: a categorical (or few-valued
 * numeric) column with as many distinct values as the target, each recurring,
 * whose every value maps to exactly one target value is a renamed copy of the
 * target, e.g. `label_text` next to `label`. Returns the number dropped. */
int data_drop_leaks(Table *t, int target)
{
	Col *tc = &t->col[target];
	char b1[64], b2[64];
	int dropped = 0;
	for (int ci = 0; ci < t->ncols; ci++) {
		Col *c = &t->col[ci];
		if (ci == target || !(c->kind == COL_CAT || (c->kind == COL_NUM && c->nlevels <= 64)) || c->nlevels < 2) continue;
		/* a renamed copy maps one-to-one onto the target values, and each value
		 * recurs; a column of mostly-unique values separates any small sample */
		if (c->nlevels != tc->nlevels) continue;
		int rare = 0;
		for (int i = 0; i < c->nlevels; i++) rare |= c->counts[i] < 2;
		if (rare) continue;
		int *map = xmalloc((size_t)c->nlevels * sizeof *map), ok = 1, pairs = 0;
		for (int i = 0; i < c->nlevels; i++) map[i] = -1;
		for (int r = 0; r < t->nrows && ok; r++) {
			const Cell *x = &c->cell[r], *y = &tc->cell[r];
			if (x->t == CELL_NULL || y->t == CELL_NULL) continue;
			const char *xs = cell_str(x, b1, sizeof b1), *ys = cell_str(y, b2, sizeof b2);
			int li = -1, ti = -1;
			for (int i = 0; i < c->nlevels; i++)
				if (!strcmp(c->levels[i], xs)) li = i;
			for (int i = 0; i < tc->nlevels; i++)
				if (!strcmp(tc->levels[i], ys)) ti = i;
			if (li < 0 || ti < 0) continue;
			if (map[li] < 0) map[li] = ti;
			else if (map[li] != ti) ok = 0;
			pairs++;
		}
		xfree(map);
		if (ok && pairs > 0) {
			c->kind = COL_DROP;
			c->why = "determines the target exactly (would leak the answer)";
			dropped++;
		}
	}
	return dropped;
}
