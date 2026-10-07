/* tgc: the Technograd compiler driver. */
#include "tg.h"

#include <stdlib.h>
#include <string.h>

static void usage(void)
{
	fputs("usage: tgc <command> <file.tg|file.tgir> [args]\n"
	      "  check <file>              parse, lower and verify\n"
	      "  ir    <file>              print canonical TGIR (S-expressions)\n"
	      "  plan  <file>              print the static memory plan\n"
	      "  run   <file> <in>... [-o out]   interpret; each input is comma-separated values or @file\n"
	      "  batch <file> <data> [-o out]    one sample per text row, or a raw .bin/.f32 stream\n"
	      "  c     <file> [-o out.c]   emit a freestanding C unit\n"
	      "  fix   <file> [-o out.tg]  repair compile errors with the neural-forest fixer\n"
	      "  fixer-train -o forest.tg <corpus.tg>...   train the fixer (self-supervised)\n"
	      "  fixer-eval <corpus.tg>...                 measure repair rates on corrupted programs\n"
	      "options:\n"
	      "  --autofix   on a compile error, repair in memory and continue (source is not modified)\n"
	      "  -v, --verbose   trace compiler stages and think loops to stderr; -vv adds every VM value\n"
	      "  --log FILE  append a JSON Lines log of all events and errors (debug events need -vv)\n",
	      stderr);
	exit(2);
}

static void parse_input(const char *s, float *v, int n, const char *name)
{
	int k = 0;
	char *e;
	for (;;) {
		if (k == n) die(NULL, 0, "input '%s' has more than %d values", name, n);
		v[k++] = strtof(s, &e);
		if (e == s) die(NULL, 0, "input '%s': bad number near '%s'", name, s);
		if (!*e) break;
		if (*e != ',') die(NULL, 0, "input '%s': expected ',' near '%s'", name, e);
		s = e + 1;
	}
	if (k != n) die(NULL, 0, "input '%s' needs %d values, got %d", name, n, k);
}

/* A trailing `-o PATH` pair, removed from argv. */
static const char *take_output(int *argc, char **argv)
{
	if (*argc >= 5 && strcmp(argv[*argc - 2], "-o") == 0) {
		*argc -= 2;
		return argv[*argc + 1];
	}
	return NULL;
}

static FILE *open_out(const char *path)
{
	FILE *f = fopen(path, io_is_binary(path) ? "wb" : "w");
	if (!f) die(NULL, 0, "cannot write '%s'", path);
	return f;
}

static void close_out(FILE *f, const char *path)
{
	if (ferror(f) | fclose(f)) die(NULL, 0, "write failed '%s'", path);
}

/* An input is inline comma-separated values, or @PATH to read them from a file. */
static const float *load_input(const char *arg, const Value *x)
{
	int n = shape_numel(&x->sh);
	if (arg[0] == '@') {
		int cnt;
		float *v = io_load(arg + 1, &cnt);
		if (cnt != n) die(NULL, 0, "input '%s' needs %d values, '%s' has %d", x->name, n, arg + 1, cnt);
		return v;
	}
	float *buf = xmalloc((size_t)n * sizeof *buf);
	parse_input(arg, buf, n, x->name);
	return buf;
}

static const Ins *find_think(const Block *b, int tid)
{
	for (int i = 0; i < b->len; i++) {
		const Ins *in = &b->v[i];
		if (in->op != OP_THINK) continue;
		if (in->tid == tid) return in;
		const Ins *r = find_think(in->body, tid);
		if (r) return r;
	}
	return NULL;
}

static void trace_plan(const Module *m, double ms)
{
	int unshared = 0;
	for (int v = 0; v < m->nval; v++)
		if (m->val[v].kind == V_TMP && m->val[v].def >= 0)
			unshared += (shape_numel(&m->val[v].sh) + TG_ALIGN - 1) / TG_ALIGN * TG_ALIGN;
	tr_begin(1, "plan");
	tr_num("arena_bytes", m->arena * 4);
	tr_num("unshared_bytes", unshared * 4);
	tr_num("ms", ms);
	tr_end("arena %d bytes (%d unshared, %.0f%% saved) in %.2f ms", m->arena * 4, unshared * 4,
	       unshared ? 100.0 * (unshared - m->arena) / unshared : 0.0, ms);
}

static void trace_run(const Module *m, const int *steps, double ms)
{
	tr_begin(1, "run");
	tr_str("model", m->name);
	tr_num("ms", ms);
	tr_end("vm executed %s in %.3f ms", m->name, ms);
	for (int t = 0; t < m->nthink; t++) {
		const Ins *in = find_think(&m->top, t);
		float d = vm_last_delta(t);
		int halted = in->eps >= 0 && d <= in->eps;
		tr_begin(1, "think");
		tr_num("loop", t);
		tr_num("steps", steps[t]);
		tr_num("budget", in->maxit);
		tr_num("final_delta", d);
		tr_str("stop", halted ? "converged" : "budget");
		tr_end("think[%d]: %d/%d iteration(s), final delta %.3g, %s", t, steps[t], in->maxit, (double)d,
		       halted ? "converged" : in->eps >= 0 ? "budget exhausted before convergence" : "fixed budget");
	}
}

/* Per-loop step statistics over a batch. */
typedef struct {
	int rows;
	long *sum;
	int *mn, *mx;
} Acc;

static void acc_add(Acc *a, const int *steps, int nthink)
{
	for (int t = 0; t < nthink; t++) {
		a->sum[t] += steps[t];
		if (!a->rows || steps[t] < a->mn[t]) a->mn[t] = steps[t];
		if (!a->rows || steps[t] > a->mx[t]) a->mx[t] = steps[t];
	}
	a->rows++;
}

static void trace_batch(const Module *m, const Acc *a, double ms)
{
	tr_begin(1, "batch");
	tr_num("rows", a->rows);
	tr_num("ms", ms);
	tr_num("us_per_row", a->rows ? 1e3 * ms / a->rows : 0);
	tr_end("%d row(s) in %.2f ms (%.1f us/row)", a->rows, ms, a->rows ? 1e3 * ms / a->rows : 0.0);
	for (int t = 0; t < m->nthink && a->rows; t++) {
		tr_begin(1, "think");
		tr_num("loop", t);
		tr_num("steps_min", a->mn[t]);
		tr_num("steps_mean", (double)a->sum[t] / a->rows);
		tr_num("steps_max", a->mx[t]);
		tr_end("think[%d] over batch: steps min %d mean %.2f max %d", t, a->mn[t], (double)a->sum[t] / a->rows, a->mx[t]);
	}
}

static int is_ir(const char *src)
{
	while (*src == ' ' || *src == '\t' || *src == '\n' || *src == '\r') src++;
	return *src == '(' || *src == ';';
}

static Module *load_fixed(const char *path)
{
	char *src = read_file(path, NULL);
	if (is_ir(src)) return load_module(path);
	int n;
	char *fixed = autofix(src, path, 1, &n);
	if (!fixed) return load_module(path); /* reports the original error */
	if (n) fprintf(stderr, "autofix: applied %d repair(s) in memory; run `tgc fix %s -o <out>` to keep them\n", n, path);
	return lower(surface_parse(fixed, path), path);
}

int main(int argc, char **argv)
{
	int fix = 0, k = 1;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--autofix") == 0) fix = 1;
		else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) tg_verbose = tg_verbose > 1 ? tg_verbose : 1;
		else if (strcmp(argv[i], "-vv") == 0) tg_verbose = 2;
		else if (strcmp(argv[i], "--log") == 0) {
			if (++i == argc) usage();
			tr_open_log(argv[i]);
			atexit(tr_close_log);
		} else argv[k++] = argv[i];
	}
	argc = k;
	if (argc < 3) usage();
	const char *cmd = argv[1];

	if (strcmp(cmd, "fixer-train") == 0) {
		if (argc < 5 || strcmp(argv[2], "-o") != 0) usage();
		return fixer_train(argv[3], argv + 4, argc - 4);
	}
	if (strcmp(cmd, "fixer-eval") == 0) return fixer_eval(argv + 2, argc - 2);
	if (strcmp(cmd, "fix") == 0) {
		if (argc != 3 && !(argc == 5 && strcmp(argv[3], "-o") == 0)) usage();
		char *src = read_file(argv[2], NULL);
		if (is_ir(src)) die(NULL, 0, "fix works on surface (.tg) programs");
		int n;
		char *fixed = autofix(src, argv[2], 1, &n);
		if (!fixed) return 1;
		FILE *f = argc == 5 ? fopen(argv[4], "w") : stdout;
		if (!f) die(NULL, 0, "cannot write '%s'", argv[4]);
		fputs(fixed, f);
		if (f != stdout && fclose(f) != 0) die(NULL, 0, "write failed '%s'", argv[4]);
		fprintf(stderr, "autofix: %d repair(s)\n", n);
		return 0;
	}

	Module *m = fix ? load_fixed(argv[2]) : load_module(argv[2]);
	double tp = tr_now_ms();
	plan(m);
	m->trace = 1;
	if (tr_on(1)) trace_plan(m, tr_now_ms() - tp);

	if (strcmp(cmd, "check") == 0) {
		if (argc != 3) usage();
		printf("ok: model %s, %d input(s), %d think loop(s), arena %d bytes\n", m->name, m->ninputs, m->nthink, m->arena * 4);
	} else if (strcmp(cmd, "ir") == 0) {
		if (argc != 3) usage();
		ir_write(m, stdout);
	} else if (strcmp(cmd, "plan") == 0) {
		if (argc != 3) usage();
		plan_dump(m, stdout);
	} else if (strcmp(cmd, "run") == 0) {
		const char *opath = take_output(&argc, argv);
		if (argc != 3 + m->ninputs) die(NULL, 0, "model %s takes %d input(s), got %d", m->name, m->ninputs, argc - 3);
		const float **in = xmalloc((size_t)m->ninputs * sizeof *in);
		for (int i = 0; i < m->ninputs; i++) in[i] = load_input(argv[3 + i], &m->val[m->inputs[i]]);
		int n = shape_numel(&m->val[m->output].sh);
		float *out = xmalloc((size_t)n * sizeof *out);
		int *steps = xmalloc((size_t)(m->nthink ? m->nthink : 1) * sizeof *steps);
		double t0 = tr_now_ms();
		vm_run(m, in, out, steps);
		if (tr_on(1)) trace_run(m, steps, tr_now_ms() - t0);
		if (opath) {
			int bin = io_is_binary(opath);
			FILE *f = open_out(opath);
			io_write_row(f, bin, out, n, steps, bin ? 0 : m->nthink);
			close_out(f, opath);
		} else {
			for (int i = 0; i < n; i++) printf("%s%.9g", i ? " " : "", (double)out[i]);
			putchar('\n');
			for (int i = 0; i < m->nthink; i++) printf("steps %d %d\n", i, steps[i]);
		}
	} else if (strcmp(cmd, "batch") == 0) {
		const char *opath = take_output(&argc, argv);
		if (argc != 4) usage();
		const char *data = argv[3];
		int per = 0, *off = xmalloc((size_t)m->ninputs * sizeof *off);
		for (int i = 0; i < m->ninputs; i++) {
			off[i] = per;
			per += shape_numel(&m->val[m->inputs[i]].sh);
		}
		int n = shape_numel(&m->val[m->output].sh);
		float *out = xmalloc((size_t)n * sizeof *out);
		int *steps = xmalloc((size_t)(m->nthink ? m->nthink : 1) * sizeof *steps);
		const float **in = xmalloc((size_t)m->ninputs * sizeof *in);
		int obin = opath && io_is_binary(opath);
		int nt = m->nthink ? m->nthink : 1;
		Acc acc = { 0, xmalloc((size_t)nt * sizeof(long)), xmalloc((size_t)nt * sizeof(int)), xmalloc((size_t)nt * sizeof(int)) };
		double tb = tr_now_ms();
		FILE *f = NULL; /* opened only after the whole input validates */
		if (io_is_binary(data)) {
			int cnt;
			float *all = io_load(data, &cnt);
			if (cnt % per) die(NULL, 0, "'%s': %d values is not a multiple of the %d per sample", data, cnt, per);
			f = opath ? open_out(opath) : stdout;
			for (int r = 0; r < cnt / per; r++) {
				for (int i = 0; i < m->ninputs; i++) in[i] = all + (size_t)r * (size_t)per + off[i];
				vm_run(m, in, out, steps);
				acc_add(&acc, steps, m->nthink);
				io_write_row(f, obin, out, n, steps, obin ? 0 : m->nthink);
			}
		} else {
			char *raw = read_file(data, NULL);
			float *row = xmalloc((size_t)per * sizeof *row);
			for (int i = 0; i < m->ninputs; i++) in[i] = row + off[i];
			/* pass 0 validates every row; pass 1 computes, so a bad row yields no output */
			for (int pass = 0; pass < 2; pass++) {
				if (pass == 1) f = opath ? open_out(opath) : stdout;
				int line = 1;
				for (char *s = raw; *s; line++) {
					char *nl = strchr(s, '\n');
					size_t len = nl ? (size_t)(nl - s) : strlen(s);
					char save = s[len];
					s[len] = 0;
					const char *bad;
					int k = io_parse_row(s, NULL, 0, &bad);
					if (k < 0) die(data, line, "bad number near '%.20s'", bad);
					if (k > 0 && k != per) die(data, line, "row has %d values, model %s needs %d per sample", k, m->name, per);
					if (k > 0 && pass == 1) {
						io_parse_row(s, row, per, &bad);
						vm_run(m, in, out, steps);
				acc_add(&acc, steps, m->nthink);
						io_write_row(f, obin, out, n, steps, obin ? 0 : m->nthink);
					}
					s[len] = save;
					if (!nl) break;
					s = nl + 1;
				}
			}
		}
		if (opath) close_out(f, opath);
		if (tr_on(1)) trace_batch(m, &acc, tr_now_ms() - tb);
	} else if (strcmp(cmd, "c") == 0) {
		FILE *f = stdout;
		if (argc == 5 && strcmp(argv[3], "-o") == 0) {
			f = fopen(argv[4], "w");
			if (!f) die(NULL, 0, "cannot write '%s'", argv[4]);
		} else if (argc != 3) {
			usage();
		}
		cgen(m, f);
		if (tr_on(1)) {
			tr_begin(1, "cgen");
			tr_str("output", f == stdout ? "-" : argv[4]);
			tr_num("arena_bytes", m->arena * 4);
			tr_end("emitted C unit for %s to %s", m->name, f == stdout ? "stdout" : argv[4]);
		}
		if (f != stdout && fclose(f) != 0) die(NULL, 0, "write failed '%s'", argv[4]);
	} else {
		usage();
	}
	return 0;
}
