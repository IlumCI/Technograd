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
	      "  run   <file> <in>...      interpret; each input is comma-separated f32 values\n"
	      "  c     <file> [-o out.c]   emit a freestanding C unit\n"
	      "  fix   <file> [-o out.tg]  repair compile errors with the neural-forest fixer\n"
	      "  fixer-train -o forest.tg <corpus.tg>...   train the fixer (self-supervised)\n"
	      "  fixer-eval <corpus.tg>...                 measure repair rates on corrupted programs\n"
	      "options:\n"
	      "  --autofix   on a compile error, repair in memory and continue (source is not modified)\n",
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
		else argv[k++] = argv[i];
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
	plan(m);

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
		if (argc != 3 + m->ninputs) die(NULL, 0, "model %s takes %d input(s), got %d", m->name, m->ninputs, argc - 3);
		const float **in = xmalloc((size_t)m->ninputs * sizeof *in);
		for (int i = 0; i < m->ninputs; i++) {
			const Value *x = &m->val[m->inputs[i]];
			float *buf = xmalloc((size_t)shape_numel(&x->sh) * sizeof *buf);
			parse_input(argv[3 + i], buf, shape_numel(&x->sh), x->name);
			in[i] = buf;
		}
		int n = shape_numel(&m->val[m->output].sh);
		float *out = xmalloc((size_t)n * sizeof *out);
		int *steps = xmalloc((size_t)(m->nthink ? m->nthink : 1) * sizeof *steps);
		vm_run(m, in, out, steps);
		for (int i = 0; i < n; i++) printf("%s%.9g", i ? " " : "", (double)out[i]);
		putchar('\n');
		for (int i = 0; i < m->nthink; i++) printf("steps %d %d\n", i, steps[i]);
	} else if (strcmp(cmd, "c") == 0) {
		FILE *f = stdout;
		if (argc == 5 && strcmp(argv[3], "-o") == 0) {
			f = fopen(argv[4], "w");
			if (!f) die(NULL, 0, "cannot write '%s'", argv[4]);
		} else if (argc != 3) {
			usage();
		}
		cgen(m, f);
		if (f != stdout && fclose(f) != 0) die(NULL, 0, "write failed '%s'", argv[4]);
	} else {
		usage();
	}
	return 0;
}
