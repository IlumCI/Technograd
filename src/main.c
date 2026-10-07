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
	      "  c     <file> [-o out.c]   emit a freestanding C unit\n",
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

int main(int argc, char **argv)
{
	if (argc < 3) usage();
	const char *cmd = argv[1];
	Module *m = load_module(argv[2]);
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
