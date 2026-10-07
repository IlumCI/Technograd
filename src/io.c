/* Tensor file I/O for the CLI.
 *
 * Two formats, chosen by extension:
 *   .bin / .f32  raw little-endian IEEE-754 f32, no header (the format that
 *                param `file("...")` initializers read)
 *   anything else  text: numbers separated by commas and/or whitespace,
 *                `#` starts a comment that runs to end of line
 *
 * Text written by tgc is one comma-separated row per sample. */
#include "tg.h"

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int io_is_binary(const char *path)
{
	const char *dot = strrchr(path, '.');
	return dot && (strcmp(dot, ".bin") == 0 || strcmp(dot, ".f32") == 0);
}

/* Parse the numbers in one text line (up to the end of the string or a '#').
 * Returns the count, or -1 with *bad pointing at the offending text. */
int io_parse_row(const char *s, float *v, int max, const char **bad)
{
	int n = 0;
	for (;;) {
		while (*s == ' ' || *s == '\t' || *s == ',' || *s == '\r' || *s == '\n') s++;
		if (!*s || *s == '#') return n;
		char *e;
		float x = strtof(s, &e);
		if (e == s || (*e && !strchr(" \t,\r\n#", *e))) {
			*bad = s;
			return -1;
		}
		if (n < max) v[n] = x;
		n++;
		s = e;
	}
}

float *io_load(const char *path, int *count)
{
	size_t len;
	char *raw = read_file(path, &len);
	float *v;
	if (io_is_binary(path)) {
		if (len % 4) die(NULL, 0, "'%s': %zu bytes is not a whole number of f32 values", path, len);
		int n = (int)(len / 4);
		v = xmalloc((size_t)(n ? n : 1) * sizeof *v);
		const unsigned char *b = (const unsigned char *)raw;
		for (int i = 0; i < n; i++) {
			uint32_t w = (uint32_t)b[4 * i] | (uint32_t)b[4 * i + 1] << 8 | (uint32_t)b[4 * i + 2] << 16 | (uint32_t)b[4 * i + 3] << 24;
			memcpy(&v[i], &w, 4);
		}
		*count = n;
	} else {
		int cap = 64, n = 0, line = 1;
		v = xmalloc((size_t)cap * sizeof *v);
		for (char *s = raw; *s; line++) {
			char *nl = strchr(s, '\n');
			if (nl) *nl = 0;
			const char *bad;
			int k = io_parse_row(s, NULL, 0, &bad);
			if (k < 0) die(path, line, "bad number near '%.20s'", bad);
			while (n + k > cap) v = xrealloc(v, (size_t)(cap *= 2) * sizeof *v);
			io_parse_row(s, v + n, k, &bad);
			n += k;
			if (!nl) break;
			s = nl + 1;
		}
		*count = n;
	}
	xfree(raw);
	return v;
}

void io_write_row(FILE *f, int binary, const float *v, int n, const int *steps, int nsteps)
{
	if (binary) {
		for (int i = 0; i < n; i++) {
			uint32_t w;
			memcpy(&w, &v[i], 4);
			unsigned char b[4] = { (unsigned char)w, (unsigned char)(w >> 8), (unsigned char)(w >> 16), (unsigned char)(w >> 24) };
			fwrite(b, 1, 4, f);
		}
		return;
	}
	for (int i = 0; i < n; i++) fprintf(f, "%s%.9g", i ? "," : "", (double)v[i]);
	for (int i = 0; i < nsteps; i++) fprintf(f, ",%d", steps[i]);
	fputc('\n', f);
}
