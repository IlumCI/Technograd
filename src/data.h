/* Datasets: loading, column inference, featurization (data.c). */
#ifndef TG_DATA_H
#define TG_DATA_H

#include <stdio.h>

typedef enum { CELL_NULL, CELL_NUM, CELL_STR, CELL_LIST, CELL_OTHER } CellType;

typedef struct {
	CellType t;
	double n;
	char *s;
	float *l;
	int nl;
} Cell;

typedef enum { COL_DROP, COL_NUM, COL_CAT, COL_TEXT, COL_VEC } ColKind;

typedef struct {
	char *name;
	Cell *cell;
	ColKind kind;
	const char *why; /* reason for COL_DROP */
	char **levels;   /* distinct values, first-seen order */
	int *counts;
	int nlevels, missing, veclen, intlike;
} Col;

typedef struct {
	char *source;
	const char *format;
	Col *col;
	int ncols, nrows, caprows;
} Table;

typedef struct {
	char *name;
	ColKind kind;
	int dim;
	double *mean, *std; /* numeric, vector */
	char **levels;      /* categorical one-hot */
	int nlev;
	int flag; /* numeric: one extra 0/1 feature marking a missing value */
} Feat;

typedef struct {
	char *target;
	int classify, nclass;
	char **classes;
	double tmean, tstd; /* regression target standardization */
	Feat *f;
	int nf, dim;
} Spec;

Table *data_load(const char *src, int max_rows);
void data_analyze(Table *t);
int data_pick_target(const Table *t, const char *want); /* want may be NULL */
int data_drop_leaks(Table *t, int target);
void data_describe(const Table *t, int target, FILE *f);
Spec *spec_fit(const Table *t, int target, const int *rows, int nrows, int text_dim);
int spec_apply(const Spec *s, const Table *t, int row, float *x, float *y, int *label);
/* Word buckets of text feature f (index into s->f) in reading order: writes
 * up to max ids, returns the number of words. */
int spec_tokens(const Spec *s, const Table *t, int row, int f, int *ids, int max);
/* Bucket ids of the words of any text, hashed exactly as spec_tokens hashes a text column of dim buckets. */
int text_tokens(const char *text, int dim, int *ids, int max);
void spec_save(const Spec *s, const char *path);
Spec *spec_load(const char *path);

#endif
