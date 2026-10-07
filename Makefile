CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS  = -lm

SRC = src/util.c src/sexp.c src/ops.c src/parse.c src/lower.c src/ir.c src/plan.c src/vm.c src/cgen.c src/fixer.c src/io.c src/trace.c src/main.c
OBJ = $(SRC:src/%.c=build/%.o)

all: build/tgc

build/tgc: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDLIBS)

build/%.o: src/%.c src/tg.h runtime/tg_rt.h build/rt_embed.h build/forest_embed.h | build
	$(CC) $(CFLAGS) -Ibuild -c -o $@ $<

# Embed the runtime kernels as C string lines so emitted units are self-contained.
build/rt_embed.h: runtime/tg_rt.h | build
	perl -ne 'BEGIN { print "static const char *const tg_rt_lines[] = {\n" } chomp; s/\\/\\\\/g; s/"/\\"/g; print "\t\"$$_\\n\",\n"; END { print "\t0\n};\n" }' < $< > $@

# The auto-fix ranker is a Technograd program, embedded as source. Before the
# first `make fixer` the list is empty and autofix reports that no model exists.
FOREST = $(wildcard fixer/forest.tg)
build/forest_embed.h: $(FOREST) | build
	perl -e 'print "static const char *const tg_forest_lines[] = {\n"; for $$f (@ARGV) { open F, "<", $$f or die; while (<F>) { chomp; s/\\/\\\\/g; s/"/\\"/g; print "\t\"$$_\\n\",\n" } } print "\t0\n};\n"' $(FOREST) > $@

TRAIN_CORPUS = $(wildcard fixer/corpus/*.tg) examples/xor.tg examples/newton.tg examples/latent_reasoner.tg
EVAL_CORPUS  = $(wildcard tests/cases/*.tg)

# Train the ranker on corrupted copies of TRAIN_CORPUS, rebuild with it, then
# evaluate on corruptions of programs it has never seen (EVAL_CORPUS).
fixer: build/tgc
	build/tgc fixer-train -o fixer/forest.tg $(TRAIN_CORPUS)
	$(MAKE) build/tgc
	build/tgc fixer-eval $(EVAL_CORPUS)

build:
	mkdir -p build

test: build/tgc
	sh tests/run.sh

clean:
	rm -rf build

.PHONY: all test clean fixer
