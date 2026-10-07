CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS  = -lm

SRC = src/util.c src/sexp.c src/ops.c src/parse.c src/lower.c src/ir.c src/plan.c src/vm.c src/cgen.c src/main.c
OBJ = $(SRC:src/%.c=build/%.o)

all: build/tgc

build/tgc: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDLIBS)

build/%.o: src/%.c src/tg.h runtime/tg_rt.h build/rt_embed.h | build
	$(CC) $(CFLAGS) -Ibuild -c -o $@ $<

# Embed the runtime kernels as C string lines so emitted units are self-contained.
build/rt_embed.h: runtime/tg_rt.h | build
	perl -ne 'BEGIN { print "static const char *const tg_rt_lines[] = {\n" } chomp; s/\\/\\\\/g; s/"/\\"/g; print "\t\"$$_\\n\",\n"; END { print "\t0\n};\n" }' < $< > $@

build:
	mkdir -p build

test: build/tgc
	sh tests/run.sh

clean:
	rm -rf build

.PHONY: all test clean
