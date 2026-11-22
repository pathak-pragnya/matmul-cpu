CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

.PHONY: all test test_suite clean

all: matmul

matmul: matmul.c matmul_optimal.c matmul_optimal.h Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ matmul.c matmul_optimal.c $(LDLIBS)

# Example: make test ARGS="100 80 60"
test: matmul
	@./matmul $(ARGS)

# Square sizes through 2048: powers of two, multiples of 13 through 512,
# and multiples of 389 above 512.
test_suite: matmul
	@set --; size=1; power=1; \
	while [ $$size -le 2048 ]; do \
		step=13; if [ $$size -gt 512 ]; then step=389; fi; \
		if [ $$size -eq $$power ] || [ $$((size % step)) -eq 0 ]; then \
			set -- "$$@" $$size; \
		fi; \
		if [ $$size -eq $$power ]; then power=$$((power * 2)); fi; \
		size=$$((size + 1)); \
	done; \
	./matmul --test-suite "$$@"

clean:
	$(RM) matmul
