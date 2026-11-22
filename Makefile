CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

.PHONY: all test clean

all: matmul

matmul: matmul.c matmul_optimal.c matmul_optimal.h Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ matmul.c matmul_optimal.c $(LDLIBS)

# Example: make test ARGS="100 80 60"
test: matmul
	@./matmul $(ARGS)

clean:
	$(RM) matmul
