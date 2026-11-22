#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matmul_optimal.h"

/* Build: make
 * Run: ./matmul [M N K]
 */

void matmul( int M, int N,int K,int a[][K], int b[][N], int c[M][N]) {
	for (int r=0;r<M;r++)
	{
		for (int cl=0; cl<N; cl++)
		{
			c[r][cl]=0;
		}
	}
	for(int i=0;i<M;i++){
		for(int j=0; j<N;j++){
			for(int k=0; k<K; k++){
				c[i][j]+=a[i][k]*b[k][j];		
			}
		}
	}
}
static double seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static int parse_dimension(const char *text, int *dimension) {
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || value <= 0 || value > INT_MAX)
        return 0;
    *dimension = (int)value;
    return 1;
}

static void *matrix_alloc(int rows, int cols, size_t element_size) {
    if ((size_t)rows > SIZE_MAX / (size_t)cols / element_size) {
        fprintf(stderr, "Matrix allocation size overflows size_t\n");
        exit(EXIT_FAILURE);
    }
    void *data = malloc((size_t)rows * (size_t)cols * element_size);
    if (!data) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    return data;
}

static int run_test(int M, int N, int K, int value_limit) {
    /* Bound inputs so every integer accumulation fits in int. */
    if (K > INT_MAX / value_limit / value_limit) {
        fprintf(stderr, "K is too large for the configured integer input range\n");
        return 0;
    }
    int (*a)[K] = matrix_alloc(M, K, sizeof(int));
    int (*b)[N] = matrix_alloc(K, N, sizeof(int));
    int (*c)[N] = matrix_alloc(M, N, sizeof(int));
    int (*expected)[N] = matrix_alloc(M, N, sizeof(int));
    int passed = 1;

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            int value = 1 + (int)(((size_t)i + (size_t)k) % value_limit);
            if ((i % 2) != (k % 2))
                value = -value;
            a[i][k] = value;
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            int value = 1 + (int)(((size_t)k * N + j) % value_limit);
            if (j % 2)
                value = -value;
            b[k][j] = value;
        }
    }

    double start = seconds();
    matmul(M, N, K, a, b, c);
    double matmul_time = seconds() - start;

    start = seconds();
    matmul_optimal(M, N, K, a, b, expected);
    double optimal_time = seconds() - start;

    for (int i = 0; i < M && passed; ++i) {
        for (int j = 0; j < N; ++j) {
            if (c[i][j] != expected[i][j]) {
                passed = 0;
                break;
            }
        }
    }
    /* Report equivalent GFLOP/s: two integer operations per multiply-add. */
    double giga_operations = 2.0 * M * N * K / 1e9;
    printf("M=%d N=%d K=%d | %s | matmul %.3f GFLOP/s | optimal %.3f GFLOP/s\n",
           M, N, K, passed ? "PASS" : "FAIL",
           matmul_time > 0.0 ? giga_operations / matmul_time : 0.0,
           optimal_time > 0.0 ? giga_operations / optimal_time : 0.0);
    free(expected);
    free(c);
    free(b);
    free(a);
    return passed;
}

int main(int argc, char **argv) {
    /* Test configuration; command-line dimensions override the defaults. */
#define DEFAULT_M 10
#define DEFAULT_N 10
#define DEFAULT_K 5
#define TEST_VALUE_LIMIT 8
#define ARBITRARY_SIZE_CASES { {1, 1, 1}, {1, 7, 3}, {7, 1, 3}, \
                               {7, 13, 1}, {7, 13, 11}, {31, 17, 19} }
    int M = DEFAULT_M, N = DEFAULT_N, K = DEFAULT_K;
    const int sizes[][3] = ARBITRARY_SIZE_CASES;

    if (argc != 1 && argc != 4) {
        fprintf(stderr, "Usage: %s [M N K]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4 && (!parse_dimension(argv[1], &M) ||
                      !parse_dimension(argv[2], &N) ||
                      !parse_dimension(argv[3], &K))) {
        fprintf(stderr, "M, N, and K must be positive integers <= INT_MAX\n");
        return EXIT_FAILURE;
    }
    if (!run_test(M, N, K, TEST_VALUE_LIMIT))
        return EXIT_FAILURE;
    if (argc == 4)
        return EXIT_SUCCESS;
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        if (!run_test(sizes[i][0], sizes[i][1], sizes[i][2], TEST_VALUE_LIMIT))
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
