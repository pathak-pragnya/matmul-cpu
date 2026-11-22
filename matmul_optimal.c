#include <stddef.h>
#include <string.h>
#include "matmul_optimal.h"

/* Cache blocking and a four-row microkernel reuse B across four output rows.
 * The contiguous column loop allows ordinary CPU SIMD auto-vectorization.
 * No BLAS library, accelerator calls, OpenMP, or worker threads are used.
 */
#define BLOCK_M 32
#define BLOCK_N 128
#define BLOCK_K 64

static size_t block_end(size_t start, size_t length, size_t limit) {
    return limit - start < length ? limit : start + length;
}

void matmul_optimal(int M, int N, int K,
                    int a[restrict M][K], int b[restrict K][N],
                    int c[restrict M][N]) {
    memset(c, 0, (size_t)M * (size_t)N * sizeof(int));
    for (size_t ii = 0; ii < (size_t)M; ii += BLOCK_M) {
        size_t imax = block_end(ii, BLOCK_M, (size_t)M);
        for (size_t kk = 0; kk < (size_t)K; kk += BLOCK_K) {
            size_t kmax = block_end(kk, BLOCK_K, (size_t)K);
            for (size_t jj = 0; jj < (size_t)N; jj += BLOCK_N) {
                size_t jmax = block_end(jj, BLOCK_N, (size_t)N);
                size_t i = ii;
                for (; i + 4 <= imax; i += 4) {
                    for (size_t k = kk; k < kmax; ++k) {
                        int a0 = a[i][k], a1 = a[i + 1][k];
                        int a2 = a[i + 2][k], a3 = a[i + 3][k];
                        for (size_t j = jj; j < jmax; ++j) {
                            int value = b[k][j];
                            c[i][j] += a0 * value;
                            c[i + 1][j] += a1 * value;
                            c[i + 2][j] += a2 * value;
                            c[i + 3][j] += a3 * value;
                        }
                    }
                }
                /* Handle dimensions that are not multiples of four. */
                for (; i < imax; ++i) {
                    for (size_t k = kk; k < kmax; ++k) {
                        int value = a[i][k];
                        for (size_t j = jj; j < jmax; ++j)
                            c[i][j] += value * b[k][j];
                    }
                }
            }
        }
    }
}
