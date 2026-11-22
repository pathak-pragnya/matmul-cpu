#ifndef MATMUL_OPTIMAL_H
#define MATMUL_OPTIMAL_H

/* Row-major C = A * B. Positive dimensions, disjoint buffers, int-safe sums. */
void matmul_optimal(int M, int N, int K,
                    int a[restrict M][K], int b[restrict K][N],
                    int c[restrict M][N]);

#endif
