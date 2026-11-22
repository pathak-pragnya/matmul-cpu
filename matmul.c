#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "matmul_optimal.h"

/* Build: make
 * Run: ./matmul [M N K] or ./matmul --test-suite SIZE...
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

#define GRAPH_WIDTH 24

struct test_result {
    int M, N, K, passed;
    double matmul_gflops, optimal_gflops;
};

struct report {
    struct test_result *rows;
    size_t count;
    double matmul_scale, optimal_scale;
    int interactive;
    int terminal_columns, terminal_rows;
    size_t screen_lines;
};

/* Round up to a readable 1, 2, or 5 times a power of ten. */
static double graph_scale(double value) {
    if (value <= 0.0)
        return 1.0;
    double scale = 1.0;
    while (scale < value)
        scale *= 10.0;
    while (scale / 10.0 >= value)
        scale /= 10.0;
    if (scale / 5.0 >= value)
        return scale / 5.0;
    if (scale / 2.0 >= value)
        return scale / 2.0;
    return scale;
}

static void print_graph(double value, double scale) {
    int position = scale > 0.0 ? (int)(value / scale * GRAPH_WIDTH + 0.5) : 0;
    putchar('|');
    for (int i = 0; i <= GRAPH_WIDTH; ++i)
        putchar(i == position ? '>' : (i < position ? '-' : ' '));
}

static int print_header(const struct report *report) {
    char matmul_graph[64], optimal_graph[64];
    snprintf(matmul_graph, sizeof(matmul_graph), "graph (0..%.3g GFLOP/s)",
             report->matmul_scale);
    snprintf(optimal_graph, sizeof(optimal_graph), "graph (0..%.3g GFLOP/s)",
             report->optimal_scale);
    return printf("%6s, %6s, %6s, %-6s, %18s, %-26s, %18s, %-26s\n",
                  "M", "N", "K", "Result", "matmul (GFLOP/s)", matmul_graph,
                  "optimal (GFLOP/s)", optimal_graph) - 1;
}

static int print_result(const struct test_result *row, const struct report *report) {
    int length = printf("%6d, %6d, %6d, %-6s, %18.3f, ", row->M, row->N, row->K,
                        row->passed ? "PASS" : "FAIL", row->matmul_gflops);
    print_graph(row->matmul_gflops, report->matmul_scale);
    length += printf(", %18.3f, ", row->optimal_gflops);
    print_graph(row->optimal_gflops, report->optimal_scale);
    putchar('\n');
    return length + 2 * (GRAPH_WIDTH + 2);
}

static size_t screen_lines(int length, int columns) {
    return columns > 0 && length > 0 ? 1 + (size_t)(length - 1) / columns : 1;
}

static void print_report(struct report *report) {
    report->screen_lines = screen_lines(print_header(report), report->terminal_columns);
    for (size_t i = 0; i < report->count; ++i)
        report->screen_lines += screen_lines(print_result(&report->rows[i], report),
                                             report->terminal_columns);
    fflush(stdout);
}

static void record_result(struct report *report, struct test_result row) {
    int rescale = report->count == 0;
    report->rows[report->count++] = row;
    if (row.matmul_gflops > report->matmul_scale || report->count == 1) {
        report->matmul_scale = graph_scale(row.matmul_gflops);
        rescale = 1;
    }
    if (row.optimal_gflops > report->optimal_scale || report->count == 1) {
        report->optimal_scale = graph_scale(row.optimal_gflops);
        rescale = 1;
    }
    if (report->interactive) {
        struct winsize window = {0};
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &window);
        int same_size = window.ws_col > 0 && window.ws_row > 0 &&
                        window.ws_col == report->terminal_columns &&
                        window.ws_row == report->terminal_rows;
        if (rescale) {
            /* Redraw saved results; never rerun or change the measurements. */
            if (report->count > 1 && same_size && report->screen_lines < window.ws_row)
                printf("\033[%zuA\r\033[J", report->screen_lines);
            /* If rows have scrolled away or the terminal resized, append a
             * refreshed table rather than risk overwriting command history. */
            report->terminal_columns = window.ws_col;
            report->terminal_rows = window.ws_row;
            print_report(report);
        } else {
            if (!same_size) {
                report->terminal_columns = 0;
                report->terminal_rows = 0;
            }
            report->screen_lines += screen_lines(print_result(&row, report),
                                                 report->terminal_columns);
            fflush(stdout);
        }
    }
}

static int run_test(int M, int N, int K, int value_limit, struct report *report) {
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
    record_result(report, (struct test_result){
        M, N, K, passed,
        matmul_time > 0.0 ? giga_operations / matmul_time : 0.0,
        optimal_time > 0.0 ? giga_operations / optimal_time : 0.0
    });
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
    int suite = argc >= 2 && strcmp(argv[1], "--test-suite") == 0;

    if ((suite && argc < 3) || (!suite && argc != 1 && argc != 4)) {
        fprintf(stderr, "Usage: %s [M N K] or %s --test-suite SIZE...\n",
                argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    if (!suite && argc == 4 && (!parse_dimension(argv[1], &M) ||
                      !parse_dimension(argv[2], &N) ||
                      !parse_dimension(argv[3], &K))) {
        fprintf(stderr, "M, N, and K must be positive integers <= INT_MAX\n");
        return EXIT_FAILURE;
    }
    size_t capacity = suite ? (size_t)(argc - 2) : 1 + sizeof(sizes) / sizeof(sizes[0]);
    const char *term = getenv("TERM");
    struct report report = { .rows = matrix_alloc(1, (int)capacity, sizeof(struct test_result)),
        .interactive = isatty(STDOUT_FILENO) && (!term || strcmp(term, "dumb") != 0) };
    int passed = 1;
    if (suite) {
        for (int i = 2; i < argc; ++i) {
            if (!parse_dimension(argv[i], &M)) {
                fprintf(stderr, "Suite sizes must be positive integers <= INT_MAX\n");
                passed = 0;
                break;
            }
            if (!run_test(M, M, M, TEST_VALUE_LIMIT, &report)) {
                passed = 0;
                break;
            }
        }
    } else {
        passed = run_test(M, N, K, TEST_VALUE_LIMIT, &report);
        if (argc == 1) {
            for (size_t i = 0; passed && i < sizeof(sizes) / sizeof(sizes[0]); ++i)
                passed = run_test(sizes[i][0], sizes[i][1], sizes[i][2],
                                  TEST_VALUE_LIMIT, &report);
        }
    }
    /* Files and pipes get one clean CSV table using the final scales. */
    if (!report.interactive && report.count > 0)
        print_report(&report);
    free(report.rows);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
