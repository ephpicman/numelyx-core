/**
 * @file matrix_benchmark.c
 * @brief Diagnostic benchmark for Matrix construction and numerical operations.
 *
 * @details This standalone program measures selected Matrix operations for diagnostic comparison. Benchmark results are informational and are not correctness assertions or stable performance guarantees.
 */

#include "../numerus_matrix.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 64
#define SMALL_SIZE 32
#define READ_ITERATIONS 10
#define OPERATION_ITERATIONS 3

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void __real_free(void *pointer);

static size_t allocation_calls;
static size_t allocation_bytes;
static volatile double benchmark_sink;

void *__wrap_malloc(size_t size)
{
    allocation_calls++;
    if (size <= SIZE_MAX - allocation_bytes) {
        allocation_bytes += size;
    } else {
        allocation_bytes = SIZE_MAX;
    }

    return __real_malloc(size);
}

void *__wrap_calloc(size_t count, size_t size)
{
    allocation_calls++;
    if (size == 0 || count <= SIZE_MAX / size) {
        size_t bytes = count * size;
        if (bytes <= SIZE_MAX - allocation_bytes) {
            allocation_bytes += bytes;
        } else {
            allocation_bytes = SIZE_MAX;
        }
    } else {
        allocation_bytes = SIZE_MAX;
    }

    return __real_calloc(count, size);
}

void __wrap_free(void *pointer)
{
    __real_free(pointer);
}

static void reset_allocation_stats(void)
{
    allocation_calls = 0;
    allocation_bytes = 0;
}

static void report_measurement(
    const char *name,
    size_t rows,
    size_t columns,
    size_t iterations,
    clock_t start,
    clock_t end
)
{
    size_t calls = allocation_calls;
    size_t bytes = allocation_bytes;
    double seconds = (double) (end - start) / (double) CLOCKS_PER_SEC;

    printf(
        "%-24s shape=%zux%zu iterations=%zu seconds=%.6f "
        "allocations=%zu allocated_bytes=%zu sink=%.6f\n",
        name,
        rows,
        columns,
        iterations,
        seconds,
        calls,
        bytes,
        (double) benchmark_sink
    );
}

static void benchmark_reads(
    const char *name,
    const numerus_matrix *matrix,
    size_t iterations
)
{
    size_t rows = numerus_matrix_rows(matrix);
    size_t columns = numerus_matrix_columns(matrix);
    size_t iteration;
    size_t row;
    size_t column;
    clock_t start;
    clock_t end;

    reset_allocation_stats();
    start = clock();

    for (iteration = 0; iteration < iterations; iteration++) {
        for (row = 0; row < rows; row++) {
            for (column = 0; column < columns; column++) {
                double value;
                int status = numerus_matrix_get(
                    matrix, row, column, &value
                );

                if (status != NUMERUS_MATRIX_SUCCESS) {
                    fprintf(stderr, "read failed in benchmark '%s': %d\n", name, status);
                    exit(EXIT_FAILURE);
                }

                benchmark_sink += value;
            }
        }
    }

    end = clock();
    report_measurement(name, rows, columns, iterations, start, end);
}

static void benchmark_materialize(
    const numerus_matrix *source,
    size_t iterations
)
{
    size_t iteration;
    size_t rows = numerus_matrix_rows(source);
    size_t columns = numerus_matrix_columns(source);
    clock_t start;
    clock_t end;

    reset_allocation_stats();
    start = clock();

    for (iteration = 0; iteration < iterations; iteration++) {
        numerus_matrix *result = NULL;
        double value;
        int status = numerus_matrix_materialize(source, &result);

        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "materialization benchmark failed: %d\n", status);
            exit(EXIT_FAILURE);
        }

        status = numerus_matrix_get(result, 0, 0, &value);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "materialized result read failed: %d\n", status);
            exit(EXIT_FAILURE);
        }
        benchmark_sink += value;
        numerus_matrix_destroy(result);
    }

    end = clock();
    report_measurement(
        "materialize view", rows, columns, iterations, start, end
    );
}

#define MULTIPLY_BENCHMARK_SAMPLES 5

static void benchmark_multiplication(
    const char *name,
    const numerus_matrix *left,
    const numerus_matrix *right,
    size_t iterations
)
{
    double seconds[MULTIPLY_BENCHMARK_SAMPLES];
    size_t calls[MULTIPLY_BENCHMARK_SAMPLES];
    size_t bytes[MULTIPLY_BENCHMARK_SAMPLES];
    size_t sample;
    size_t rows = numerus_matrix_rows(left);
    size_t columns = numerus_matrix_columns(right);

    for (sample = 0; sample < MULTIPLY_BENCHMARK_SAMPLES; sample++) {
        size_t iteration;
        clock_t start;
        clock_t end;

        reset_allocation_stats();
        start = clock();
        for (iteration = 0; iteration < iterations; iteration++) {
            numerus_matrix *result = NULL;
            double value;
            int status = numerus_matrix_multiply(left, right, &result);

            if (status != NUMERUS_MATRIX_SUCCESS) {
                fprintf(stderr, "multiplication benchmark failed: %d\n", status);
                exit(EXIT_FAILURE);
            }
            status = numerus_matrix_get(result, 0, 0, &value);
            if (status != NUMERUS_MATRIX_SUCCESS) {
                fprintf(stderr, "product read failed: %d\n", status);
                numerus_matrix_destroy(result);
                exit(EXIT_FAILURE);
            }
            benchmark_sink += value;
            numerus_matrix_destroy(result);
        }
        end = clock();
        seconds[sample] = (double) (end - start) / (double) CLOCKS_PER_SEC;
        calls[sample] = allocation_calls;
        bytes[sample] = allocation_bytes;
    }

    for (sample = 0; sample < MULTIPLY_BENCHMARK_SAMPLES; sample++) {
        size_t other;
        for (other = sample + 1; other < MULTIPLY_BENCHMARK_SAMPLES; other++) {
            if (seconds[other] < seconds[sample]) {
                double temp = seconds[sample];
                seconds[sample] = seconds[other];
                seconds[other] = temp;
            }
            if (calls[other] < calls[sample]) {
                size_t temp = calls[sample];
                calls[sample] = calls[other];
                calls[other] = temp;
            }
            if (bytes[other] < bytes[sample]) {
                size_t temp = bytes[sample];
                bytes[sample] = bytes[other];
                bytes[other] = temp;
            }
        }
    }

    printf(
        "%-24s shape=%zux%zu samples=%d iterations_per_sample=%zu "
        "median_seconds=%.6f min_seconds=%.6f max_seconds=%.6f "
        "median_allocations_per_sample=%zu median_allocated_bytes_per_sample=%zu sink=%.6f\n",
        name, rows, columns, MULTIPLY_BENCHMARK_SAMPLES, iterations,
        seconds[MULTIPLY_BENCHMARK_SAMPLES / 2], seconds[0],
        seconds[MULTIPLY_BENCHMARK_SAMPLES - 1],
        calls[MULTIPLY_BENCHMARK_SAMPLES / 2],
        bytes[MULTIPLY_BENCHMARK_SAMPLES / 2], (double) benchmark_sink
    );
}

static void benchmark_inverse(
    const numerus_matrix *matrix,
    size_t iterations
)
{
    numerus_matrix **fresh;
    size_t iteration;
    size_t size = numerus_matrix_rows(matrix);
    clock_t start;
    clock_t end;

    fresh = calloc(iterations, sizeof(*fresh));
    if (fresh == NULL) {
        fprintf(stderr, "inverse benchmark setup allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (iteration = 0; iteration < iterations; iteration++) {
        int status = numerus_matrix_materialize(matrix, &fresh[iteration]);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "inverse benchmark setup failed: %d\n", status);
            while (iteration > 0) numerus_matrix_destroy(fresh[--iteration]);
            free(fresh);
            exit(EXIT_FAILURE);
        }
    }

    reset_allocation_stats();
    start = clock();
    for (iteration = 0; iteration < iterations; iteration++) {
        numerus_matrix *inverse = NULL;
        double value;
        int status = numerus_matrix_inverse(fresh[iteration], &inverse);

        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "inverse benchmark failed: %d\n", status);
            exit(EXIT_FAILURE);
        }
        status = numerus_matrix_get(inverse, 0, 0, &value);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "inverse result read failed: %d\n", status);
            numerus_matrix_destroy(inverse);
            exit(EXIT_FAILURE);
        }
        benchmark_sink += value;
        numerus_matrix_destroy(inverse);
    }

    end = clock();
    report_measurement("LU inverse (cold)", size, size, iterations, start, end);
    for (iteration = 0; iteration < iterations; iteration++) {
        numerus_matrix_destroy(fresh[iteration]);
    }
    free(fresh);
}

static void benchmark_inverse_cache_hit(
    const numerus_matrix *matrix,
    size_t iterations
)
{
    size_t iteration;
    size_t size = numerus_matrix_rows(matrix);
    numerus_matrix *warm = NULL;
    clock_t start;
    clock_t end;

    if (numerus_matrix_inverse(matrix, &warm) != NUMERUS_MATRIX_SUCCESS) {
        fprintf(stderr, "inverse cache warmup failed\n");
        exit(EXIT_FAILURE);
    }
    numerus_matrix_destroy(warm);

    reset_allocation_stats();
    start = clock();
    for (iteration = 0; iteration < iterations; iteration++) {
        numerus_matrix *inverse = NULL;
        double value;
        int status = numerus_matrix_inverse(matrix, &inverse);

        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "inverse cache-hit benchmark failed: %d\n", status);
            exit(EXIT_FAILURE);
        }
        status = numerus_matrix_get(inverse, 0, 0, &value);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            numerus_matrix_destroy(inverse);
            exit(EXIT_FAILURE);
        }
        benchmark_sink += value;
        numerus_matrix_destroy(inverse);
    }
    end = clock();
    report_measurement("inverse cache hit", size, size, iterations, start, end);
}


static void benchmark_solve_lu_cache(
    const numerus_matrix *matrix,
    size_t iterations
)
{
    size_t iteration;
    size_t size = numerus_matrix_rows(matrix);
    double *rhs_values = calloc(size, sizeof(*rhs_values));
    numerus_matrix *rhs = NULL;
    numerus_matrix **fresh = NULL;
    numerus_matrix *warm_solution = NULL;
    clock_t start;
    clock_t end;

    if (rhs_values == NULL) {
        fprintf(stderr, "solve benchmark RHS allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (iteration = 0; iteration < size; iteration++) {
        rhs_values[iteration] = 1.0;
    }
    if (numerus_matrix_create_dense(size, 1, rhs_values, &rhs) !=
        NUMERUS_MATRIX_SUCCESS) {
        free(rhs_values);
        exit(EXIT_FAILURE);
    }
    free(rhs_values);

    fresh = calloc(iterations, sizeof(*fresh));
    if (fresh == NULL) {
        numerus_matrix_destroy(rhs);
        exit(EXIT_FAILURE);
    }
    for (iteration = 0; iteration < iterations; iteration++) {
        if (numerus_matrix_materialize(matrix, &fresh[iteration]) !=
            NUMERUS_MATRIX_SUCCESS) {
            while (iteration > 0) numerus_matrix_destroy(fresh[--iteration]);
            free(fresh);
            numerus_matrix_destroy(rhs);
            exit(EXIT_FAILURE);
        }
    }

    reset_allocation_stats();
    start = clock();
    for (iteration = 0; iteration < iterations; iteration++) {
        numerus_matrix *solution = NULL;
        double value;
        int status = numerus_matrix_solve(fresh[iteration], rhs, &solution);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "cold solve benchmark failed: %d\n", status);
            exit(EXIT_FAILURE);
        }
        status = numerus_matrix_get(solution, 0, 0, &value);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            numerus_matrix_destroy(solution);
            exit(EXIT_FAILURE);
        }
        benchmark_sink += value;
        numerus_matrix_destroy(solution);
    }
    end = clock();
    report_measurement("solve (cold LU)", size, 1, iterations, start, end);
    for (iteration = 0; iteration < iterations; iteration++) {
        numerus_matrix_destroy(fresh[iteration]);
    }
    free(fresh);

    if (numerus_matrix_solve(matrix, rhs, &warm_solution) !=
        NUMERUS_MATRIX_SUCCESS) {
        numerus_matrix_destroy(rhs);
        exit(EXIT_FAILURE);
    }
    numerus_matrix_destroy(warm_solution);

    reset_allocation_stats();
    start = clock();
    for (iteration = 0; iteration < iterations; iteration++) {
        numerus_matrix *solution = NULL;
        double value;
        int status = numerus_matrix_solve(matrix, rhs, &solution);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "cached LU solve benchmark failed: %d\n", status);
            numerus_matrix_destroy(rhs);
            exit(EXIT_FAILURE);
        }
        status = numerus_matrix_get(solution, 0, 0, &value);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            numerus_matrix_destroy(solution);
            numerus_matrix_destroy(rhs);
            exit(EXIT_FAILURE);
        }
        benchmark_sink += value;
        numerus_matrix_destroy(solution);
    }
    end = clock();
    report_measurement("solve (cached LU)", size, 1, iterations, start, end);
    numerus_matrix_destroy(rhs);
}

static void benchmark_condition_estimate(
    const numerus_matrix *matrix,
    size_t iterations
)
{
    size_t iteration;
    size_t size = numerus_matrix_rows(matrix);
    clock_t start;
    clock_t end;

    reset_allocation_stats();
    start = clock();

    for (iteration = 0; iteration < iterations; iteration++) {
        double condition_estimate;
        int status = numerus_matrix_condition_estimate_one(
            matrix, &condition_estimate
        );

        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "condition estimate benchmark failed: %d\n", status);
            exit(EXIT_FAILURE);
        }
        benchmark_sink += condition_estimate;
    }

    end = clock();
    report_measurement(
        "condition estimate", size, size, iterations, start, end
    );
}

#define CACHE_BENCHMARK_ITERATIONS 128

static void benchmark_analysis_cache(numerus_matrix *source)
{
    numerus_matrix *fresh[CACHE_BENCHMARK_ITERATIONS] = {NULL};
    numerus_matrix_flags flags;
    size_t i;
    size_t size = numerus_matrix_rows(source);
    clock_t start;
    clock_t end;
    int status;

    /* Prepare independent matrices outside the measured interval. */
    for (i = 0; i < CACHE_BENCHMARK_ITERATIONS; i++) {
        status = numerus_matrix_materialize(source, &fresh[i]);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "cache benchmark setup failed: %d\n", status);
            while (i > 0) numerus_matrix_destroy(fresh[--i]);
            return;
        }
    }

    /* Warm the source explicitly so the hit cases are genuinely cache hits. */
    status = numerus_matrix_get_flags(source, &flags);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        fprintf(stderr, "structural flags cache benchmark warmup failed: %d\n", status);
        goto cleanup;
    }
    {
        double determinant;
        status = numerus_matrix_determinant(source, &determinant);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "determinant cache benchmark warmup failed: %d\n", status);
            goto cleanup;
        }
    }

    reset_allocation_stats();
    start = clock();
    for (i = 0; i < CACHE_BENCHMARK_ITERATIONS; i++) {
        status = numerus_matrix_get_flags(fresh[i], &flags);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "structural flags cache-miss benchmark failed: %d\n", status);
            goto cleanup;
        }
        benchmark_sink += flags.identity ? 1.0 : 0.0;
    }
    end = clock();
    report_measurement("flags cache miss", size, size,
        CACHE_BENCHMARK_ITERATIONS, start, end);

    reset_allocation_stats();
    start = clock();
    for (i = 0; i < CACHE_BENCHMARK_ITERATIONS; i++) {
        status = numerus_matrix_get_flags(source, &flags);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "structural flags cache-hit benchmark failed: %d\n", status);
            goto cleanup;
        }
        benchmark_sink += flags.identity ? 1.0 : 0.0;
    }
    end = clock();
    report_measurement("flags cache hit", size, size,
        CACHE_BENCHMARK_ITERATIONS, start, end);

    reset_allocation_stats();
    start = clock();
    for (i = 0; i < CACHE_BENCHMARK_ITERATIONS; i++) {
        double determinant;
        status = numerus_matrix_determinant(fresh[i], &determinant);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "determinant cache-miss benchmark failed: %d\n", status);
            goto cleanup;
        }
        benchmark_sink += determinant;
    }
    end = clock();
    report_measurement("determinant cache miss", size, size,
        CACHE_BENCHMARK_ITERATIONS, start, end);

    reset_allocation_stats();
    start = clock();
    for (i = 0; i < CACHE_BENCHMARK_ITERATIONS; i++) {
        double determinant;
        status = numerus_matrix_determinant(source, &determinant);
        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "determinant cache-hit benchmark failed: %d\n", status);
            goto cleanup;
        }
        benchmark_sink += determinant;
    }
    end = clock();
    report_measurement("determinant cache hit", size, size,
        CACHE_BENCHMARK_ITERATIONS, start, end);

cleanup:
    for (i = 0; i < CACHE_BENCHMARK_ITERATIONS; i++) {
        numerus_matrix_destroy(fresh[i]);
    }
}

static void verify_inverse_residual(const numerus_matrix *matrix)
{
    numerus_matrix *inverse = NULL;
    size_t size = numerus_matrix_rows(matrix);
    size_t row;
    double maximum_residual = 0.0;
    int status = numerus_matrix_inverse(matrix, &inverse);

    if (status != NUMERUS_MATRIX_SUCCESS) {
        fprintf(stderr, "inverse residual check failed: %d\n", status);
        exit(EXIT_FAILURE);
    }

    for (row = 0; row < size; row++) {
        size_t column;

        for (column = 0; column < size; column++) {
            size_t k;
            double product = 0.0;
            double expected = row == column ? 1.0 : 0.0;

            for (k = 0; k < size; k++) {
                double left;
                double right;

                status = numerus_matrix_get(matrix, row, k, &left);
                if (status != NUMERUS_MATRIX_SUCCESS) goto failure;
                status = numerus_matrix_get(inverse, k, column, &right);
                if (status != NUMERUS_MATRIX_SUCCESS) goto failure;
                product += left * right;
            }

            if (fabs(product - expected) > maximum_residual) {
                maximum_residual = fabs(product - expected);
            }
        }
    }

    printf("LU inverse residual: max|A*A^-1-I|=%.6e\n", maximum_residual);
    numerus_matrix_destroy(inverse);
    return;

failure:
    fprintf(stderr, "inverse residual read failed: %d\n", status);
    numerus_matrix_destroy(inverse);
    exit(EXIT_FAILURE);
}

static void benchmark_scale_view_creation(
    numerus_matrix *parent,
    size_t iterations
)
{
    size_t iteration;
    size_t rows = numerus_matrix_rows(parent);
    size_t columns = numerus_matrix_columns(parent);
    clock_t start;
    clock_t end;

    reset_allocation_stats();
    start = clock();

    for (iteration = 0; iteration < iterations; iteration++) {
        numerus_matrix *view = NULL;
        int status = numerus_matrix_create_scale(parent, 2.0, &view);

        if (status != NUMERUS_MATRIX_SUCCESS) {
            fprintf(stderr, "scale view construction failed: %d\n", status);
            exit(EXIT_FAILURE);
        }
        numerus_matrix_destroy(view);
    }

    end = clock();
    report_measurement(
        "scale view creation", rows, columns, iterations, start, end
    );
}

int main(void)
{
    static double dense_values[MATRIX_SIZE * MATRIX_SIZE];
    static double diagonal_values[MATRIX_SIZE];
    static double upper_values[MATRIX_SIZE * (MATRIX_SIZE + 1) / 2];
    static numerus_storage_sparse_entry sparse_entries[MATRIX_SIZE];
    static double small_values[SMALL_SIZE * SMALL_SIZE];
    static double ill_conditioned_values[SMALL_SIZE * SMALL_SIZE];

    numerus_matrix *dense = NULL;
    numerus_matrix *diagonal = NULL;
    numerus_matrix *upper = NULL;
    numerus_matrix *sparse = NULL;
    numerus_matrix *identity = NULL;
    numerus_matrix *zero = NULL;
    numerus_matrix *transpose = NULL;
    numerus_matrix *nested_view = NULL;
    numerus_matrix *joined = NULL;
    numerus_matrix *scaled = NULL;
    numerus_matrix *sum = NULL;
    numerus_matrix *small = NULL;
    numerus_matrix *ill_conditioned = NULL;
    size_t row;
    size_t column;
    size_t packed_index = 0;
    int status;

    printf("Numerus Matrix benchmark baseline\n");
    printf("compiler=%s matrix_size=%d small_size=%d\n", __VERSION__,
        MATRIX_SIZE, SMALL_SIZE);
    printf("timing uses clock(); allocation metrics wrap malloc/calloc in the standalone test build\n");

    for (row = 0; row < MATRIX_SIZE; row++) {
        diagonal_values[row] = (double) (row + 1);
        sparse_entries[row].index = row * MATRIX_SIZE + row;
        sparse_entries[row].value = (double) (row + 1);

        for (column = row; column < MATRIX_SIZE; column++) {
            upper_values[packed_index++] =
                (double) ((row + column) % 11 + 1);
        }

        for (column = 0; column < MATRIX_SIZE; column++) {
            dense_values[row * MATRIX_SIZE + column] =
                (double) ((row + column) % 13 + 1) / 4.0;
        }
    }

    for (row = 0; row < SMALL_SIZE; row++) {
        for (column = 0; column < SMALL_SIZE; column++) {
            small_values[row * SMALL_SIZE + column] = row == column
                ? (double) (SMALL_SIZE + row + 1)
                : (double) ((row + column) % 13 + 1) / 16.0;
        }
    }

    for (row = 0; row < SMALL_SIZE; row++) {
        for (column = 0; column < SMALL_SIZE; column++) {
            ill_conditioned_values[row * SMALL_SIZE + column] =
                row == column
                    ? (row == SMALL_SIZE - 1 ? 1e-7 : 1.0)
                    : 0.0;
        }
    }

    status = numerus_matrix_create_dense(
        MATRIX_SIZE, MATRIX_SIZE, dense_values, &dense
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_diagonal(
        MATRIX_SIZE, diagonal_values, &diagonal
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_upper_triangular(
        MATRIX_SIZE, upper_values, &upper
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_sparse(
        MATRIX_SIZE, MATRIX_SIZE, 0.0,
        sparse_entries, MATRIX_SIZE, &sparse
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_identity(MATRIX_SIZE, &identity);
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_zero(MATRIX_SIZE, MATRIX_SIZE, &zero);
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_transpose(dense, &transpose);
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_transpose(transpose, &nested_view);
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_join_horizontal(dense, transpose, &joined);
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_scale(dense, 2.0, &scaled);
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_add(dense, transpose, &sum);
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;
    status = numerus_matrix_create_dense(
        SMALL_SIZE, SMALL_SIZE, small_values, &small
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;

    status = numerus_matrix_create_dense(
        SMALL_SIZE, SMALL_SIZE, ill_conditioned_values, &ill_conditioned
    );
    if (status != NUMERUS_MATRIX_SUCCESS) goto fail;

    benchmark_reads("dense read", dense, READ_ITERATIONS);
    benchmark_reads("diagonal read", diagonal, READ_ITERATIONS);
    benchmark_reads("triangular read", upper, READ_ITERATIONS);
    benchmark_reads("sparse read", sparse, READ_ITERATIONS);
    benchmark_reads("transpose view read", transpose, READ_ITERATIONS);
    benchmark_reads("nested transpose read", nested_view, READ_ITERATIONS);
    benchmark_reads("joined view read", joined, READ_ITERATIONS);
    benchmark_reads("scaled view read", scaled, READ_ITERATIONS);
    benchmark_reads("binary view read", sum, READ_ITERATIONS);
    benchmark_scale_view_creation(dense, 1000);
    benchmark_materialize(transpose, OPERATION_ITERATIONS);
    benchmark_materialize(nested_view, OPERATION_ITERATIONS);
    benchmark_multiplication("dense x dense", dense, dense, OPERATION_ITERATIONS);
    benchmark_multiplication("dense x identity", dense, identity, OPERATION_ITERATIONS);
    benchmark_multiplication("identity x dense", identity, dense, OPERATION_ITERATIONS);
    benchmark_multiplication("dense x zero", dense, zero, OPERATION_ITERATIONS);
    benchmark_multiplication("dense x diagonal", dense, diagonal, OPERATION_ITERATIONS);
    benchmark_multiplication("dense x triangular", dense, upper, OPERATION_ITERATIONS);
    benchmark_multiplication("dense x sparse", dense, sparse, OPERATION_ITERATIONS);
    benchmark_multiplication("small dense multiply", small, small, OPERATION_ITERATIONS);
    benchmark_analysis_cache(small);
    benchmark_inverse(small, OPERATION_ITERATIONS);
    benchmark_inverse_cache_hit(small, OPERATION_ITERATIONS);
    benchmark_solve_lu_cache(small, OPERATION_ITERATIONS);
    benchmark_condition_estimate(small, OPERATION_ITERATIONS);
    verify_inverse_residual(small);

    benchmark_inverse(ill_conditioned, OPERATION_ITERATIONS);
    benchmark_condition_estimate(ill_conditioned, OPERATION_ITERATIONS);
    verify_inverse_residual(ill_conditioned);

    numerus_matrix_destroy(ill_conditioned);
    numerus_matrix_destroy(small);
    numerus_matrix_destroy(sum);
    numerus_matrix_destroy(scaled);
    numerus_matrix_destroy(joined);
    numerus_matrix_destroy(nested_view);
    numerus_matrix_destroy(transpose);
    numerus_matrix_destroy(zero);
    numerus_matrix_destroy(identity);
    numerus_matrix_destroy(sparse);
    numerus_matrix_destroy(upper);
    numerus_matrix_destroy(diagonal);
    numerus_matrix_destroy(dense);

    return EXIT_SUCCESS;

fail:
    fprintf(stderr, "benchmark setup failed: %d\n", status);
    numerus_matrix_destroy(ill_conditioned);
    numerus_matrix_destroy(small);
    numerus_matrix_destroy(sum);
    numerus_matrix_destroy(scaled);
    numerus_matrix_destroy(joined);
    numerus_matrix_destroy(transpose);
    numerus_matrix_destroy(zero);
    numerus_matrix_destroy(identity);
    numerus_matrix_destroy(sparse);
    numerus_matrix_destroy(upper);
    numerus_matrix_destroy(diagonal);
    numerus_matrix_destroy(dense);
    return EXIT_FAILURE;
}
