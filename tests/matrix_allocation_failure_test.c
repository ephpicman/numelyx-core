/**
 * @file matrix_allocation_failure_test.c
 * @brief Native tests for allocation-failure injection, cleanup, and output-state guarantees.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void __real_free(void *pointer);

static long allocations_before_failure = -1;

static int should_fail_allocation(void)
{
    if (allocations_before_failure < 0) {
        return 0;
    }
    if (allocations_before_failure == 0) {
        allocations_before_failure = -1;
        return 1;
    }
    allocations_before_failure--;
    return 0;
}

void *__wrap_malloc(size_t size)
{
    return should_fail_allocation() ? NULL : __real_malloc(size);
}

void *__wrap_calloc(size_t count, size_t size)
{
    return should_fail_allocation() ? NULL : __real_calloc(count, size);
}

void __wrap_free(void *pointer)
{
    __real_free(pointer);
}

static void test_materialize_allocation_failures(void)
{
    const double values[] = {2.0, -1.0, 4.0, 3.0, 5.0, 6.0};
    numerus_matrix *source = NULL;
    long fail_after;
    int allocation_failure_observed = 0;
    int successful_materialization_observed = 0;

    allocations_before_failure = -1;
    assert(numerus_matrix_create_dense(2, 3, values, &source) ==
        NUMERUS_MATRIX_SUCCESS);

    for (fail_after = 0; fail_after < 16; fail_after++) {
        numerus_matrix *copy = (numerus_matrix *) 1;
        numerus_matrix_status status;

        allocations_before_failure = fail_after;
        status = numerus_matrix_materialize(source, &copy);

        if (status == NUMERUS_MATRIX_SUCCESS) {
            assert(copy != NULL);
            numerus_matrix_destroy(copy);
            successful_materialization_observed = 1;
            break;
        }

        assert(status == NUMERUS_MATRIX_OUT_OF_MEMORY);
        assert(copy == NULL);
        allocation_failure_observed = 1;
    }

    assert(allocation_failure_observed);
    assert(successful_materialization_observed);

    numerus_matrix_destroy(source);
}

static void test_addition_allocation_failures(void)
{
    const double left_values[] = {1.0, 2.0, 3.0, 4.0};
    const double right_values[] = {5.0, 6.0, 7.0, 8.0};
    numerus_matrix *left = NULL;
    numerus_matrix *right = NULL;
    long fail_after;
    int allocation_failure_observed = 0;
    int successful_addition_observed = 0;

    allocations_before_failure = -1;
    assert(numerus_matrix_create_dense(2, 2, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, right_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);

    for (fail_after = 0; fail_after < 16; fail_after++) {
        numerus_matrix *sum = (numerus_matrix *) 1;
        numerus_matrix_status status;

        allocations_before_failure = fail_after;
        status = numerus_matrix_create_add(left, right, &sum);

        if (status == NUMERUS_MATRIX_SUCCESS) {
            assert(sum != NULL);
            numerus_matrix_destroy(sum);
            successful_addition_observed = 1;
            break;
        }

        assert(status == NUMERUS_MATRIX_OUT_OF_MEMORY);
        assert(sum == NULL);
        allocation_failure_observed = 1;
    }

    assert(allocation_failure_observed);
    assert(successful_addition_observed);

    numerus_matrix_destroy(right);
    numerus_matrix_destroy(left);
}


static void test_inverse_allocation_failures(void)
{
    const double values[] = {4.0, 1.0, 2.0, 3.0};
    numerus_matrix *source = NULL;
    long fail_after;
    int allocation_failure_observed = 0;
    int successful_inverse_observed = 0;

    allocations_before_failure = -1;
    assert(numerus_matrix_create_dense(2, 2, values, &source) ==
        NUMERUS_MATRIX_SUCCESS);

    for (fail_after = 0; fail_after < 32; fail_after++) {
        numerus_matrix *inverse = (numerus_matrix *) 1;
        numerus_matrix_status status;

        allocations_before_failure = fail_after;
        status = numerus_matrix_inverse(source, &inverse);

        if (status == NUMERUS_MATRIX_SUCCESS) {
            assert(inverse != NULL);
            numerus_matrix_destroy(inverse);
            successful_inverse_observed = 1;
            break;
        }

        assert(status == NUMERUS_MATRIX_OUT_OF_MEMORY);
        assert(inverse == NULL);
        allocation_failure_observed = 1;
    }

    allocations_before_failure = -1;
    assert(allocation_failure_observed);
    assert(successful_inverse_observed);

    /* Failed construction must leave the source valid and reusable. */
    {
        numerus_matrix *inverse = NULL;
        double value;

        assert(numerus_matrix_get(source, 0, 0, &value) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(value == 4.0);
        assert(numerus_matrix_inverse(source, &inverse) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(inverse != NULL);
        numerus_matrix_destroy(inverse);
    }

    numerus_matrix_destroy(source);
}

static void test_solve_allocation_failures(void)
{
    const double matrix_values[] = {4.0, 1.0, 2.0, 3.0};
    const double rhs_values[] = {1.0, 2.0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    long fail_after;
    int allocation_failure_observed = 0;
    int successful_solve_observed = 0;

    allocations_before_failure = -1;
    assert(numerus_matrix_create_dense(2, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);

    for (fail_after = 0; fail_after < 32; fail_after++) {
        numerus_matrix *solution = (numerus_matrix *) 1;
        numerus_matrix_status status;

        allocations_before_failure = fail_after;
        status = numerus_matrix_solve(matrix, rhs, &solution);

        if (status == NUMERUS_MATRIX_SUCCESS) {
            assert(solution != NULL);
            numerus_matrix_destroy(solution);
            successful_solve_observed = 1;
            break;
        }

        assert(status == NUMERUS_MATRIX_OUT_OF_MEMORY);
        assert(solution == NULL);
        allocation_failure_observed = 1;
    }

    allocations_before_failure = -1;
    assert(allocation_failure_observed);
    assert(successful_solve_observed);

    /* The same operands must still work after injected failures. */
    {
        numerus_matrix *solution = NULL;

        assert(numerus_matrix_solve(matrix, rhs, &solution) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(solution != NULL);
        numerus_matrix_destroy(solution);
    }

    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}


static void test_selection_view_allocation_failures(void)
{
    const double values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const size_t selected_rows[] = {1, 0};
    numerus_matrix *source = NULL;
    long fail_after;
    int allocation_failure_observed = 0;
    int successful_view_observed = 0;

    allocations_before_failure = -1;
    assert(numerus_matrix_create_dense(3, 2, values, &source) ==
        NUMERUS_MATRIX_SUCCESS);

    for (fail_after = 0; fail_after < 16; fail_after++) {
        numerus_matrix *view = (numerus_matrix *) 1;
        numerus_matrix_status status;

        allocations_before_failure = fail_after;
        status = numerus_matrix_create_select_rows(
            source, selected_rows, 2, &view
        );

        if (status == NUMERUS_MATRIX_SUCCESS) {
            double value;

            assert(view != NULL);
            assert(numerus_matrix_get(view, 0, 0, &value) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(value == 3.0);
            numerus_matrix_destroy(view);
            successful_view_observed = 1;
            break;
        }

        assert(status == NUMERUS_MATRIX_OUT_OF_MEMORY);
        assert(view == NULL);
        allocation_failure_observed = 1;
    }

    allocations_before_failure = -1;
    assert(allocation_failure_observed);
    assert(successful_view_observed);

    /* Failed view construction must not damage the borrowed parent. */
    {
        double value;

        assert(numerus_matrix_get(source, 2, 1, &value) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(value == 6.0);
    }

    numerus_matrix_destroy(source);
}


static void test_block_grid_allocation_failures(void)
{
    const double a_values[] = {1.0, 2.0};
    const double b_values[] = {3.0};
    const double c_values[] = {4.0, 5.0, 6.0, 7.0};
    const double d_values[] = {8.0, 9.0};
    numerus_matrix *a = NULL;
    numerus_matrix *b = NULL;
    numerus_matrix *c = NULL;
    numerus_matrix *d = NULL;
    numerus_matrix *blocks[4];
    long fail_after;
    int allocation_failure_observed = 0;
    int successful_view_observed = 0;

    allocations_before_failure = -1;
    assert(numerus_matrix_create_dense(1, 2, a_values, &a) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 1, b_values, &b) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, c_values, &c) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, d_values, &d) ==
        NUMERUS_MATRIX_SUCCESS);

    blocks[0] = a;
    blocks[1] = b;
    blocks[2] = c;
    blocks[3] = d;

    for (fail_after = 0; fail_after < 24; fail_after++) {
        numerus_matrix *grid = (numerus_matrix *) 1;
        numerus_matrix_status status;

        allocations_before_failure = fail_after;
        status = numerus_matrix_create_block_grid(blocks, 2, 2, &grid);

        if (status == NUMERUS_MATRIX_SUCCESS) {
            double value;

            assert(grid != NULL);
            assert(numerus_matrix_rows(grid) == 3);
            assert(numerus_matrix_columns(grid) == 3);
            assert(numerus_matrix_get(grid, 2, 2, &value) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(value == 9.0);
            numerus_matrix_destroy(grid);
            successful_view_observed = 1;
            break;
        }

        assert(status == NUMERUS_MATRIX_OUT_OF_MEMORY);
        assert(grid == NULL);
        allocation_failure_observed = 1;
    }

    allocations_before_failure = -1;
    assert(allocation_failure_observed);
    assert(successful_view_observed);

    /* The view borrows its blocks; failed construction must preserve them. */
    {
        double value;

        assert(numerus_matrix_get(d, 1, 0, &value) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(value == 9.0);
    }

    numerus_matrix_destroy(d);
    numerus_matrix_destroy(c);
    numerus_matrix_destroy(b);
    numerus_matrix_destroy(a);
}

int main(void)
{
    const double values[] = {1.0, 2.0, 3.0, 4.0};
    long fail_after;
    int allocation_failure_observed = 0;
    int successful_creation_observed = 0;

    /*
     * Fail each allocation in turn until the operation succeeds. This makes
     * the test sensitive to added allocation points without hard-coding the
     * current allocation count.
     */
    for (fail_after = 0; fail_after < 16; fail_after++) {
        numerus_matrix *matrix = (numerus_matrix *) 1;
        numerus_matrix_status status;

        allocations_before_failure = fail_after;
        status = numerus_matrix_create_dense(2, 2, values, &matrix);

        if (status == NUMERUS_MATRIX_SUCCESS) {
            assert(matrix != NULL);
            numerus_matrix_destroy(matrix);
            successful_creation_observed = 1;
            break;
        }

        assert(status == NUMERUS_MATRIX_OUT_OF_MEMORY);
        assert(matrix == NULL);
        allocation_failure_observed = 1;
    }

    assert(allocation_failure_observed);
    assert(successful_creation_observed);

    test_materialize_allocation_failures();
    test_addition_allocation_failures();
    test_inverse_allocation_failures();
    test_solve_allocation_failures();
    test_selection_view_allocation_failures();
    test_block_grid_allocation_failures();
    allocations_before_failure = -1;

    /* A failure must not poison later operations. */
    {
        numerus_matrix *matrix = NULL;

        assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(matrix != NULL);
        numerus_matrix_destroy(matrix);
    }

    puts("Matrix allocation-failure tests passed.");
    return 0;
}
