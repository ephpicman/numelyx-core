/**
 * @file matrix_trace_test.c
 * @brief Native tests for Matrix trace and square-dimension requirements.
 *
 * @details These native tests define regression coverage for the named API
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static numerus_matrix_status fail_on_row_one(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) context;

    if (row == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void test_trace_values_and_nonfinite_behavior(void)
{
    const double values[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    const double nonfinite_values[] = {
        NAN, 0,
        0, INFINITY
    };
    numerus_matrix *root = NULL;
    numerus_matrix *nonfinite = NULL;
    double trace = -123.0;

    assert(numerus_matrix_create_dense(3, 3, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_trace(root, &trace) == NUMERUS_MATRIX_SUCCESS);
    assert(trace == 15.0);

    assert(numerus_matrix_create_dense(
        2, 2, nonfinite_values, &nonfinite
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_trace(nonfinite, &trace) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(trace));

    numerus_matrix_destroy(nonfinite);
    numerus_matrix_destroy(root);
}

static void test_validation_and_read_failure(void)
{
    const double square_values[] = {1, 2, 3, 4};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *rectangular = NULL;
    double trace = 7654.25;

    assert(numerus_matrix_trace(NULL, &trace) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(trace == 7654.25);
    assert(numerus_matrix_trace(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_dense(
        2, 3, rectangular_values, &rectangular
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_trace(rectangular, &trace) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(trace == 7654.25);

    assert(numerus_matrix_create_dense(2, 2, square_values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_trace(failing, &trace) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(trace == 7654.25);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
    numerus_matrix_destroy(rectangular);
}

int main(void)
{
    test_trace_values_and_nonfinite_behavior();
    test_validation_and_read_failure();
    puts("Matrix trace tests passed.");
    return 0;
}
