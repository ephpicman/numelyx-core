/**
 * @file matrix_system_test.c
 * @brief Native tests for classifying linear systems by consistency and solution uniqueness.
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
    (void) column;
    (void) context;
    if (row == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void test_unique_and_infinite_systems(void)
{
    const double identity_values[] = {1, 0, 0, 1};
    const double multiple_rhs_values[] = {3, 4, 5, 6};
    const double underdetermined_values[] = {1, 1};
    const double underdetermined_rhs_values[] = {2};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix_solution_kind kind = NUMERUS_MATRIX_SOLUTION_INCONSISTENT;

    assert(numerus_matrix_create_dense(2, 2, identity_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, multiple_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(kind == NUMERUS_MATRIX_SOLUTION_UNIQUE);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        1, 2, underdetermined_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(
        1, 1, underdetermined_rhs_values, &rhs
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INFINITE);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_inconsistent_and_multiple_rhs(void)
{
    const double column_values[] = {1, 1};
    const double inconsistent_rhs_values[] = {1, 2};
    const double multiple_rhs_values[] = {1, 1, 0, 1};
    const double single_rhs_values[] = {1, 0};
    const double unique_column_values[] = {1, 0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix_solution_kind kind = NUMERUS_MATRIX_SOLUTION_UNIQUE;

    assert(numerus_matrix_create_dense(2, 1, column_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(
        2, 1, inconsistent_rhs_values, &rhs
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INCONSISTENT);
    numerus_matrix_destroy(rhs);

    assert(numerus_matrix_create_dense(2, 2, multiple_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INCONSISTENT);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 1, unique_column_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, single_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(kind == NUMERUS_MATRIX_SOLUTION_UNIQUE);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_zero_matrix_and_tolerance(void)
{
    const double zero_values[] = {0, 0, 0, 0};
    const double zero_rhs_values[] = {0, 0};
    const double nonzero_rhs_values[] = {0, 1};
    const double matrix_values[] = {1, 0};
    const double near_rhs_values[] = {1, 1e-12};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix_solution_kind kind;

    assert(numerus_matrix_create_dense(2, 2, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, zero_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INFINITE);
    numerus_matrix_destroy(rhs);

    assert(numerus_matrix_create_dense(2, 1, nonzero_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INCONSISTENT);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 1, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, near_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(kind == NUMERUS_MATRIX_SOLUTION_UNIQUE);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_errors_preserve_classification(void)
{
    const double matrix_values[] = {1, 2, 3, 4};
    const double rhs_values[] = {1, 2, 3};
    const double nan_rhs_values[] = {NAN, 1};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix_solution_kind kind = NUMERUS_MATRIX_SOLUTION_INFINITE;

    assert(numerus_matrix_classify_system(NULL, NULL, &kind) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INFINITE);

    assert(numerus_matrix_create_dense(2, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INFINITE);
    numerus_matrix_destroy(rhs);

    assert(numerus_matrix_create_dense(2, 1, nan_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(matrix, rhs, &kind) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INFINITE);
    numerus_matrix_destroy(rhs);

    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, matrix_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_classify_system(failing, rhs, &kind) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(kind == NUMERUS_MATRIX_SOLUTION_INFINITE);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_unique_and_infinite_systems();
    test_inconsistent_and_multiple_rhs();
    test_zero_matrix_and_tolerance();
    test_errors_preserve_classification();
    puts("Matrix system classification tests passed.");
    return 0;
}
