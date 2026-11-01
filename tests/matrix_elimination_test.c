/**
 * @file matrix_elimination_test.c
 * @brief Native tests for Gaussian elimination, row operations, and echelon-form invariants.
 *
 * @details These native tests define regression coverage for the named Matrix
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

static void assert_value(
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    double expected
)
{
    double value = 0.0;
    assert(numerus_matrix_get(matrix, row, column, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - expected) < 1e-9);
}

static void test_ref_and_rref(void)
{
    const double values[] = {
        0, 2, 4,
        1, 3, 5
    };
    numerus_matrix *matrix = NULL;
    numerus_matrix *ref = NULL;
    numerus_matrix *rref = NULL;

    assert(numerus_matrix_create_dense(2, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_row_echelon_form(matrix, &ref) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(ref) == 2);
    assert(numerus_matrix_columns(ref) == 3);
    assert_value(ref, 0, 0, 1);
    assert_value(ref, 0, 1, 3);
    assert_value(ref, 0, 2, 5);
    assert_value(ref, 1, 0, 0);
    assert_value(ref, 1, 1, 2);
    assert_value(ref, 1, 2, 4);

    assert(numerus_matrix_reduced_row_echelon_form(matrix, &rref) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_value(rref, 0, 0, 1);
    assert_value(rref, 0, 1, 0);
    assert_value(rref, 0, 2, -1);
    assert_value(rref, 1, 0, 0);
    assert_value(rref, 1, 1, 1);
    assert_value(rref, 1, 2, 2);

    numerus_matrix_destroy(rref);
    numerus_matrix_destroy(ref);
    numerus_matrix_destroy(matrix);
}

static void test_zero_pivot_and_scale_threshold(void)
{
    const double leading_zero[] = {0, 1, 0, 2};
    const double near_singular[] = {1, 0, 0, 1e-12};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rref = NULL;

    assert(numerus_matrix_create_dense(2, 2, leading_zero, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_reduced_row_echelon_form(matrix, &rref) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_value(rref, 0, 0, 0);
    assert_value(rref, 0, 1, 1);
    assert_value(rref, 1, 0, 0);
    assert_value(rref, 1, 1, 0);
    numerus_matrix_destroy(rref);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, near_singular, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_row_echelon_form(matrix, &rref) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_value(rref, 1, 1, 0);
    numerus_matrix_destroy(rref);
    numerus_matrix_destroy(matrix);
}

static void test_nonfinite_and_failure_paths(void)
{
    const double nan_values[] = {NAN, 1, 2, 3};
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *matrix = NULL;
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *result = (void *) 1;

    assert(numerus_matrix_row_echelon_form(NULL, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);

    assert(numerus_matrix_create_dense(2, 2, nan_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_reduced_row_echelon_form(matrix, &result) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(result == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_row_echelon_form(failing, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result == NULL);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_ref_and_rref();
    test_zero_pivot_and_scale_threshold();
    test_nonfinite_and_failure_paths();
    puts("Matrix echelon form tests passed.");
    return 0;
}
