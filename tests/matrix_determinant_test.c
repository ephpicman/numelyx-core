/**
 * @file matrix_determinant_test.c
 * @brief Native tests for determinant calculation, singular matrices, and numeric edge cases.
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

static void assert_determinant(
    numerus_matrix *matrix,
    double expected
)
{
    double determinant = -123.0;

    assert(numerus_matrix_determinant(matrix, &determinant) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(determinant - expected) < 1e-12);
}

static void test_basic_and_specialized_cases(void)
{
    const double identity_values[] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    const double zero_values[] = {0, 0, 0, 0};
    const double singular_values[] = {1, 2, 2, 4};
    const double triangular_values[] = {2, 3, 0, 4};
    const double pivot_values[] = {
        0, 2, 1,
        3, 4, 5,
        1, 0, 6
    };
    numerus_matrix *matrix = NULL;
    numerus_matrix_flags flags = {0};

    assert(numerus_matrix_create_dense(3, 3, identity_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_determinant(matrix, 1.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_determinant(matrix, 0.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, singular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_determinant(matrix, 0.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, triangular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get_flags(matrix, &flags) == NUMERUS_MATRIX_SUCCESS);
    assert(flags.upper_triangular);
    assert_determinant(matrix, 8.0);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(3, 3, pivot_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_determinant(matrix, -30.0);
    numerus_matrix_destroy(matrix);
}

static void test_scale_and_near_singular_cases(void)
{
    const double small_values[] = {1e-150, 0, 0, 1e-150};
    const double large_values[] = {1e150, 0, 0, 1e150};
    const double near_singular_values[] = {1, 1, 1, 1 + 1e-12};
    numerus_matrix *matrix = NULL;
    double determinant = 0.0;

    assert(numerus_matrix_create_dense(2, 2, small_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_determinant(matrix, &determinant) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(determinant > 0.0);
    assert(determinant < 1e-299);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, large_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_determinant(matrix, &determinant) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isfinite(determinant));
    assert(fabs(determinant / 1e300 - 1.0) < 1e-12);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, near_singular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_determinant(matrix, &determinant) ==
        NUMERUS_MATRIX_SUCCESS);
    /*
     * The pivot decision is exact-zero, not an absolute epsilon threshold.
     * This small but representable determinant must not be classified as zero.
     */
    assert(determinant > 0.0);
    assert(determinant < 2e-12);
    numerus_matrix_destroy(matrix);
}

static void test_failure_preserves_output(void)
{
    const double values[] = {1, 2, 3, 4};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    double determinant = 765.25;

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_determinant(failing, &determinant) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(determinant == 765.25);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_basic_and_specialized_cases();
    test_scale_and_near_singular_cases();
    test_failure_preserves_output();
    puts("Matrix determinant edge-case tests passed.");
    return 0;
}
