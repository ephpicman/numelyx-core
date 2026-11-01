/**
 * @file matrix_ldlt_test.c
 * @brief Native tests for LDLᵀ factorization and symmetric-system numerical properties.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static numelyx_core_matrix_status fail_on_row_one(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) column;
    (void) context;
    if (row == 1) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void assert_ldlt_reconstructs(
    const numelyx_core_matrix *matrix,
    const numelyx_core_matrix *lower,
    const numelyx_core_matrix *diagonal,
    double tolerance
)
{
    size_t size = numelyx_core_matrix_rows(matrix);
    size_t row;
    size_t column;

    for (row = 0; row < size; row++) {
        double unit;
        assert(numelyx_core_matrix_get(lower, row, row, &unit) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(fabs(unit - 1.0) < 1e-12);

        for (column = 0; column < size; column++) {
            size_t k;
            double product = 0.0;
            double expected;

            for (k = 0; k < size; k++) {
                double left;
                double d;
                double right;
                assert(numelyx_core_matrix_get(lower, row, k, &left) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numelyx_core_matrix_get(diagonal, k, k, &d) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numelyx_core_matrix_get(lower, column, k, &right) ==
                    NUMERUS_MATRIX_SUCCESS);
                product += left * d * right;
            }
            assert(numelyx_core_matrix_get(
                matrix, row, column, &expected
            ) == NUMERUS_MATRIX_SUCCESS);
            assert(fabs(product - expected) <=
                tolerance * (1.0 + fabs(expected)));
        }
    }

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            double value;
            assert(numelyx_core_matrix_get(diagonal, row, column, &value) ==
                NUMERUS_MATRIX_SUCCESS);
            if (row != column) assert(value == 0.0);
        }
    }
}

static void test_positive_definite_and_indefinite(void)
{
    const double positive_values[] = {
        4, 12, -16,
        12, 37, -43,
        -16, -43, 98
    };
    const double indefinite_values[] = {1, 2, 2, 1};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *lower = NULL;
    numelyx_core_matrix *diagonal = NULL;
    double value;

    assert(numelyx_core_matrix_create_dense(3, 3, positive_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(matrix, &lower, &diagonal) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(diagonal, 2, 2, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 9.0) < 1e-12);
    assert_ldlt_reconstructs(matrix, lower, diagonal, 1e-10);
    numelyx_core_matrix_destroy(diagonal);
    numelyx_core_matrix_destroy(lower);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, indefinite_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(matrix, &lower, &diagonal) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_get(diagonal, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value + 3.0) < 1e-12);
    assert_ldlt_reconstructs(matrix, lower, diagonal, 1e-10);
    numelyx_core_matrix_destroy(diagonal);
    numelyx_core_matrix_destroy(lower);
    numelyx_core_matrix_destroy(matrix);
}

static void test_pivoting_limit_and_scale(void)
{
    const double requires_pivot_values[] = {0, 1, 1, 0};
    const double singular_values[] = {1, 2, 2, 4};
    const double small_values[] = {1e-100, 0, 0, 2e-100};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *lower = (void *) 1;
    numelyx_core_matrix *diagonal = (void *) 1;

    assert(numelyx_core_matrix_create_dense(
        2, 2, requires_pivot_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(matrix, &lower, &diagonal) ==
        NUMERUS_MATRIX_PIVOT_TOO_SMALL);
    assert(lower == NULL && diagonal == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, singular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(matrix, &lower, &diagonal) ==
        NUMERUS_MATRIX_PIVOT_TOO_SMALL);
    assert(lower == NULL && diagonal == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, small_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(matrix, &lower, &diagonal) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_ldlt_reconstructs(matrix, lower, diagonal, 1e-9);
    numelyx_core_matrix_destroy(diagonal);
    numelyx_core_matrix_destroy(lower);
    numelyx_core_matrix_destroy(matrix);
}

static void test_validation_and_read_failures(void)
{
    const double nonsymmetric_values[] = {1, 0, 1, 1};
    const double nonfinite_values[] = {1, NAN, NAN, 1};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *failing = NULL;
    numelyx_core_matrix *lower = (void *) 1;
    numelyx_core_matrix *diagonal = (void *) 1;

    assert(numelyx_core_matrix_ldlt_decompose(NULL, &lower, &diagonal) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(lower == NULL && diagonal == NULL);

    assert(numelyx_core_matrix_create_dense(
        2, 3, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(matrix, &lower, &diagonal) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(lower == NULL && diagonal == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(
        2, 2, nonsymmetric_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(matrix, &lower, &diagonal) ==
        NUMERUS_MATRIX_NOT_SYMMETRIC);
    assert(lower == NULL && diagonal == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(matrix, &lower, &diagonal) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(lower == NULL && diagonal == NULL);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(
        2, 2, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_ldlt_decompose(failing, &lower, &diagonal) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(lower == NULL && diagonal == NULL);
    numelyx_core_matrix_destroy(failing);
    numelyx_core_matrix_destroy(matrix);
}

int main(void)
{
    test_positive_definite_and_indefinite();
    test_pivoting_limit_and_scale();
    test_validation_and_read_failures();
    puts("Matrix LDLT tests passed.");
    return 0;
}
