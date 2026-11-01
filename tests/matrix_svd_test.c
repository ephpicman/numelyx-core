/**
 * @file matrix_svd_test.c
 * @brief Native tests for singular value decomposition, reconstruction, and orthogonality.
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
    if (row == 1) return NUMERUS_MATRIX_INVALID_ARGUMENT;
    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void assert_svd(
    const numerus_matrix *matrix,
    const numerus_matrix *u,
    const numerus_matrix *singular,
    const numerus_matrix *vt,
    double tolerance
)
{
    size_t rows = numerus_matrix_rows(matrix);
    size_t columns = numerus_matrix_columns(matrix);
    size_t k = rows < columns ? rows : columns;
    size_t row;
    size_t column;
    double previous = INFINITY;

    assert(numerus_matrix_rows(u) == rows);
    assert(numerus_matrix_columns(u) == k);
    assert(numerus_matrix_rows(singular) == k);
    assert(numerus_matrix_columns(singular) == 1);
    assert(numerus_matrix_rows(vt) == k);
    assert(numerus_matrix_columns(vt) == columns);

    for (column = 0; column < k; column++) {
        double value;
        assert(numerus_matrix_get(singular, column, 0, &value) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(value >= 0.0);
        assert(value <= previous + tolerance);
        previous = value;
    }

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            size_t index;
            double product = 0.0;
            double expected;

            for (index = 0; index < k; index++) {
                double u_value;
                double s_value;
                double vt_value;
                assert(numerus_matrix_get(u, row, index, &u_value) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(singular, index, 0, &s_value) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(vt, index, column, &vt_value) ==
                    NUMERUS_MATRIX_SUCCESS);
                product += u_value * s_value * vt_value;
            }

            assert(numerus_matrix_get(matrix, row, column, &expected) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(fabs(product - expected) <=
                tolerance * (1.0 + fabs(expected)));
        }
    }

    for (row = 0; row < k; row++) {
        for (column = 0; column < k; column++) {
            size_t index;
            double u_dot = 0.0;
            double vt_dot = 0.0;

            for (index = 0; index < rows; index++) {
                double left;
                double right;
                assert(numerus_matrix_get(u, index, row, &left) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(u, index, column, &right) ==
                    NUMERUS_MATRIX_SUCCESS);
                u_dot += left * right;
            }
            for (index = 0; index < columns; index++) {
                double left;
                double right;
                assert(numerus_matrix_get(vt, row, index, &left) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(vt, column, index, &right) ==
                    NUMERUS_MATRIX_SUCCESS);
                vt_dot += left * right;
            }
            assert(fabs(u_dot - (row == column ? 1.0 : 0.0)) < tolerance);
            assert(fabs(vt_dot - (row == column ? 1.0 : 0.0)) < tolerance);
        }
    }
}

static void test_tall_and_wide_matrices(void)
{
    const double tall_values[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 10,
        2, -1, 1
    };
    const double wide_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *matrix = NULL;
    numerus_matrix *u = NULL;
    numerus_matrix *singular = NULL;
    numerus_matrix *vt = NULL;

    assert(numerus_matrix_create_dense(4, 3, tall_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_svd(matrix, &u, &singular, &vt) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_svd(matrix, u, singular, vt, 1e-8);
    numerus_matrix_destroy(vt);
    numerus_matrix_destroy(singular);
    numerus_matrix_destroy(u);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 3, wide_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_svd(matrix, &u, &singular, &vt) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_svd(matrix, u, singular, vt, 1e-8);
    numerus_matrix_destroy(vt);
    numerus_matrix_destroy(singular);
    numerus_matrix_destroy(u);
    numerus_matrix_destroy(matrix);
}

static void test_rank_deficient_and_zero_matrices(void)
{
    const double deficient_values[] = {1, 2, 2, 4, 3, 6};
    const double zero_values[] = {0, 0, 0, 0, 0, 0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *u = NULL;
    numerus_matrix *singular = NULL;
    numerus_matrix *vt = NULL;

    assert(numerus_matrix_create_dense(3, 2, deficient_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_svd(matrix, &u, &singular, &vt) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_svd(matrix, u, singular, vt, 1e-8);
    numerus_matrix_destroy(vt);
    numerus_matrix_destroy(singular);
    numerus_matrix_destroy(u);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(3, 2, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_svd(matrix, &u, &singular, &vt) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_svd(matrix, u, singular, vt, 1e-8);
    numerus_matrix_destroy(vt);
    numerus_matrix_destroy(singular);
    numerus_matrix_destroy(u);
    numerus_matrix_destroy(matrix);
}

static void test_failure_contracts(void)
{
    const double values[] = {1, 2, 3, 4};
    const double nonfinite_values[] = {1, NAN, 3, 4};
    numerus_matrix *matrix = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *u = (void *) 1;
    numerus_matrix *singular = (void *) 1;
    numerus_matrix *vt = (void *) 1;

    assert(numerus_matrix_svd(NULL, &u, &singular, &vt) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(u == NULL && singular == NULL && vt == NULL);
    assert(numerus_matrix_svd(NULL, &u, &u, &vt) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(u == NULL && vt == NULL);

    assert(numerus_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_svd(matrix, &u, &singular, &vt) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(u == NULL && singular == NULL && vt == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_svd(failing, &u, &singular, &vt) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(u == NULL && singular == NULL && vt == NULL);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_tall_and_wide_matrices();
    test_rank_deficient_and_zero_matrices();
    test_failure_contracts();
    puts("Matrix SVD tests passed.");
    return 0;
}
