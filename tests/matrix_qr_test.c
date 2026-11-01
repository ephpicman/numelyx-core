/**
 * @file matrix_qr_test.c
 * @brief Native tests for QR factorization correctness and edge cases.
 *
 * @details This file is part of Numerus's internal C implementation. Keep its
 * contracts aligned with the Matrix/Storage interfaces and the corresponding
 * implementation units; these declarations do not constitute a stable public
 * C ABI unless explicitly documented otherwise.
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

static void assert_qr_reconstructs(
    const numerus_matrix *matrix,
    const numerus_matrix *q,
    const numerus_matrix *r
)
{
    size_t rows = numerus_matrix_rows(matrix);
    size_t columns = numerus_matrix_columns(matrix);
    size_t k = rows < columns ? rows : columns;
    size_t row;
    size_t column;

    assert(numerus_matrix_rows(q) == rows);
    assert(numerus_matrix_columns(q) == k);
    assert(numerus_matrix_rows(r) == k);
    assert(numerus_matrix_columns(r) == columns);

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            size_t index;
            double actual;
            double expected;
            double product = 0.0;

            for (index = 0; index < k; index++) {
                double q_value;
                double r_value;
                assert(numerus_matrix_get(q, row, index, &q_value) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(r, index, column, &r_value) ==
                    NUMERUS_MATRIX_SUCCESS);
                product += q_value * r_value;
            }

            assert(numerus_matrix_get(
                matrix, row, column, &expected
            ) == NUMERUS_MATRIX_SUCCESS);
            actual = product;
            assert(fabs(actual - expected) <= 1e-9 * (1.0 + fabs(expected)));
        }
    }

    for (row = 0; row < k; row++) {
        for (column = 0; column < k; column++) {
            size_t index;
            double dot = 0.0;

            for (index = 0; index < rows; index++) {
                double left;
                double right;
                assert(numerus_matrix_get(q, index, row, &left) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(q, index, column, &right) ==
                    NUMERUS_MATRIX_SUCCESS);
                dot += left * right;
            }
            assert(fabs(dot - (row == column ? 1.0 : 0.0)) < 1e-9);
        }
    }

    for (row = 1; row < k; row++) {
        for (column = 0; column < row && column < columns; column++) {
            double value;
            assert(numerus_matrix_get(r, row, column, &value) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(value == 0.0);
        }
    }
}

static void test_tall_and_wide_matrices(void)
{
    const double tall_values[] = {
        12, -51, 4,
        6, 167, -68,
        -4, 24, -41,
        1, 1, 1
    };
    const double wide_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *matrix = NULL;
    numerus_matrix *q = NULL;
    numerus_matrix *r = NULL;

    assert(numerus_matrix_create_dense(4, 3, tall_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_qr_decompose(matrix, &q, &r) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_qr_reconstructs(matrix, q, r);
    numerus_matrix_destroy(r);
    numerus_matrix_destroy(q);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 3, wide_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_qr_decompose(matrix, &q, &r) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_qr_reconstructs(matrix, q, r);
    numerus_matrix_destroy(r);
    numerus_matrix_destroy(q);
    numerus_matrix_destroy(matrix);
}

static void test_rank_deficient_and_zero_matrices(void)
{
    const double deficient_values[] = {
        1, 2, 3,
        2, 4, 6,
        0, 0, 0
    };
    const double zero_values[] = {0, 0, 0, 0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *q = NULL;
    numerus_matrix *r = NULL;

    assert(numerus_matrix_create_dense(3, 3, deficient_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_qr_decompose(matrix, &q, &r) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_qr_reconstructs(matrix, q, r);
    numerus_matrix_destroy(r);
    numerus_matrix_destroy(q);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_qr_decompose(matrix, &q, &r) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_qr_reconstructs(matrix, q, r);
    numerus_matrix_destroy(r);
    numerus_matrix_destroy(q);
    numerus_matrix_destroy(matrix);
}

static void test_failure_contracts(void)
{
    const double values[] = {1, 2, 3, 4};
    const double nonfinite_values[] = {1, NAN, 3, 4};
    numerus_matrix *matrix = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *q = (void *) 1;
    numerus_matrix *r = (void *) 1;

    assert(numerus_matrix_qr_decompose(NULL, &q, &r) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(q == NULL && r == NULL);
    assert(numerus_matrix_qr_decompose(NULL, &q, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(q == NULL);
    assert(numerus_matrix_qr_decompose(NULL, &q, &q) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(q == NULL);

    assert(numerus_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_qr_decompose(matrix, &q, &r) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(q == NULL && r == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_qr_decompose(failing, &q, &r) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(q == NULL && r == NULL);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_tall_and_wide_matrices();
    test_rank_deficient_and_zero_matrices();
    test_failure_contracts();
    puts("Matrix QR tests passed.");
    return 0;
}
