/**
 * @file matrix_inverse_test.c
 * @brief Native tests for Matrix inversion, identity reconstruction, and singular matrices.
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

static double determinant_small(double values[3][3], size_t size)
{
    if (size == 1) {
        return values[0][0];
    }
    if (size == 2) {
        return values[0][0] * values[1][1] -
            values[0][1] * values[1][0];
    }

    return values[0][0] * (
            values[1][1] * values[2][2] - values[1][2] * values[2][1]
        ) -
        values[0][1] * (
            values[1][0] * values[2][2] - values[1][2] * values[2][0]
        ) +
        values[0][2] * (
            values[1][0] * values[2][1] - values[1][1] * values[2][0]
        );
}

/*
 * Tiny, test-only cofactor reference. Keep it independent from the production
 * LU path; this is deliberately limited to 1x1..3x3 matrices.
 */
static void cofactor_inverse_reference(
    double input[3][3],
    size_t size,
    double output[3][3]
)
{
    double determinant = determinant_small(input, size);

    assert(size >= 1 && size <= 3);
    assert(determinant != 0.0);

    for (size_t row = 0; row < size; row++) {
        for (size_t column = 0; column < size; column++) {
            double minor[3][3] = {{0.0}};
            size_t minor_row = 0;

            if (size == 1) {
                output[column][row] = 1.0 / determinant;
                continue;
            }

            for (size_t source_row = 0; source_row < size; source_row++) {
                size_t minor_column = 0;
                if (source_row == row) {
                    continue;
                }
                for (size_t source_column = 0; source_column < size; source_column++) {
                    if (source_column == column) {
                        continue;
                    }
                    minor[minor_row][minor_column++] =
                        input[source_row][source_column];
                }
                minor_row++;
            }

            {
                double cofactor = determinant_small(minor, size - 1);
                if ((row + column) % 2 != 0) {
                    cofactor = -cofactor;
                }
                output[column][row] = cofactor / determinant;
            }
        }
    }
}

static void assert_matches_cofactor_reference(
    const numerus_matrix *matrix,
    const numerus_matrix *inverse
)
{
    double input[3][3] = {{0.0}};
    double expected[3][3] = {{0.0}};
    size_t size = numerus_matrix_rows(matrix);

    assert(size >= 1 && size <= 3);
    for (size_t row = 0; row < size; row++) {
        for (size_t column = 0; column < size; column++) {
            assert(numerus_matrix_get(
                matrix, row, column, &input[row][column]
            ) == NUMERUS_MATRIX_SUCCESS);
        }
    }
    cofactor_inverse_reference(input, size, expected);

    for (size_t row = 0; row < size; row++) {
        for (size_t column = 0; column < size; column++) {
            double actual;
            assert(numerus_matrix_get(
                inverse, row, column, &actual
            ) == NUMERUS_MATRIX_SUCCESS);
            assert(fabs(actual - expected[row][column]) < 1e-9);
        }
    }
}

static void assert_identity_product(
    const numerus_matrix *matrix,
    const numerus_matrix *inverse
)
{
    size_t size = numerus_matrix_rows(matrix);
    size_t row;
    size_t column;

    for (row = 0; row < size; row++) {
        for (column = 0; column < size; column++) {
            size_t k;
            double value = 0.0;

            for (k = 0; k < size; k++) {
                double left;
                double right;
                assert(numerus_matrix_get(matrix, row, k, &left) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(inverse, k, column, &right) ==
                    NUMERUS_MATRIX_SUCCESS);
                value += left * right;
            }
            assert(fabs(value - (row == column ? 1.0 : 0.0)) < 1e-9);
        }
    }
}

static void test_known_and_pivoted_inverses(void)
{
    const double values[] = {4, 7, 2, 6};
    const double pivot_values[] = {0, 1, 1, 0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *inverse = NULL;

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(inverse) == 2);
    assert(numerus_matrix_columns(inverse) == 2);
    assert_identity_product(matrix, inverse);
    assert_matches_cofactor_reference(matrix, inverse);
    numerus_matrix_destroy(inverse);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, pivot_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert_identity_product(matrix, inverse);
    numerus_matrix_destroy(inverse);
    numerus_matrix_destroy(matrix);
}

static void test_three_by_three_cofactor_reference(void)
{
    const double values[] = {3, 0, 2, 2, 0, -2, 0, 1, 1};
    numerus_matrix *matrix = NULL;
    numerus_matrix *inverse = NULL;

    assert(numerus_matrix_create_dense(3, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) == NUMERUS_MATRIX_SUCCESS);
    assert_identity_product(matrix, inverse);
    assert_matches_cofactor_reference(matrix, inverse);

    numerus_matrix_destroy(inverse);
    numerus_matrix_destroy(matrix);
}


static void test_inverse_cache_returns_independent_results(void)
{
    const double values[] = {4, 7, 2, 6};
    numerus_matrix *matrix = NULL;
    numerus_matrix *first = NULL;
    numerus_matrix *second = NULL;
    double value;

    assert(numerus_matrix_create_dense(2, 2, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &first) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &second) == NUMERUS_MATRIX_SUCCESS);
    assert(first != second);
    assert_identity_product(matrix, first);
    assert_identity_product(matrix, second);

    /* Both returned matrices survive destruction of the cache owner. */
    numerus_matrix_destroy(matrix);
    assert(numerus_matrix_get(first, 0, 0, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 0.6) < 1e-12);
    assert(numerus_matrix_get(second, 1, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 0.4) < 1e-12);

    numerus_matrix_destroy(first);
    numerus_matrix_destroy(second);
}

static void test_singularity_and_nonfinite_semantics(void)
{
    const double singular_values[] = {1, 2, 2, 4};
    const double near_singular_values[] = {1, 0, 0, 1e-12};
    const double nan_values[] = {1, NAN, 0, 1};
    numerus_matrix *matrix = NULL;
    numerus_matrix *inverse = (void *) 1;

    assert(numerus_matrix_create_dense(2, 2, singular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SINGULAR);
    assert(inverse == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        2, 2, near_singular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SINGULAR);
    assert(inverse == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, nan_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(inverse == NULL);
    numerus_matrix_destroy(matrix);
}

static void test_validation_and_scale(void)
{
    const double small_values[] = {1e-100, 0, 0, 2e-100};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *matrix = NULL;
    numerus_matrix *inverse = (void *) 1;
    double value;

    assert(numerus_matrix_inverse(NULL, &inverse) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(inverse == NULL);
    assert(numerus_matrix_inverse(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_dense(
        2, 3, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(inverse == NULL);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, small_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(inverse, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 1e100) / 1e100 < 1e-12);
    assert(numerus_matrix_get(inverse, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 5e99) / 5e99 < 1e-12);
    assert_identity_product(matrix, inverse);
    numerus_matrix_destroy(inverse);
    numerus_matrix_destroy(matrix);
}


static void test_one_by_one_inverse(void)
{
    const double input[] = {4.0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *inverse = NULL;
    double value;

    assert(numerus_matrix_create_dense(1, 1, input, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(inverse) == 1);
    assert(numerus_matrix_columns(inverse) == 1);
    assert(numerus_matrix_get(inverse, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 0.25) < 1e-12);
    assert_matches_cofactor_reference(matrix, inverse);
    assert_identity_product(matrix, inverse);

    numerus_matrix_destroy(inverse);
    numerus_matrix_destroy(matrix);
}


static void test_inverse_overflow_returns_nonfinite(void)
{
    const double values[] = {1e-320};
    numerus_matrix *matrix = NULL;
    numerus_matrix *inverse = (void *) 1;

    assert(numerus_matrix_create_dense(1, 1, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &inverse) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(inverse == NULL);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_inverse_overflow_returns_nonfinite();
    test_one_by_one_inverse();
    test_known_and_pivoted_inverses();
    test_three_by_three_cofactor_reference();
    test_inverse_cache_returns_independent_results();
    test_singularity_and_nonfinite_semantics();
    test_validation_and_scale();
    puts("Matrix inverse tests passed.");
    return 0;
}
