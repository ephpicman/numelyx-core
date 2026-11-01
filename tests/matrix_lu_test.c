/**
 * @file matrix_lu_test.c
 * @brief Native tests for LU factorization, pivot permutations, and solve/reconstruction behavior.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix_lu.h"

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

static void test_factor_reconstruction_and_pivoting(void)
{
    const double values[] = {
        0, 2, 1,
        3, 4, 5,
        1, 0, 6
    };
    numerus_matrix *matrix = NULL;
    numerus_matrix_lu_factorization *lu = NULL;
    size_t size;
    size_t row;
    size_t column;

    assert(numerus_matrix_create_dense(3, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_lu_factorize(matrix, &lu) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(lu != NULL);
    assert(numerus_matrix_lu_size(lu) == 3);
    assert(numerus_matrix_lu_rank(lu) == 3);
    assert(numerus_matrix_lu_permutation_sign(lu) == -1);

    size = numerus_matrix_lu_size(lu);
    for (row = 0; row < size; row++) {
        size_t source_row;
        assert(numerus_matrix_lu_get_permutation(
            lu, row, &source_row
        ) == NUMERUS_MATRIX_SUCCESS);

        for (column = 0; column < size; column++) {
            double sum = 0.0;
            double expected;
            double actual;

            assert(numerus_matrix_get(
                matrix, source_row, column, &expected
            ) == NUMERUS_MATRIX_SUCCESS);

            for (size_t k = 0; k < size; k++) {
                double lower;
                double upper;
                assert(numerus_matrix_lu_get_lower(lu, row, k, &lower) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_lu_get_upper(lu, k, column, &upper) ==
                    NUMERUS_MATRIX_SUCCESS);
                sum += lower * upper;
            }

            actual = sum;
            assert(fabs(actual - expected) < 1e-12);
        }
    }

    {
        double value = 321.0;
        assert(numerus_matrix_lu_get_lower(lu, 3, 0, &value) ==
            NUMERUS_MATRIX_OUT_OF_BOUNDS);
        assert(value == 321.0);
    }

    numerus_matrix_lu_destroy(lu);
    numerus_matrix_destroy(matrix);
}

static void test_rank_deficiency_and_zero_pivot(void)
{
    const double singular_values[] = {1, 2, 2, 4};
    const double leading_zero_values[] = {0, 1, 0, 2};
    numerus_matrix *matrix = NULL;
    numerus_matrix_lu_factorization *lu = NULL;

    assert(numerus_matrix_create_dense(2, 2, singular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_lu_factorize(matrix, &lu) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_lu_rank(lu) == 1);
    numerus_matrix_lu_destroy(lu);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, leading_zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_lu_factorize(matrix, &lu) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_lu_rank(lu) == 1);
    assert(numerus_matrix_lu_permutation_sign(lu) == 1);
    numerus_matrix_lu_destroy(lu);
    numerus_matrix_destroy(matrix);
}

static void test_validation_and_read_failure(void)
{
    const double values[] = {1, 2, 3, 4};
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    numerus_matrix *root = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *rectangular = NULL;
    numerus_matrix_lu_factorization *lu = (void *) 1;

    assert(numerus_matrix_lu_factorize(NULL, &lu) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(lu == NULL);
    assert(numerus_matrix_lu_factorize(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numerus_matrix_create_dense(
        2, 3, rectangular_values, &rectangular
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_lu_factorize(rectangular, &lu) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(lu == NULL);
    numerus_matrix_destroy(rectangular);

    assert(numerus_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_lu_factorize(failing, &lu) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(lu == NULL);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(root);
}

int main(void)
{
    test_factor_reconstruction_and_pivoting();
    test_rank_deficiency_and_zero_pivot();
    test_validation_and_read_failure();
    puts("Matrix LU factorization tests passed.");
    return 0;
}
