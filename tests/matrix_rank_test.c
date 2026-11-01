/**
 * @file matrix_rank_test.c
 * @brief Native tests for numerical rank estimation and tolerance-sensitive matrices.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
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

    if (row == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void test_full_and_deficient_rank(void)
{
    const double full_values[] = {1, 2, 3, 4};
    const double singular_values[] = {1, 2, 2, 4};
    const double rectangular_values[] = {1, 0, 0, 1, 1, 1};
    const double zero_values[] = {0, 0, 0, 0, 0, 0};
    numelyx_core_matrix *matrix = NULL;
    size_t rank = 99;

    assert(numelyx_core_matrix_create_dense(2, 2, full_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 2);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, singular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 1);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 3, rectangular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 2);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 3, zero_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 0);
    numelyx_core_matrix_destroy(matrix);
}

static void test_scale_aware_threshold(void)
{
    const double near_singular[] = {1, 0, 0, 1e-12};
    const double scaled_near_singular[] = {1e100, 0, 0, 1e88};
    const double scaled_full_rank[] = {1e-100, 0, 0, 2e-100};
    numelyx_core_matrix *matrix = NULL;
    size_t rank = 99;

    assert(numelyx_core_matrix_create_dense(2, 2, near_singular, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 1);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(
        2, 2, scaled_near_singular, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 1);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, scaled_full_rank, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 2);
    numelyx_core_matrix_destroy(matrix);
}

static void test_diagonal_rank_fast_path_contract(void)
{
    const double full_values[] = {1e100, 0.0, 0.0, 2e100};
    const double deficient_values[] = {1.0, 0.0, 0.0, 1e-12};
    const double rectangular_values[] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    numelyx_core_matrix *matrix = NULL;
    size_t rank = 99;

    assert(numelyx_core_matrix_create_diagonal(
        2, full_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 2);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_diagonal(
        2, deficient_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 1);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(
        2, 3, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_SUCCESS);
    assert(rank == 1);
    numelyx_core_matrix_destroy(matrix);
}

static void test_nonfinite_and_failure_paths(void)
{
    const double nan_values[] = {1, NAN, 3, 4};
    const double infinity_values[] = {1, INFINITY, 3, 4};
    const double values[] = {1, 2, 3, 4};
    numelyx_core_matrix *matrix = NULL;
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *failing = NULL;
    numelyx_core_matrix *huge = NULL;
    size_t rank = 765;

    assert(numelyx_core_matrix_rank(NULL, &rank) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(rank == 765);
    assert(numelyx_core_matrix_rank(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numelyx_core_matrix_create_dense(2, 2, nan_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_NON_FINITE);
    assert(rank == 765);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, infinity_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(matrix, &rank) == NUMERUS_MATRIX_NON_FINITE);
    assert(rank == 765);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(failing, &rank) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(rank == 765);
    numelyx_core_matrix_destroy(failing);
    numelyx_core_matrix_destroy(root);

    assert(numelyx_core_matrix_create_dense(1, 1, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_from_parent(
        root, SIZE_MAX, 2, &huge
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_rank(huge, &rank) == NUMERUS_MATRIX_OVERFLOW);
    assert(rank == 765);
    numelyx_core_matrix_destroy(huge);
    numelyx_core_matrix_destroy(root);
}

int main(void)
{
    test_full_and_deficient_rank();
    test_scale_aware_threshold();
    test_diagonal_rank_fast_path_contract();
    test_nonfinite_and_failure_paths();
    puts("Matrix numerical rank tests passed.");
    return 0;
}
