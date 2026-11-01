/**
 * @file matrix_structural_predicate_test.c
 * @brief Native tests for Matrix structural predicates such as diagonal, triangular, and symmetric.
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

    if (row == 1) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }

    *parent_row = row;
    *parent_column = column;
    return NUMERUS_MATRIX_SUCCESS;
}

static void test_skew_symmetric(void)
{
    const double values[] = {
         0,  2, -3,
        -2,  0,  4,
         3, -4,  0
    };
    const double non_skew_values[] = {0, 1, 2, 0};
    const double rectangular_values[] = {0, 1, 2, 0, 0, 3};
    numelyx_core_matrix *matrix = NULL;
    bool result = false;

    assert(numelyx_core_matrix_create_dense(3, 3, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_is_skew_symmetric(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(result);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, non_skew_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_is_skew_symmetric(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 3, rectangular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_is_skew_symmetric(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    numelyx_core_matrix_destroy(matrix);
}

static void test_orthogonal(void)
{
    const double identity_values[] = {1, 0, 0, 1};
    const double rotation_values[] = {0, -1, 1, 0};
    const double non_orthogonal_values[] = {1, 1, 0, 1};
    const double rectangular_values[] = {1, 0, 0, 1, 1, 0};
    numelyx_core_matrix *matrix = NULL;
    bool result = false;

    assert(numelyx_core_matrix_create_dense(2, 2, identity_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_is_orthogonal(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(result);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, rotation_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_is_orthogonal(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(result);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 2, non_orthogonal_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_is_orthogonal(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    numelyx_core_matrix_destroy(matrix);

    assert(numelyx_core_matrix_create_dense(2, 3, rectangular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_is_orthogonal(matrix, &result) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!result);
    numelyx_core_matrix_destroy(matrix);
}

static void test_validation_and_read_failure(void)
{
    const double values[] = {0, -1, 1, 0};
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *failing = NULL;
    bool result = true;

    assert(numelyx_core_matrix_is_skew_symmetric(NULL, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numelyx_core_matrix_is_orthogonal(NULL, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numelyx_core_matrix_is_orthogonal(NULL, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    assert(numelyx_core_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_from_parent_with_transforms(
        root, 2, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    result = true;
    assert(numelyx_core_matrix_is_skew_symmetric(failing, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numelyx_core_matrix_is_orthogonal(failing, &result) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);

    numelyx_core_matrix_destroy(failing);
    numelyx_core_matrix_destroy(root);
}

int main(void)
{
    test_skew_symmetric();
    test_orthogonal();
    test_validation_and_read_failure();
    puts("Matrix structural predicate tests passed.");
    return 0;
}
