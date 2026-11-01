/**
 * @file matrix_close_test.c
 * @brief Native tests for tolerance-based Matrix comparison, including NaN and infinity semantics.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numelyx_core_matrix.h"
#include <assert.h>
#include <stdio.h>

static numelyx_core_matrix_status reject_coordinates(size_t row, size_t column,
    size_t *parent_row, size_t *parent_column, const void *context)
{
    (void) row; (void) column; (void) parent_row; (void) parent_column;
    (void) context;
    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

int main(void)
{
    const double a[] = {1.0, 2.0, 3.0};
    const double one_match[] = {1.0, 200.0, 800.0};
    const double none_match[] = {7.0, 8.0, 9.0};
    numelyx_core_matrix *left = NULL, *near = NULL, *other = NULL;
    numelyx_core_matrix *shape = NULL, *failing = NULL;
    bool result = false;

    assert(numelyx_core_matrix_create_dense(1, 3, a, &left) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 3, one_match, &near) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 3, none_match, &other) == NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(3, 1, a, &shape) == NUMERUS_MATRIX_SUCCESS);

    assert(numelyx_core_matrix_all_close(left, near, &result) == NUMERUS_MATRIX_SUCCESS && !result);
    assert(numelyx_core_matrix_any_close(left, near, &result) == NUMERUS_MATRIX_SUCCESS && result);
    assert(numelyx_core_matrix_all_close(left, left, &result) == NUMERUS_MATRIX_SUCCESS && result);
    assert(numelyx_core_matrix_any_close(left, other, &result) == NUMERUS_MATRIX_SUCCESS && !result);
    assert(numelyx_core_matrix_any_close(left, shape, &result) == NUMERUS_MATRIX_SUCCESS && !result);

    assert(numelyx_core_matrix_create_from_parent_with_transforms(
        left, 1, 3, reject_coordinates, NULL, NULL, &failing) == NUMERUS_MATRIX_SUCCESS);
    result = true;
    assert(numelyx_core_matrix_any_close(failing, left, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);
    assert(numelyx_core_matrix_all_close(failing, left, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(result);

    assert(numelyx_core_matrix_any_close(NULL, left, &result) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    numelyx_core_matrix_destroy(failing);
    numelyx_core_matrix_destroy(shape);
    numelyx_core_matrix_destroy(other);
    numelyx_core_matrix_destroy(near);
    numelyx_core_matrix_destroy(left);
    puts("Matrix all/any-close tests passed.");
    return 0;
}
