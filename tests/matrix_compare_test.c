/**
 * @file matrix_compare_test.c
 * @brief Native tests for exact Matrix equality and its shape and non-finite-value behavior.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numelyx_core_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static numelyx_core_matrix_status reject_coordinates(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
)
{
    (void) row;
    (void) column;
    (void) parent_row;
    (void) parent_column;
    (void) context;

    return NUMERUS_MATRIX_INVALID_ARGUMENT;
}

static void test_exact_equality(void)
{
    const double left_values[] = {1.0, INFINITY, 0.0, -2.0};
    const double equal_values[] = {1.0, INFINITY, -0.0, -2.0};
    const double different_values[] = {1.0 + 5e-10, INFINITY, 0.0, -2.0};
    numelyx_core_matrix *left = NULL;
    numelyx_core_matrix *equal_matrix = NULL;
    numelyx_core_matrix *different = NULL;
    bool equal = false;

    assert(numelyx_core_matrix_create_dense(1, 4, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 4, equal_values, &equal_matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 4, different_values, &different) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numelyx_core_matrix_is_equal(left, equal_matrix, &equal) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(equal); /* +0 and -0 compare equal; equal infinities compare equal. */

    assert(numelyx_core_matrix_is_equal(left, different, &equal) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!equal);

    numelyx_core_matrix_destroy(different);
    numelyx_core_matrix_destroy(equal_matrix);
    numelyx_core_matrix_destroy(left);
}

static void test_approximate_equality_and_nan(void)
{
    const double left_values[] = {1.0, 1e12, INFINITY, NAN};
    const double close_values[] = {1.0 + 5e-10, 1e12 + 500.0, INFINITY, NAN};
    numelyx_core_matrix *left = NULL;
    numelyx_core_matrix *right = NULL;
    bool close = false;

    assert(numelyx_core_matrix_create_dense(1, 4, left_values, &left) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 4, close_values, &right) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numelyx_core_matrix_is_close(left, right, &close) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!close); /* NaN is unequal to itself, even for approximate equality. */

    numelyx_core_matrix_destroy(right);
    numelyx_core_matrix_destroy(left);

    {
        const double finite_left[] = {1.0, 1e12, INFINITY};
        const double finite_right[] = {1.0 + 5e-10, 1e12 + 500.0, INFINITY};

        assert(numelyx_core_matrix_create_dense(1, 3, finite_left, &left) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numelyx_core_matrix_create_dense(1, 3, finite_right, &right) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numelyx_core_matrix_is_close(left, right, &close) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(close);
        numelyx_core_matrix_destroy(right);
        numelyx_core_matrix_destroy(left);
    }
}

static void test_shape_mismatch_and_failure_preservation(void)
{
    const double values[] = {1, 2, 3, 4};
    numelyx_core_matrix *root = NULL;
    numelyx_core_matrix *row = NULL;
    numelyx_core_matrix *failing = NULL;
    bool equal = true;
    bool close = false;

    assert(numelyx_core_matrix_create_dense(2, 2, values, &root) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numelyx_core_matrix_create_dense(1, 4, values, &row) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numelyx_core_matrix_is_equal(root, row, &equal) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!equal);
    assert(numelyx_core_matrix_is_close(root, row, &close) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(!close);

    assert(numelyx_core_matrix_create_from_parent_with_transforms(
        root, 2, 2, reject_coordinates, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);

    equal = true;
    assert(numelyx_core_matrix_is_equal(failing, root, &equal) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(equal);
    close = false;
    assert(numelyx_core_matrix_is_close(root, failing, &close) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(!close);

    assert(numelyx_core_matrix_is_equal(NULL, root, &equal) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numelyx_core_matrix_is_close(root, NULL, &close) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(numelyx_core_matrix_is_equal(root, root, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numelyx_core_matrix_destroy(failing);
    numelyx_core_matrix_destroy(row);
    numelyx_core_matrix_destroy(root);
}

int main(void)
{
    test_exact_equality();
    test_approximate_equality_and_nan();
    test_shape_mismatch_and_failure_preservation();
    puts("Matrix comparison tests passed.");
    return 0;
}
