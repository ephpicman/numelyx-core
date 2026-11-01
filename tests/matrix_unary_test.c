/**
 * @file matrix_unary_test.c
 * @brief Native tests for unary Matrix operations and value/shape preservation.
 *
 * @details These native tests define regression coverage for the named API
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static int matrix_value_equals(
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    double expected
)
{
    double actual = 0.0;

    return numerus_matrix_get(
        matrix, row, column, &actual
    ) == NUMERUS_MATRIX_SUCCESS && actual == expected;
}

static void test_scalar_multiplication_and_negation_views(void)
{
    const double source[] = {1.5, -2.0, 0.0, 4.0, NAN, INFINITY};
    const double scaled[] = {3.0, -4.0, 0.0, 8.0, NAN, INFINITY};
    const double negated[] = {-1.5, 2.0, -0.0, -4.0, NAN, -INFINITY};
    numerus_matrix *root = NULL;
    numerus_matrix *scaled_view = NULL;
    numerus_matrix *negated_view = NULL;
    numerus_matrix *nested = NULL;
    double value = 123.0;

    assert(numerus_matrix_create_dense(2, 3, source, &root) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_create_scale(root, 2.0, &scaled_view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(scaled_view, 0, 0, scaled[0]));
    assert(matrix_value_equals(scaled_view, 0, 1, scaled[1]));
    assert(matrix_value_equals(scaled_view, 0, 2, scaled[2]));
    assert(matrix_value_equals(scaled_view, 1, 0, scaled[3]));
    assert(numerus_matrix_get(scaled_view, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));
    assert(numerus_matrix_get(scaled_view, 1, 2, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isinf(value) && value > 0.0);
    assert(matrix_value_equals(root, 0, 0, 1.5));
    assert(matrix_value_equals(root, 1, 0, 4.0));

    assert(numerus_matrix_create_negate(root, &negated_view) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(negated_view, 0, 0, negated[0]));
    assert(matrix_value_equals(negated_view, 0, 1, negated[1]));
    assert(matrix_value_equals(negated_view, 0, 2, negated[2]));
    assert(matrix_value_equals(negated_view, 1, 0, negated[3]));
    assert(numerus_matrix_get(negated_view, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));
    assert(numerus_matrix_get(negated_view, 1, 2, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isinf(value) && value < 0.0);
    assert(matrix_value_equals(root, 0, 1, -2.0));

    /* Nested lazy transforms compose without materializing or borrowing stack data. */
    assert(numerus_matrix_create_scale(scaled_view, 0.5, &nested) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(matrix_value_equals(nested, 0, 0, 1.5));
    assert(matrix_value_equals(nested, 1, 0, 4.0));
    assert(numerus_matrix_get(nested, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));

    numerus_matrix_destroy(nested);
    numerus_matrix_destroy(negated_view);

    scaled_view = root;
    assert(numerus_matrix_create_scale(NULL, 2.0, &scaled_view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(scaled_view == NULL);
    assert(numerus_matrix_create_negate(NULL, &scaled_view) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(scaled_view == NULL);
    assert(numerus_matrix_create_scale(NULL, 2.0, NULL) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);

    numerus_matrix_destroy(scaled_view);
    numerus_matrix_destroy(root);
}

 
static void test_scalar_addition_and_subtraction_views(void)
{
    const double source[] = {1.5, -2.0, 0.0, 4.0, NAN, INFINITY};
    numerus_matrix *root = NULL;
    numerus_matrix *added = NULL;
    numerus_matrix *subtracted = NULL;
    double value = 0.0;

    assert(numerus_matrix_create_dense(2, 3, source, &root) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_scalar_add(root, 2.0, &added) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_scalar_subtract(root, 2.0, &subtracted) == NUMERUS_MATRIX_SUCCESS);

    assert(matrix_value_equals(added, 0, 0, 3.5));
    assert(matrix_value_equals(added, 0, 1, 0.0));
    assert(matrix_value_equals(added, 0, 2, 2.0));
    assert(matrix_value_equals(subtracted, 0, 0, -0.5));
    assert(matrix_value_equals(subtracted, 0, 1, -4.0));
    assert(matrix_value_equals(subtracted, 0, 2, -2.0));
    assert(numerus_matrix_get(added, 1, 1, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(isnan(value));
    assert(numerus_matrix_get(subtracted, 1, 2, &value) == NUMERUS_MATRIX_SUCCESS);
    assert(isinf(value) && value > 0.0);
    assert(matrix_value_equals(root, 0, 0, 1.5));

    numerus_matrix_destroy(added);
    numerus_matrix_destroy(subtracted);
    numerus_matrix_destroy(root);

    added = (void *) 1;
    assert(numerus_matrix_create_scalar_add(NULL, 1.0, &added) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(added == NULL);
    assert(numerus_matrix_create_scalar_subtract(NULL, 1.0, &added) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(added == NULL);
}

int main(void)
{
    test_scalar_multiplication_and_negation_views();
    test_scalar_addition_and_subtraction_views();
    puts("Matrix unary operation tests passed.");
    return 0;
}
