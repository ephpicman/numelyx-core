/**
 * @file matrix_weighted_least_squares_test.c
 * @brief Native tests for weighted least-squares solutions and weight validation.
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

static void test_equal_weights_match_ordinary_least_squares(void)
{
    const double matrix_values[] = {
        1, 0,
        0, 1,
        1, 1
    };
    const double rhs_values[] = {1, 2, 4};
    const double weights[] = {1, 1, 1};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *ordinary = NULL;
    numerus_matrix *weighted = NULL;
    size_t row;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(matrix, rhs, &ordinary) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_weighted_least_squares(
        matrix, rhs, weights, 3, &weighted
    ) == NUMERUS_MATRIX_SUCCESS);

    for (row = 0; row < 2; row++) {
        double left;
        double right;
        assert(numerus_matrix_get(ordinary, row, 0, &left) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_get(weighted, row, 0, &right) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(fabs(left - right) < 1e-10);
    }

    numerus_matrix_destroy(weighted);
    numerus_matrix_destroy(ordinary);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_zero_weight_excludes_observation(void)
{
    const double matrix_values[] = {
        1, 0,
        0, 1,
        1, 1
    };
    const double rhs_values[] = {1, 2, 100};
    const double weights[] = {1, 1, 0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = NULL;
    double value;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_weighted_least_squares(
        matrix, rhs, weights, 3, &solution
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 1.0) < 1e-10);
    assert(numerus_matrix_get(solution, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 2.0) < 1e-10);

    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_unequal_weights_and_multiple_rhs(void)
{
    const double matrix_values[] = {
        1, 0,
        0, 1,
        1, 1
    };
    const double rhs_values[] = {
        1, 2,
        2, 4,
        4, 8
    };
    const double weights[] = {1, 2, 3};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = NULL;
    double value;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 2, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_weighted_least_squares(
        matrix, rhs, weights, 3, &solution
    ) == NUMERUS_MATRIX_SUCCESS);

    /* Exact weighted normal-equation solution is [17/11, 25/11]^T.
     * The second RHS is twice the first, so its coefficient column doubles.
     * This catches implementations that accidentally ignore unequal weights.
     */
    assert(numerus_matrix_rows(solution) == 2);
    assert(numerus_matrix_columns(solution) == 2);
    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 17.0 / 11.0) < 1e-9);
    assert(numerus_matrix_get(solution, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 25.0 / 11.0) < 1e-9);
    assert(numerus_matrix_get(solution, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 34.0 / 11.0) < 1e-9);
    assert(numerus_matrix_get(solution, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 50.0 / 11.0) < 1e-9);

    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_infinite_weight_is_rejected(void)
{
    const double matrix_values[] = {1, 0, 0, 1, 1, 1};
    const double rhs_values[] = {1, 2, 4};
    const double weights[] = {1, INFINITY, 1};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_weighted_least_squares(
        matrix, rhs, weights, 3, &solution
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);

    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_validation_and_rank_failure(void)
{
    const double matrix_values[] = {1, 0, 1, 0, 1, 1};
    const double rhs_values[] = {1, 2, 3};
    const double bad_weights[] = {1, -1, 1};
    const double nan_weights[] = {1, NAN, 1};
    const double zero_weights[] = {1, 0, 0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_weighted_least_squares(
        matrix, rhs, bad_weights, 3, &solution
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(solution == NULL);

    assert(numerus_matrix_weighted_least_squares(
        matrix, rhs, nan_weights, 3, &solution
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);

    assert(numerus_matrix_weighted_least_squares(
        matrix, rhs, zero_weights, 2, &solution
    ) == NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(solution == NULL);

    assert(numerus_matrix_weighted_least_squares(
        matrix, rhs, zero_weights, 3, &solution
    ) == NUMERUS_MATRIX_RANK_DEFICIENT);
    assert(solution == NULL);

    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_rhs_dimension_and_nonfinite_data(void)
{
    const double matrix_values[] = {1, 0, 0, 1, 1, 1};
    const double rhs_values[] = {1, 2, 4};
    const double short_rhs_values[] = {1, 2};
    const double weights[] = {1, 2, 3};
    const double nonfinite_matrix_values[] = {1, 0, 0, 1, NAN, 1};
    const double nonfinite_rhs_values[] = {1, INFINITY, 4};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *short_rhs = NULL;
    numerus_matrix *bad_matrix = NULL;
    numerus_matrix *bad_rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, short_rhs_values, &short_rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_weighted_least_squares(
        matrix, short_rhs, weights, 3, &solution
    ) == NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(solution == NULL);

    assert(numerus_matrix_create_dense(
        3, 2, nonfinite_matrix_values, &bad_matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_weighted_least_squares(
        bad_matrix, rhs, weights, 3, &solution
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);

    assert(numerus_matrix_create_dense(
        3, 1, nonfinite_rhs_values, &bad_rhs
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_weighted_least_squares(
        matrix, bad_rhs, weights, 3, &solution
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);

    numerus_matrix_destroy(bad_rhs);
    numerus_matrix_destroy(bad_matrix);
    numerus_matrix_destroy(short_rhs);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_nonfinite_and_read_failures(void)
{
    const double matrix_values[] = {1, 0, 0, 1, 1, 1};
    const double rhs_values[] = {1, 2, 3};
    const double weights[] = {1, 1, 1};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 3, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_weighted_least_squares(
        failing, rhs, weights, 3, &solution
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(solution == NULL);

    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_equal_weights_match_ordinary_least_squares();
    test_zero_weight_excludes_observation();
    test_unequal_weights_and_multiple_rhs();
    test_infinite_weight_is_rejected();
    test_validation_and_rank_failure();
    test_rhs_dimension_and_nonfinite_data();
    test_nonfinite_and_read_failures();
    puts("Matrix weighted least-squares tests passed.");
    return 0;
}
