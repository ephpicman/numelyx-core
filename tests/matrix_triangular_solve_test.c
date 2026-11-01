/**
 * @file matrix_triangular_solve_test.c
 * @brief Native tests for forward/back substitution and triangular-system edge cases.
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

static void test_lower_solve_multiple_rhs(void)
{
    const double lower_values[] = {2, 0, 3, 1};
    const double rhs_values[] = {2, 4, 5, 7};
    numerus_matrix *lower = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = NULL;
    double value;

    assert(numerus_matrix_create_dense(2, 2, lower_values, &lower) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        lower, rhs, true, false, &solution
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 1.0);
    assert(numerus_matrix_get(solution, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 2.0);
    assert(numerus_matrix_get(solution, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 2.0);
    assert(numerus_matrix_get(solution, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 1.0);

    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(lower);
}

static void test_transpose_and_upper_solve(void)
{
    const double lower_values[] = {2, 0, 3, 1};
    const double upper_values[] = {2, 3, 0, 1};
    const double transpose_rhs_values[] = {5, 4};
    const double upper_rhs_values[] = {8, 2};
    numerus_matrix *factor = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = NULL;
    double value;

    assert(numerus_matrix_create_dense(2, 2, lower_values, &factor) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, transpose_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        factor, rhs, true, true, &solution
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value + 3.5) < 1e-12);
    assert(numerus_matrix_get(solution, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 4.0);
    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(factor);

    assert(numerus_matrix_create_dense(2, 2, upper_values, &factor) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, upper_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        factor, rhs, false, false, &solution
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 1.0);
    assert(numerus_matrix_get(solution, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 2.0);

    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(factor);
}


static void test_diagonal_covariance_whitening_reference(void)
{
    const double covariance_values[] = {
        1, 0, 0,
        0, 4, 0,
        0, 0, 9
    };
    const double design_values[] = {1, 1, 1};
    const double observation_values[] = {1, 2, 4};
    numerus_matrix *covariance = NULL;
    numerus_matrix *lower = NULL;
    numerus_matrix *design = NULL;
    numerus_matrix *observations = NULL;
    numerus_matrix *whitened_design = NULL;
    numerus_matrix *whitened_observations = NULL;
    numerus_matrix *coefficients = NULL;
    double value;

    assert(numerus_matrix_create_dense(
        3, 3, covariance_values, &covariance
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_cholesky(covariance, &lower) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, design_values, &design) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(
        3, 1, observation_values, &observations
    ) == NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_solve_triangular(
        lower, design, true, false, &whitened_design
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        lower, observations, true, false, &whitened_observations
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(
        whitened_design, whitened_observations, &coefficients
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(coefficients, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 10.0 / 7.0) < 1e-12);

    numerus_matrix_destroy(coefficients);
    numerus_matrix_destroy(whitened_observations);
    numerus_matrix_destroy(whitened_design);
    numerus_matrix_destroy(observations);
    numerus_matrix_destroy(design);
    numerus_matrix_destroy(lower);
    numerus_matrix_destroy(covariance);
}

static void test_validation_and_output_preservation(void)
{
    const double malformed_values[] = {2, 1, 3, 1};
    const double zero_diagonal_values[] = {2, 0, 3, 0};
    const double nan_values[] = {2, 0, NAN, 1};
    const double rhs_values[] = {1, 2};
    const double mismatched_rhs_values[] = {1, 2, 3};
    const double non_square_values[] = {1, 0, 0, 1, 1, 1};
    numerus_matrix *factor = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(2, 2, malformed_values, &factor) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        factor, rhs, true, false, &solution
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(factor);

    assert(numerus_matrix_create_dense(2, 2, zero_diagonal_values, &factor) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        factor, rhs, true, false, &solution
    ) == NUMERUS_MATRIX_SINGULAR);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(factor);

    assert(numerus_matrix_create_dense(2, 2, nan_values, &factor) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        factor, rhs, true, false, &solution
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(factor);

    assert(numerus_matrix_create_dense(
        2, 2, (const double[]){2, 0, 3, 1}, &factor
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(
        3, 1, mismatched_rhs_values, &rhs
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        factor, rhs, true, false, &solution
    ) == NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(factor);

    assert(numerus_matrix_create_dense(
        2, 3, non_square_values, &factor
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        factor, rhs, true, false, &solution
    ) == NUMERUS_MATRIX_NOT_SQUARE);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(factor);

    assert(numerus_matrix_solve_triangular(
        NULL, NULL, true, false, &solution
    ) == NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(solution == NULL);
}

static void test_nonfinite_rhs(void)
{
    const double lower_values[] = {2, 0, 3, 1};
    const double rhs_values[] = {NAN, 1};
    numerus_matrix *lower = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(2, 2, lower_values, &lower) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        lower, rhs, true, false, &solution
    ) == NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);

    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(lower);
}

int main(void)
{
    test_lower_solve_multiple_rhs();
    test_transpose_and_upper_solve();
    test_diagonal_covariance_whitening_reference();
    test_validation_and_output_preservation();
    test_nonfinite_rhs();
    puts("Matrix triangular solve tests passed.");
    return 0;
}
