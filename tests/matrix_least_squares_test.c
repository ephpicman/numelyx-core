/**
 * @file matrix_least_squares_test.c
 * @brief Native tests for least-squares solutions, residuals, and rank-related failure semantics.
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

static void assert_residual(
    const numerus_matrix *matrix,
    const numerus_matrix *solution,
    const numerus_matrix *rhs,
    double tolerance
)
{
    size_t rows = numerus_matrix_rows(matrix);
    size_t rhs_columns = numerus_matrix_columns(rhs);
    size_t row;
    size_t column;

    for (row = 0; row < rows; row++) {
        for (column = 0; column < rhs_columns; column++) {
            size_t k;
            double value = 0.0;
            double expected;

            for (k = 0; k < numerus_matrix_columns(matrix); k++) {
                double a_value;
                double x_value;
                assert(numerus_matrix_get(matrix, row, k, &a_value) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(solution, k, column, &x_value) ==
                    NUMERUS_MATRIX_SUCCESS);
                value += a_value * x_value;
            }
            assert(numerus_matrix_get(rhs, row, column, &expected) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(fabs(value - expected) <= tolerance);
        }
    }
}

static void test_exact_multiple_rhs(void)
{
    const double matrix_values[] = {
        1, 0,
        0, 1,
        1, 1
    };
    const double rhs_values[] = {
        1, 2,
        2, 4,
        3, 6
    };
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = NULL;
    double value;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 2, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(solution) == 2);
    assert(numerus_matrix_columns(solution) == 2);
    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 1.0) < 1e-10);
    assert(numerus_matrix_get(solution, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 2.0) < 1e-10);
    assert(numerus_matrix_get(solution, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 2.0) < 1e-10);
    assert(numerus_matrix_get(solution, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 4.0) < 1e-10);
    assert_residual(matrix, solution, rhs, 1e-9);

    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_noisy_least_squares_solution(void)
{
    const double matrix_values[] = {
        1, 0,
        0, 1,
        1, 1
    };
    const double rhs_values[] = {1, 2, 4};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = NULL;
    double first;
    double second;
    double residual_sum = 0.0;
    size_t row;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(solution, 0, 0, &first) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(solution, 1, 0, &second) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(first - 4.0 / 3.0) < 1e-9);
    assert(fabs(second - 7.0 / 3.0) < 1e-9);

    for (row = 0; row < 3; row++) {
        double a0;
        double a1;
        double b;
        double residual;
        assert(numerus_matrix_get(matrix, row, 0, &a0) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_get(matrix, row, 1, &a1) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(numerus_matrix_get(rhs, row, 0, &b) ==
            NUMERUS_MATRIX_SUCCESS);
        residual = a0 * first + a1 * second - b;
        residual_sum += residual * residual;
    }
    assert(fabs(residual_sum - 1.0 / 3.0) < 1e-9);

    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_rank_and_dimension_errors(void)
{
    const double deficient_values[] = {1, 2, 2, 4};
    const double underdetermined_values[] = {1, 1};
    const double rhs_values[] = {1, 2};
    const double mismatched_rhs_values[] = {1, 2, 3};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(2, 2, deficient_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_RANK_DEFICIENT);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(
        1, 2, underdetermined_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_RANK_DEFICIENT);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, deficient_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(
        3, 1, mismatched_rhs_values, &rhs
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_nonfinite_and_read_failures(void)
{
    const double matrix_values[] = {1, 0, 0, 1, 1, 1};
    const double nan_rhs_values[] = {1, NAN, 3};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *failing = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(3, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, nan_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);

    assert(numerus_matrix_create_from_parent_with_transforms(
        matrix, 3, 2, fail_on_row_one, NULL, NULL, &failing
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, matrix_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(failing, rhs, &solution) ==
        NUMERUS_MATRIX_INVALID_ARGUMENT);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(failing);
    numerus_matrix_destroy(matrix);
}

static void test_nonfinite_coefficients(void)
{
    const double nonfinite_values[] = {
        1, NAN,
        0, 1,
        1, 1
    };
    const double rhs_values[] = {1, 2, 3};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(3, 2, nonfinite_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_least_squares(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);

    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}


static void test_minimum_norm_composition_for_underdetermined_system(void)
{
    const double matrix_values[] = {1.0, 1.0};
    const double rhs_values[] = {2.0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *pseudoinverse = NULL;
    numerus_matrix *solution = NULL;
    numerus_matrix *reconstructed = NULL;
    double first;
    double second;
    double reconstructed_rhs;
    double solution_norm_squared;
    double alternative_norm_squared;

    assert(numerus_matrix_create_dense(1, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);

    /* The QR least-squares API intentionally rejects underdetermined input. */
    assert(numerus_matrix_least_squares(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_RANK_DEFICIENT);
    assert(solution == NULL);

    /* Compose A+ b for the documented minimum-norm solution. */
    assert(numerus_matrix_pseudoinverse(matrix, &pseudoinverse) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_multiply(pseudoinverse, rhs, &solution) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(solution) == 2);
    assert(numerus_matrix_columns(solution) == 1);
    assert(numerus_matrix_get(solution, 0, 0, &first) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(solution, 1, 0, &second) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(first - 1.0) < 1e-10);
    assert(fabs(second - 1.0) < 1e-10);

    assert(numerus_matrix_multiply(matrix, solution, &reconstructed) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(reconstructed, 0, 0, &reconstructed_rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(reconstructed_rhs - 2.0) < 1e-10);

    solution_norm_squared = first * first + second * second;
    /* [2, 0] is another exact solution, but has a larger Euclidean norm. */
    alternative_norm_squared = 2.0 * 2.0;
    assert(solution_norm_squared < alternative_norm_squared);

    numerus_matrix_destroy(reconstructed);
    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(pseudoinverse);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_exact_multiple_rhs();
    test_noisy_least_squares_solution();
    test_rank_and_dimension_errors();
    test_nonfinite_and_read_failures();
    test_minimum_norm_composition_for_underdetermined_system();
    test_nonfinite_coefficients();
    puts("Matrix least-squares tests passed.");
    return 0;
}
