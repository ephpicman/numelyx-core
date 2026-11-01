/**
 * @file matrix_solve_test.c
 * @brief Native tests for linear-system solves, residuals, and singular/rank-deficient cases.
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

static void assert_solution(
    const numerus_matrix *matrix,
    const numerus_matrix *solution,
    const numerus_matrix *right_hand_side
)
{
    size_t rows = numerus_matrix_rows(matrix);
    size_t columns = numerus_matrix_columns(right_hand_side);
    size_t row;
    size_t column;

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            size_t k;
            double value = 0.0;
            double expected;

            for (k = 0; k < rows; k++) {
                double left;
                double right;
                assert(numerus_matrix_get(matrix, row, k, &left) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(solution, k, column, &right) ==
                    NUMERUS_MATRIX_SUCCESS);
                value += left * right;
            }
            assert(numerus_matrix_get(
                right_hand_side, row, column, &expected
            ) == NUMERUS_MATRIX_SUCCESS);
            assert(fabs(value - expected) < 1e-9);
        }
    }
}

static void test_multiple_right_hand_sides(void)
{
    const double matrix_values[] = {3, 1, 1, 2};
    const double rhs_values[] = {9, 4, 8, 3};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = NULL;
    double value;

    assert(numerus_matrix_create_dense(2, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_SUCCESS);

    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 2.0) < 1e-12);
    assert(numerus_matrix_get(solution, 1, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 3.0) < 1e-12);
    assert(numerus_matrix_get(solution, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 1.0) < 1e-12);
    assert(numerus_matrix_get(solution, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 1.0) < 1e-12);

    assert_solution(matrix, solution, rhs);
    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_row_pivoting_and_errors(void)
{
    const double pivot_values[] = {0, 1, 1, 0};
    const double rhs_values[] = {2, 4, 3, 5};
    const double singular_values[] = {1, 2, 2, 4};
    const double mismatched_values[] = {1, 2, 3};
    const double nan_rhs_values[] = {NAN, 1};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;
    double value;

    assert(numerus_matrix_create_dense(2, 2, pivot_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 3.0);
    assert(numerus_matrix_get(solution, 1, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(value == 4.0);
    assert_solution(matrix, solution, rhs);
    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, singular_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 2, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_SINGULAR);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);

    assert(numerus_matrix_create_dense(2, 2, pivot_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(3, 1, mismatched_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_DIMENSION_MISMATCH);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);

    assert(numerus_matrix_create_dense(2, 1, nan_rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}


static void test_lu_cache_reuse_and_source_lifetime(void)
{
    const double matrix_values[] = {4, 1, 2, 3};
    const double rhs_values[] = {1, 2};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *first = NULL;
    numerus_matrix *second = NULL;
    double a, b;

    assert(numerus_matrix_create_dense(2, 2, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);

    /* Determinant, solve and inverse share the source's retained LU factors. */
    {
        double determinant;
        assert(numerus_matrix_determinant(matrix, &determinant) ==
            NUMERUS_MATRIX_SUCCESS);
        assert(fabs(determinant - 10.0) < 1e-12);
    }
    assert(numerus_matrix_solve(matrix, rhs, &first) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_inverse(matrix, &second) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(first, 0, 0, &a) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_get(first, 1, 0, &b) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(a - 0.1) < 1e-12);
    assert(fabs(b - 0.6) < 1e-12);

    numerus_matrix_destroy(matrix);
    assert(numerus_matrix_get(first, 0, 0, &a) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(a - 0.1) < 1e-12);
    assert(numerus_matrix_get(second, 0, 0, &a) == NUMERUS_MATRIX_SUCCESS);
    assert(fabs(a - 0.3) < 1e-12);

    numerus_matrix_destroy(first);
    numerus_matrix_destroy(second);
    numerus_matrix_destroy(rhs);
}

static void test_nonfinite_coefficients_and_nonsquare_input(void)
{
    const double rectangular_values[] = {1, 2, 3, 4, 5, 6};
    const double nonfinite_values[] = {1, NAN, 0, 1};
    const double rhs_values[] = {1, 2};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(
        2, 3, rectangular_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_NOT_SQUARE);
    assert(solution == NULL);
    numerus_matrix_destroy(matrix);
    numerus_matrix_destroy(rhs);

    assert(numerus_matrix_create_dense(
        2, 2, nonfinite_values, &matrix
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(2, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    solution = (void *) 1;
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}


static void test_one_by_one_system(void)
{
    const double matrix_values[] = {4.0};
    const double rhs_values[] = {10.0, -6.0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = NULL;
    double value;

    assert(numerus_matrix_create_dense(1, 1, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 2, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_rows(solution) == 1);
    assert(numerus_matrix_columns(solution) == 2);

    assert(numerus_matrix_get(solution, 0, 0, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value - 2.5) < 1e-12);
    assert(numerus_matrix_get(solution, 0, 1, &value) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(fabs(value + 1.5) < 1e-12);
    assert_solution(matrix, solution, rhs);

    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}


static void test_solve_overflow_returns_nonfinite(void)
{
    const double matrix_values[] = {1e-320};
    const double rhs_values[] = {1.0};
    numerus_matrix *matrix = NULL;
    numerus_matrix *rhs = NULL;
    numerus_matrix *solution = (void *) 1;

    assert(numerus_matrix_create_dense(1, 1, matrix_values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_create_dense(1, 1, rhs_values, &rhs) ==
        NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve(matrix, rhs, &solution) ==
        NUMERUS_MATRIX_NON_FINITE);
    assert(solution == NULL);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

int main(void)
{
    test_solve_overflow_returns_nonfinite();
    test_one_by_one_system();
    test_lu_cache_reuse_and_source_lifetime();
    test_multiple_right_hand_sides();
    test_row_pivoting_and_errors();
    test_nonfinite_coefficients_and_nonsquare_input();
    puts("Matrix linear solve tests passed.");
    return 0;
}
