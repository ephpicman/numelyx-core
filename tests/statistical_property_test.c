/**
 * @file statistical_property_test.c
 * @brief Property-based tests for statistical numerical primitives and distribution invariants.
 *
 * @details These native tests define regression coverage for the named API
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_numeric.h"
#include "../numerus_optimizer.h"
#include "../numerus_probability.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static void assert_relative_close(double actual, double expected, double tolerance)
{
    double scale = fmax(1.0, fabs(expected));
    assert(fabs(actual - expected) <= tolerance * scale);
}

static numerus_matrix *make_dense(
    size_t rows,
    size_t columns,
    const double *values
)
{
    numerus_matrix *matrix = NULL;
    assert(numerus_matrix_create_dense(rows, columns, values, &matrix) ==
        NUMERUS_MATRIX_SUCCESS);
    return matrix;
}

static void test_cholesky_reconstruction_and_solve_residuals(void)
{
    const double matrix_values[] = {
        6, 2, 1,
        2, 5, 2,
        1, 2, 4
    };
    const double rhs_values[] = {1, -2, 3};
    numerus_matrix *matrix = make_dense(3, 3, matrix_values);
    numerus_matrix *rhs = make_dense(3, 1, rhs_values);
    numerus_matrix *lower = NULL;
    numerus_matrix *solution = NULL;
    numerus_matrix *transpose_solution = NULL;
    size_t row;
    size_t column;
    size_t inner;

    assert(numerus_matrix_cholesky(matrix, &lower) ==
        NUMERUS_MATRIX_SUCCESS);

    for (row = 0; row < 3; row++) {
        for (column = 0; column < 3; column++) {
            double reconstructed = 0.0;
            double expected;

            for (inner = 0; inner < 3; inner++) {
                double left;
                double right;
                assert(numerus_matrix_get(lower, row, inner, &left) ==
                    NUMERUS_MATRIX_SUCCESS);
                assert(numerus_matrix_get(lower, column, inner, &right) ==
                    NUMERUS_MATRIX_SUCCESS);
                reconstructed += left * right;
            }

            assert(numerus_matrix_get(matrix, row, column, &expected) ==
                NUMERUS_MATRIX_SUCCESS);
            assert_relative_close(reconstructed, expected, 1e-13);
        }
    }

    assert(numerus_matrix_solve_triangular(
        lower, rhs, true, false, &solution
    ) == NUMERUS_MATRIX_SUCCESS);
    assert(numerus_matrix_solve_triangular(
        lower, rhs, true, true, &transpose_solution
    ) == NUMERUS_MATRIX_SUCCESS);

    for (row = 0; row < 3; row++) {
        double reconstructed = 0.0;
        double transpose_reconstructed = 0.0;

        for (column = 0; column <= row; column++) {
            double coefficient;
            double value;
            assert(numerus_matrix_get(lower, row, column, &coefficient) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(numerus_matrix_get(solution, column, 0, &value) ==
                NUMERUS_MATRIX_SUCCESS);
            reconstructed += coefficient * value;
        }
        for (column = row; column < 3; column++) {
            double coefficient;
            double value;
            assert(numerus_matrix_get(lower, column, row, &coefficient) ==
                NUMERUS_MATRIX_SUCCESS);
            assert(numerus_matrix_get(
                transpose_solution, column, 0, &value
            ) == NUMERUS_MATRIX_SUCCESS);
            transpose_reconstructed += coefficient * value;
        }

        assert_relative_close(reconstructed, rhs_values[row], 1e-13);
        assert_relative_close(
            transpose_reconstructed, rhs_values[row], 1e-13
        );
    }

    numerus_matrix_destroy(transpose_solution);
    numerus_matrix_destroy(solution);
    numerus_matrix_destroy(lower);
    numerus_matrix_destroy(rhs);
    numerus_matrix_destroy(matrix);
}

static void test_scalar_functional_identities(void)
{
    int step;

    for (step = -400; step <= 400; step++) {
        double x = (double) step / 4.0;
        double sigmoid_x;
        double sigmoid_negative_x;
        double softplus_x;
        double softplus_negative_x;
        double log_sigmoid_x;

        assert(numerus_numeric_sigmoid(x, &sigmoid_x) ==
            NUMERUS_NUMERIC_SUCCESS);
        assert(numerus_numeric_sigmoid(-x, &sigmoid_negative_x) ==
            NUMERUS_NUMERIC_SUCCESS);
        assert(numerus_numeric_softplus(x, &softplus_x) ==
            NUMERUS_NUMERIC_SUCCESS);
        assert(numerus_numeric_softplus(-x, &softplus_negative_x) ==
            NUMERUS_NUMERIC_SUCCESS);
        assert(numerus_numeric_log_sigmoid(x, &log_sigmoid_x) ==
            NUMERUS_NUMERIC_SUCCESS);

        assert_relative_close(
            sigmoid_x + sigmoid_negative_x, 1.0, 1e-14
        );
        assert_relative_close(
            softplus_x - softplus_negative_x, x, 1e-13
        );
        assert_relative_close(
            log_sigmoid_x + softplus_negative_x, 0.0, 1e-13
        );
    }
}

static void test_rng_replay_and_sampling_invariants(void)
{
    size_t seed_index;

    for (seed_index = 0; seed_index < 16; seed_index++) {
        numerus_rng *first = NULL;
        numerus_rng *second = NULL;
        size_t draw_index;

        assert(numerus_rng_create(
            (uint64_t) (seed_index + 1),
            (uint64_t) (seed_index + 101),
            &first
        ) == NUMERUS_RNG_SUCCESS);
        assert(numerus_rng_create(
            (uint64_t) (seed_index + 1),
            (uint64_t) (seed_index + 101),
            &second
        ) == NUMERUS_RNG_SUCCESS);

        for (draw_index = 0; draw_index < 100; draw_index++) {
            uint32_t first_value;
            uint32_t second_value;
            assert(numerus_rng_next_u32(first, &first_value) ==
                NUMERUS_RNG_SUCCESS);
            assert(numerus_rng_next_u32(second, &second_value) ==
                NUMERUS_RNG_SUCCESS);
            assert(first_value == second_value);
        }

        numerus_rng_destroy(second);
        numerus_rng_destroy(first);
    }

    {
        numerus_rng *rng = NULL;
        size_t *indices = NULL;
        size_t index;
        size_t other;

        assert(numerus_rng_create(42, 54, &rng) == NUMERUS_RNG_SUCCESS);
        assert(numerus_rng_sample_indices(
            rng, 32, 12, false, &indices
        ) == NUMERUS_RNG_SUCCESS);
        for (index = 0; index < 12; index++) {
            assert(indices[index] < 32);
            for (other = index + 1; other < 12; other++) {
                assert(indices[index] != indices[other]);
            }
        }
        numerus_rng_free_indices(indices);
        indices = NULL;

        assert(numerus_rng_sample_indices(
            rng, 5, 1000, true, &indices
        ) == NUMERUS_RNG_SUCCESS);
        for (index = 0; index < 1000; index++) {
            assert(indices[index] < 5);
        }

        numerus_rng_free_indices(indices);
        numerus_rng_destroy(rng);
    }
}

typedef struct {
    double target_x;
    double target_y;
    double y_weight;
} property_quadratic_context;

static bool property_quadratic_objective(
    const double *parameters,
    size_t count,
    const void *context,
    double *value
)
{
    const property_quadratic_context *settings = context;
    double dx;
    double dy;

    if (count != 2 || value == NULL) {
        return false;
    }

    dx = parameters[0] - settings->target_x;
    dy = parameters[1] - settings->target_y;
    *value = dx * dx + settings->y_weight * dy * dy;
    return true;
}

static bool property_quadratic_gradient(
    const double *parameters,
    size_t count,
    const void *context,
    double *gradient
)
{
    const property_quadratic_context *settings = context;

    if (count != 2 || gradient == NULL) {
        return false;
    }

    gradient[0] = 2.0 * (parameters[0] - settings->target_x);
    gradient[1] = 2.0 * settings->y_weight *
        (parameters[1] - settings->target_y);
    return true;
}

static void test_optimizer_invariants_over_parameter_grid(void)
{
    numerus_optimizer_options options;
    size_t fixture;

    assert(numerus_optimizer_default_options(&options) ==
        NUMERUS_OPTIMIZER_SUCCESS);

    for (fixture = 0; fixture < 12; fixture++) {
        double initial[] = {
            (double) fixture - 4.0,
            5.0 - (double) fixture * 0.25
        };
        property_quadratic_context context = {
            (double) fixture * 0.2 - 1.0,
            2.0 - (double) fixture * 0.1,
            0.5 + (double) fixture * 0.25
        };
        numerus_optimizer_result *result = NULL;
        numerus_optimizer_termination termination;
        double initial_objective;
        double final_objective;
        double x;
        double y;
        double gradient_x;
        double gradient_y;

        assert(property_quadratic_objective(
            initial, 2, &context, &initial_objective
        ));
        assert(numerus_optimizer_minimize(
            property_quadratic_objective,
            property_quadratic_gradient,
            &context,
            initial,
            2,
            &options,
            &result
        ) == NUMERUS_OPTIMIZER_SUCCESS);
        assert(numerus_optimizer_result_get_objective(
            result, &final_objective
        ) == NUMERUS_OPTIMIZER_SUCCESS);
        assert(final_objective <= initial_objective);
        assert(numerus_optimizer_result_get_parameter(result, 0, &x) ==
            NUMERUS_OPTIMIZER_SUCCESS);
        assert(numerus_optimizer_result_get_parameter(result, 1, &y) ==
            NUMERUS_OPTIMIZER_SUCCESS);
        gradient_x = 2.0 * (x - context.target_x);
        gradient_y = 2.0 * context.y_weight * (y - context.target_y);
        assert(fabs(gradient_x) < 1e-5);
        assert(fabs(gradient_y) < 1e-5);
        assert(numerus_optimizer_result_get_termination(
            result, &termination
        ) == NUMERUS_OPTIMIZER_SUCCESS);
        assert(termination == NUMERUS_OPTIMIZER_CONVERGED_GRADIENT);

        numerus_optimizer_result_destroy(result);
    }
}

static void test_gaussian_log_density_symmetry_and_distance(void)
{
    const double mean_values[] = {0.0, 0.0};
    const double covariance_values[] = {
        1, 0,
        0, 1
    };
    const double origin_values[] = {0.0, 0.0};
    const double point_values[] = {1.0, 2.0};
    const double reflected_values[] = {-1.0, -2.0};
    numerus_matrix *mean = make_dense(2, 1, mean_values);
    numerus_matrix *covariance = make_dense(2, 2, covariance_values);
    numerus_matrix *origin = make_dense(2, 1, origin_values);
    numerus_matrix *point = make_dense(2, 1, point_values);
    numerus_matrix *reflected = make_dense(2, 1, reflected_values);
    double origin_log_density;
    double point_log_density;
    double reflected_log_density;

    assert(numerus_multivariate_gaussian_log_density(
        origin, mean, covariance, &origin_log_density
    ) == NUMERUS_PROBABILITY_SUCCESS);
    assert(numerus_multivariate_gaussian_log_density(
        point, mean, covariance, &point_log_density
    ) == NUMERUS_PROBABILITY_SUCCESS);
    assert(numerus_multivariate_gaussian_log_density(
        reflected, mean, covariance, &reflected_log_density
    ) == NUMERUS_PROBABILITY_SUCCESS);

    assert(origin_log_density > point_log_density);
    assert_relative_close(
        point_log_density, reflected_log_density, 1e-14
    );
    assert_relative_close(
        point_log_density - origin_log_density, -2.5, 1e-14
    );

    numerus_matrix_destroy(reflected);
    numerus_matrix_destroy(point);
    numerus_matrix_destroy(origin);
    numerus_matrix_destroy(covariance);
    numerus_matrix_destroy(mean);
}

int main(void)
{
    test_cholesky_reconstruction_and_solve_residuals();
    test_scalar_functional_identities();
    test_rng_replay_and_sampling_invariants();
    test_optimizer_invariants_over_parameter_grid();
    test_gaussian_log_density_symmetry_and_distance();
    puts("Statistical property tests passed.");
    return 0;
}
