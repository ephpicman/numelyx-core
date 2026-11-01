/**
 * @file numerus_matrix_operations.h
 * @brief Matrix arithmetic and element-wise operations.
 */
#ifndef NUMERUS_MATRIX_OPERATIONS_H
#define NUMERUS_MATRIX_OPERATIONS_H

#include "numerus_matrix_types.h"

/**
 * Create a lazy scalar-multiplication view.
 *
 * The scalar is copied into the view; the parent is borrowed and must outlive
 * the result. IEEE-754 multiplication semantics apply, including NaN and
 * infinities. The output pointer is set to NULL before work begins.
 */
int numerus_matrix_create_scale(
    numerus_matrix *parent,
    double scalar,
    numerus_matrix **matrix
);

/**
 * Create a lazy view that adds a scalar to every matrix element.
 * The scalar is copied into the view and IEEE-754 addition semantics apply.
 */
int numerus_matrix_create_scalar_add(
    numerus_matrix *parent,
    double scalar,
    numerus_matrix **matrix
);

/**
 * Create a lazy view that subtracts a scalar from every matrix element.
 * This computes A[i,j] - scalar (not scalar - A[i,j]).
 */
int numerus_matrix_create_scalar_subtract(
    numerus_matrix *parent,
    double scalar,
    numerus_matrix **matrix
);

/** Create a lazy unary-negation view. */
int numerus_matrix_create_negate(
    numerus_matrix *parent,
    numerus_matrix **matrix
);

/**
 * Create a lazy element-wise sum of two equally sized Matrices.
 *
 * Both parent references are non-owning and must remain alive for the result's
 * lifetime. Addition follows IEEE-754 double semantics.
 */
int numerus_matrix_create_add(
    numerus_matrix *left,
    numerus_matrix *right,
    numerus_matrix **matrix
);

/**
 * Create a lazy element-wise difference of two equally sized Matrices.
 *
 * Both parent references are non-owning and must remain alive for the result's
 * lifetime. Subtraction follows IEEE-754 double semantics.
 */
int numerus_matrix_create_subtract(
    numerus_matrix *left,
    numerus_matrix *right,
    numerus_matrix **matrix
);

/**
 * Create a lazy Hadamard (element-wise) product of equally sized Matrices.
 *
 * This is element-wise multiplication, not matrix multiplication. Both parent
 * references are non-owning and must remain alive for the result's lifetime.
 * IEEE-754 double semantics apply.
 */
int numerus_matrix_create_hadamard_product(
    numerus_matrix *left,
    numerus_matrix *right,
    numerus_matrix **matrix
);

/**
 * Create a lazy element-wise quotient of equally sized Matrices.
 *
 * Division follows IEEE-754 double semantics: zero divisors produce the
 * corresponding infinities or NaN rather than a Matrix status error. Both
 * parent references are non-owning and must remain alive for the result.
 */
int numerus_matrix_create_divide(
    numerus_matrix *left,
    numerus_matrix *right,
    numerus_matrix **matrix
);

/**
 * @brief Create an independent dense Matrix containing element-wise minima.
 *
 * Each output element is min(left[i,j], right[i,j]). Inputs must have equal
 * dimensions; NaN propagates from either operand. The output pointer remains
 * NULL on failure.
 */
int numerus_matrix_create_elementwise_min(
    const numerus_matrix *left, const numerus_matrix *right,
    numerus_matrix **matrix
);

/**
 * Create an independent dense Matrix containing element-wise maxima.
 * Inputs must have equal dimensions; NaN propagates from either operand.
 * The output pointer remains NULL on failure.
 */
int numerus_matrix_create_elementwise_max(
    const numerus_matrix *left, const numerus_matrix *right,
    numerus_matrix **matrix
);

/**
 * Clamp values to [lower, upper] in an independent dense Matrix. NaN values
 * propagate; NaN bounds or lower > upper are invalid. Infinite bounds are OK.
 */
int numerus_matrix_create_clamp(
    const numerus_matrix *source, double lower, double upper,
    numerus_matrix **matrix
);

/**
 * Create a lazy map view. Callback and context must remain deterministic and
 * logically unchanged for the view lifetime; parent is borrowed and must live.
 */
int numerus_matrix_create_map(
    const numerus_matrix *parent, numerus_value_transform_fn transform,
    const void *context, numerus_matrix **matrix
);

/**
 * Eagerly apply a deterministic callback into independent dense Storage.
 * Callback context is borrowed for the duration of this call only.
 */
int numerus_matrix_apply(
    const numerus_matrix *source, numerus_value_transform_fn transform,
    const void *context, numerus_matrix **matrix
);

/**
 * Normalize globally, by rows, or by columns using the selected norm. Inputs
 * must be finite; zero-norm vectors return DIVISION_BY_ZERO. Result is dense.
 */
int numerus_matrix_create_normalized(
    const numerus_matrix *source,
    numerus_matrix_normalize_axis axis,
    numerus_matrix_normalize_norm norm,
    numerus_matrix **matrix
);

/**
 * Materialize the logical values of a Matrix into independent dense Storage.
 *
 * Works for roots, nested views, joins, and binary arithmetic nodes. The result
 * owns its Storage and does not retain the source or its parents. On failure,
 * the output pointer remains NULL.
 */
int numerus_matrix_materialize(
    const numerus_matrix *source,
    numerus_matrix **matrix
);

/**
 * Compute the matrix product left × right into independent dense Storage.
 *
 * The left column count must equal the right row count. The operation is
 * materialized rather than a lazy view because a lazy getter would repeat the
 * dot product for every read. On failure, the output pointer remains NULL.
 */
int numerus_matrix_multiply(
    const numerus_matrix *left,
    const numerus_matrix *right,
    numerus_matrix **matrix
);

/**
 * Create a lazy view dividing every parent value by a scalar.
 *
 * A positive or negative zero divisor is rejected with
 * NUMERUS_MATRIX_DIVISION_BY_ZERO. Other values use IEEE-754 double
 * division semantics. The parent is borrowed and must outlive the view.
 */
int numerus_matrix_create_divide_scalar(
    numerus_matrix *parent,
    double divisor,
    numerus_matrix **matrix
);

/**
 * Compare two Matrices using exact C double equality.
 *
 * Shape mismatch is a successful comparison with result=false. NaN compares
 * unequal; equal infinities and signed zero follow C double == semantics.
 * The output is unchanged if a Matrix read fails.
 */
numerus_matrix_status numerus_matrix_is_equal(
    const numerus_matrix *left,
    const numerus_matrix *right,
    bool *equal
);

/**
 * Compare two Matrices using NUMERUS_EPSILON's combined tolerance.
 *
 * Shape mismatch is a successful comparison with result=false. NaN compares
 * unequal, and equal infinities compare equal. The output is unchanged if a
 * Matrix read fails.
 */
numerus_matrix_status numerus_matrix_all_close(
    const numerus_matrix *left, const numerus_matrix *right, bool *close
);

/**
 * Return true when at least one corresponding element pair is close under
 * NUMERUS_EPSILON's combined tolerance. Shape mismatch returns success/false;
 * NaN values compare unequal. Output is unchanged on read failure.
 */
numerus_matrix_status numerus_matrix_any_close(
    const numerus_matrix *left, const numerus_matrix *right, bool *close
);

/**
 * Compare all corresponding elements using NUMERUS_EPSILON's combined
 * tolerance. This is the all-elements predicate; use any_close() when one
 * matching element is sufficient. Shape mismatch returns success/false.
 */
numerus_matrix_status numerus_matrix_is_close(
    const numerus_matrix *left,
    const numerus_matrix *right,
    bool *close
);

/**
 * Compute base raised to a non-negative integer exponent.
 *
 * The base must be square. Exponent zero returns an identity Matrix; exponent
 * one returns an independent materialized copy. Larger powers use
 * exponentiation by squaring and return independent dense Storage.
 */
int numerus_matrix_power(
    const numerus_matrix *base,
    size_t exponent,
    numerus_matrix **matrix
);

/**
 * Evaluate c[0] + c[1]A + ... + c[count - 1]A^(count - 1) by Horner's
 * method. Coefficients are ordered from the constant term upward. The base
 * must be square and count must be positive; the result is independent dense
 * Storage. The output pointer remains NULL on failure.
 */
int numerus_matrix_polynomial(
    const numerus_matrix *base,
    const double *coefficients,
    size_t coefficient_count,
    numerus_matrix **matrix
);

/**
 * Compute exp(A) using scaling-and-squaring with a degree-13 Padé
 * approximant. Input must be square and finite; the result is independent
 * dense Storage. Non-finite intermediate results and solver failures are
 * reported as statuses, and the output pointer remains NULL on failure.
 */
int numerus_matrix_exponential(
    const numerus_matrix *matrix,
    numerus_matrix **exponential
);

/**
 * Compute base raised to a signed integer exponent.
 *
 * Negative exponents invert the square base once, then use exponentiation by
 * squaring. A singular or numerically singular base returns
 * NUMERUS_MATRIX_SINGULAR. The output is NULL on failure.
 */
int numerus_matrix_power_signed(
    const numerus_matrix *base,
    int64_t exponent,
    numerus_matrix **matrix
);

/**
 * Compute the Kronecker product left ⊗ right into independent dense Storage.
 *
 * Result dimensions are left.rows × right.rows by left.columns × right.columns.
 * All dimension, element-count, and byte-count arithmetic is checked. On
 * failure, the output pointer remains NULL.
 */
int numerus_matrix_kronecker_product(
    const numerus_matrix *left,
    const numerus_matrix *right,
    numerus_matrix **matrix
);

/**
 * Compute the Kronecker sum left ⊕ right = left ⊗ I + I ⊗ right.
 *
 * Both inputs must be square. The result is an independent dense Matrix with
 * dimension left.rows * right.rows. Dimensions and allocation sizes are
 * checked; the output pointer remains NULL on failure.
 */
int numerus_matrix_kronecker_sum(
    const numerus_matrix *left,
    const numerus_matrix *right,
    numerus_matrix **matrix
);

#endif
