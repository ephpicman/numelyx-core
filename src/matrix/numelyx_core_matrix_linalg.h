/**
 * @file numelyx_core_matrix_linalg.h
 * @brief Matrix analysis and linear algebra API.
 */
#ifndef NUMERUS_MATRIX_LINALG_H
#define NUMERUS_MATRIX_LINALG_H

#include "numelyx_core_matrix_types.h"

/**
 * @brief Compute or retrieve cached structural properties.
 *
 * Properties describe the logical values exposed by this Matrix. The output
 * is unchanged if the scan fails.
 */
numelyx_core_matrix_status numelyx_core_matrix_get_flags(
    numelyx_core_matrix *matrix,
    numelyx_core_matrix_flags *flags
);

/**
 * @brief Compute or retrieve the cached determinant of a square Matrix.
 *
 * Uses partial-pivoted Gaussian elimination in the general case, with cheaper
 * paths for zero, identity, or triangular matrices when structural properties
 * have already been cached. Successful results, including zero, are cached. Failures are not.
 * The output is unchanged on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_determinant(
    numelyx_core_matrix *matrix,
    double *determinant
);

/**
 * Compute the signed log-absolute-determinant of a finite square Matrix.
 *
 * For nonsingular input, sign is +1 or -1 and log_abs_determinant is
 * log(abs(det(A))). Singular input returns SINGULAR; non-finite input or
 * elimination intermediates return NON_FINITE. Both scalar outputs remain
 * unchanged on failure. The implementation uses LU factorization rather
 * than forming the determinant product.
 */
numelyx_core_matrix_status numelyx_core_matrix_log_determinant(
    const numelyx_core_matrix *matrix,
    int *sign,
    double *log_abs_determinant
);

/**
 * Materialize a row-echelon form using scale-aware pivoting.
 * The result is independent dense Storage. Non-finite input or intermediates
 * return NUMERUS_MATRIX_NON_FINITE; on failure, *result remains NULL.
 */
numelyx_core_matrix_status numelyx_core_matrix_row_echelon_form(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **result
);

/**
 * Materialize a reduced row-echelon form using scale-aware pivoting.
 * The result is independent dense Storage. Non-finite input or intermediates
 * return NUMERUS_MATRIX_NON_FINITE; on failure, *result remains NULL.
 */
numelyx_core_matrix_status numelyx_core_matrix_reduced_row_echelon_form(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **result
);

/**
 * Return a basis for the numerical column space of a Matrix.
 *
 * The basis columns are selected from the input columns using column-pivoted
 * Householder QR. If the numerical rank is zero, success is represented by
 * *basis == NULL and *dimension == 0. On failure, *basis is NULL and
 * *dimension is unchanged.
 */
numelyx_core_matrix_status numelyx_core_matrix_column_space_basis(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **basis,
    size_t *dimension
);

/**
 * Return a basis for the numerical row space of a Matrix.
 *
 * The basis rows are selected from the input rows using column-pivoted QR of
 * the transpose. If the numerical rank is zero, success is represented by
 * *basis == NULL and *dimension == 0. On failure, *basis is NULL and
 * *dimension is unchanged.
 */
numelyx_core_matrix_status numelyx_core_matrix_row_space_basis(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **basis,
    size_t *dimension
);

/**
 * Return a basis for the numerical null space as columns.
 * For a trivial null space, success is represented by *basis == NULL and
 * *nullity == 0. On failure, *basis remains NULL and *nullity is unchanged.
 */
numelyx_core_matrix_status numelyx_core_matrix_null_space(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **basis,
    size_t *nullity
);

/**
 * Compute the spectral norm (largest singular value) of any real Matrix.
 * The scalar output is unchanged on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_spectral_norm(
    const numelyx_core_matrix *matrix,
    double *norm
);

/**
 * Compute the spectral radius for a real symmetric Matrix only.
 * The radius is the largest absolute eigenvalue. Non-symmetric input returns
 * NUMERUS_MATRIX_NOT_SYMMETRIC; output is unchanged on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_symmetric_spectral_radius(
    const numelyx_core_matrix *matrix,
    double *radius
);

/**
 * Compute real eigenvalues and orthonormal eigenvectors for a symmetric Matrix.
 * Eigenvalues are descending in algebraic order; eigenvectors are columns.
 * Non-symmetric input and non-convergence return explicit statuses; outputs
 * remain NULL on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_symmetric_eigen(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **eigenvalues,
    numelyx_core_matrix **eigenvectors
);

/**
 * Compute the principal square root of a real symmetric positive-semidefinite
 * matrix by spectral decomposition. Small negative eigenvalues within the
 * scale-relative tolerance are clamped to zero; materially negative values
 * return NUMERUS_MATRIX_NOT_POSITIVE_SEMIDEFINITE.
 */
numelyx_core_matrix_status numelyx_core_matrix_symmetric_square_root(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **square_root
);

/**
 * Compute the real matrix logarithm of a symmetric positive-definite matrix
 * by spectral decomposition. Eigenvalues at or below the scale-relative
 * positivity tolerance return NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE.
 */
numelyx_core_matrix_status numelyx_core_matrix_symmetric_logarithm(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **logarithm
);

/** Compute sin(A) for a real symmetric Matrix using its spectral decomposition. */
numelyx_core_matrix_status numelyx_core_matrix_symmetric_sine(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **sine
);

/** Compute cos(A) for a real symmetric Matrix using its spectral decomposition. */
numelyx_core_matrix_status numelyx_core_matrix_symmetric_cosine(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **cosine
);

/**
 * Compute the Moore-Penrose pseudoinverse using the reduced real SVD.
 *
 * Singular values at or below NUMERUS_EPSILON * max(rows, columns) *
 * largest_singular_value are treated as zero. The output has shape
 * columns(A)-by-rows(A), is independent dense Storage, and remains NULL
 * on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_pseudoinverse(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **pseudoinverse
);

/**
 * Compute the reduced real SVD A = U * Sigma * Vᵀ.
 * Singular values are nonnegative and sorted descending. U, the singular-value
 * column vector, and Vᵀ are independent dense outputs; all remain NULL on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_svd(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **u,
    numelyx_core_matrix **singular_values,
    numelyx_core_matrix **vt
);

/**
 * Compute an unpivoted LDLᵀ factorization of a symmetric Matrix.
 * Matrices requiring symmetric pivoting may return PIVOT_TOO_SMALL despite
 * being nonsingular. Both outputs remain NULL on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_ldlt_decompose(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **lower,
    numelyx_core_matrix **diagonal
);

/**
 * Compute lower-triangular L such that A is approximately L * Lᵀ.
 * The input must be symmetric positive definite under the documented numerical
 * checks. The output remains NULL on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_cholesky(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **lower
);

/**
 * Classify a finite real symmetric Matrix as positive definite and/or positive
 * semidefinite using eigenvalues and a scale-relative tolerance.
 * Eigenvalues within NUMERUS_EPSILON × n × max(|lambda_i|) count as zero.
 * Both outputs are unchanged on failure and must be distinct.
 */
numelyx_core_matrix_status numelyx_core_matrix_classify_definiteness(
    const numelyx_core_matrix *matrix,
    int *positive_definite,
    int *positive_semidefinite
);

/**
 * Compute reduced, unpivoted Householder QR: A = Q * R.
 * Q is rows(A)×min(rows(A), columns(A)); R is min(rows(A), columns(A))×columns(A).
 * This API is not rank-revealing. Both outputs remain NULL on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_qr_decompose(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **q,
    numelyx_core_matrix **r
);

/**
 * Estimate the 1-norm condition number ||A||₁ ||A⁻¹||₁.
 * This implementation uses the inverse-based estimate; singular/numerically
 * singular inputs can yield infinity. The scalar output is unchanged on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_condition_estimate_one(
    const numelyx_core_matrix *matrix,
    double *condition_estimate
);

/**
 * Classify the numerical solutions to A X = B.
 *
 * Supports rectangular A and multiple right-hand-side columns in B. Consistency
 * is tested by projecting each RHS onto the numerical column space obtained
 * from column-pivoted QR. The classification output is unchanged on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_classify_system(
    const numelyx_core_matrix *matrix,
    const numelyx_core_matrix *right_hand_side,
    numelyx_core_matrix_solution_kind *classification
);

/**
 * Solve weighted least squares: minimize sum_i weights[i] * ||A_i X - B_i||².
 *
 * weights_count must equal rows(A); weights must be finite and nonnegative.
 * Zero weights exclude rows. The implementation scales rows by sqrt(weight)
 * and uses the QR least-squares solver, not normal equations. On failure,
 * *solution remains NULL.
 */
numelyx_core_matrix_status numelyx_core_matrix_weighted_least_squares(
    const numelyx_core_matrix *matrix,
    const numelyx_core_matrix *right_hand_side,
    const double *weights,
    size_t weights_count,
    numelyx_core_matrix **solution
);

/**
 * Solve min_X ||A X - B||₂ using column-pivoted QR.
 * Requires full column rank and supports overdetermined/square A and multiple
 * RHS columns. Rank-deficient or underdetermined inputs return RANK_DEFICIENT.
 * For a minimum-Euclidean-norm solution when A is rank deficient or
 * underdetermined, compose numelyx_core_matrix_pseudoinverse(A) * B. The
 * pseudoinverse applies the documented SVD singular-value threshold.
 */
numelyx_core_matrix_status numelyx_core_matrix_least_squares(
    const numelyx_core_matrix *matrix,
    const numelyx_core_matrix *right_hand_side,
    numelyx_core_matrix **solution
);

/**
 * Solve A X = B for square, numerically nonsingular A using one LU factorization.
 * Multiple RHS columns share that factorization. The result is independent
 * dense Storage; the output remains NULL on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_solve(
    const numelyx_core_matrix *matrix,
    const numelyx_core_matrix *right_hand_side,
    numelyx_core_matrix **solution
);

/**
 * Solve T X = B or Tᵀ X = B by forward/back substitution.
 *
 * The stored T must be square and triangular in the orientation selected by
 * lower; entries in its structurally zero half must be exactly zero. When
 * transpose is true, the solve uses Tᵀ but lower still describes stored T.
 * The diagonal must be finite and nonzero. Multiple RHS columns are supported.
 * The result is independent dense Storage; *solution is NULL on failure.
 * Malformed triangular input returns INVALID_ARGUMENT, a zero diagonal returns
 * SINGULAR, and non-finite input/intermediate values return NON_FINITE.
 */
numelyx_core_matrix_status numelyx_core_matrix_solve_triangular(
    const numelyx_core_matrix *triangular,
    const numelyx_core_matrix *right_hand_side,
    bool lower,
    bool transpose,
    numelyx_core_matrix **solution
);

/**
 * Compute an inverse using Gauss–Jordan elimination with partial pivoting.
 * This alternative is primarily useful for comparison and education; the
 * LU-based numelyx_core_matrix_inverse() remains the default production path.
 * Singular and non-finite inputs return explicit statuses; output stays NULL
 * on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_inverse_gauss_jordan(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **inverse
);

numelyx_core_matrix_status numelyx_core_matrix_inverse(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **inverse
);

/**
 * Estimate numerical rank using scale-aware complete-pivoting elimination.
 * This is a tolerance-dependent numerical classification, not symbolic rank.
 * The output is unchanged on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_rank(
    const numelyx_core_matrix *matrix,
    size_t *rank
);

/**
 * Return whether a square Matrix is skew-symmetric within NUMERUS_EPSILON.
 * A non-square Matrix returns success with false. Output is unchanged on read
 * failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_is_skew_symmetric(
    const numelyx_core_matrix *matrix,
    bool *is_skew_symmetric
);

/**
 * Return whether a square Matrix has orthonormal rows within
 * NUMERUS_EPSILON. A non-square Matrix returns success with false. This check
 * uses O(n^3) time and O(1) auxiliary memory; output is unchanged on read
 * failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_is_orthogonal(
    const numelyx_core_matrix *matrix,
    bool *is_orthogonal
);

/**
 * Return whether any element is NaN. The output is unchanged if a read fails.
 * The scan stops at the first matching element.
 */
numelyx_core_matrix_status numelyx_core_matrix_has_nan(
    const numelyx_core_matrix *matrix,
    bool *has_nan
);

/**
 * Return whether any element is positive or negative infinity. The output is
 * unchanged if a read fails; the scan stops at the first matching element.
 */
numelyx_core_matrix_status numelyx_core_matrix_has_infinity(
    const numelyx_core_matrix *matrix,
    bool *has_infinity
);

/**
 * Return whether every element is finite (neither NaN nor infinity).
 * The output is unchanged if a read fails; the scan stops at the first
 * non-finite element.
 */
numelyx_core_matrix_status numelyx_core_matrix_is_finite(
    const numelyx_core_matrix *matrix,
    bool *is_finite
);

/**
 * Compute the Frobenius norm using a stable hypot-based accumulation.
 * If any element is NaN, the result is NaN; infinities otherwise follow
 * IEEE-754 behavior. Output is unchanged if a read fails.
 */
numelyx_core_matrix_status numelyx_core_matrix_norm_frobenius(
    const numelyx_core_matrix *matrix,
    double *norm
);

/**
 * Compute the induced matrix 1-norm (maximum absolute column sum).
 * Any NaN element makes the result NaN; output is unchanged on read failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_norm_one(
    const numelyx_core_matrix *matrix,
    double *norm
);

/**
 * Compute the induced matrix infinity-norm (maximum absolute row sum).
 * Any NaN element makes the result NaN; output is unchanged on read failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_norm_infinity(
    const numelyx_core_matrix *matrix,
    double *norm
);

/**
 * Compute the trace of a square Matrix.
 *
 * Diagonal elements are accumulated in increasing index order using ordinary
 * double addition. NaN and infinities follow IEEE-754 arithmetic. The output
 * is unchanged if validation or any element read fails.
 */
numelyx_core_matrix_status numelyx_core_matrix_trace(
    const numelyx_core_matrix *matrix,
    double *trace
);

/**
 * Compute the sum of all elements using row-major double accumulation.
 * NaN and infinities follow IEEE-754 arithmetic; output is unchanged on error.
 */
numelyx_core_matrix_status numelyx_core_matrix_sum(
    const numelyx_core_matrix *matrix,
    double *sum
);

/**
 * Compute the minimum element. If any element is NaN, the result is NaN.
 * Output is unchanged if an element read fails.
 */
numelyx_core_matrix_status numelyx_core_matrix_min(
    const numelyx_core_matrix *matrix,
    double *minimum
);

/**
 * Compute the maximum element. If any element is NaN, the result is NaN.
 * Output is unchanged if an element read fails.
 */
numelyx_core_matrix_status numelyx_core_matrix_max(
    const numelyx_core_matrix *matrix,
    double *maximum
);

/** Compute the arithmetic mean using row-major sum divided by element count. */
numelyx_core_matrix_status numelyx_core_matrix_variance(const numelyx_core_matrix *matrix, bool sample, double *variance);

/**
 * Compute the population or sample standard deviation.
 *
 * Uses the corresponding variance definition selected by sample and returns
 * its square root. Input elements must be finite; sample mode requires at
 * least two elements. The output is unchanged on failure.
 */
numelyx_core_matrix_status numelyx_core_matrix_standard_deviation(const numelyx_core_matrix *matrix, bool sample, double *standard_deviation);

/**
 * Compute the arithmetic mean using row-major double accumulation.
 * NaN and infinities follow IEEE-754 arithmetic; output is unchanged on error.
 */
numelyx_core_matrix_status numelyx_core_matrix_mean(
    const numelyx_core_matrix *matrix,
    double *mean
);

/**
 * Return per-row sums as an independent rows×1 dense Matrix.
 * NaN and infinities follow IEEE-754 arithmetic.
 */
numelyx_core_matrix_status numelyx_core_matrix_row_sums(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **sums
);

/**
 * Return per-column sums as an independent 1×columns dense Matrix.
 * NaN and infinities follow IEEE-754 arithmetic.
 */
numelyx_core_matrix_status numelyx_core_matrix_column_sums(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **sums
);

/** Return per-row arithmetic means as an independent rows×1 dense Matrix. */
numelyx_core_matrix_status numelyx_core_matrix_row_means(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **means
);

/** Return per-column arithmetic means as an independent 1×columns dense Matrix. */
numelyx_core_matrix_status numelyx_core_matrix_column_means(
    const numelyx_core_matrix *matrix,
    numelyx_core_matrix **means
);

#endif
