/**
 * @file numelyx_core_matrix_types.h
 * @brief Shared Matrix types and contracts.
 *
 * Matrix is the logical layer above immutable Storage. A root Matrix owns its
 * Storage; a derived Matrix retains its parent Matrix and delegates element
 * reads to it. Each Matrix starts with one caller-owned reference. Creating a
 * dependent Matrix retains its parents; destroying a Matrix releases one
 * reference, and the object is reclaimed after its final reference is released.
 */
#ifndef NUMERUS_MATRIX_TYPES_H
#define NUMERUS_MATRIX_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "numelyx_core_storage.h"

/**
 * @brief Status codes returned by Matrix operations.
 */
typedef enum {
    /** Operation completed successfully. */
    NUMERUS_MATRIX_SUCCESS = 0,
    /** A required argument is NULL or otherwise invalid. */
    NUMERUS_MATRIX_INVALID_ARGUMENT,
    /** A size calculation would overflow size_t. */
    NUMERUS_MATRIX_OVERFLOW,
    /** Memory allocation failed. */
    NUMERUS_MATRIX_OUT_OF_MEMORY,
    /** A matrix coordinate is outside the valid range. */
    NUMERUS_MATRIX_OUT_OF_BOUNDS,
    /** The requested Storage representation requires a square Matrix. */
    NUMERUS_MATRIX_NOT_SQUARE,
    NUMERUS_MATRIX_DIMENSION_MISMATCH,
    /** Scalar division rejected a zero divisor. */
    NUMERUS_MATRIX_DIVISION_BY_ZERO,
    /** A numerical analysis cannot classify non-finite input or intermediate values. */
    NUMERUS_MATRIX_NON_FINITE,
    /** The operation requires a nonsingular Matrix, but numerical rank is deficient. */
    NUMERUS_MATRIX_SINGULAR,
    /** The requested numerical solve requires full column rank, but rank is deficient. */
    NUMERUS_MATRIX_RANK_DEFICIENT,
    /** The Matrix is not symmetric within the numerical symmetry tolerance. */
    NUMERUS_MATRIX_NOT_SYMMETRIC,
    /** The Matrix is not numerically positive definite. */
    NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE,
    /** An unpivoted decomposition encountered a zero or numerically small pivot. */
    NUMERUS_MATRIX_PIVOT_TOO_SMALL,
    /** A numerical iteration failed to converge within its documented limit. */
    NUMERUS_MATRIX_NO_CONVERGENCE,
    /** A symmetric matrix has a materially negative eigenvalue and is not PSD. */
    NUMERUS_MATRIX_NOT_POSITIVE_SEMIDEFINITE
} numelyx_core_matrix_status;

typedef enum {
    NUMERUS_MATRIX_NORMALIZE_WHOLE = 0,
    NUMERUS_MATRIX_NORMALIZE_ROWS,
    NUMERUS_MATRIX_NORMALIZE_COLUMNS
} numelyx_core_matrix_normalize_axis;

typedef enum {
    NUMERUS_MATRIX_NORMALIZE_L1 = 0,
    NUMERUS_MATRIX_NORMALIZE_L2,
    NUMERUS_MATRIX_NORMALIZE_INFINITY
} numelyx_core_matrix_normalize_norm;

/**
 * Numerical classification of the solutions to A X = B.
 */
typedef enum {
    /** Every right-hand side has exactly one solution. */
    NUMERUS_MATRIX_SOLUTION_UNIQUE = 0,
    /** Every right-hand side is consistent, but solutions are not unique. */
    NUMERUS_MATRIX_SOLUTION_INFINITE,
    /** At least one right-hand side is outside the numerical column space of A. */
    NUMERUS_MATRIX_SOLUTION_INCONSISTENT
} numelyx_core_matrix_solution_kind;

/**
 * @brief Layout used to join two Matrices.
 */
typedef enum {
    /** Append the second Matrix to the right of the first. */
    NUMERUS_MATRIX_JOIN_HORIZONTAL = 0,
    /** Append the second Matrix below the first. */
    NUMERUS_MATRIX_JOIN_VERTICAL
} numelyx_core_matrix_join_type;

/**
 * @brief Cached structural properties of a Matrix.
 *
 * Floating-point properties use NUMERUS_EPSILON from numelyx_core_numeric.h.
 * A square zero Matrix is also diagonal, upper-triangular, lower-triangular,
 * and symmetric.
 */
typedef struct {
    bool square;
    bool zero;
    bool diagonal;
    bool upper_triangular;
    bool lower_triangular;
    bool symmetric;
    bool identity;
} numelyx_core_matrix_flags;

/**
 * @brief Parameters consumed by the general Matrix factory.
 *
 * Only fields required by the selected Storage kind are used. The factory
 * delegates representation-specific construction to Storage.
 */
typedef struct {
    /** Value buffer for dense, triangular, diagonal, symmetric and banded Storage. */
    const double *values;
    /** Single value used by constant and scaled-identity Storage. */
    double value;
    /** Explicit entries used by sparse Storage. */
    const numelyx_core_storage_sparse_entry *entries;
    /** Number of sparse entries. */
    size_t count;
    /** Default value used by sparse Storage. */
    double default_value;
    /** Lower bandwidth used by banded Storage. */
    size_t lower_bandwidth;
    /** Upper bandwidth used by banded Storage. */
    size_t upper_bandwidth;
} numelyx_core_matrix_data;

/**
 * @brief Opaque immutable Matrix object.
 */
typedef struct numelyx_core_matrix numelyx_core_matrix;

/**
 * @brief Map a child Matrix coordinate to its parent's coordinate.
 *
 * The callback must write both parent coordinates on success. It must return
 * a Matrix status on failure; the status is propagated to the caller.
 * The callback must be deterministic and must not depend on mutable external
 * state. The context is borrowed, read-only, and must remain valid and
 * logically unchanged for the lifetime of the child Matrix. Because a Matrix
 * may be read concurrently, callback implementations and their contexts must
 * be safe for concurrent invocation. During cache computation, callbacks must
 * not enter cache operations on another Matrix in a way that can invert lock
 * acquisition order.
 */
typedef numelyx_core_matrix_status (*numelyx_core_coordinate_transform_fn)(
    size_t row,
    size_t column,
    size_t *parent_row,
    size_t *parent_column,
    const void *context
);

/**
 * @brief Transform a value read from the parent.
 *
 * row and column are the original coordinates requested from the child.
 * The callback writes result only when it returns NUMERUS_MATRIX_SUCCESS.
 * The callback must be deterministic and must not depend on mutable external
 * state. The context is borrowed, read-only, and must remain valid and
 * logically unchanged for the lifetime of the child Matrix.
 */
typedef numelyx_core_matrix_status (*numelyx_core_value_transform_fn)(
    size_t row,
    size_t column,
    double parent_value,
    double *result,
    const void *context
);

#endif
