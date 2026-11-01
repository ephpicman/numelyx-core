/**
 * @file numelyx_core_matrix_solve.c
 * @brief Structure-aware dispatch and general LU solve for square systems.
 */

#include "numelyx_core_matrix.h"
#include "numelyx_core_matrix_internal.h"
#include "numelyx_core_matrix_lu.h"
#include "numelyx_core_size.h"

#include <math.h>
#include <stdlib.h>

/*
 * Select algorithms from verified logical properties, not merely from the
 * root Storage tag: Matrix views and value transforms can change the logical
 * structure while retaining their parent's Storage.
 */
static numelyx_core_matrix_status try_structured_solve(
    const numelyx_core_matrix *matrix,
    const numelyx_core_matrix *right_hand_side,
    numelyx_core_matrix **solution,
    bool *handled
)
{
    numelyx_core_matrix_flags flags;
    numelyx_core_matrix *lower = NULL;
    numelyx_core_matrix *intermediate = NULL;
    numelyx_core_matrix_status status;

    *handled = false;
    status = numelyx_core_matrix_get_flags(
        (numelyx_core_matrix *) matrix, &flags
    );
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }

    /*
     * Triangular solve is O(n^2 * nrhs), versus O(n^3) for LU. The dedicated
     * routine performs exact structural validation, so approximate flag
     * classifications never silently discard small off-triangle entries.
     * INVALID_ARGUMENT here means the exact triangular contract was not met;
     * in that case continue with the general solver.
     */
    if (flags.lower_triangular || flags.upper_triangular) {
        status = numelyx_core_matrix_solve_triangular(
            matrix,
            right_hand_side,
            flags.lower_triangular,
            false,
            solution
        );
        if (status == NUMERUS_MATRIX_SUCCESS) {
            *handled = true;
            return status;
        }
        if (status != NUMERUS_MATRIX_INVALID_ARGUMENT) {
            return status;
        }
    }

    /*
     * Cholesky is both cheaper and more storage-efficient than pivoted LU for
     * symmetric positive-definite systems. Symmetry is only a candidate test:
     * Cholesky performs its own finite, symmetry, and scale-relative pivot
     * validation. Indefinite or nonsymmetric matrices fall through to LU.
     */
    if (flags.symmetric) {
        status = numelyx_core_matrix_cholesky(matrix, &lower);
        if (status == NUMERUS_MATRIX_SUCCESS) {
            status = numelyx_core_matrix_solve_triangular(
                lower, right_hand_side, true, false, &intermediate
            );
            if (status == NUMERUS_MATRIX_SUCCESS) {
                status = numelyx_core_matrix_solve_triangular(
                    lower, intermediate, true, true, solution
                );
            }
            numelyx_core_matrix_destroy(intermediate);
            numelyx_core_matrix_destroy(lower);
            if (status == NUMERUS_MATRIX_SUCCESS) {
                *handled = true;
            }
            return status;
        }
        if (status != NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE &&
            status != NUMERUS_MATRIX_NOT_SYMMETRIC) {
            return status;
        }
    }

    return NUMERUS_MATRIX_SUCCESS;
}

numelyx_core_matrix_status numelyx_core_matrix_solve(
    const numelyx_core_matrix *matrix,
    const numelyx_core_matrix *right_hand_side,
    numelyx_core_matrix **solution
)
{
    numelyx_core_matrix_lu_factorization *factorization = NULL;
    size_t size;
    size_t right_hand_sides;
    size_t element_count;
    size_t value_bytes;
    size_t work_bytes;
    size_t column;
    size_t rank;
    double *values = NULL;
    double *work = NULL;
    numelyx_core_matrix_status status;
    bool handled;

    if (solution == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    *solution = NULL;

    if (matrix == NULL || right_hand_side == NULL) {
        return NUMERUS_MATRIX_INVALID_ARGUMENT;
    }
    if (numelyx_core_matrix_rows(matrix) != numelyx_core_matrix_columns(matrix)) {
        return NUMERUS_MATRIX_NOT_SQUARE;
    }

    size = numelyx_core_matrix_rows(matrix);
    if (numelyx_core_matrix_rows(right_hand_side) != size) {
        return NUMERUS_MATRIX_DIMENSION_MISMATCH;
    }
    right_hand_sides = numelyx_core_matrix_columns(right_hand_side);

    status = try_structured_solve(matrix, right_hand_side, solution, &handled);
    if (status != NUMERUS_MATRIX_SUCCESS || handled) {
        return status;
    }

    /* Preserve the existing numerical-rank contract for the general LU path. */
    status = numelyx_core_matrix_rank(matrix, &rank);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        return status;
    }
    if (rank != size) {
        return NUMERUS_MATRIX_SINGULAR;
    }

    if (!numelyx_core_size_multiply(size, right_hand_sides, &element_count) ||
        !numelyx_core_size_multiply(element_count, sizeof(*values), &value_bytes) ||
        !numelyx_core_size_multiply(size, sizeof(*work), &work_bytes)) {
        return NUMERUS_MATRIX_OVERFLOW;
    }

    values = malloc(value_bytes);
    if (values == NULL) {
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }
    work = malloc(work_bytes);
    if (work == NULL) {
        free(values);
        return NUMERUS_MATRIX_OUT_OF_MEMORY;
    }

    status = numelyx_core_matrix_get_or_factorize_lu(matrix, &factorization);
    if (status != NUMERUS_MATRIX_SUCCESS) {
        free(work);
        free(values);
        return status;
    }
    if (numelyx_core_matrix_lu_rank(factorization) != size) {
        numelyx_core_matrix_lu_destroy(factorization);
        free(work);
        free(values);
        return NUMERUS_MATRIX_SINGULAR;
    }

    for (column = 0; column < right_hand_sides; column++) {
        size_t row;

        /* Forward substitution: L y = P b. */
        for (row = 0; row < size; row++) {
            size_t source_row;
            size_t k;
            double value;

            status = numelyx_core_matrix_lu_get_permutation(
                factorization, row, &source_row
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }
            status = numelyx_core_matrix_get(
                right_hand_side, source_row, column, &value
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto failure;
            }

            for (k = 0; k < row; k++) {
                double lower_value;
                status = numelyx_core_matrix_lu_get_lower(
                    factorization, row, k, &lower_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }
                value -= lower_value * work[k];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto failure;
                }
            }

            work[row] = value;
        }

        /* Back substitution: U x = y. */
        row = size;
        while (row > 0) {
            size_t k;
            double value;
            double diagonal;

            row--;
            value = work[row];

            for (k = row + 1; k < size; k++) {
                double upper_value;
                status = numelyx_core_matrix_lu_get_upper(
                    factorization, row, k, &upper_value
                );
                if (status != NUMERUS_MATRIX_SUCCESS) {
                    goto failure;
                }
                value -= upper_value * work[k];
                if (!isfinite(value)) {
                    status = NUMERUS_MATRIX_NON_FINITE;
                    goto failure;
                }
            }

            status = numelyx_core_matrix_lu_get_upper(
                factorization, row, row, &diagonal
            );
            if (status != NUMERUS_MATRIX_SUCCESS) {
                goto failure;
            }
            if (diagonal == 0.0) {
                status = NUMERUS_MATRIX_SINGULAR;
                goto failure;
            }

            value /= diagonal;
            if (!isfinite(value)) {
                status = NUMERUS_MATRIX_NON_FINITE;
                goto failure;
            }
            work[row] = value;
            values[row * right_hand_sides + column] = value;
        }
    }

    numelyx_core_matrix_lu_destroy(factorization);
    free(work);
    status = (numelyx_core_matrix_status) numelyx_core_matrix_create_dense(
        size, right_hand_sides, values, solution
    );
    free(values);
    return status;

failure:
    numelyx_core_matrix_lu_destroy(factorization);
    free(work);
    free(values);
    return status;
}
