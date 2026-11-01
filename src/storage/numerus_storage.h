/**
 * @file numerus_storage.h
 * @brief Immutable matrix storage backends used by Numerus.
 *
 * Storage is an internal numerical representation. It is intentionally opaque
 * to callers: users interact with it through the constructors, accessors and
 * metadata functions declared here rather than depending on a representation.
 */
#ifndef NUMERUS_STORAGE_H
#define NUMERUS_STORAGE_H

#include <stddef.h>

/**
 * @brief Status codes returned by Storage operations.
 */
typedef enum {
    /** Operation completed successfully. */
    NUMERUS_STORAGE_SUCCESS = 0,
    /** A required argument was NULL or otherwise invalid. */
    NUMERUS_STORAGE_INVALID_ARGUMENT,
    /** A size calculation would overflow size_t. */
    NUMERUS_STORAGE_OVERFLOW,
    /** Memory allocation failed. */
    NUMERUS_STORAGE_OUT_OF_MEMORY,
    /** A matrix coordinate or sparse index is outside the valid range. */
    NUMERUS_STORAGE_OUT_OF_BOUNDS
} numerus_storage_status;

/**
 * @brief Concrete representation used by an immutable Storage instance.
 *
 * Several kinds are implicit and therefore require little or no value storage.
 * All representations expose the same logical matrix through
 * numerus_storage_get().
 */
typedef enum {
    /** Fully populated row-major storage. */
    NUMERUS_STORAGE_DENSE,
    /** Values on and above the main diagonal; below it reads as zero. */
    NUMERUS_STORAGE_UPPER_TRIANGULAR,
    /** Values on and below the main diagonal; above it reads as zero. */
    NUMERUS_STORAGE_LOWER_TRIANGULAR,
    /** Main diagonal values; all off-diagonal reads return zero. */
    NUMERUS_STORAGE_DIAGONAL,
    /** Identity matrix; stores no matrix values. */
    NUMERUS_STORAGE_IDENTITY,
    /** Every element has the same value. */
    NUMERUS_STORAGE_CONSTANT,
    /** Every element is zero; stores no matrix values. */
    NUMERUS_STORAGE_ZERO,
    /** Diagonal matrix with one shared diagonal value. */
    NUMERUS_STORAGE_SCALED_IDENTITY,
    /** Default value plus explicitly stored sparse overrides. */
    NUMERUS_STORAGE_SPARSE,
    /** Symmetric matrix stored as one triangular half. */
    NUMERUS_STORAGE_SYMMETRIC,
    /** Matrix with fixed lower and upper bandwidths. */
    NUMERUS_STORAGE_BANDED
} numerus_storage_kind;

/**
 * @brief Opaque immutable matrix storage object.
 *
 * The representation is private to numerus_storage.c so other Numerus
 * components cannot couple themselves to a particular storage layout.
 */
typedef struct numerus_storage numerus_storage;

/**
 * @brief One explicit entry in sparse storage.
 *
 * The index is a row-major linear index:
 * row * columns + column.
 */
typedef struct {
    size_t index;
    double value;
} numerus_storage_sparse_entry;

/**
 * @brief Create dense row-major storage.
 *
 * @param rows Number of rows.
 * @param columns Number of columns.
 * @param values Row-major values to copy into the storage.
 * @param storage Receives the newly allocated storage.
 * @return NUMERUS_STORAGE_SUCCESS on success, otherwise an error status.
 *
 * The input buffer is copied and is not retained by Storage. A non-zero
 * element count therefore requires values to be non-NULL.
 */
int numerus_storage_create_dense(size_t rows, size_t columns, const double *values, numerus_storage **storage);

/**
 * @brief Create packed upper-triangular storage.
 *
 * Values are supplied row-by-row for the elements on and above the main
 * diagonal. Reads below the diagonal return zero.
 */
int numerus_storage_create_upper_triangular(size_t size, const double *values, numerus_storage **storage);

/**
 * @brief Create packed lower-triangular storage.
 *
 * Values are supplied row-by-row for the elements on and below the main
 * diagonal. Reads above the diagonal return zero.
 */
int numerus_storage_create_lower_triangular(size_t size, const double *values, numerus_storage **storage);

/**
 * @brief Create diagonal storage.
 *
 * @param size Matrix dimension.
 * @param values Main diagonal values to copy.
 * @param storage Receives the newly allocated storage.
 * @return NUMERUS_STORAGE_SUCCESS on success, otherwise an error status.
 *
 * Off-diagonal reads return zero.
 */
int numerus_storage_create_diagonal(size_t size, const double *values, numerus_storage **storage);

/**
 * @brief Create an identity matrix.
 *
 * The representation stores no matrix values. Diagonal reads return 1.0 and
 * off-diagonal reads return 0.0.
 */
int numerus_storage_create_identity(size_t size, numerus_storage **storage);

/**
 * @brief Create storage whose every element has the same value.
 *
 * @param rows Number of rows.
 * @param columns Number of columns.
 * @param value Value returned for every valid coordinate.
 * @param storage Receives the newly allocated storage.
 * @return NUMERUS_STORAGE_SUCCESS on success, otherwise an error status.
 */
int numerus_storage_create_constant(size_t rows, size_t columns, double value, numerus_storage **storage);

/**
 * @brief Create an all-zero matrix.
 *
 * The representation stores no matrix values.
 */
int numerus_storage_create_zero(size_t rows, size_t columns, numerus_storage **storage);

/**
 * @brief Create a scaled identity matrix.
 *
 * Diagonal reads return value; off-diagonal reads return zero.
 */
int numerus_storage_create_scaled_identity(size_t size, double value, numerus_storage **storage);

/**
 * @brief Create default-value sparse storage.
 *
 * Explicit entries override default_value. Entries are copied, sorted by
 * linear index and required to be unique. Sparse lookup is performed with
 * binary search.
 *
 * @param rows Number of rows.
 * @param columns Number of columns.
 * @param default_value Value returned for coordinates without an override.
 * @param entries Explicit row-major entries, or NULL when count is zero.
 * @param count Number of explicit entries.
 * @param storage Receives the newly allocated storage.
 * @return NUMERUS_STORAGE_SUCCESS on success, or an error status.
 */
int numerus_storage_create_sparse(size_t rows, size_t columns, double default_value, const numerus_storage_sparse_entry *entries, size_t count, numerus_storage **storage);

/**
 * @brief Create symmetric storage from one triangular half.
 *
 * The supplied values are the lower-triangular elements in packed row order.
 * Reads from the upper half are mirrored onto the lower half.
 */
int numerus_storage_create_symmetric(size_t size, const double *values, numerus_storage **storage);

/**
 * @brief Create fixed-bandwidth storage.
 *
 * Each row stores a fixed-width band containing lower_bandwidth elements below
 * the diagonal, the diagonal, and upper_bandwidth elements above it.
 * Coordinates outside the band read as zero. Bandwidths greater than the
 * matrix dimensions are clamped to the largest meaningful values.
 */
int numerus_storage_create_banded(size_t rows, size_t columns, size_t lower_bandwidth, size_t upper_bandwidth, const double *values, numerus_storage **storage);

/**
 * @brief Read one matrix element with validation.
 *
 * @param storage Storage to read.
 * @param row Zero-based row index.
 * @param column Zero-based column index.
 * @param value Receives the logical value as a double.
 * @return NUMERUS_STORAGE_SUCCESS on success, NUMERUS_STORAGE_INVALID_ARGUMENT
 *         for NULL storage/output, or NUMERUS_STORAGE_OUT_OF_BOUNDS for an
 *         invalid coordinate.
 *
 * The output parameter is left unchanged when the operation fails.
 */
numerus_storage_status numerus_storage_get(
    const numerus_storage *storage,
    size_t row,
    size_t column,
    double *value
);

/**
 * @brief Read one matrix element without validation.
 *
 * This is the hot-path accessor for callers that have already validated the
 * storage and coordinates. Passing NULL or out-of-bounds coordinates is
 * undefined behaviour.
 */
double numerus_storage_get_unchecked(
    const numerus_storage *storage,
    size_t row,
    size_t column
);

/**
 * @brief Release an immutable Storage object and its owned buffers.
 *
 * Passing NULL is allowed and has no effect.
 */
void numerus_storage_destroy(numerus_storage *storage);

/**
 * @brief Return the number of rows.
 *
 * Returns zero when storage is NULL.
 */
size_t numerus_storage_rows(const numerus_storage *storage);

/**
 * @brief Return the number of columns.
 *
 * Returns zero when storage is NULL.
 */
size_t numerus_storage_columns(const numerus_storage *storage);

/**
 * @brief Return the concrete storage representation kind.
 *
 * Returns NUMERUS_STORAGE_ZERO when storage is NULL.
 */
numerus_storage_kind numerus_storage_kind_of(const numerus_storage *storage);

#endif
