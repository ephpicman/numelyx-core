/**
 * @file numerus_storage.c
 * @brief Immutable storage representations and coordinate-based value access.
 *
 * @details Each Storage owns only the payload required by its representation.
 * Matrix code depends on this abstraction rather than the concrete union below;
 * constructors establish representation invariants and accessors translate
 * logical coordinates into the corresponding packed or implicit value.
 */

#include "numerus_storage.h"
#include "numerus_size.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef NUMERUS_STORAGE_USE_LIBC_ALLOC
# include "php.h"
# include "Zend/zend_alloc.h"
# define numerus_alloc(size) emalloc(size)
# define numerus_calloc(count, size) ecalloc((count), (size))
# define numerus_free(ptr) efree(ptr)
#else
# define numerus_alloc(size) malloc(size)
# define numerus_calloc(count, size) calloc((count), (size))
# define numerus_free(ptr) free(ptr)
#endif

/* The tag determines which union member is active. Keep this representation
 * private: all callers must go through the validated constructor/accessor API.
 */
struct numerus_storage {
    numerus_storage_kind kind;
    size_t rows;
    size_t columns;

    union {
        struct { double *values; } dense;
        struct { double *values; } upper_triangular;
        struct { double *values; } lower_triangular;
        struct { double *values; } diagonal;
        struct { double value; } constant;
        struct { double value; } scaled_identity;
        struct {
            size_t count;
            numerus_storage_sparse_entry *entries;
            double default_value;
        } sparse;
        struct { double *values; } symmetric;
        struct {
            size_t lower_bandwidth;
            size_t upper_bandwidth;
            double *values;
        } banded;
    } data;
};

/* Adapt shared checked arithmetic to Storage's status convention. */
static int size_mul(size_t a, size_t b, size_t *result)
{
    return numerus_size_multiply(a, b, result)
        ? NUMERUS_STORAGE_SUCCESS
        : NUMERUS_STORAGE_OVERFLOW;
}

static int size_add(size_t a, size_t b, size_t *result)
{
    return numerus_size_add(a, b, result)
        ? NUMERUS_STORAGE_SUCCESS
        : NUMERUS_STORAGE_OVERFLOW;
}

static int triangular_count(size_t n, size_t *result)
{
    return numerus_size_triangular_count(n, result)
        ? NUMERUS_STORAGE_SUCCESS
        : NUMERUS_STORAGE_OVERFLOW;
}

/* Allocate and initialise the opaque Storage object itself. */

static int allocate_storage(
    numerus_storage_kind kind,
    size_t rows,
    size_t columns,
    numerus_storage **storage
)
{
    numerus_storage *result;

    if (storage == NULL) {
        return NUMERUS_STORAGE_INVALID_ARGUMENT;
    }
    *storage = NULL;

    result = numerus_calloc(1, sizeof(*result));
    if (result == NULL) {
        return NUMERUS_STORAGE_OUT_OF_MEMORY;
    }

    result->kind = kind;
    result->rows = rows;
    result->columns = columns;
    *storage = result;

    return NUMERUS_STORAGE_SUCCESS;
}

/* Allocate zero-initialised storage for count double values. */
static int allocate_values(size_t count, double **values)
{
    size_t bytes;

    if (values == NULL) {
        return NUMERUS_STORAGE_INVALID_ARGUMENT;
    }

    if (size_mul(count, sizeof(double), &bytes) != NUMERUS_STORAGE_SUCCESS) {
        return NUMERUS_STORAGE_OVERFLOW;
    }

    *values = numerus_calloc(1, bytes);

    if (*values == NULL && bytes != 0) {
        return NUMERUS_STORAGE_OUT_OF_MEMORY;
    }

    return NUMERUS_STORAGE_SUCCESS;
}

/* Allocate owned storage and copy an immutable input value buffer. */
static int copy_values(double **destination, const double *source, size_t count)
{
    int status = allocate_values(count, destination);

    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    if (count != 0 && source == NULL) {
        numerus_free(*destination);
        *destination = NULL;
        return NUMERUS_STORAGE_INVALID_ARGUMENT;
    }

    if (count != 0) {
        memcpy(*destination, source, count * sizeof(double));
    }

    return NUMERUS_STORAGE_SUCCESS;
}

/* Return the packed upper-triangular offset for row <= column. The closed-form
 * prefix count skips all preceding rows; callers must validate coordinates and
 * construction must have checked the triangular allocation size first.
 */
static size_t upper_offset(size_t size, size_t row, size_t column)
{
    size_t a = row;
    size_t b = 2 * size - row + 1;
    size_t skipped;

    if (a % 2 == 0) {
        a /= 2;
    } else {
        b /= 2;
    }

    skipped = a * b;
    return skipped + (column - row);
}

/* Row-major packed lower-triangle offset: each preceding row contributes
 * 1, 2, ..., row elements. Valid coordinates require column <= row.
 */
static size_t lower_offset(size_t row, size_t column)
{
    return row * (row + 1) / 2 + column;
}

/* Return the packed lower-triangular offset used by symmetric storage. */
static size_t triangular_offset(size_t row, size_t column)
{
    return row * (row + 1) / 2 + column;
}

/** Create an immutable row-major dense Storage. */
int numerus_storage_create_dense(
    size_t rows,
    size_t columns,
    const double *values,
    numerus_storage **storage
)
{
    size_t count;
    int status;

    if (size_mul(rows, columns, &count) != NUMERUS_STORAGE_SUCCESS) {
        return NUMERUS_STORAGE_OVERFLOW;
    }

    status = allocate_storage(NUMERUS_STORAGE_DENSE, rows, columns, storage);
    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    status = copy_values(&(*storage)->data.dense.values, values, count);
    if (status != NUMERUS_STORAGE_SUCCESS) {
        numerus_storage_destroy(*storage);
        *storage = NULL;
    }

    return status;
}

/** Create immutable packed upper-triangular Storage. */
int numerus_storage_create_upper_triangular(
    size_t size,
    const double *values,
    numerus_storage **storage
)
{
    size_t count;
    int status = triangular_count(size, &count);

    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    status = allocate_storage(
        NUMERUS_STORAGE_UPPER_TRIANGULAR,
        size,
        size,
        storage
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    status = copy_values(
        &(*storage)->data.upper_triangular.values,
        values,
        count
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        numerus_storage_destroy(*storage);
        *storage = NULL;
    }

    return status;
}

/** Create immutable packed lower-triangular Storage. */
int numerus_storage_create_lower_triangular(
    size_t size,
    const double *values,
    numerus_storage **storage
)
{
    size_t count;
    int status = triangular_count(size, &count);

    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    status = allocate_storage(
        NUMERUS_STORAGE_LOWER_TRIANGULAR,
        size,
        size,
        storage
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    status = copy_values(
        &(*storage)->data.lower_triangular.values,
        values,
        count
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        numerus_storage_destroy(*storage);
        *storage = NULL;
    }

    return status;
}

/** Create immutable diagonal Storage. */
int numerus_storage_create_diagonal(
    size_t size,
    const double *values,
    numerus_storage **storage
)
{
    int status = allocate_storage(
        NUMERUS_STORAGE_DIAGONAL,
        size,
        size,
        storage
    );

    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    status = copy_values(
        &(*storage)->data.diagonal.values,
        values,
        size
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        numerus_storage_destroy(*storage);
        *storage = NULL;
    }

    return status;
}

/** Create immutable identity Storage without storing matrix values. */
int numerus_storage_create_identity(size_t size, numerus_storage **storage)
{
    return allocate_storage(
        NUMERUS_STORAGE_IDENTITY,
        size,
        size,
        storage
    );
}

/** Create immutable constant-value Storage. */
int numerus_storage_create_constant(
    size_t rows,
    size_t columns,
    double value,
    numerus_storage **storage
)
{
    int status = allocate_storage(
        NUMERUS_STORAGE_CONSTANT,
        rows,
        columns,
        storage
    );

    if (status == NUMERUS_STORAGE_SUCCESS) {
        (*storage)->data.constant.value = value;
    }

    return status;
}

/** Create immutable zero Storage without storing matrix values. */
int numerus_storage_create_zero(
    size_t rows,
    size_t columns,
    numerus_storage **storage
)
{
    return allocate_storage(
        NUMERUS_STORAGE_ZERO,
        rows,
        columns,
        storage
    );
}

/** Create immutable scaled-identity Storage. */
int numerus_storage_create_scaled_identity(
    size_t size,
    double value,
    numerus_storage **storage
)
{
    int status = allocate_storage(
        NUMERUS_STORAGE_SCALED_IDENTITY,
        size,
        size,
        storage
    );

    if (status == NUMERUS_STORAGE_SUCCESS) {
        (*storage)->data.scaled_identity.value = value;
    }

    return status;
}

/* Compare sparse entries by their row-major linear index for qsort. */
static int sparse_compare(const void *left, const void *right)
{
    const numerus_storage_sparse_entry *a = left;
    const numerus_storage_sparse_entry *b = right;

    if (a->index < b->index) {
        return -1;
    }

    if (a->index > b->index) {
        return 1;
    }

    return 0;
}

/**
 * Create immutable default-value sparse Storage and canonicalise its entries.
 * Entries are sorted by their row-major linear index so coordinate lookup can
 * use binary search. Construction also validates index bounds and resolves
 * duplicate-index policy before the object becomes observable.
 */
int numerus_storage_create_sparse(
    size_t rows,
    size_t columns,
    double default_value,
    const numerus_storage_sparse_entry *entries,
    size_t count,
    numerus_storage **storage
)
{
    size_t element_count;
    size_t bytes;
    int status;

    if (size_mul(rows, columns, &element_count) != NUMERUS_STORAGE_SUCCESS) {
        return NUMERUS_STORAGE_OVERFLOW;
    }

    if (count != 0 && entries == NULL) {
        return NUMERUS_STORAGE_INVALID_ARGUMENT;
    }

    status = allocate_storage(
        NUMERUS_STORAGE_SPARSE,
        rows,
        columns,
        storage
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    if (count != 0) {
        if (size_mul(count, sizeof(*entries), &bytes) != NUMERUS_STORAGE_SUCCESS) {
            numerus_storage_destroy(*storage);
            *storage = NULL;
            return NUMERUS_STORAGE_OVERFLOW;
        }

        (*storage)->data.sparse.entries = numerus_alloc(bytes);
        if ((*storage)->data.sparse.entries == NULL) {
            numerus_storage_destroy(*storage);
            *storage = NULL;
            return NUMERUS_STORAGE_OUT_OF_MEMORY;
        }

        memcpy((*storage)->data.sparse.entries, entries, bytes);
        qsort(
            (*storage)->data.sparse.entries,
            count,
            sizeof(*entries),
            sparse_compare
        );

        for (size_t i = 1; i < count; i++) {
            if ((*storage)->data.sparse.entries[i - 1].index ==
                (*storage)->data.sparse.entries[i].index) {
                numerus_storage_destroy(*storage);
                *storage = NULL;
                return NUMERUS_STORAGE_INVALID_ARGUMENT;
            }
        }

        for (size_t i = 0; i < count; i++) {
            if ((*storage)->data.sparse.entries[i].index >= element_count) {
                numerus_storage_destroy(*storage);
                *storage = NULL;
                return NUMERUS_STORAGE_OUT_OF_BOUNDS;
            }
        }
    }

    (*storage)->data.sparse.count = count;
    (*storage)->data.sparse.default_value = default_value;

    return NUMERUS_STORAGE_SUCCESS;
}

/** Create immutable symmetric Storage from a packed lower triangle. */
int numerus_storage_create_symmetric(
    size_t size,
    const double *values,
    numerus_storage **storage
)
{
    size_t count;
    int status = triangular_count(size, &count);

    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    status = allocate_storage(
        NUMERUS_STORAGE_SYMMETRIC,
        size,
        size,
        storage
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    status = copy_values(
        &(*storage)->data.symmetric.values,
        values,
        count
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        numerus_storage_destroy(*storage);
        *storage = NULL;
    }

    return status;
}

/** Create immutable fixed-bandwidth Storage. */
int numerus_storage_create_banded(
    size_t rows,
    size_t columns,
    size_t lower_bandwidth,
    size_t upper_bandwidth,
    const double *values,
    numerus_storage **storage
)
{
    size_t width;
    size_t count;
    int status;

    if (rows == 0 || columns == 0) {
        lower_bandwidth = 0;
        upper_bandwidth = 0;
    } else {
        if (lower_bandwidth >= rows) {
            lower_bandwidth = rows - 1;
        }

        if (upper_bandwidth >= columns) {
            upper_bandwidth = columns - 1;
        }
    }

    if (size_add(lower_bandwidth, upper_bandwidth, &width) != NUMERUS_STORAGE_SUCCESS ||
        size_add(width, 1, &width) != NUMERUS_STORAGE_SUCCESS ||
        size_mul(rows, width, &count) != NUMERUS_STORAGE_SUCCESS) {
        return NUMERUS_STORAGE_OVERFLOW;
    }

    status = allocate_storage(
        NUMERUS_STORAGE_BANDED,
        rows,
        columns,
        storage
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        return status;
    }

    (*storage)->data.banded.lower_bandwidth = lower_bandwidth;
    (*storage)->data.banded.upper_bandwidth = upper_bandwidth;

    status = copy_values(
        &(*storage)->data.banded.values,
        values,
        count
    );
    if (status != NUMERUS_STORAGE_SUCCESS) {
        numerus_storage_destroy(*storage);
        *storage = NULL;
    }

    return status;
}

/* Look up one valid sparse coordinate using binary search. */
static double sparse_get(
    const numerus_storage *storage,
    size_t row,
    size_t column
)
{
    size_t index = row * storage->columns + column;
    size_t left = 0;
    size_t right = storage->data.sparse.count;

    while (left < right) {
        size_t middle = left + (right - left) / 2;
        size_t candidate = storage->data.sparse.entries[middle].index;

        if (candidate == index) {
            return storage->data.sparse.entries[middle].value;
        }

        if (candidate < index) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }

    return storage->data.sparse.default_value;
}

/** Read a valid coordinate without repeating bounds or NULL checks. */
double numerus_storage_get_unchecked(
    const numerus_storage *storage,
    size_t row,
    size_t column
)
{
    switch (storage->kind) {
        case NUMERUS_STORAGE_DENSE:
            return storage->data.dense.values[row * storage->columns + column];

        case NUMERUS_STORAGE_UPPER_TRIANGULAR:
            return row > column
                ? 0.0
                : storage->data.upper_triangular.values[
                    upper_offset(storage->columns, row, column)
                ];

        case NUMERUS_STORAGE_LOWER_TRIANGULAR:
            return row < column
                ? 0.0
                : storage->data.lower_triangular.values[
                    lower_offset(row, column)
                ];

        case NUMERUS_STORAGE_DIAGONAL:
            return row == column
                ? storage->data.diagonal.values[row]
                : 0.0;

        case NUMERUS_STORAGE_IDENTITY:
            return row == column ? 1.0 : 0.0;

        case NUMERUS_STORAGE_CONSTANT:
            return storage->data.constant.value;

        case NUMERUS_STORAGE_ZERO:
            return 0.0;

        case NUMERUS_STORAGE_SCALED_IDENTITY:
            return row == column
                ? storage->data.scaled_identity.value
                : 0.0;

        case NUMERUS_STORAGE_SPARSE:
            return sparse_get(storage, row, column);

        case NUMERUS_STORAGE_SYMMETRIC: {
            size_t r = row;
            size_t c = column;

            if (r < c) {
                size_t tmp = r;
                r = c;
                c = tmp;
            }

            return storage->data.symmetric.values[
                triangular_offset(r, c)
            ];
        }

        case NUMERUS_STORAGE_BANDED: {
            size_t row_width =
                storage->data.banded.lower_bandwidth +
                storage->data.banded.upper_bandwidth + 1;
            size_t distance;

            if (column >= row) {
                distance = column - row;
                if (distance > storage->data.banded.upper_bandwidth) {
                    return 0.0;
                }
            } else {
                distance = row - column;
                if (distance > storage->data.banded.lower_bandwidth) {
                    return 0.0;
                }
            }

            return storage->data.banded.values[
                row * row_width +
                storage->data.banded.upper_bandwidth +
                column - row
            ];
        }
    }

    return 0.0;
}

/** Validate a coordinate and read its logical value. */
numerus_storage_status numerus_storage_get(
    const numerus_storage *storage,
    size_t row,
    size_t column,
    double *value
)
{
    if (storage == NULL || value == NULL) {
        return NUMERUS_STORAGE_INVALID_ARGUMENT;
    }

    if (row >= storage->rows || column >= storage->columns) {
        return NUMERUS_STORAGE_OUT_OF_BOUNDS;
    }

    *value = numerus_storage_get_unchecked(storage, row, column);

    return NUMERUS_STORAGE_SUCCESS;
}

/** Release the Storage object and all owned representation buffers. */
void numerus_storage_destroy(numerus_storage *storage)
{
    if (storage == NULL) {
        return;
    }

    switch (storage->kind) {
        case NUMERUS_STORAGE_DENSE:
            numerus_free(storage->data.dense.values);
            break;
        case NUMERUS_STORAGE_UPPER_TRIANGULAR:
            numerus_free(storage->data.upper_triangular.values);
            break;
        case NUMERUS_STORAGE_LOWER_TRIANGULAR:
            numerus_free(storage->data.lower_triangular.values);
            break;
        case NUMERUS_STORAGE_DIAGONAL:
            numerus_free(storage->data.diagonal.values);
            break;
        case NUMERUS_STORAGE_SPARSE:
            numerus_free(storage->data.sparse.entries);
            break;
        case NUMERUS_STORAGE_SYMMETRIC:
            numerus_free(storage->data.symmetric.values);
            break;
        case NUMERUS_STORAGE_BANDED:
            numerus_free(storage->data.banded.values);
            break;
        case NUMERUS_STORAGE_IDENTITY:
        case NUMERUS_STORAGE_CONSTANT:
        case NUMERUS_STORAGE_ZERO:
        case NUMERUS_STORAGE_SCALED_IDENTITY:
            break;
    }

    numerus_free(storage);
}

/** Return the row count, or zero for NULL. */
size_t numerus_storage_rows(const numerus_storage *storage)
{
    return storage == NULL ? 0 : storage->rows;
}

/** Return the column count, or zero for NULL. */
size_t numerus_storage_columns(const numerus_storage *storage)
{
    return storage == NULL ? 0 : storage->columns;
}

/** Return the representation kind, or ZERO for NULL. */
numerus_storage_kind numerus_storage_kind_of(const numerus_storage *storage)
{
    return storage == NULL ? NUMERUS_STORAGE_ZERO : storage->kind;
}
