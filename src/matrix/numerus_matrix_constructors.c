/**
 * @file numerus_matrix_constructors.c
 * @brief Matrix constructors and initialization of valid representations.
 *
 * @details This implementation is part of the internal Matrix numerical
 * layer. It uses the shared Matrix status model and checked-size utilities;
 * algorithm-specific failure and tolerance behavior is documented alongside
 * the relevant routines below.
 */

#include "numerus_matrix.h"

/** Create a row vector represented as a one-row dense Matrix. */
int numerus_matrix_create_row_vector(
    size_t length,
    const double *values,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_dense(1, length, values, matrix);
}

/** Create a column vector represented as a one-column dense Matrix. */
int numerus_matrix_create_column_vector(
    size_t length,
    const double *values,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create_dense(length, 1, values, matrix);
}

int numerus_matrix_create_dense(
    size_t rows,
    size_t columns,
    const double *values,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};

    data.values = values;
    return numerus_matrix_create(
        NUMERUS_STORAGE_DENSE,
        rows,
        columns,
        &data,
        matrix
    );
}

int numerus_matrix_create_identity(size_t size, numerus_matrix **matrix)
{
    return numerus_matrix_create(
        NUMERUS_STORAGE_IDENTITY,
        size,
        size,
        NULL,
        matrix
    );
}

int numerus_matrix_create_zero(
    size_t rows,
    size_t columns,
    numerus_matrix **matrix
)
{
    return numerus_matrix_create(
        NUMERUS_STORAGE_ZERO,
        rows,
        columns,
        NULL,
        matrix
    );
}

int numerus_matrix_create_zero_square(size_t size, numerus_matrix **matrix)
{
    return numerus_matrix_create_zero(size, size, matrix);
}

int numerus_matrix_create_diagonal(
    size_t size,
    const double *values,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};

    data.values = values;
    return numerus_matrix_create(
        NUMERUS_STORAGE_DIAGONAL,
        size,
        size,
        &data,
        matrix
    );
}

int numerus_matrix_create_constant(
    size_t rows,
    size_t columns,
    double value,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};

    data.value = value;
    return numerus_matrix_create(
        NUMERUS_STORAGE_CONSTANT,
        rows,
        columns,
        &data,
        matrix
    );
}

int numerus_matrix_create_scaled_identity(
    size_t size,
    double value,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};

    data.value = value;
    return numerus_matrix_create(
        NUMERUS_STORAGE_SCALED_IDENTITY,
        size,
        size,
        &data,
        matrix
    );
}

int numerus_matrix_create_upper_triangular(
    size_t size,
    const double *values,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};
    data.values = values;
    return numerus_matrix_create(
        NUMERUS_STORAGE_UPPER_TRIANGULAR, size, size, &data, matrix
    );
}

int numerus_matrix_create_lower_triangular(
    size_t size,
    const double *values,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};
    data.values = values;
    return numerus_matrix_create(
        NUMERUS_STORAGE_LOWER_TRIANGULAR, size, size, &data, matrix
    );
}

int numerus_matrix_create_sparse(
    size_t rows,
    size_t columns,
    double default_value,
    const numerus_storage_sparse_entry *entries,
    size_t count,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};
    data.default_value = default_value;
    data.entries = entries;
    data.count = count;
    return numerus_matrix_create(
        NUMERUS_STORAGE_SPARSE, rows, columns, &data, matrix
    );
}

int numerus_matrix_create_symmetric(
    size_t size,
    const double *values,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};
    data.values = values;
    return numerus_matrix_create(
        NUMERUS_STORAGE_SYMMETRIC, size, size, &data, matrix
    );
}

int numerus_matrix_create_banded(
    size_t rows,
    size_t columns,
    size_t lower_bandwidth,
    size_t upper_bandwidth,
    const double *values,
    numerus_matrix **matrix
)
{
    numerus_matrix_data data = {0};
    data.lower_bandwidth = lower_bandwidth;
    data.upper_bandwidth = upper_bandwidth;
    data.values = values;
    return numerus_matrix_create(
        NUMERUS_STORAGE_BANDED, rows, columns, &data, matrix
    );
}
