/**
 * @file numerus_matrix_core.h
 * @brief Matrix lifecycle and access API.
 */
#ifndef NUMERUS_MATRIX_CORE_H
#define NUMERUS_MATRIX_CORE_H

#include "numerus_matrix_types.h"

/**
 * @brief Create a root Matrix from a Storage representation.
 *
 * On success, Matrix owns the newly created Storage.
 *
 * @param kind Storage representation to construct.
 * @param rows Number of rows; must be greater than zero.
 * @param columns Number of columns; must be greater than zero.
 * @param data Representation-specific construction data. It may be NULL for
 *             representations that require no data.
 * @param matrix Receives the new Matrix.
 */
int numerus_matrix_create(
    numerus_storage_kind kind,
    size_t rows,
    size_t columns,
    const numerus_matrix_data *data,
    numerus_matrix **matrix
);

/**
 * @brief Create a Matrix whose values are provided by an existing parent.
 *
 * A child may have different logical dimensions from its parent. This is
 * required for derived views such as a transpose. The parent reference is
 * non-owning; the eventual object layer is responsible for keeping it alive.
 *
 * @param parent Existing parent Matrix.
 * @param rows Number of rows exposed by the child.
 * @param columns Number of columns exposed by the child.
 * @param matrix Receives the new child Matrix.
 */
int numerus_matrix_create_from_parent(
    numerus_matrix *parent,
    size_t rows,
    size_t columns,
    numerus_matrix **matrix
);

/**
 * @brief Create a child Matrix with coordinate and value transforms.
 *
 * A NULL callback selects its identity/pass-through default. The context is
 * borrowed and is never modified or freed by Matrix. It must remain valid and
 * logically unchanged for the child's lifetime. Callbacks must be deterministic
 * and must not depend on mutable external state; this contract is required for
 * the child to remain logically immutable. The const qualifier documents and
 * enforces read-only access through this API, but cannot prevent a caller from
 * mutating the same object through another alias.
 */
int numerus_matrix_create_from_parent_with_transforms(
    numerus_matrix *parent,
    size_t rows,
    size_t columns,
    numerus_coordinate_transform_fn coordinate_transform,
    numerus_value_transform_fn value_transform,
    const void *context,
    numerus_matrix **matrix
);

/** Read one Matrix element with validation. */
numerus_matrix_status numerus_matrix_get(
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    double *value
);

/**
 * @brief Read one Matrix element without validation.
 *
 * This is the hot-path accessor for callers that already validated the
 * Matrix and coordinates.
 */
numerus_matrix_status numerus_matrix_get_unchecked(
    const numerus_matrix *matrix,
    size_t row,
    size_t column,
    double *value
);

/** Return the number of rows, or zero for NULL. */
size_t numerus_matrix_rows(const numerus_matrix *matrix);

/** Return the number of columns, or zero for NULL. */
size_t numerus_matrix_columns(const numerus_matrix *matrix);

/** Return the Storage kind of the root Matrix, or ZERO for NULL. */
numerus_storage_kind numerus_matrix_storage_kind(
    const numerus_matrix *matrix
);

/**
 * @brief Release a Matrix object.
 *
 * A root Matrix releases its owned Storage. A child Matrix only releases its
 * own Matrix object; its parent remains untouched.
 */
void numerus_matrix_destroy(numerus_matrix *matrix);

#endif
