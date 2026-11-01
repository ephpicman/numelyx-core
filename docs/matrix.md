# Matrix

Matrix is Numerus's internal logical matrix layer above Storage. It provides a
uniform coordinate-based interface while allowing roots, derived views, and
joined matrices to share data without copying it.

Matrix is currently an internal C API, not a PHP-facing object API. The
function-level contract is declared in [`numerus_matrix.h`](../numerus_matrix.h).

## Object model

A Matrix has positive row and column counts and is one of four forms:

- **Root Matrix:** owns an immutable `numerus_storage`.
- **Derived view:** refers to one parent Matrix and applies coordinate and/or
  value transforms when an element is read.
- **Joined view:** refers to two parent Matrices and delegates each read to the
  appropriate parent.
- **Binary arithmetic view:** refers to two equally shaped parents and combines
  their corresponding values on each read.

The Matrix structure is opaque. Callers cannot access or change its internal
fields through the public header.

## Storage representations

Root construction delegates to Storage. Supported kinds are dense,
upper-triangular, lower-triangular, diagonal, identity, constant, zero,
scaled-identity, sparse, symmetric, and banded. Representation-specific
constraints and input layouts are documented in [Storage](storage.md).

Use the general constructor when a representation's full parameter set is
needed:

```c
numerus_matrix_data data = {0};
numerus_matrix *matrix = NULL;

data.values = values;
int status = numerus_matrix_create(
    NUMERUS_STORAGE_DENSE, rows, columns, &data, &matrix
);
```

Convenience constructors are also available for each representation, including
`numerus_matrix_create_dense()`, `numerus_matrix_create_identity()`,
`numerus_matrix_create_sparse()`, and `numerus_matrix_create_banded()`. Row- and column-vector constructors
are also available; they produce ordinary dense Matrices with shapes 1×n and
n×1 respectively.
The root owns the Storage created for it. Constructor input buffers are copied
by Storage and are not retained.

Dimensions must be positive. Constructors that use square-only representations
return `NUMERUS_MATRIX_NOT_SQUARE` when rows and columns differ.

## Reshape and flatten

`numerus_matrix_create_reshape(parent, rows, columns, &result)` creates a
lazy view with a new shape but the same logical element count. Traversal is
row-major: reading the reshaped result in row-major order yields the same
sequence as reading the parent in row-major order. Both dimensions must be
positive. Overflow in either element-count calculation returns
`NUMERUS_MATRIX_OVERFLOW`; unequal counts return
`NUMERUS_MATRIX_DIMENSION_MISMATCH`.

`numerus_matrix_create_flatten(parent, &result)` is the row-vector
specialization, producing shape 1 × (rows·columns). Both operations compose
with existing views and propagate parent read failures. They do not copy
element data; the parent is borrowed and must outlive the result.

## Row and column selection

`numerus_matrix_create_select_rows(parent, indices, count, &result)` and
`numerus_matrix_create_select_columns()` create lazy views using zero-based
index lists. The index list is copied into view-owned memory; callers may reuse
or modify their input arrays after construction. Indices are validated against
the selected parent dimension before allocation.

Selection order is preserved and duplicate indices are allowed, so a row or
column can intentionally appear more than once. Count must be positive; a
null list is invalid, an out-of-range index returns
`NUMERUS_MATRIX_OUT_OF_BOUNDS`, and index-buffer size overflow returns
`NUMERUS_MATRIX_OVERFLOW`. The parent is still borrowed and must outlive the
view. Nested selections compose through the parent accessor.

## Range slices

`numerus_matrix_create_slice(parent, row_start, row_count, column_start,
column_count, &result)` creates a lazy rectangular view using zero-based
coordinates. Row and column counts must be positive. The complete requested
range must fit within the parent; an invalid range returns
`NUMERUS_MATRIX_OUT_OF_BOUNDS`, while a zero count or null argument returns
`NUMERUS_MATRIX_INVALID_ARGUMENT`.

Slices compose with other views, including nested slices. The coordinate
offsets are stored in the slice itself, not in a caller-owned stack context.
The parent remains non-owning and must outlive the slice. Read errors from
the parent propagate without changing the caller's output value.

## Vector-shaped constructors

`numerus_matrix_create_row_vector(length, values, &matrix)` creates a dense
1×length Matrix, while `numerus_matrix_create_column_vector()` creates a
length×1 Matrix. The input buffer is copied into Storage; no separate Vector
type is introduced. Length must be positive, and a failed constructor leaves
the output pointer NULL.

## Lazy views

Views do not materialise a second matrix buffer. A read maps child coordinates
to parent coordinates, reads the parent value, and optionally transforms that
value.

Supported convenience views:

| Constructor | Result |
| --- | --- |
| `numerus_matrix_create_transpose()` | Rows and columns are swapped |
| `numerus_matrix_create_flip_rows()` | Row order is reversed |
| `numerus_matrix_create_flip_columns()` | Column order is reversed |
| `numerus_matrix_create_rotate_90_clockwise()` | Clockwise quarter-turn; dimensions are swapped |
| `numerus_matrix_create_rotate_180()` | Half-turn; dimensions are unchanged |
| `numerus_matrix_create_rotate_90_counterclockwise()` | Counter-clockwise quarter-turn; dimensions are swapped |
| `numerus_matrix_create_remove_row(parent, row, ...)` | Omits one zero-based row; result has one fewer row |
| `numerus_matrix_create_remove_column(parent, column, ...)` | Omits one zero-based column; result has one fewer column |
| `numerus_matrix_create_swap_rows(parent, row1, row2, ...)` | Exchanges two zero-based row positions |
| `numerus_matrix_create_swap_columns(parent, column1, column2, ...)` | Exchanges two zero-based column positions |
| `numerus_matrix_create_scale(parent, scalar, ...)` | Multiplies each logical value by a copied scalar |
| `numerus_matrix_create_negate(parent, ...)` | Negates each logical value |
| `numerus_matrix_create_divide_scalar(parent, divisor, ...)` | Divides each logical value by a nonzero scalar |

Row/column removal and swaps are coordinate transforms. They do not mutate the
parent or copy element data. Removing the only row or only column is rejected
with `NUMERUS_MATRIX_INVALID_ARGUMENT`, because Matrix dimensions must remain
positive. An index outside the corresponding parent dimension returns
`NUMERUS_MATRIX_OUT_OF_BOUNDS`. Swapping an index with itself is valid and
produces an identity mapping.

The convenience constructors above use the generic coordinate-transform
mechanism internally. Use `numerus_matrix_create_from_parent_with_transforms()`
when a custom coordinate mapping or value transformation is needed.

Views can be composed: a view can use another view or a joined Matrix as its
parent. This composes the read mapping without copying element data.

### Custom transforms

`numerus_matrix_create_from_parent_with_transforms()` accepts optional
coordinate and value callbacks. A NULL coordinate callback uses identity
coordinates; a NULL value callback passes the parent value through unchanged.

Callbacks must write their output parameters only on success and return a
`numerus_matrix_status`. A callback error is propagated to the caller, and
the checked getter leaves the caller's output value unchanged on failure.

The context pointer is borrowed: Matrix neither owns nor frees it. It must
remain valid and logically unchanged for the entire lifetime of the child.
Callbacks must be deterministic and must not depend on mutable external state.
The API uses `const void *` so callbacks cannot mutate the context through that
pointer; `const` does not prevent mutation through another alias. If a context
contains pointers to mutable data, the caller remains responsible for preserving
the logical immutability contract.

A custom transform can be logically mutable if its callback depends on mutable
external state; that violates the API contract even though the Matrix object's
fields themselves are opaque.

The scalar multiplication and negation convenience views store their scalar inside
the view itself, so they do not borrow a caller-owned scalar or callback context.
Their parent Matrix is still non-owning and must outlive the view. They use normal
IEEE-754 `double` multiplication semantics: zero, signed zero, NaN, and infinities
are not converted into status errors.

## Joins

Joins are lazy and hold two non-owning parent references.

- **Horizontal join:** appends the second Matrix to the right of the first.
  Parents must have equal row counts. Result rows equal the parent row count,
  and result columns are the sum of their column counts.
- **Vertical join:** appends the second Matrix below the first. Parents must
  have equal column counts. Result columns equal the parent column count, and
  result rows are the sum of their row counts.

Dimension addition is checked for `size_t` overflow. Incompatible dimensions
return `NUMERUS_MATRIX_DIMENSION_MISMATCH`; overflow returns
`NUMERUS_MATRIX_OVERFLOW`. Nested joins remain lazy.

## Matrix multiplication

`numerus_matrix_multiply(left, right, &result)` computes the mathematical
matrix product, not the element-wise Hadamard product. The left column count
must equal the right row count; a mismatch returns
`NUMERUS_MATRIX_DIMENSION_MISMATCH`. The implementation materializes an
independent dense result because a lazy getter would repeat a complete dot
product for every element read.

The current baseline uses the standard row/column/inner triple loop, accumulates
each dot product left-to-right in `double`, and runs in O(m × n × k) time for
an m×k left operand and k×n right operand. The output buffer uses O(m × n)
temporary memory in addition to the independent dense Storage. Result dimension
and byte-count arithmetic are checked before allocation. Parent read failures
are propagated and the output pointer remains NULL on failure. Non-finite
values follow IEEE-754 arithmetic; they are not status errors.

Exact identity and zero operands use guarded fast paths when the other operand
contains only finite, nonnegative values and no negative zero. Sparse-backed
right operands can use a row-compressed multiplication path when at most 25%
of logical entries are nonzero and both operands satisfy that same numeric
precondition. These paths preserve the independent dense result contract.
Negative values, signed zero, NaN, and infinities fall back to the generic loop
so implicit zero terms retain their IEEE-754 behavior.

## Materialization

`numerus_matrix_materialize(source, &result)` copies the logical values exposed
by any Matrix node into a new dense, Storage-backed Matrix. It supports roots,
nested coordinate/value views, joins, and lazy binary arithmetic nodes. The
result owns its Storage and does not retain the source or its parents.

The operation checks both element-count and byte-count arithmetic before
allocation. It reads through the checked getter and propagates callback/parent
errors. If allocation or any read fails, the temporary buffer is released and
the output pointer remains NULL; a partially populated result is never
published. Materialization has O(rows × columns) time and O(rows × columns)
temporary memory, in addition to the independent dense result Storage.

## Kronecker product

`numerus_matrix_kronecker_product(left, right, &result)` computes the
Kronecker product without requiring matching input dimensions. For A of shape
m×n and B of shape p×q, the result has shape (m·p)×(n·q), with
`C[i*p + k, j*q + l] = A[i,j] * B[k,l]`. The result is independent dense
Storage.

All output-dimension, element-count, and byte-count arithmetic is checked.
Parent read failures are propagated, temporary memory is released, and no
partial result is published. The operation uses O(m·n·p·q) time and O(m·p·n·q)
temporary/result memory. Values follow ordinary IEEE-754 multiplication
semantics, including NaN from zero multiplied by infinity.

## Kronecker sum

`numerus_matrix_kronecker_sum(left, right, &result)` computes
`left ⊗ I + I ⊗ right`. Both inputs must be square; otherwise the function
returns `NUMERUS_MATRIX_NOT_SQUARE` and leaves the output pointer NULL. For
square inputs of dimensions m and n, the result is an independent dense
(m·n)×(m·n) Matrix.

Dimension, element-count, and byte-count arithmetic is checked before
allocation. Parent read failures propagate and no partial result is published.
The implementation follows IEEE-754 arithmetic for both products, including
zero multiplied by non-finite values. Time and result storage are O(m²n²).

## Matrix polynomial evaluation

`numerus_matrix_polynomial(base, coefficients, count, &result)` evaluates
`c[0] + c[1]A + ... + c[count - 1]A^(count - 1)` using Horner's method.
Coefficients are ordered from the constant term upward, and count must be
positive. The base must be square. The result is independent dense Storage;
each Horner step uses a materialized matrix product, so intermediate lazy views
do not outlive their parents. Allocation-size overflow and parent read failures
are reported, and the output pointer remains NULL on failure. Floating-point
operations follow ordinary IEEE-754 double arithmetic.

## Signed integer matrix powers

`numerus_matrix_power_signed(base, exponent, &result)` extends the
non-negative power API to signed 64-bit exponents. For a negative exponent it
computes one LU-based inverse and then uses exponentiation by squaring on that
inverse. The base must be square and numerically nonsingular for negative
powers; singular inputs return `NUMERUS_MATRIX_SINGULAR`. The result is
independent dense Storage, and the output remains NULL on failure.

The implementation computes the magnitude without directly negating
`INT64_MIN`, avoiding signed overflow. The existing `numerus_matrix_power()`
API remains available with its non-negative `size_t` contract.

## Non-negative integer matrix powers

`numerus_matrix_power(base, exponent, &result)` requires a square Matrix.
Exponent zero returns an identity Matrix without reading the base values;
exponent one returns an independent dense materialization. Larger exponents use
exponentiation by squaring and matrix multiplication, producing independent
dense Storage rather than a lazy node that recomputes powers on reads.

The operation propagates materialization/multiplication failures and leaves the
output pointer NULL on failure. Its arithmetic inherits the numerical behavior
of the multiplication implementation. For exponent n, it uses O(log n) matrix
multiplications; the total time is O(n³ log exponent) for an n×n dense matrix,
with O(n²) auxiliary/result storage.

## Exact and approximate equality

`numerus_matrix_is_equal()` compares corresponding values with exact C
`double ==` semantics. `numerus_matrix_is_close()` uses
`numerus_double_equals()` from `numerus_numeric.h`, with the established
combined absolute/relative tolerance (`NUMERUS_EPSILON = 1e-9`). These are
separate operations: approximate equality must not silently replace exact
equality.

For both operations, a shape mismatch is a successful comparison with a
`false` result, not a status error. NaN compares unequal to itself in both
modes; equal infinities compare equal, and positive/negative zero compare equal.
If a parent read fails, the status is propagated and the caller's boolean
output is unchanged. Null arguments return `NUMERUS_MATRIX_INVALID_ARGUMENT`.

The explicit `numerus_matrix_all_close()` name is an alias for the same
all-elements predicate. `numerus_matrix_any_close()` returns true as soon as
one corresponding pair passes `numerus_double_equals()`; otherwise it scans
the complete matrix. Shape mismatches return false successfully. Both APIs
preserve the output boolean if an element read fails.

## Scalar division

`numerus_matrix_create_divide_scalar(parent, divisor, &result)` returns a lazy
value-transform view for `parent / divisor`. Unlike element-wise division,
scalar division rejects positive or negative zero with
`NUMERUS_MATRIX_DIVISION_BY_ZERO`; the output pointer remains NULL. Nonzero
divisors, including NaN and infinities, follow IEEE-754 `double` semantics.
This is direct division by the scalar, not multiplication by a precomputed
reciprocal, so rounding follows the requested operation. The parent is
non-owning and must outlive the view.

## Element-wise addition and subtraction

`numerus_matrix_create_add()`, `numerus_matrix_create_subtract()`,
`numerus_matrix_create_hadamard_product()`, and `numerus_matrix_create_divide()`
create lazy two-parent nodes. Both parents must have identical row and column
counts; otherwise the constructor returns `NUMERUS_MATRIX_DIMENSION_MISMATCH`
and leaves the output pointer NULL. Each read obtains the corresponding value
from both parents, propagates either parent's read failure, then applies ordinary
IEEE-754 `double` addition, subtraction, multiplication, or division. Hadamard
multiplication is element-wise and is not the dot-product-based matrix
multiplication operation.

Element-wise division intentionally follows IEEE-754 behavior: a nonzero value
divided by positive or negative zero produces the corresponding signed
infinity; zero divided by zero and infinity divided by infinity produce NaN;
NaN inputs propagate. These numeric results do not become Matrix status errors.
The operations do not mutate or materialize either input. Both parents must
outlive the result.

## Map and apply

`numerus_matrix_create_map(parent, callback, context, &result)` creates a lazy
value-transform view. The parent is borrowed and must outlive the view; the
callback and context must remain valid, deterministic, and logically unchanged
for the view's lifetime. Reads propagate parent and callback errors.

`numerus_matrix_apply(source, callback, context, &result)` evaluates the callback
eagerly in row-major order and materializes independent dense Storage. Its
context only needs to remain valid during the call. If a source read or callback
fails, the partial buffer is discarded and the output pointer remains NULL.
Callbacks must not mutate the source or depend on mutable external state.

## Variance and standard deviation

`numerus_matrix_variance(matrix, sample, &result)` computes population variance
when `sample=false`, and unbiased sample variance when `sample=true`.
`numerus_matrix_standard_deviation()` returns the square root of the same
quantity. Both use a one-pass Welford accumulation in row-major order, reject
non-finite inputs/intermediates, and leave outputs unchanged on failure.
Sample statistics require at least two elements.

## Normalization

`numerus_matrix_create_normalized(source, axis, norm, &result)` returns an
independent dense Matrix normalized globally, per row, or per column. Supported
norms are L1, L2, and infinity norm. Every input element must be finite; a
non-finite input or overflowing norm returns `NUMERUS_MATRIX_NON_FINITE`.
If any selected vector has zero norm, the whole operation returns
`NUMERUS_MATRIX_DIVISION_BY_ZERO` and no partial result. L2 uses iterative
`hypot()` accumulation to reduce overflow/underflow risk.

## Element-wise minimum, maximum, and clamp

`numerus_matrix_create_elementwise_min()` and
`numerus_matrix_create_elementwise_max()` require equal shapes and produce
independent dense matrices. NaN propagates if either corresponding operand is
NaN; this deliberately differs from C `fmin()`/`fmax()` behavior. Otherwise
ordinary floating-point ordering applies, including infinities.

`numerus_matrix_create_clamp(source, lower, upper, &result)` returns an
independent dense matrix with each non-NaN value limited to the inclusive
interval `[lower, upper]`. NaN values propagate. NaN bounds and
`lower > upper` return `NUMERUS_MATRIX_INVALID_ARGUMENT`; infinite bounds are
allowed. Source-read failures propagate and output pointers remain NULL on
failure. Successful results do not depend on input lifetimes.

## Ownership and lifetime

Ownership is deliberately simple but important:

- A root Matrix owns and destroys its Storage.
- A derived view does not own its parent.
- A joined view does not own either parent.
- A binary arithmetic view does not own either parent.
- Destroying a view does not destroy its parent.
- Every parent must remain alive while any dependent view can be read.

The C caller is responsible for satisfying these lifetime rules. The planned
PHP object layer must retain parent objects (including both parents of a join)
so that the underlying C pointers cannot outlive their targets.

Destroying a parent while a child or joined Matrix still refers to it leaves a
dangling pointer and makes later reads invalid. Do not rely on destruction
order being inferred or managed by the C Matrix API.

## Reading values and errors

`numerus_matrix_get()` validates the Matrix pointer, output pointer, and
coordinates. It returns a `numerus_matrix_status` and writes the output value
only on success.

Relevant statuses:

- `NUMERUS_MATRIX_SUCCESS`
- `NUMERUS_MATRIX_INVALID_ARGUMENT`
- `NUMERUS_MATRIX_OVERFLOW`
- `NUMERUS_MATRIX_OUT_OF_MEMORY`
- `NUMERUS_MATRIX_OUT_OF_BOUNDS`
- `NUMERUS_MATRIX_NOT_SQUARE`
- `NUMERUS_MATRIX_DIMENSION_MISMATCH`
- `NUMERUS_MATRIX_DIVISION_BY_ZERO`\n- `NUMERUS_MATRIX_NON_FINITE`

A numeric zero is a valid value, not an error signal.

`numerus_matrix_get_unchecked()` skips validation and is intended for callers
that have already established valid pointers and coordinates. Despite its name,
it returns a status because custom callbacks and parent reads can still fail.
Only use it when the Matrix and coordinate preconditions are satisfied.

The metadata functions `numerus_matrix_rows()` and
`numerus_matrix_columns()` return zero for a NULL pointer.
`numerus_matrix_storage_kind()` reports the root Storage kind for ordinary
derived views and returns `NUMERUS_STORAGE_ZERO` for NULL. For a joined or binary arithmetic Matrix,
the implementation currently follows the first parent; therefore, this value
does not describe both sides of the node and should not be treated as a complete
summary of its input representations.


## Linear-system solve

`numerus_matrix_solve(A, B, &X)` solves (A X = B) for square,
numerically nonsingular (A). The right-hand side may contain multiple
columns; the implementation factorizes (A) once and reuses that LU
factorization for every column. It returns an independent dense solution Matrix
and does not form (A^{-1} B).

The solver uses the same scale-aware singularity threshold as numerical rank.
It returns `NUMERUS_MATRIX_NOT_SQUARE` for non-square (A),
`NUMERUS_MATRIX_DIMENSION_MISMATCH` when the right-hand-side row count differs,
`NUMERUS_MATRIX_SINGULAR` for a deficient numerical rank, and
`NUMERUS_MATRIX_NON_FINITE` for non-finite inputs or intermediates. On
failure, the output pointer remains NULL.

## Weighted least-squares solve

`numerus_matrix_weighted_least_squares(A, B, weights, count, &X)` minimizes
the weighted squared residual sum. The weight count must equal rows(A), and
each weight must be finite and nonnegative. Zero-weight rows are excluded by
multiplying each row of A and B by sqrt(weight); the resulting problem is
solved by the same pivoted-QR least-squares path. This avoids normal equations.

The current implementation materializes the scaled inputs, so temporary
memory is O(m*n + m*p) for m rows, n coefficients, and p RHS columns. If the
weighted design matrix loses numerical column rank, the call returns
`NUMERUS_MATRIX_RANK_DEFICIENT`. Invalid negative weights, non-finite values,
and source read failures return explicit statuses; on failure, the output
pointer remains NULL.

## Least-squares solve

`numerus_matrix_least_squares(A, B, &X)` computes a minimizer of
`||A X - B||₂` using column-pivoted Householder QR. It supports overdetermined
and square systems with one or multiple right-hand-side columns. The result is
a dense Matrix of shape columns(A)-by-columns(B). It does not form normal
equations or an inverse.

The current contract requires full numerical column rank. Underdetermined or
rank-deficient inputs return `NUMERUS_MATRIX_RANK_DEFICIENT` rather than
silently selecting an arbitrary minimizer; a future minimum-norm solution
belongs with pseudoinverse/SVD work. Non-finite inputs/intermediates and source
read failures propagate their statuses, and the output pointer remains NULL
on failure.

## Linear-system classification

`numerus_matrix_classify_system(A, B, &kind)` classifies the numerical
solutions to A X = B for rectangular A and one or multiple right-hand-side
columns. The result is one of `NUMERUS_MATRIX_SOLUTION_UNIQUE`,
`NUMERUS_MATRIX_SOLUTION_INFINITE`, or
`NUMERUS_MATRIX_SOLUTION_INCONSISTENT`.

The implementation uses column-pivoted Householder QR to estimate the
numerical column space, then projects each right-hand side onto that space.
Consistency is based on the residual norm relative to the right-hand-side
norm and the shared epsilon policy. This is a numerical classification, not
symbolic algebra: very ill-conditioned systems can be sensitive to the rank
and residual thresholds. All right-hand sides are inspected; a non-finite
value or source read failure returns its status even if an earlier column was
already found inconsistent. The output classification is unchanged on failure.

## Row-space and column-space bases

`numerus_matrix_column_space_basis(A, &basis, &dimension)` returns selected
independent columns of A, so the basis is shaped rows(A)-by-rank(A).
`numerus_matrix_row_space_basis(A, &basis, &dimension)` returns selected
independent rows, shaped rank(A)-by-columns(A). Both use column-pivoted
Householder QR; the row-space implementation factors a lazy transpose view.
The same scale-aware numerical-rank threshold is used for both operations.

A zero numerical rank is represented by success with `basis == NULL` and
`dimension == 0`; otherwise the basis is independent dense Storage. Failures
leave the dimension output unchanged and return a NULL basis. These are
numerical subspaces, so results near the threshold depend on the documented
floating-point tolerance.

## Numerical null-space basis

`numerus_matrix_null_space(A, &basis, &nullity)` returns columns that span
the numerical null space of A. When nullity is positive, the basis is an
independent dense Matrix of shape columns(A)-by-nullity. If A has full column
rank, the null space is trivial: the call succeeds with `basis == NULL` and
`nullity == 0`. This explicitly represents the empty basis without relaxing
the Matrix invariant that dimensions are positive.

The implementation uses column-pivoted Householder QR and solves the resulting
triangular system for each free variable. Numerical rank uses a scale-aware
threshold based on `NUMERUS_EPSILON`, the largest absolute input value, and
the larger input dimension. Consequently, the result is a **numerical**
null-space basis: vectors may have small nonzero residuals for matrices near
the rank threshold. Tests verify `A*basis` residuals for rectangular,
rank-deficient, zero, and near-singular inputs. Non-finite values and source
read failures propagate their status; on failure, the basis is NULL and the
nullity output is unchanged.

## Spectral norm and spectral radius

`numerus_matrix_spectral_norm(A, &norm)` returns the largest singular value
for any real Matrix, using the SVD implementation. It is a scalar result and
is distinct from the Frobenius norm.

`numerus_matrix_symmetric_spectral_radius(A, &radius)` returns the largest
absolute eigenvalue for real symmetric matrices only. It deliberately rejects
non-symmetric input rather than pretending the symmetric eigensolver handles
general real matrices, whose eigenvalues may be complex. Both scalar outputs
remain unchanged on failure.

These APIs currently compute the full factorization for a clear baseline.
A specialized largest-singular-value estimator can be considered later if
benchmarks show that full SVD is a material cost for real workloads.

## Symmetric eigendecomposition

`numerus_matrix_symmetric_eigen(A, &eigenvalues, &eigenvectors)` computes
the real eigendecomposition of a finite symmetric square Matrix. Eigenvalues
are returned in descending algebraic order as an n-by-1 vector, and matching
orthonormal eigenvectors are columns of an n-by-n Matrix V, with
A approximately equal to V D Vᵀ.

The implementation uses cyclic Jacobi rotations, with symmetry tolerance and
off-diagonal convergence threshold scaled to the maximum input magnitude and
matrix dimension. It returns `NUMERUS_MATRIX_NOT_SYMMETRIC` for input outside
the symmetry tolerance and `NUMERUS_MATRIX_NO_CONVERGENCE` if the bounded
rotation sweeps do not converge. Eigenvalues may be negative or repeated.
General real matrices are not accepted here; their eigenvalues may be complex.

## Reduced singular value decomposition (SVD)

`numerus_matrix_svd(A, &U, &singular_values, &Vt)` computes the reduced real
factorization A approximately equal to U times Sigma times Vt. If
k = min(rows(A), columns(A)), U has shape rows(A)-by-k, singular values are a
k-by-1 column vector sorted in descending order, and Vt has shape k-by-columns(A).
The columns of U and rows of Vt are orthonormal within floating-point error.

The implementation uses one-sided Jacobi rotations on the tall orientation,
transposing wide inputs as needed. This avoids forming AᵀA, which squares the
condition number. Zero singular values receive orthonormal completion vectors
in U. A bounded sweep count returns `NUMERUS_MATRIX_NO_CONVERGENCE` if
orthogonalization does not converge. Non-finite values and source read failures
propagate their statuses, and all three output pointers remain NULL on failure.

The current implementation is real-valued only. Singular values are
nonnegative; complex singular vectors are not needed for a real SVD.

## Moore–Penrose pseudoinverse

`numerus_matrix_pseudoinverse(A, &A_plus)` computes the real Moore–Penrose
pseudoinverse from the reduced SVD, `A⁺ = V Σ⁺ Uᵀ`. The result is an
independent dense Matrix of shape columns(A)-by-rows(A). Singular values at or
below `NUMERUS_EPSILON * max(rows, columns) * sigma_max` are treated as zero;
this is a scale-aware numerical-rank decision, not symbolic rank.

The operation supports tall, wide, rank-deficient, and all-zero matrices.
Non-finite inputs, SVD convergence failures, allocation failures, and source
read failures propagate their statuses. The output pointer remains NULL on
failure. Native tests cover all four Moore–Penrose conditions.

## LDLᵀ decomposition

`numerus_matrix_ldlt_decompose(A, &L, &D)` computes A approximately equal
to L times D times the transpose of L. L is unit lower triangular and D uses
diagonal Storage. The input must be square, finite, and symmetric within a
scale-aware tolerance. D may contain negative entries, so the algorithm
supports some indefinite symmetric matrices as well as positive-definite ones.

This implementation deliberately does **not** pivot. If a leading pivot is
zero or numerically small, it returns `NUMERUS_MATRIX_PIVOT_TOO_SMALL`, even
when a symmetric permutation could make the matrix factorable. This limitation
is explicit rather than misreporting every such case as mathematically
singular. A pivoted Bunch–Kaufman implementation is deferred until a concrete
consumer justifies its extra block-pivot and permutation API.

## Cholesky decomposition

`numerus_matrix_cholesky(A, &L)` computes a lower-triangular factor such
that A is approximately L times its transpose. The input must be square,
finite, and symmetric within a scale-aware tolerance. Each diagonal pivot must
exceed a threshold based on the largest absolute input element, matrix size,
and `NUMERUS_EPSILON`; semidefinite, indefinite, or numerically singular
inputs return `NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE`. Non-symmetric inputs
return `NUMERUS_MATRIX_NOT_SYMMETRIC`. The factor is an independent dense
Matrix, and the output remains NULL on failure.

Cholesky is O(n³) time and O(n²) workspace. It is intended for symmetric
positive-definite systems; use LU or QR for general matrices.

## Reduced Householder QR decomposition

`numerus_matrix_qr_decompose(A, &Q, &R)` computes a reduced Householder
factorization with A approximately equal to Q times R. If m is the row count,
n the column count, and k = min(m, n), then Q has shape m-by-k and R has
shape k-by-n. The columns of Q are orthonormal within floating-point error;
R is upper trapezoidal.

The implementation uses unpivoted Householder reflectors and materializes
both outputs as independent dense Matrices. It supports tall, wide,
rank-deficient, and zero matrices. It is **not rank-revealing**: column
permutation and numerical-rank classification are deliberately separate
concerns. Non-finite inputs or intermediates return
`NUMERUS_MATRIX_NON_FINITE`; source read failures propagate unchanged. Both
output pointers are NULL on failure.

The algorithm costs O(m*n*k) arithmetic and O(m*n + m*k + k*n) temporary
storage, in addition to the two output Matrices.

## Inverse benchmark coverage

The standalone diagnostic benchmark measures the current LU-based inverse and
one-norm condition estimate on both a diagonally dominant 32×32 matrix and a
diagonal matrix with condition number around 1e7. It reports CPU time and
allocation calls/bytes and checks the maximum residual `max|A*A⁻¹-I|` outside
the timed interval. These measurements are smoke diagnostics, not statistically
reliable performance claims or CI thresholds. A second inverse algorithm is
not added solely to create a comparison; it needs a clear educational or
measured engineering benefit first.

## Condition estimate

`numerus_matrix_condition_estimate_one()` estimates the 1-norm condition
number `||A||₁ × ||A⁻¹||₁` using the implemented inverse. Values near one
indicate a better-conditioned matrix; large values indicate that small
perturbations may be amplified. A singular or numerically singular Matrix
returns success with an infinite estimate. Non-square input and failures
during inversion return statuses, leaving the output unchanged.

This estimate is not a proof of forward accuracy, and its singularity decision
inherits the scale-aware threshold used by `numerus_matrix_rank()`. The
current implementation materializes an inverse, prioritizing a clear baseline
over efficiency; a cheaper estimator can be considered if benchmarks justify
the added algorithmic complexity.

## Matrix inverse

`numerus_matrix_inverse()` computes the inverse through LU factorization and
triangular solves for each identity-column right-hand side. It does not form an
adjugate or multiply by the inverse of the determinant, and it returns a new
dense Matrix independent of the input.

Numerical singularity uses `numerus_matrix_rank()` and its scale-aware
threshold. A deficient rank returns `NUMERUS_MATRIX_SINGULAR`; non-square
input returns `NUMERUS_MATRIX_NOT_SQUARE`; non-finite values and allocation or
read failures propagate their specific status. On any failure, the output
pointer remains NULL. Inverse results are not cached.

## Row-echelon forms

`numerus_matrix_row_echelon_form()` and
`numerus_matrix_reduced_row_echelon_form()` return independent dense Matrices
with the same shape as the input. Both use partial row pivoting and the
scale-aware threshold `NUMERUS_EPSILON * max(rows, columns)` relative to the
largest absolute input value. The reduced form normalizes pivots to one and
eliminates above and below each pivot; the ordinary row-echelon form eliminates
only below pivots.

Non-finite input or intermediate values return
`NUMERUS_MATRIX_NON_FINITE`. Read, allocation, and size failures propagate
their specific status; the output pointer is NULL on failure. These routines
do not mutate or retain the input Matrix.

## Numerical rank

`numerus_matrix_rank()` estimates rank for square or rectangular Matrices using
complete-pivoting elimination. Its relative threshold is
`NUMERUS_EPSILON * max(rows, columns)` relative to the largest absolute input
element, making classification invariant under uniform finite scaling within
ordinary floating-point limits. This is a numerical estimate, not an exact
symbolic rank and not an SVD-quality rank-revealing decomposition.

A zero Matrix has rank zero. NaN/infinity input or non-finite elimination
intermediates return `NUMERUS_MATRIX_NON_FINITE`; element-read failures and
allocation/size failures return their specific statuses. The output rank is
unchanged on failure. The operation uses O(rows×columns) workspace and
O(min(rows, columns)×rows×columns) time.

## Internal LU factorization

The private `numerus_matrix_lu.h` interface factors a square Matrix into
packed lower/upper triangular data and a row permutation. The permutation maps
each factor row to its original source row, so reconstruction tests verify
(P A = L U). The factorization owns its work buffers and does not retain or
mutate the input Matrix.

Partial pivoting selects the largest absolute value in the current column.
Exact-zero pivots are skipped, allowing rank-deficient inputs to produce a
factorization object; the reported rank is the number of non-zero pivots under
that exact-zero policy, not a scale-aware numerical-rank estimate. Read and
allocation failures clean up the workspace and leave the output pointer NULL.
The API is private infrastructure for later determinant, solve, and rank work.

The general determinant path now uses the private LU factorization, multiplying
its diagonal pivots and permutation parity. The cached zero/identity/triangular
fast paths remain in place. This keeps pivot selection and elimination logic
shared with future solve and rank operations.

## Determinant edge cases

The determinant uses Gaussian elimination with partial pivoting and an exact
zero-pivot check. It does not apply `NUMERUS_EPSILON` as a singularity cutoff:
a small but representable determinant can be valid. Results still follow
ordinary double arithmetic, so sufficiently large products may overflow and
sufficiently small products may underflow. Singular matrices return a successful
numeric zero; invalid input and element-read failures return statuses and leave
the output scalar unchanged. Successfully computed results, including zero,
are cached per Matrix.

## Cached analysis

Each Matrix has a per-object analysis cache. It stores structural metadata and
scalar results, not element values, so it does not change the lazy-view model.

`numerus_matrix_get_flags()` computes and caches structural properties:
square, zero, diagonal, upper-triangular, lower-triangular, symmetric, and
identity. These flags describe the logical values exposed by that Matrix,
including views and joins. Floating-point classification uses the comparison
policy in `numerus_numeric.h` (`NUMERUS_EPSILON = 1e-9`): zero checks use an
absolute tolerance, while equality checks use combined absolute/relative
tolerance. A square zero Matrix is also diagonal, upper-triangular,
lower-triangular, and symmetric.

`numerus_matrix_determinant()` caches a successfully computed determinant.
Structural flags use `NUMERUS_EPSILON` (`1e-9`) with an absolute tolerance
for zero checks and a scale-aware combined tolerance for equality checks.
The determinant elimination pivot check remains exact-zero: applying a fixed
epsilon there would incorrectly classify small but valid matrices as singular.
The general path uses Gaussian elimination with partial pivoting. If flags
have already been computed, zero, identity, and triangular matrices can use
cheaper paths. A determinant of zero is a valid cached result. Allocation
failures and element-read failures are not cached and can be retried.

Each Matrix has its own cache. A View or Join does not reuse its parent's
analysis cache because its logical values may differ. These APIs do not
establish thread safety: concurrent access while a cache is being populated is
not guaranteed safe. Keep the Matrix and all parents alive during every call.

## Concurrency

This documentation does not promise general thread safety or concurrent-read
safety. Immutability prevents mutation through the Storage API, but it does not
manage parent lifetime, synchronise custom callbacks, or make concurrent
destruction safe. Keep objects alive for the duration of every operation and
supply callbacks/context that satisfy their documented lifetime and
determinism requirements.

## Padding and zero extension

`numerus_matrix_create_pad()` creates a lazy view with configurable top,
bottom, left, and right padding and a caller-selected constant value.
`numerus_matrix_create_zero_extend()` is the zero-filled convenience form.
The original Matrix remains centered at the requested offsets; reads within
the original region delegate to the borrowed parent, while reads outside it
return the padding value without allocating a dense buffer. Parent read errors
are propagated unchanged. The parent must outlive the view, and padded
dimension arithmetic is checked for overflow.

## Repetition and block-diagonal views

`numerus_matrix_create_repeat()` lazily tiles the parent by positive row and
column repetition counts. `numerus_matrix_create_block_diagonal()` lazily
places repeated copies of the parent along the diagonal and returns zero for
off-diagonal blocks. Block-diagonal construction also supports rectangular
parent matrices; the output shape scales both parent dimensions by the
repetition count. Both operations check dimension multiplication for overflow,
propagate parent read failures, allocate no element buffer, and borrow the
parent, which must outlive the resulting view.

## General block-grid assembly

`numerus_matrix_create_block_grid()` builds a lazy view from a row-major
array of Matrix pointers. Every block must be present: NULL/missing blocks are
rejected instead of silently choosing a fill value. Within each block row,
all blocks must have equal heights; within each block column, all blocks must
have equal widths. The API copies the pointer array and cumulative offsets,
not the matrix data, so every source Matrix must outlive the assembled view.
Block counts, allocation sizes, and total dimensions are checked for overflow.
Reads are dispatched to the corresponding source block and propagate its errors.

## Diagonal extraction and construction

`numerus_matrix_create_diagonal_extract()` returns a lazy 1×N row-vector
view of a selected diagonal. Offset zero selects the main diagonal; positive
offsets select diagonals above it, and negative offsets select diagonals below
it. Offsets with no cells in the parent return `NUMERUS_MATRIX_OUT_OF_BOUNDS`.

`numerus_matrix_create_diagonal_from_vector()` creates a lazy square view
from either a row or column vector. The vector values occupy the requested
main or offset diagonal, and all other cells read as zero. Its side length is
the vector length plus the absolute offset, with overflow checked. Both APIs
borrow their parent and propagate parent read errors.

## Row and column permutations

`numerus_matrix_create_permute_rows()` and
`numerus_matrix_create_permute_columns()` reorder every row or column using
a full zero-based permutation. The list length must match the corresponding
parent dimension; every index must be in range and appear exactly once.
Repeated indices are rejected here (unlike selection views, which intentionally
allow duplicates). The index list is copied, the view remains lazy, and the
parent must outlive it.

## Additional structural predicates

`numerus_matrix_is_skew_symmetric()` checks a square Matrix against
(A^T = -A), using the shared absolute/relative comparison policy and the
absolute zero tolerance for diagonal elements. `numerus_matrix_is_orthogonal()`
checks whether the row dot products equal the identity matrix within the same
comparison tolerance. Non-square inputs return success with `false`, matching
the structural-flag API's predicate behavior. Read failures propagate without
changing the output boolean.

The skew-symmetric check is O(n²). The orthogonal check is O(n³) time and O(1)
auxiliary memory; it does not allocate a temporary product Matrix.

## Finite-value predicates

`numerus_matrix_has_nan()` reports whether at least one element is NaN;
`numerus_matrix_has_infinity()` detects either positive or negative infinity;
and `numerus_matrix_is_finite()` is true only when every element is finite.
These are scalar predicates, not transformations, and do not allocate a new
Matrix. Each scan stops as soon as its answer is determined. If a required
element read fails before the answer is determined, its status is propagated
and the output boolean remains unchanged.

## Matrix norms

`numerus_matrix_norm_frobenius()` computes the Frobenius norm using
`hypot()` accumulation to avoid unnecessary overflow and underflow from
squaring values. The induced 1-norm is the maximum absolute column sum, and
the induced infinity-norm is the maximum absolute row sum.

All three functions return scalar `double` results. Any NaN element makes the
norm NaN; otherwise infinities and overflow in induced absolute sums follow
IEEE-754 behavior. Failed element reads propagate their status and leave the
output scalar unchanged. Each norm scans the logical Matrix once and uses
constant auxiliary memory.

## Trace

`numerus_matrix_trace()` returns the scalar sum of the main diagonal and
requires a square Matrix. It accumulates in increasing diagonal index order
using ordinary `double` addition; NaN and infinities therefore follow
IEEE-754 arithmetic. A failed element read leaves the output scalar unchanged.

## Scalar and axis aggregates

`numerus_matrix_sum()`, `numerus_matrix_min()`, `numerus_matrix_max()`,
and `numerus_matrix_mean()` return scalar results. Sum and mean use ordinary
row-major double accumulation; min/max return NaN if any element is NaN.
Infinities follow normal IEEE-754 arithmetic. A failed read leaves scalar
outputs unchanged.

`numerus_matrix_row_sums()` and `numerus_matrix_row_means()` return an
independent dense rows×1 Matrix. The column variants return an independent
dense 1×columns Matrix. These functions materialize their small output vectors
and propagate parent read errors; on failure the output Matrix pointer remains
NULL.

## Algorithm selection, complexity, and numerical limitations

The following costs describe the current dense numerical algorithms at a high
level. Let m and n be the input dimensions, and let r be the number of
right-hand-side columns. Costs exclude the cost of reading a lazy source,
which may itself perform additional mapping work.

| Operation | Typical time cost | Important limitation |
| --- | --- | --- |
| Element-wise scans, aggregates, and norms | O(mn) | Floating-point accumulation order affects rounding; NaN/infinity behavior is operation-specific. |
| Dense matrix multiplication | O(mkn) for an m×k by k×n product | Uses ordinary double arithmetic; output is materialized. |
| LU factorization, determinant, inverse | O(n³) for n×n input | LU uses partial pivoting; inverse rejects matrices classified as numerically singular. |
| Square solve with r RHS columns | O(n³ + n²r) | One LU factorization is shared across RHS columns; solving does not form an inverse. |
| REF/RREF and numerical rank | O(mn·min(m,n)) | Rank is a scale-aware numerical estimate, not symbolic rank. |
| Reduced Householder QR | O(mn·min(m,n)) | The unpivoted QR API is not rank-revealing. |
| Cholesky and unpivoted LDLᵀ | O(n³) | Cholesky requires positive definiteness; LDLᵀ may reject matrices that need symmetric pivoting. |
| Reduced SVD | Iteration-dependent; each Jacobi sweep is O(mn·min(m,n)) | Convergence is bounded and failure is reported; near-zero singular values are threshold-dependent. |
| Symmetric Jacobi eigendecomposition | O(n³) per sweep | Accepts real symmetric matrices only; general real matrices may have complex eigenvalues and are not supported by this API. |
| Materializing a view | O(mn) time and O(mn) additional storage | The result is independent of its source and no longer depends on source lifetime. |

These are algorithmic estimates, not runtime guarantees. Allocation cost,
Storage kind, view depth, compiler, and floating-point hardware can materially
change observed performance. Use the benchmark harness to compare representative
cases; do not add timing thresholds to CI based on one machine.

### Numerical interpretation

- Exact equality uses C double equality. Approximate comparisons use the
  documented combined tolerance; neither operation proves mathematical
  equivalence.
- Pivot, rank, symmetry, positive-definiteness, and singular-value decisions
  are numerical classifications with documented scale-aware thresholds.
  They are not exact symbolic statements.
- Partial pivoting, Householder QR, and SVD are different numerical tools; do
  not substitute one for another solely because their signatures appear similar.
  In particular, unpivoted QR and unpivoted LDLᵀ have explicit limitations.
- A successful result does not guarantee small forward error for an
  ill-conditioned problem. Inspect residuals and conditioning when accuracy
  matters. A condition estimate is a diagnostic, not an error bound.
- Non-finite input handling is operation-specific. Operations that reject
  non-finite values return a status; arithmetic operations may instead preserve
  IEEE-754 NaN/infinity behavior. Check each function contract.

### Runnable C example: solve without forming an inverse

This example solves A x = b directly. It uses the same public C API exercised
by the native test suite.

```c
#include "numerus_matrix.h"
#include <stdio.h>

int main(void)
{
    const double a_values[] = {3.0, 1.0, 1.0, 2.0};
    const double b_values[] = {9.0, 8.0};
    numerus_matrix *a = NULL;
    numerus_matrix *b = NULL;
    numerus_matrix *x = NULL;
    int status = numerus_matrix_create_dense(2, 2, a_values, &a);

    if (status == NUMERUS_MATRIX_SUCCESS) {
        status = numerus_matrix_create_column_vector(2, b_values, &b);
    }
    if (status == NUMERUS_MATRIX_SUCCESS) {
        status = numerus_matrix_solve(a, b, &x);
    }
    if (status == NUMERUS_MATRIX_SUCCESS) {
        double x0, x1;
        status = (int) numerus_matrix_get(x, 0, 0, &x0);
        if (status == NUMERUS_MATRIX_SUCCESS) {
            status = (int) numerus_matrix_get(x, 1, 0, &x1);
        }
        if (status == NUMERUS_MATRIX_SUCCESS) {
            printf("x = [%.6f, %.6f]\\n", x0, x1);
        }
    }

    numerus_matrix_destroy(x);
    numerus_matrix_destroy(b);
    numerus_matrix_destroy(a);
    return status == NUMERUS_MATRIX_SUCCESS ? 0 : 1;
}
```

Expected solution: x = [2.000000, 3.000000]. In production code, report the
status explicitly rather than silently collapsing every failure into a generic
exit code.

## Native tests

The native Matrix tests cover storage-backed constructors, orientation
transforms, row/column removal and swaps, composed views, lazy scalar
multiplication, powers, Kronecker products, range slices, indexed row/column
selection, reshape/flatten, padding, repetition, block-diagonal, and block-grid, diagonal, permutation views, trace and aggregate analysis, scalar division, equality, vector constructors,
element-wise arithmetic, materialization, and overflow/read-failure handling.

Run them from the repository root:

```sh
cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_test tests/matrix_test.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_unary_test tests/matrix_unary_test.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_unary_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_binary_test tests/matrix_binary_test.c numerus_matrix.c numerus_matrix_binary.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_binary_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_hadamard_test tests/matrix_hadamard_test.c numerus_matrix.c numerus_matrix_binary.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_hadamard_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_division_test tests/matrix_division_test.c numerus_matrix.c numerus_matrix_binary.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_division_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_materialize_test tests/matrix_materialize_test.c numerus_matrix.c numerus_matrix_binary.c numerus_matrix_materialize.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_materialize_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_multiply_test tests/matrix_multiply_test.c numerus_matrix.c numerus_matrix_binary.c numerus_matrix_materialize.c numerus_matrix_multiply.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_multiply_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_scalar_division_test tests/matrix_scalar_division_test.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_scalar_division_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_compare_test tests/matrix_compare_test.c numerus_matrix.c numerus_matrix_compare.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_compare_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_vector_constructor_test tests/matrix_vector_constructor_test.c numerus_matrix.c numerus_matrix_constructors.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_vector_constructor_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_power_test tests/matrix_power_test.c numerus_matrix.c numerus_matrix_binary.c numerus_matrix_materialize.c numerus_matrix_multiply.c numerus_matrix_power.c numerus_matrix_inverse.c numerus_matrix_analysis.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_power_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_kronecker_test tests/matrix_kronecker_test.c numerus_matrix.c numerus_matrix_kronecker.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_kronecker_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_slice_test tests/matrix_slice_test.c numerus_matrix.c numerus_matrix_views.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_slice_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_selection_test tests/matrix_selection_test.c numerus_matrix.c numerus_matrix_views.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_selection_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_reshape_test tests/matrix_reshape_test.c numerus_matrix.c numerus_matrix_views.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_reshape_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_padding_test tests/matrix_padding_test.c numerus_matrix.c numerus_matrix_views.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_padding_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_repeat_test tests/matrix_repeat_test.c numerus_matrix.c numerus_matrix_views.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_repeat_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_block_grid_test tests/matrix_block_grid_test.c numerus_matrix.c numerus_matrix_views.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_block_grid_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_diagonal_test tests/matrix_diagonal_test.c numerus_matrix.c numerus_matrix_views.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_diagonal_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_permutation_test tests/matrix_permutation_test.c numerus_matrix.c numerus_matrix_views.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_permutation_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_trace_test tests/matrix_trace_test.c numerus_matrix.c numerus_matrix_analysis.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_trace_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_aggregate_test tests/matrix_aggregate_test.c numerus_matrix.c numerus_matrix_analysis.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_aggregate_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_norm_test tests/matrix_norm_test.c numerus_matrix.c numerus_matrix_analysis.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_norm_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_finite_test tests/matrix_finite_test.c numerus_matrix.c numerus_matrix_analysis.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_finite_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_determinant_test tests/matrix_determinant_test.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c
./tests/matrix_determinant_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_lu_test tests/matrix_lu_test.c numerus_matrix_lu.c numerus_matrix.c numerus_storage.c
./tests/matrix_lu_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_rank_test tests/matrix_rank_test.c numerus_matrix.c numerus_matrix_analysis.c numerus_matrix_lu.c numerus_storage.c -lm
./tests/matrix_rank_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_elimination_test tests/matrix_elimination_test.c numerus_matrix_elimination.c numerus_matrix_analysis.c numerus_matrix_lu.c numerus_matrix.c numerus_storage.c -lm
./tests/matrix_elimination_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_inverse_test tests/matrix_inverse_test.c numerus_matrix_inverse.c numerus_matrix_analysis.c numerus_matrix_lu.c numerus_matrix.c numerus_storage.c -lm
./tests/matrix_inverse_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_solve_test tests/matrix_solve_test.c numerus_matrix_solve.c numerus_matrix_analysis.c numerus_matrix_lu.c numerus_matrix.c numerus_storage.c -lm
./tests/matrix_solve_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_condition_test tests/matrix_condition_test.c numerus_matrix_condition.c numerus_matrix_inverse.c numerus_matrix_analysis.c numerus_matrix_lu.c numerus_matrix.c numerus_storage.c -lm
./tests/matrix_condition_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_qr_test tests/matrix_qr_test.c numerus_matrix_qr.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_qr_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_cholesky_test tests/matrix_cholesky_test.c numerus_matrix_cholesky.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_cholesky_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_ldlt_test tests/matrix_ldlt_test.c numerus_matrix_ldlt.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_ldlt_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_svd_test tests/matrix_svd_test.c numerus_matrix_svd.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_svd_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_pseudoinverse_test tests/matrix_pseudoinverse_test.c numerus_matrix_pseudoinverse.c numerus_matrix_svd.c numerus_matrix_multiply.c numerus_matrix_views.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_pseudoinverse_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_symmetric_eigen_test tests/matrix_symmetric_eigen_test.c numerus_matrix_symmetric_eigen.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_symmetric_eigen_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_spectral_test tests/matrix_spectral_test.c numerus_matrix_spectral.c numerus_matrix_svd.c numerus_matrix_symmetric_eigen.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_spectral_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_null_space_test tests/matrix_null_space_test.c numerus_matrix_null_space.c numerus_matrix_qr.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_null_space_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_spaces_test tests/matrix_spaces_test.c numerus_matrix_spaces.c numerus_matrix_qr.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_spaces_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_system_test tests/matrix_system_test.c numerus_matrix_system.c numerus_matrix_qr.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_system_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_least_squares_test tests/matrix_least_squares_test.c numerus_matrix_least_squares.c numerus_matrix_qr.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_least_squares_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_weighted_least_squares_test tests/matrix_weighted_least_squares_test.c numerus_matrix_weighted_least_squares.c numerus_matrix_least_squares.c numerus_matrix_qr.c numerus_matrix.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_weighted_least_squares_test

cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -DNUMERUS_MATRIX_USE_LIBC_ALLOC \
  -DNUMERUS_STORAGE_USE_LIBC_ALLOC \
  -o tests/matrix_structural_predicate_test tests/matrix_structural_predicate_test.c numerus_matrix.c numerus_matrix_analysis.c numerus_storage.c numerus_matrix_lu.c -lm
./tests/matrix_structural_predicate_test
```

The standalone allocator defines are only for native tests. Extension builds
use the Zend Memory Manager.
