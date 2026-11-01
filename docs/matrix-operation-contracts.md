# Matrix Operation Contracts

This document establishes the conventions new numerical operations must follow. It complements the implementation roadmap and the declarations in `numerus_matrix.h`.

## Result types

- An operation that mathematically produces a matrix returns a `numerus_matrix *` through an output parameter.
- Scalar analyses (determinant, trace, norms, rank, aggregate values and condition estimates) return their scalar value through a typed output parameter plus a status.
- Decompositions and analyses with multiple outputs use an explicit result structure or separate documented outputs. Do not encode scalar values as 1×1 matrices merely to force a uniform return type.

## Dimensions and indexing

- Dimensions are strictly positive; empty matrices are not currently supported.
- Public matrix coordinates and selection indices are zero-based.
- Every dimension addition, dimension product, element count, and byte count must be checked before allocation or indexing.
- Use `numerus_size.h` for checked `size_t` addition, multiplication, and triangular counts. Do not repeat ad-hoc overflow expressions in new algorithms.
- A failed operation must not expose a partially initialized Matrix. Initialize output pointers to `NULL` as soon as the output argument itself has been validated, and publish the result only after construction succeeds.
- For non-pointer output parameters, preserve the caller's previous value on failure.

## Status and numeric failure policy

The current Matrix status enum distinguishes invalid arguments, out-of-bounds access, incompatible dimensions, non-square inputs, size overflow, allocation failure, scalar division by zero, and success. Scalar division rejects a zero divisor with `NUMERUS_MATRIX_DIVISION_BY_ZERO`; element-wise division instead follows IEEE-754 and returns infinities/NaN as successful numeric values. These meanings must remain distinct.

New status values should be added when a concrete operation requires them:
- A singular matrix is a valid input for which an inverse may not exist; it is not an invalid argument and must not be reported as a successful zero-valued inverse.
- Failure to converge is different from invalid input and from a valid numerical result.
- A numerical breakdown or unsupported input class must be documented separately from allocation or dimension errors.

Do not add speculative statuses before an operation needs them. When required, update both Matrix and Storage mapping where applicable, document the exact trigger, and test that failure leaves outputs unchanged. Never use a numeric result such as `0.0`, `NAN`, or infinity as the sole error signal.

## Floating-point policy

- Matrix values are C `double`; integer-valued inputs are represented as doubles. Complex numbers and arbitrary precision are out of scope.
- Use `numerus_numeric.h` for approximate equality and structural classification.
- Do not automatically use the structural epsilon for pivot selection, singularity classification, numerical rank, or convergence. Those algorithms need scale-aware criteria specific to their numerical method.
- Define NaN/infinity behavior for each operation that compares, aggregates, or transforms values.
- Exact equality and approximate equality are different operations and must remain separately named/documented.

## Laziness and materialization

- A lazy view is appropriate when reading one result element maps cheaply to its parent(s) without recomputing an expensive algorithm.
- An operation such as matrix multiplication should not be represented by a naive lazy getter if each element read repeats an entire dot product. Prefer a materialized independent Storage-backed result unless a measured alternative avoids repeated work.
- Every view must document non-owning parent/context lifetime requirements. The future PHP object layer must retain referenced parent objects.
- Cache flags and results only after successful computation. Cache state must distinguish “not computed” from valid false/zero results. A failed operation must remain retryable unless documented otherwise.
- Do not add a generic cache or per-element cache without benchmark evidence and a clear memory/ownership policy.

## Testing and performance acceptance

Every operation PR should include:
- happy-path and shape-validation tests;
- bounds, overflow, and output-preservation tests where applicable;
- numerical edge cases (including NaN/infinity, scale extremes, and ill-conditioned inputs where relevant);
- tests for lazy-view composition and lifetime assumptions where relevant;
- native tests with `-std=c11 -Wall -Wextra -Wpedantic -Werror`, extension/PHPT checks, supported PHP CI jobs, and Debug build;
- benchmark results when performance determines whether to use a view, materialize a result, or cache an analysis.

Benchmarks should compare representative sizes and storage kinds, report compiler/build context and allocations where feasible, and avoid unstable timing thresholds as hard CI gates until the harness is reliable.

## Positive-definiteness classification

`numerus_matrix_classify_definiteness()` requires a finite, real symmetric square Matrix. It reports two predicates through distinct output pointers and leaves both values unchanged on failure. Eigenvalues within `NUMERUS_EPSILON × n × max(|λᵢ|)` of zero are treated as zero. The zero matrix is positive semidefinite but not positive definite. This is a numerical classification, not a symbolic proof.


## Alternative inverse algorithm: Gauss–Jordan

`numerus_matrix_inverse_gauss_jordan()` is an explicit alternative to the default
LU-based inverse. It uses partial pivoting and the same scale-aware numerical
rank classification as the existing inverse contract. It is intended for
algorithm comparison and education, not as a claim of better stability or
performance. It returns an independently materialized dense matrix; on invalid,
non-square, singular, non-finite, overflow, or allocation failure, the output
pointer remains NULL. The LU-based inverse remains the recommended default.


## Scalar addition and subtraction

`numerus_matrix_create_scalar_add(A, c)` returns a lazy view whose elements are
`A[i,j] + c`; `numerus_matrix_create_scalar_subtract(A, c)` returns a lazy view
whose elements are `A[i,j] - c`. These are distinct from element-wise
matrix/matrix addition and subtraction and from scalar multiplication. The
scalar is copied into the allocated view, not borrowed from the caller.
Operations use ordinary IEEE-754 `double` semantics; NaN and infinities are
successful values, and the borrowed parent must outlive the view.


## Inverse-result cache

Each Matrix may retain one owned dense value buffer containing its inverse after
a successful LU inverse. The buffer is owned by the source Matrix and contains
no Matrix/parent pointers, so it cannot form a parent/cache cycle. A cache hit
constructs a fresh independent dense Matrix; callers never own or destroy the
cache buffer itself. Cache population is best-effort: if the
extra cache allocation fails, the already-computed inverse still succeeds.
Destroying the source Matrix destroys its cache. As with existing mutable
structural/determinant caches, concurrent cache population is not thread-safe.
The cache retains at most 64 KiB of inverse element payload per Matrix; larger
inverses are returned normally but are not retained. This bounds each entry,
though total cache memory still scales with the number of eligible Matrix
instances. Use the benchmark's cold-vs-hit comparison when deciding whether to
keep or revise this policy.


## LU factorization cache

A Matrix may retain one LU factorization for reuse by determinant, inverse, and
solve operations. LU factorization storage is reference-counted: the source
Matrix owns one reference, while each active operation owns its own reference.
The factorization contains only packed numeric values and a permutation, never
a pointer to the source Matrix. Cache entries are retained only when the packed
values plus permutation fit within the same 64 KiB per-entry payload limit.
Allocation or retention failure skips caching rather than failing the numerical
operation. As with other lazy caches, concurrent cache population is not
thread-safe.


## Matrix exponential

`numerus_matrix_exponential(A)` computes the real matrix exponential with
scaling-and-squaring and a degree-13 Padé approximant. It requires a square,
finite input and returns an independent dense Matrix. It checks the 1-norm,
all generated linear combinations and products, and every squaring step for
non-finite values. Solver failures are propagated, and the output remains NULL
on failure. This implementation does not claim arbitrary-precision behavior;
results use C `double` arithmetic.


## Symmetric matrix functions

The first real matrix square-root/logarithm APIs deliberately support a clearly
defined class rather than pretending to handle arbitrary non-normal matrices:
`numerus_matrix_symmetric_square_root()` accepts symmetric positive-semidefinite
input and returns its principal symmetric square root;
`numerus_matrix_symmetric_logarithm()` accepts symmetric positive-definite
input. Both use the symmetric eigendecomposition and a scale-relative
eigenvalue tolerance. A square root clamps only small negative eigenvalues
within tolerance; a materially negative eigenvalue returns
`NUMERUS_MATRIX_NOT_POSITIVE_SEMIDEFINITE`. The logarithm rejects eigenvalues
at or below its positivity tolerance with
`NUMERUS_MATRIX_NOT_POSITIVE_DEFINITE`.

`numerus_matrix_symmetric_sine()` and
`numerus_matrix_symmetric_cosine()` evaluate sine/cosine spectrally for real
symmetric inputs. All four operations return independent dense results and
leave output pointers NULL on failure. General non-symmetric square roots,
logarithms, sine and cosine are not claimed by these APIs.
