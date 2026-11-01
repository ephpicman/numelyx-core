# Matrix Storage-Aware Optimization Audit

> **Snapshot dated 2026-10-09:** this document records the optimization review at that point in development. For current Matrix behavior and contracts, use [Matrix documentation](matrix.md) and [Matrix operation contracts](matrix-operation-contracts.md); do not treat this snapshot alone as proof that an optimization remains current or enabled.

Date: 2026-10-09

## Current representations

The Storage implementation currently supports these immutable representations:

| Kind | Value storage | Read behavior / implications |
| --- | --- | --- |
| Dense | O(rows × columns) doubles | Row-major direct lookup. |
| Upper triangular | Packed triangular values | Reads below the diagonal as zero. |
| Lower triangular | Packed triangular values | Reads above the diagonal as zero. |
| Diagonal | O(n) doubles | Reads off-diagonal elements as zero. |
| Identity | No value buffer | Computes 1 on the diagonal and 0 elsewhere. |
| Constant | One double | Every coordinate returns the same value. |
| Zero | No value buffer | Every coordinate returns zero. |
| Scaled identity | One double | Shared diagonal value, zero elsewhere. |
| Sparse | O(explicit entries) entries | Binary-search lookup; absent coordinates use the configured default. |
| Symmetric | Packed triangular values | Reflects coordinates across the diagonal. |
| Banded | O(rows × (lower bandwidth + upper bandwidth + 1)) doubles | Uses fixed row width and returns zero outside the configured band. |

The representation is private to `numerus_storage.c`; callers use the logical accessor and do not depend on internal buffers.

## Current operation behavior

- `numerus_matrix_multiply()` uses a generic triple loop. It checks the inner dimensions, reads both operands through `numerus_matrix_get()`, accumulates each dot product left-to-right in `double`, and materializes independent dense output.
- The multiplication algorithm does not dispatch on Storage kind. Identity, zero, diagonal, triangular, sparse, symmetric, and banded operands still go through the same O(m × k × n) loop and coordinate reads.
- Existing benchmark coverage measures reads for dense, diagonal, upper-triangular, and sparse matrices, but multiplication timing currently covers only dense 32×32 operands. The initial measurements are a single smoke run and are not enough to justify algorithm selection.
- Element-wise binary operations are lazy views; they preserve the logical Storage semantics but do not currently specialize their arithmetic by representation.

## Correctness constraints before adding fast paths

The current implementation evaluates IEEE-754 operations even where an implicit Storage element is zero. For example, a generic dot product can evaluate `0 × INFINITY` or `0 × NaN`, producing NaN. A shortcut that returns the other operand for multiplication by identity, returns an all-zero matrix for multiplication by zero, or skips zero terms may therefore change observable results for non-finite inputs.

Any specialized path must choose and document one of these policies before implementation:

1. Preserve the existing IEEE-754 result exactly, including NaN propagation from zero-times-non-finite terms; or
2. Explicitly narrow the fast-path precondition (for example, prove operands finite) and fall back to the generic implementation otherwise.

Do not silently change the numerical contract in the name of performance. Preserve dimension/status behavior, output-pointer failure guarantees, and materialized-result ownership.

## Initial multiplication measurements

The benchmark harness was extended to five samples of three multiplications each
for 64×64 operands. Results below are medians from one PHP 8.5 CI run, not
portable performance claims. Each sample includes result allocation and
destruction; allocation metrics are per sample.

| Workload | Median seconds / 3 products | Median allocations | Median requested bytes |
| --- | ---: | ---: | ---: |
| Dense × dense | 0.010159 | 12 | 197,520 |
| Dense × identity | 0.010724 | 12 | 197,520 |
| Identity × dense | 0.010673 | 12 | 197,520 |
| Dense × zero | 0.010654 | 12 | 197,520 |
| Dense × diagonal | 0.010737 | 12 | 197,520 |
| Dense × upper triangular | 0.010917 | 12 | 197,520 |
| Dense × sparse | 0.017660 | 12 | 197,520 |

Source: [CI benchmark run](https://github.com/ephpicman/Numerus/actions/runs/37986233768).

These measurements show no useful speedup from the generic loop on structured
operands; sparse reads are noticeably slower. They justify testing identity and
zero fast paths, but not assuming diagonal or triangular shortcuts will win.
The implemented identity/zero and sparse-right paths are guarded by exact structure checks
and a finite, nonnegative operand precondition. Negative values, signed zero,
NaN, and infinities retain the generic implementation's IEEE-754 behavior.


## Specialized constructor scope decision

The current API already has row/column permutation views and structured
Storage for diagonal, triangular, symmetric, banded, identity, constant, zero,
and sparse matrices. No current algorithm or test suite consumes Toeplitz,
Hankel, Vandermonde, or Hilbert matrices as named concepts.

- **Toeplitz/Hankel:** defer until a convolution, signal-processing, or structured
  solver use case needs the API; dense construction alone does not exploit the
  structure.
- **Vandermonde/Hilbert:** defer. They are useful in specific numerical methods,
  but can be badly conditioned and would expand the API without a current
  consumer or a dedicated stability story.
- **Permutation matrix:** defer the separate constructor. Existing row/column
  permutation views cover the immediate operation; a standalone constructor
  would currently materialize dense values unless a new Storage kind were
  designed. Add it when a factorization or decomposition needs it directly.

This closes the scope review, not an implementation commitment. Revisit each
constructor when a concrete consumer and tests can justify its API and Storage
cost.

## Results after guarded fast paths

A later PHP 8.5 CI run measured the implemented paths with the same 64×64
workloads, five samples of three products each. These are medians from one run;
the dense×dense workload is the generic-loop reference in this same run.

| Workload | Median seconds / 3 products | Median allocations | Median requested bytes |
| --- | ---: | ---: | ---: |
| Dense × dense (generic reference) | 0.012838 | 12 | 197,520 |
| Dense × identity | 0.000355 | 12 | 197,520 |
| Identity × dense | 0.000355 | 12 | 197,520 |
| Dense × zero | 0.000247 | 12 | 197,520 |
| Dense × sparse (row-compressed path) | 0.000645 | 21 | 202,152 |
| Dense × diagonal (generic path) | 0.012137 | 12 | 197,520 |
| Dense × upper triangular (generic path) | 0.013245 | 12 | 197,520 |

Source: [CI benchmark run after specialization](https://github.com/ephpicman/Numerus/actions/runs/37988391181).

Within this run, identity multiplication is about 36× faster than the dense
reference, zero multiplication about 52× faster, and the sparse-right path
about 20× faster. Sparse scratch allocation is only 4,632 bytes above the
generic case after sizing temporary arrays to the actual nonzero count. These
numbers validate the fast paths on this workload; they are not a universal
performance guarantee. Diagonal and triangular multiplication remain generic
because no comparably compelling specialized algorithm has been validated.


## Sparse factorization boundary

Sparse-aware multiplication is now implemented for sufficiently sparse right
operands under a finite, nonnegative numeric precondition. Sparse factorization
is deliberately not inferred from that result. A real sparse LU/rank
implementation needs a sparse elimination representation, pivoting rules,
fill-in accounting, rank tolerance semantics, and memory-growth benchmarks.
Routing sparse input through the existing dense LU implementation would not
provide sparse memory complexity and must not be marketed as sparse support.

Keep Phase 10.5 open until those algorithm and memory contracts are designed and
tested against matrices with controlled fill-in. The same evidence-first rule
applies to the advanced decompositions and matrix equations in Phase 10.7.

## Follow-up work

1. **Completed:** expand the benchmark to compare generic multiplication for
   dense, identity, zero, diagonal, triangular, and sparse operands.
2. **Completed:** record repeated samples, medians, timing ranges, and
   allocation counts/bytes; add finite and non-finite correctness tests.
3. **Completed:** implement exact identity/zero shortcuts and a sparse-right
   row-compressed path only under finite, nonnegative preconditions. The generic
   implementation remains the fallback.
4. **Deferred:** diagonal/triangular/symmetric/banded fast paths until a
   dedicated algorithm demonstrates a meaningful win. Sparse factorization and
   rank remain separate work because fill-in and output representation need
   their own design.

The audit and measurements are evidence for the current limited set of
specializations, not a blanket promise that every structured Storage kind has
an optimized multiplication algorithm.
