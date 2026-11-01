# BLAS and LAPACK backend

Numelyx Core uses BLAS and LAPACK as required numerical dependencies. The
public Matrix API and Storage representations remain owned by Numelyx Core;
vendor-specific routines are confined to the internal numerical layer.

## Required interfaces

- BLAS through the CBLAS C interface.
- LAPACK through the LAPACKE C interface.
- CMake locates BLAS and LAPACK. On non-Windows platforms it also locates
  `cblas.h`, `lapacke.h`, and the LAPACKE library.
- Ubuntu development packages: `libopenblas-dev` and `liblapacke-dev`.
- Windows CI uses the conda-forge BLAS selector, which selects Intel oneMKL.
  The development package supplies `mkl.h`; CBLAS and LAPACKE declarations are
  exposed through that header and linked through the selected MKL BLAS/LAPACK
  libraries. A separate generic `lapacke.lib` is not required.
- CI exercises both dependency setups.

## Current dispatch

| Operation | Dispatch |
| --- | --- |
| Dense, finite matrix multiplication | CBLAS `cblas_dgemm` (row-major) |
| Exact diagonal × finite nonnegative matrix | Numelyx Core row/column scaling kernel |
| Identity, zero, and sparse multiplication shortcuts | Existing Numelyx Core fast paths, guarded for IEEE-754 equivalence |
| Nonsingular finite square LU factorization | LAPACKE `LAPACKE_dgetrf` (row-major) |
| Cholesky factorization | LAPACKE `LAPACKE_dpotrf` (row-major), with Numelyx Core symmetry and scale-relative pivot checks |
| Unpivoted reduced QR | LAPACKE `LAPACKE_dgeqrf` + `LAPACKE_dorgqr` (row-major) |
| Reduced SVD | LAPACKE `LAPACKE_dgesdd` (row-major); one-sided Jacobi remains the non-convergence fallback |
| Symmetric eigen decomposition | LAPACKE `LAPACKE_dsyevd` (row-major); cyclic Jacobi remains the non-convergence fallback |
| Triangular solve | LAPACKE `LAPACKE_dtrtrs` (row-major), after exact structural validation |
| General square solve | Structure-aware dispatch: exact triangular solve; otherwise Cholesky for symmetric positive-definite candidates; otherwise pivoted LU |
| Singular LU inputs, non-finite data, or dimensions beyond the conservative 32-bit LAPACK boundary | Existing native LU path (preserves exact-pivot rank semantics) |
| Pivoted QR (rank-revealing) | Existing native implementation; its permutation/rank contract remains separate |
| Non-finite dense multiplication and non-dense multiplication | Existing native multiplication path |

The solve dispatcher uses cached logical Matrix flags only to identify candidate
structures. The triangular kernel validates the zero half exactly before
solving, so tolerance-based flag classification cannot discard small,
nonzero coefficients. The Cholesky routine independently checks finiteness,
symmetry, and scale-relative positive pivots. A non-positive-definite or
nonsymmetric candidate falls through to the general LU path; allocation,
read, and other operational failures are propagated.

Diagonal multiplication is restricted to exact off-diagonal zeros and finite,
nonnegative operands. This restriction preserves the generic kernel's
IEEE-754 behavior for omitted zero-times-value terms and signed zero. Other
diagonal products use the generic path.

Dense inputs are gathered through the Matrix accessor before calling the
backend. This preserves the opacity of Storage and supports logical dense
views without assuming that every Matrix is a contiguous root allocation.
CBLAS/LAPACKE integer dimensions are range-checked before conversion.

## Numerical and ownership contracts

- Backend calls operate on temporary row-major buffers; they never receive
  pointers into immutable shared Storage.
- Output Matrix objects are published only after the backend operation
  completes successfully.
- Native implementations remain available for the documented fallback cases
  and as differential-test references during the migration.
- BLAS/LAPACK may use different reduction orders from the native kernels.
  Floating-point comparisons must therefore use numerical tolerances except
  where exact behavior is part of a structural fast-path contract.
- A backend implementation may use internal worker threads. Applications
  combining Numelyx-level parallelism with multithreaded BLAS should configure
  vendor thread counts to avoid oversubscription.

Numerical rank estimation now detects exact diagonal logical matrices during its existing finite-value scan and counts diagonal pivots using the unchanged global scale-relative threshold. This avoids allocating/eliminating a second cubic work phase for diagonal inputs; the detection scan remains O(m*n), and transformed views are evaluated through logical Matrix accessors rather than trusting their parent Storage tag.

## Remaining optimization work

The structure-aware solve dispatcher currently covers exact triangular
systems and symmetric positive-definite systems. Dedicated sparse and banded
solvers, block-diagonal decomposition, and benchmark-derived size thresholds
for native-versus-BLAS dispatch remain future work. They should be added only
with explicit numerical contracts and differential tests; Storage kinds alone
must not be treated as proof of the logical structure of a transformed view.
