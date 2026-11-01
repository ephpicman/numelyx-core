# GLS Whitening and Log-Determinant Contracts

Status: design contract for issues #113 and #114; this document does not claim either missing API is implemented.

## Existing source and tests

- `numerus_matrix_cholesky()` in `numerus_matrix_cholesky.c` returns a dense lower-triangular factor for a symmetric positive-definite input. It rejects non-finite entries, checks symmetry with a scale-aware tolerance, and rejects pivots at/below its scale-aware threshold.
- `numerus_matrix_solve()` in `numerus_matrix_solve.c` solves square nonsingular systems with multiple right-hand sides using LU with partial pivoting. It is not a triangular-solve API and does not itself provide covariance whitening.
- `numerus_matrix_least_squares()` in `numerus_matrix_least_squares.c` uses pivoted QR and requires full column rank.
- `numerus_matrix_determinant()` and LU are present, but no stable log-determinant API was found.
- Relevant tests: `tests/matrix_cholesky_test.c`, `tests/matrix_solve_test.c`, `tests/matrix_least_squares_test.c`, `tests/matrix_determinant_test.c`.

## Contract A — triangular solve

Proposed C API:

```c
numerus_matrix_status numerus_matrix_solve_triangular(
    const numerus_matrix *triangular,
    const numerus_matrix *right_hand_side,
    bool lower,
    bool transpose,
    numerus_matrix **solution
);
```

The signature is a design proposal; implementation may adjust naming only if the same semantics remain explicit.

### Semantics

- Solve (T X = B) when `transpose == false`; solve (T^T X = B) when true. `lower` describes the stored input `T`, not its transposed orientation.
- `T` must be square, and `B` must have the same row count. Any positive number of RHS columns is supported.
- The stored matrix must be triangular in the declared orientation. Entries in the structurally zero half must be exactly zero; do not silently discard small nonzero entries. Factorization outputs already use exact structural zeros.
- Every diagonal entry must be finite and nonzero. This is a triangular algebraic solve, not a positive-definiteness check; callers requiring SPD must validate/factorize first.
- All referenced input and intermediate values must be finite. If an arithmetic operation overflows or produces NaN/infinity, return `NUMERUS_MATRIX_NON_FINITE`.
- Shape mismatch returns `NUMERUS_MATRIX_DIMENSION_MISMATCH`; non-square input returns `NUMERUS_MATRIX_NOT_SQUARE`; malformed triangle/zero diagonal returns a documented argument or singularity status consistently. Prefer `NUMERUS_MATRIX_SINGULAR` for a zero diagonal.
- Checked size arithmetic is mandatory. Allocation failure returns `NUMERUS_MATRIX_OUT_OF_MEMORY`; size overflow returns `NUMERUS_MATRIX_OVERFLOW`.
- `solution` is required and is set to NULL before any operation that can fail. The result is newly owned, dense, and independent of the inputs. Inputs are never mutated.
- Complexity should be (O(n^2 r)) for an (n 	imes n) factor and (r) right-hand sides; do not form an inverse.

### Why one API

A single orientation/transpose contract avoids separate lower/upper and transposed variants while covering the immediate reusable operations: whitening with (L^{-1}), back-substitution with (R), and Gaussian density quadratic forms. Unit-diagonal behavior is deliberately excluded from the first contract because the immediate GLS/Gaussian consumers use Cholesky factors with explicit nonzero diagonals. Add it only if a concrete consumer requires it.

## Contract B — general signed log determinant

Proposed C API:

```c
numerus_matrix_status numerus_matrix_log_determinant(
    const numerus_matrix *matrix,
    int *sign,
    double *log_abs_determinant
);
```

### Semantics

- Input must be a finite square matrix.
- For a nonsingular matrix, return `NUMERUS_MATRIX_SUCCESS`, set `sign` to exactly `+1` or `-1`, and set `log_abs_determinant` to (log(|det(A)|)).
- Compute from a factorization, e.g. pivoted LU: permutation parity times the signs of the diagonal pivots determines the determinant sign; sum (log(|U_{ii}|)) for the log magnitude. Do not multiply pivots and then take a logarithm; that reintroduces avoidable overflow/underflow.
- A singular matrix returns `NUMERUS_MATRIX_SINGULAR`; both output parameters remain unchanged. This API reports a finite log magnitude only for nonsingular matrices and does not encode failure as (-infty).
- A zero-sized matrix is not constructible through the current Matrix contract; the function therefore receives positive dimensions only.
- Invalid pointers return `NUMERUS_MATRIX_INVALID_ARGUMENT`; non-square input returns `NUMERUS_MATRIX_NOT_SQUARE`; non-finite input/intermediate returns `NUMERUS_MATRIX_NON_FINITE`; checked-size overflow and allocation failure retain their dedicated statuses.
- Both outputs are required. They are written only after the computation succeeds. The matrix is not mutated.
- The function is a signed log-absolute-determinant, not simply (log(det(A))). For a covariance matrix accepted by Cholesky, the sign must be positive. The API must not claim that an LU factorization proves positive definiteness or supplies a condition estimate.

### Why not an SPD-only function

A signed general log determinant is reusable beyond covariance matrices and explicitly handles negative determinant sign without producing NaN from `log(det(A))`. Gaussian density still requires a covariance contract: use Cholesky/SPD validation and reject invalid covariance before using the log determinant. The general log determinant must not be used as a substitute for covariance validation.

## Contract C — GLS covariance whitening

For SPD covariance (Sigma), obtain (L) from (Sigma = LL^T), then solve:

[
X_w = L^{-1}X,qquad y_w = L^{-1}y
]

and pass ((X_w,y_w)) to ordinary least squares. This is equivalent to minimizing ((y-Xeta)^TSigma^{-1}(y-Xeta)) without constructing (Sigma^{-1}).

- The covariance must be square, finite, symmetric within the existing Cholesky tolerance, and positive definite under that factorization's pivot policy.
- Covariance dimensions must match both the rows of (X) and the rows of (y); (y) may contain multiple RHS columns.
- Factorization or solve failure aborts the workflow. Do not return partial whitened outputs.
- Whitening must use the same Cholesky factor for both (X) and (y); independently factorizing or using inconsistent tolerances is not acceptable.
- The caller owns each returned Matrix and must destroy intermediate matrices on every success/failure path.
- Zero/negative weights belong to the separate diagonal WLS API and are not a substitute for a general covariance matrix.

## Small reference cases for tests

These are acceptance fixtures for #113/#114; implementation PRs must add executable native tests.

1. **Lower solve, multiple RHS:** (L=egin{bmatrix}2&0\3&1end{bmatrix}), (B=egin{bmatrix}2&4\5&7end{bmatrix}). Expected (X=egin{bmatrix}1&2\2&1end{bmatrix}).
2. **Transpose solve:** with the same (L), solve (L^T X=B) for at least one RHS and verify the residual (L^T X-B).
3. **Triangular failure cases:** non-square factor, mismatched RHS rows, nonzero value in the forbidden triangle, zero diagonal, NaN/infinity, allocation failure, and output remains NULL on failure.
4. **Positive log determinant:** (A=operatorname{diag}(2,3)) gives sign (+1), log magnitude (log 6).
5. **Negative log determinant:** (A=operatorname{diag}(-2,3)) gives sign (-1), log magnitude (log 6).
6. **Scale safety:** (A=operatorname{diag}(10^{200},10^{-200})) gives sign (+1), log magnitude approximately (0), without multiplying the pivots first.
7. **Singular log determinant:** (A=operatorname{diag}(1,0)) returns `NUMERUS_MATRIX_SINGULAR` and leaves sentinel output values unchanged.
8. **GLS reference:** choose small SPD (Sigma), full-rank (X), and (y); compare the whitened solution to an independently calculated small reference using (X^TSigma^{-1}X) only in the test/reference calculation. Production implementation must not explicitly form the inverse.

## Issue disposition

- **#113 is confirmed as an implementation gap**. Implement only the triangular solve/whitening path specified above; do not add redundant general solve APIs.
- **#114 is confirmed as an implementation gap**. Implement signed log-absolute-determinant from factorization, with the status/output semantics above.
- **#122 remains necessary** for composition-level OLS/WLS/GLS reference tests; it is not a model-fitting API.
