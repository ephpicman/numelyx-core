# Generic Optimizer Callback and Result Contract

Status: accepted v1 contract implemented by #119 and covered by native tests. See [current statistical foundation capabilities](statistical-foundation.md) for an integrated example.

## Current implementation

The internal C API in `src/statistics/numerus_optimizer.h` implements the optimizer, objective/gradient callbacks, owned result type, and distinct termination reasons described below. Matrix status codes are not used as a substitute for optimizer termination reasons. The PHP façade does not currently expose optimizer callbacks.

## Algorithm decision

The v1 starting point is **unconstrained BFGS with an Armijo backtracking line search**.

- Objective: minimize a caller-defined scalar function. Maximizing log likelihood is done by minimizing its negation; Numerus does not add model-specific likelihood APIs.
- Gradient: caller-supplied gradient when provided; otherwise central finite differences. No automatic differentiation is implied.
- Line search: start from a positive configured step, require Armijo sufficient decrease with fixed (c_1=10^{-4}), and halve the step up to a configured limit.
- Direction: use the inverse-BFGS approximation. If the computed direction is not a finite descent direction, reset the inverse approximation to identity and retry steepest descent. If that also fails, report numerical breakdown.
- Update: apply the inverse-BFGS update only when curvature (y^Ts) is sufficiently positive relative to (|y||s|); otherwise reset the approximation to identity rather than dividing by a near-zero curvature term.
- Convergence: the primary convergence criterion is the infinity norm of the gradient at/below the configured gradient tolerance. Reaching a small step without meeting that criterion is not convergence; report a distinct small-step termination.
- No bounds, constraints, Hessian callback, stochastic objective, parallel evaluation, or large-scale limited-memory mode in the first implementation. BFGS uses (O(n^2)) storage and is intended for moderate parameter dimensions. Revisit L-BFGS or constraints only with a concrete consumer.
- The objective is expected to be deterministic for the same parameters and context during one optimization run. A noisy/stochastic objective is outside this contract.

This algorithm is a defensible baseline for smooth unconstrained likelihood objectives, while keeping the first implementation and tests bounded. It does not claim to be a universal optimizer.

## Callback contract

The implemented callback types are:

```c
typedef bool (*numerus_objective_fn)(
    const double *parameters,
    size_t parameter_count,
    const void *context,
    double *value
);

typedef bool (*numerus_gradient_fn)(
    const double *parameters,
    size_t parameter_count,
    const void *context,
    double *gradient
);
```

- The parameter vector is borrowed, read-only and valid only for the callback duration. Callbacks must not retain its pointer or mutate it.
- `context` is borrowed and read-only through this pointer; its owner keeps it alive and logically unchanged for the entire optimization call. Mutation through another alias is not prevented by `const`.
- Callbacks may be invoked repeatedly at the same point and in an implementation-dependent order. They must be deterministic, reentrant if the caller runs independent optimizations concurrently, and free of dependence on hidden global optimizer state.
- Return `true` only when the callback successfully writes its complete output. Return `false` for an evaluation the callback cannot perform. A false return at the initial point is a callback failure; a false/non-finite trial objective during line search rejects that trial and triggers backtracking.
- Every objective value and every gradient component used by the algorithm must be finite. An initially invalid/non-finite objective or a non-finite gradient is not converted to NaN-based success.
- When `gradient == NULL`, central finite differences use (h_i = ho max(1, |x_i|)), with a documented default (ho = sqrt[3]{mathrm{DBL_EPSILON}}). Both perturbed objective evaluations must succeed and be finite. No silent one-sided fallback is used; if a required perturbation cannot be evaluated, report gradient-evaluation failure.
- No PHP callback bridge is implied. PHP exceptions, bailout/reentrancy and callback overhead need separate wrapper design under the API-boundary decision.

## Options

The implementation exposes an explicit options struct rather than hiding algorithmically important defaults:

- maximum accepted iterations, greater than zero;
- maximum line-search trials per iteration, greater than zero;
- gradient infinity-norm tolerance, finite and greater than zero;
- minimum line-search step, finite and greater than zero;
- parameter-step tolerance, finite and greater than zero, used only to report `STEP_TOO_SMALL` without calling it convergence;
- initial line-search step, finite and greater than zero;
- relative finite-difference step, finite and greater than zero (used only when no gradient callback is supplied).

The Armijo constant and backtracking ratio remain fixed at (10^{-4}) and (1/2) for v1; exposing every tuning constant is unnecessary until a consumer demonstrates the need. Options must be validated before invoking callbacks.

## Result and ownership

The result owns its final parameter vector and reports:

- parameter count and final accepted parameter vector;
- objective value plus an explicit validity flag;
- accepted iteration count;
- objective and gradient evaluation counts (finite-difference evaluations count as objective evaluations);
- termination reason.

The caller receives an opaque result handle and releases it with a matching destroy function. No pointer into the optimizer's temporary work arrays may escape.

A valid optimization request produces a result even when the algorithm stops without convergence (e.g. iteration limit or line-search failure). The result contains the last accepted parameters, never an unaccepted trial. If the initial objective cannot be evaluated, the result marks the objective value invalid and preserves the initial parameter vector as the last known candidate.

## Termination versus API failure

The API status reports whether the request could be initialized and executed safely. Invalid arguments, dimension/size overflow and allocation failure are API failures; the result termination reason reports the numerical outcome.

The result termination taxonomy distinguishes:

- `CONVERGED_GRADIENT`;
- `ITERATION_LIMIT`;
- `STEP_TOO_SMALL` (not convergence);
- `LINE_SEARCH_FAILED`;
- `OBJECTIVE_CALLBACK_FAILED` (initial/current required evaluation);
- `GRADIENT_CALLBACK_FAILED` (caller-supplied gradient callback returns false);
- `GRADIENT_EVALUATION_FAILED` (finite-difference objective perturbation cannot be evaluated);
- `NON_FINITE_OBJECTIVE`;
- `NON_FINITE_GRADIENT`;
- `NUMERICAL_BREAKDOWN`.

Allocation failure must remain a distinct API status, not a termination reason encoded in a result containing partially initialized memory. An invalid argument or allocation failure leaves the caller's result output NULL. On valid invocation, the result is fully initialized before it is published.

## Output preservation and dimensions

- Parameter count must be positive. All option values and initial parameters are validated before any result is published.
- All (n^2) workspace size calculations use checked arithmetic; overflow is detected before allocation.
- The initial parameter vector is copied; caller memory is never mutated.
- The final result vector is owned by the result handle and is independent of the input.
- If API initialization fails, the output handle remains NULL. Once a result is published, its fields are internally consistent even for non-converged termination.
- Callback output buffers are temporary. A callback returning false or writing non-finite values must not publish a partially computed gradient to the algorithm.

## Native test coverage for #119

1. Quadratic objective with known minimizer, both analytic gradient and finite-difference gradient.
2. Rosenbrock or another nonlinear smooth objective with a known minimum, with conservative tolerance and iteration limits.
3. Objective/gradient evaluation counters and result iteration fields.
4. Initial callback failure, initial NaN/infinity, non-finite gradient, finite-difference perturbation failure, and line-search trial failure.
5. Iteration limit and small-step termination are distinct from convergence.
6. Curvature degeneracy, non-descent direction reset, numerical breakdown, and output-preservation behavior.
7. Invalid options, zero dimension, size overflow where constructible, allocation failure, and destroy-NULL.
8. Ensure input parameters are unchanged and the returned parameter vector remains valid after workspaces are released.
9. Strict-warning native build; sanitizer/debug coverage where available. PHPT tests are required only if a PHP-visible wrapper is later approved.

## Scope decisions

- Accepted: unconstrained BFGS, Armijo backtracking, optional caller gradient, deterministic central finite-difference fallback, explicit result and termination taxonomy.
- Deferred: constraints/bounds, Hessian APIs, automatic differentiation, stochastic objectives, PHP callback bridge, L-BFGS, and model-specific estimators.
