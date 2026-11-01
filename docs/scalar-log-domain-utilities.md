# Scalar and Log-Domain Utility Selection

Status: selected v1 scope, implemented in #118 and covered by native tests.

## Decision summary

Select five composable scalar functions:

1. **Stable sigmoid** (sigma(x)=1/(1+e^{-x})).
2. **Stable logit** (log(p)-log(1-p)).
3. **Softplus** (log(1+e^x)).
4. **Log-sigmoid** (log(sigma(x))=-operatorname{softplus}(-x)).
5. **Log-sum-exp** (log(sum_i e^{x_i})) over a finite non-empty input array.

These support generic binary likelihoods, log-probability evaluation, and likelihood/posterior accumulation without requiring a distribution framework. The initial optimizer itself does not require a sigmoid or log-sum-exp; they are selected because they are broadly useful to multiple external statistical consumers and make stable likelihood composition possible.

Do **not** add Numerus wrappers for `log1p()` or `expm1()` in v1. They are standard C math functions available to internal C implementations; use them where appropriate rather than duplicating the standard library surface. Do not add log-gamma, digamma, erf-family, a full special-functions library, or a distribution-object hierarchy until a selected v1 algorithm demonstrates a concrete need.

## Contracts

All selected APIs are C-internal first, consistent with [the API boundary decision](statistical-api-boundary.md). Use a focused numerical status contract rather than leaking Matrix status codes. Required outputs are written only on success.

### Stable sigmoid

- Input: one finite `double`.
- Compute by sign-aware branches, e.g. for (x ge 0), (1/(1+exp(-x))); otherwise (exp(x)/(1+exp(x))).
- Result is finite and in ([0,1]). For very large finite magnitudes, rounding/saturation to exactly 0 or 1 is expected and is not an error.
- Reject NaN and either infinity under the selected strict finite-input policy.

### Stable logit

- Input must be finite and strictly inside (0 < p < 1). Endpoints are domain errors rather than implicit infinities.
- Use (log(p)-log1p(-p)), not (log(p/(1-p))), to avoid avoidable ratio overflow/underflow.
- The result must be finite for accepted inputs. If arithmetic unexpectedly produces a non-finite result, report numerical failure and preserve output.

### Softplus

- Input must be finite.
- Use a stable formulation such as (max(x,0)+log1p(exp(-|x|))).
- Avoid direct (log(1+exp(x))), which overflows for large positive finite inputs.
- For extreme positive input the result may round to (x); for extreme negative input it may round to zero. Both are expected finite floating-point results.

### Log-sigmoid

- Input must be finite.
- Use (-operatorname{softplus}(-x)) or an equivalent stable sign-aware formula; do not compute (log(operatorname{sigmoid}(x))), because sigmoid can round to zero for large negative inputs.
- Result is finite for every finite input, including very large positive/negative values.

### Log-sum-exp

- Input pointer must be non-NULL and count must be greater than zero. All input elements must be finite; NaN and infinities are rejected rather than given special extended-real semantics in v1.
- Compute (m+log(sum_i exp(x_i-m))), where (m=max_i x_i). Never exponentiate the original unshifted values.
- Do not allocate memory. One pass finds the maximum and a second pass accumulates shifted exponentials.
- If the final addition is non-finite for an extreme input set, report numerical failure and leave the output unchanged.
- A singleton input returns that value (subject to finite validation). Permutation of inputs should not materially change the result within documented floating-point tolerance.
- Empty input is an invalid argument, not (-infty).

## Status and output semantics

The implemented functions use the focused `numerus_numeric_status` declared in `src/core/numerus_numeric.h`, with distinct success, invalid-argument, domain-error, non-finite-input, and numerical-failure outcomes. They do not reuse `numerus_matrix_status`.

- Required output pointers are validated before reading input.
- Output values remain unchanged on every non-success status.
- No function uses NaN or infinity as a hidden error sentinel.
- These functions allocate no memory; there is no allocation-failure path for their scalar computations.
- Do not promise correctly rounded results beyond the guarantees of the supported C math library.

## Consumer mapping

| Function | Concrete consumer |
|---|---|
| sigmoid | Generic binary probability/objective evaluation; reusable outside a specific estimator |
| logit | Convert an open-interval probability to an unconstrained scalar parameter in userland models |
| softplus | Stable positive parameter transforms and log-domain objectives |
| log-sigmoid | Stable Bernoulli/logistic log-likelihood composition without taking `log(sigmoid(x))` |
| log-sum-exp | Stable normalization and accumulation of multiple log weights/log probabilities |

## Native test coverage for #118

- Sigmoid at zero, large positive/negative finite inputs, monotonicity, finite range and expected saturation.
- Logit at (p=0.5), values close to but strictly inside both endpoints, and rejection of (p le 0), (p ge 1), NaN and infinities.
- Softplus/log-sigmoid at zero, (pm 1), and large finite magnitudes near the representable range; ensure no avoidable overflow/underflow and finite outputs.
- Log-sum-exp for a singleton, equal values, values far below zero, large similar positive values, permutation, empty input, NULL input, non-finite elements, and output preservation.
- Cross-check moderate inputs against direct high-precision/reference calculations and use tolerance-based comparisons, not bitwise assumptions about libm.
- Strict-warning native tests; PHPT only if a later API-boundary decision explicitly exposes these helpers to PHP.

## Deferred

- Numerus wrappers for `log1p`/`expm1`; call the C standard functions internally.
- Log-gamma, digamma, erf/erfc, inverse special functions, and broad special-function support.
- General probability distributions, distribution objects and automatic differentiation.
- Extended-real NaN/infinity semantics for log-sum-exp.
