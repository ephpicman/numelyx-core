# Native test coverage

The `Native coverage` GitHub Actions workflow builds all native C translation
units under `src/` (excluding `src/numerus.c`, which is the PHP module entry
point) with GCC coverage instrumentation. It links the source units into one
instrumented executable and runs deterministic Storage, Matrix property,
Matrix adversarial, and statistical property suites.

The workflow publishes these artifacts:

- `summary.txt`: line and branch coverage summary;
- `coverage.xml`: machine-readable Cobertura report;
- `index.html` and per-source HTML pages: detailed file-level coverage.

The report intentionally includes source units that are linked but not reached
by the selected suites, so untested files and branches remain visible instead
of disappearing from the report. The selected suites are a baseline, not a
claim that every existing native test binary runs in the coverage job.

## How to use the report

1. Start with per-file line and branch coverage under `src/`; do not optimize
   only for an aggregate percentage.
2. Inspect uncovered branches against the relevant function contract and
   existing dedicated tests.
3. Add deterministic tests for a documented behavior or meaningful invariant.
4. Keep numerical assertions tolerance-aware and allocation-failure tests
   focused on cleanup and output-state guarantees.
5. Do not set a minimum percentage until the baseline and report limitations
   have been reviewed.

The PHP module entry point is excluded from this native report because its
behavior is validated by the PHPT and extension-load checks in the main test
workflow. The current PHPT suite is intentionally small while the numerical
API remains C-internal; this report does not replace future PHPT coverage when
a PHP-facing API is introduced.
