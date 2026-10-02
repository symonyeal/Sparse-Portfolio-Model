# Validation

## Formulation

All optimization routes reach `pf/rw.py` for the budget, weight bounds and cardinality row.
The indicator count is distinguished from the number of positive weights when the floor is
zero. F4 retains the source's regression approximation with the owner-approved zero-cost
budget; the original transaction-cost results are not targets for this restriction.

## Algorithms

F1/F1p use SCIP. F1 also has independent extended cutting-plane bounds from HiGHS MILP
masters. F2/F3 recompute their risk measure directly from the returned weights. F4 recomputes
OLS coefficients and tracking error. F5 compares its maximizing dual bound with every point
of a small-support simplex grid. Grid points are feasible lower bounds, not certificates of
global optimality.

## Gates

| Gate | Required evidence |
| --- | --- |
| G1 | OR-Library unconstrained frontier at its printed coordinate precision; also report raw differences and KKT residuals |
| G2 | F1 objective agreement within 1e-6 relative wherever SCIP and ECP both certify; otherwise report the valid ECP bracket |
| F1p | equality with F1's integer optimum; root-bound comparison reported independently |
| F2/F3 | returned primal agrees with independently recomputed absolute deviation or fractional-tail CVaR |
| F4 restricted | both lexicographic stages certified, source regression identity checked, tracking error recomputed |
| F5 | SCIP upper bound does not lie below the best small-support grid value |
| G3 | licensed CRSP replication against Bessembinder Table 4 under PLAN.md; currently blocked |

The source frontier prints both coordinates with finite precision. A variance difference
at the printed mean is reported separately from agreement with the rectangle defined by
rounding both coordinates. A test that accepts a larger unexplained tolerance cannot
establish the printed-precision gate.

## Computation

The authoritative measured tables are [generated from admitted bundles](results/README.md).
Each bundle records code, input hashes, parameters and host. Tables must reject stale
bundles; a clean source digest alone cannot establish that the inputs still match.

The final local verification record and benchmark coverage are added after the fresh run.
No CRSP or out-of-sample empirical run has been performed.
