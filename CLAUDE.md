# Working contract

Read before changing anything. [README](README.md) states what the models are;
[Decisions](docs/DECISIONS.md) states why they are that way; [Validation](docs/VALIDATION.md)
states what has been checked. This file states the rules a change must not break.

Codex reads the same file through [AGENTS.md](AGENTS.md).

## Always on

| Rule | Reason |
| --- | --- |
| Report no solve outcome that was not read back from the result: status, primal value, dual bound, achieved gap. | A requested tolerance is not a result. HiGHS stops at `mip_rel_gap = 1e-4` by default. |
| Take a bound only from the solver's dual bound (`mip_dual_bound`; SCIP `getDualbound`), never from an incumbent. | An incumbent can understate the optimum. |
| Name a failed solve by the solver's status: infeasible, limit, unbounded, numerical. | Bad data and an empty feasible set both end in failure. |
| Re-solve with presolve off whenever infeasibility is reported. | HiGHS presolve declared feasible programs infeasible in Electricity-Clearing-and-Pricing. |
| Reject a certificate whose bounds cross by more than the tolerance. | A dual bound past the primal value is a defect. |
| Report a solve that hit its time limit as `limit`, with its gap. | It is not optimal. |
| Cite only what [Sources](docs/SOURCES.md) records as opened, and add the entry in the change that uses it. | A formulation is validated by the paper it implements, not by the solver that ran it. |
| No CRSP or other WRDS row-level data in the repository, its history or CI logs. | The data are licensed to the owner. |
| No future data in selection; the set T enters `pf/ev.py` only. | A test asserts it. |
| No empirical claim from survivor-only (OR-Library) or synthetic data. | Both universes exclude the firms the question is about. |
| Add a dependency only after measurement; no commercial solver as a requirement. | SciPy solves every LP and MILP; PySCIPOpt is admitted for MIQP and MINLP only. |
| State what was measured and what was not. | "Not measured" is a finding. |

## The feasible set is written once

`pf/rw.py` holds X(K) (`X`) and the return row (`R`). Every formulation reaches them through
`rw.pad` and `rw.stack` (HiGHS) or `rw.scip` (SCIP):

| Module | Model | Solver |
| --- | --- | --- |
| `pf/mv.py` | F1, F1p | SCIP |
| `pf/ecp.py` | F1 by extended cutting planes | HiGHS master |
| `pf/mad.py`, `pf/cv.py`, `pf/it.py` | F2, F3, F4 | HiGHS |
| `pf/sk.py` | F5 | SCIP |

`rw.ok` checks membership from the definition, not from the row matrix; the tests use it. The
independent checks (`mv.qp`, `mad.dev`, `cv.cvar`, `it.ols`, `sk.grid`) recompute from data and
the returned weights and never build rows.

## Before a change is finished

```text
python -m pytest -q --basetemp "<Claude Func Folder>\pytest-basetemp"
python -m ruff check .
python -m pip check
```

On this machine pytest's default temporary directory is denied; on CI no basetemp is needed.
Result bundles under `docs/results/` carry the revision that produced them; `run_bench.py`
refuses to write them from a dirty tree. Before any batch longer than 30 minutes in total, give
a time estimate and ask the owner; offload the tail to the workflow's `bench` job.
