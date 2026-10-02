# Sparse-Portfolio-Model

Cardinality-constrained portfolio models with solver bounds, independent numerical checks,
and an evaluation framework for long-horizon outcomes on a universe that includes delisted firms.

The repository implements mean–variance, perspective mean–variance, absolute-deviation,
CVaR, restricted index-tracking and third-central-moment formulations. Public benchmark
results test the implementation. **No CRSP empirical result has been produced.** The owner
will supply the licensed extract separately; the replication and empirical comparison phases
remain blocked until then.

## Background

A concentrated portfolio holds a small part of the equity market. Its performance depends on
the names selected, the weights assigned and the rebalancing rule. Index sampling seeks to
track a benchmark with fewer holdings; tracking error measures variation in returns relative
to that benchmark. Active Share measures the difference in portfolio weights from the index.
[Canakgoz and Beasley (2009)](https://doi.org/10.1016/j.ejor.2008.03.015),
[Cremers and Petajisto (2009)](https://doi.org/10.1093/rfs/hhp057).

Long-horizon stock wealth is positively skewed: a small group of firms accounts for the
market's net wealth creation above Treasury bills. Omitting delisted firms introduces
survivorship bias into an evaluation of stock selection. Rebalancing also changes the
distribution of compound returns, so a small rebalanced portfolio and a small buy-and-hold
portfolio answer different questions.
[Bessembinder (2018)](https://doi.org/10.1016/j.jfineco.2018.06.004),
[Farago and Hjalmarsson (2023)](https://doi.org/10.1093/rapstu/raac020).

This project asks whether optimizing a limited set of holdings changes the probability of
underperforming the value-weighted market or Treasury bills, and how much the answer depends
on the risk measure and the eligible universe. It makes no claim of priority over the
[existing literature](docs/SOURCES.md).

## Models and evidence

| Model | Criterion | Reference route | Independent check |
| --- | --- | --- | --- |
| F1 | mean–variance | SCIP | HiGHS cutting-plane bounds; continuous KKT frontier |
| F1p | perspective reformulation of F1 | SCIP | objective agreement with F1 |
| F2 | mean absolute deviation | HiGHS | deviation recomputed from returns |
| F3 | CVaR | HiGHS | sorted losses with a fractional tail atom |
| F4 | lexicographic regression tracking, zero costs | HiGHS | regression and tracking error from returns |
| F5 | mean, variance and third central moment | SCIP | exhaustive small-support simplex grid |

The [model](docs/MODEL.md) states the equations and restrictions. F4 follows the owner's
fully invested, zero-cost restriction and therefore has no identical target in the paper's
transaction-cost table. F5 is a utility in central moments; its default coefficients do not
make it a literal CRRA Taylor expansion about unit wealth.

[Validation](docs/VALIDATION.md) records the gates and checks;
[benchmark tables](docs/results/README.md) are generated from result bundles carrying the
code revision, input hashes, parameters, solver certificate and host. A limit is reported with
its achieved gap. [Open issues](docs/ISSUES.md) identifies unavailable empirical evidence.

## Reproduction

Use Python with the dependencies in `requirements.txt`. No commercial solver is required.

```text
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check .
python -m pip check
python run_bench.py --run --only g1-port1 --time 10
python run_bench.py --run --only g2-port1-lam0.25 --time 60 --missing
python run_bench.py
```

On the project owner's Windows machine, pass pytest an explicit `--basetemp` under
`Claude Func Folder`. Benchmark bundles require a committed, clean code tree. Markdown-only
changes do not dirty code identity. `--missing` reuses only admitted bundles with matching
code, parameters and inputs. `--jobs` controls parallel cases; each completed case is saved.
The manual GitHub Actions workflow runs one benchmark case per job and uploads its bundles.

OR-Library data are included with their MIT notice and SHA-256 pins. They are survivor
universes and support computational validation only. The French risk-free file is fetched
at runtime; CRSP data must remain outside the repository. [Data](docs/DATA.md) defines the
accepted schema and delisting treatment.

`run_boot.py` is the Bessembinder replication entry point; `run_oos.py` is the empirical
comparison entry point. Both stop without a licensed extract. Read the
[pre-registration](docs/PLAN.md) before either empirical run. Synthetic fixtures test
arithmetic, selection timing and numerical properties; they support no investment claim.

Code: [MIT](LICENSE). OR-Library data: [separate MIT notice](data/orlib/LICENSE).
