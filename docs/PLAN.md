# Empirical pre-registration

Recorded on 2 October 2026, before any CRSP out-of-sample run. P5 and P6 are blocked while
the licensed extract is unavailable. Public OR-Library computations are model validation,
not out-of-sample finance results. Any amendment below must precede the affected run and
appear in the deviation register. The empirical bundles must include this file's hash.

## Estimands and comparison set

The primary estimand is the fraction of monthly start dates at which a strategy's terminal
buy-and-hold wealth is below B1, and separately below B5. Differences between strategies use
paired start dates. Ties do not count as underperformance. An unavailable or failed selection
is recorded; it must not disappear selectively from one strategy's comparison sample.

| Item | Setting |
| --- | --- |
| Replication sample | July 1926–December 2016; SHRCD 10, 11 and 12; include supplied delisting returns |
| RQ1 sizes | 1, 5, 25, 50, 100, as Bessembinder Table 4 |
| RQ1 draws | 20,000 replications per size; monthly redraw, beginning-of-month value weights; seed 0 |
| RQ1 horizons | one year, ten years, full available replication span |
| RQ1 finite-horizon starts | uniform feasible monthly start per replication; a project convention pending resolution of the source's unstated start-window rule |
| G3 tolerance | absolute difference at most 0.02 in each probability of beating T-bills and the market in Table 4; also report Monte Carlo standard error and all signed differences |
| Research sizes K | 5, 10, 25, 50, 65, 100 |
| Estimation windows M | 60 and 120 months |
| Minimum history m | M/2 and M; every additional complete-case restriction is recorded separately |
| Universe caps Nc | candidate caps 100 and 200; freeze after a timed CRSP solver probe, before empirical comparisons |
| Holding H | 12 months, 120 months, and to the sample end; monthly starts |
| Holding convention | buy-and-hold primary; annual rebalancing is a separate variant |
| Baselines | B1 full value-weighted market; B2 1/N eligible; B3 cap-weighted top K; B4 equal-weighted random K separately on eligible and full universes; B5 T-bill |
| Selected strategies | F1, F2, F3, F5 at equal K; F1p is a computational equivalent of F1; F4 is a separate tracking validation |
| F1 | lambda 0.5; long only, maximum K, zero floor and ceiling 1 |
| F2, F3 | required mean return equal to the median eligible sample mean; F3 beta 0.95 |
| F5 | gamma 3, kappa 12; central-moment utility, not a literal CRRA Taylor expansion |
| Covariance | Ledoit–Wolf 2004 constant-correlation shrinkage; sample covariance sensitivity |
| Delisting sensitivity | supplied returns for RQ1; performance-delisting fill and no-fill runs for RQ2–RQ4 |
| Composition | full universe versus deliberately survivor-only universe; eligible versus full-universe B4; disclose every history filter |
| Secondary metrics | wealth-creator hit rate by PERMCO, terminal-wealth skewness and CVaR, Sharpe ratio and turnover |

The replication's monthly redraw and value weights are distinct from B4's equal-weighted
buy-and-hold extension. The finite-horizon convention has not been verified against the
author's original simulation code. G3 cannot be described as an exact protocol replication
until that ambiguity is resolved. No empirical result may be calibrated to the unpublished
manuscript.

## Information and inference

Selections at month t receive only observations dated through t. Eligibility is based on a
valid current price, history within the estimation window and current market capitalization.
An explicit survivor-only experiment may use future survival to define its biased universe;
it is never the primary selection universe. Future wealth-creator labels are computed after
selection and enter evaluation only. Delisting proceeds earn T-bills after the delisting
return. Missing returns for continuing holdings are errors, not zero returns.

Probability differences use paired stationary-bootstrap resamples over start dates, with
2,000 resamples, seed 0 and two-sided 95% intervals. Mean block lengths of 12 and 120 months
are the primary choices for their corresponding holding windows; report sensitivity to
half and twice the block length. To-end horizons have varying lengths and are reported
separately. Report the number of comparable starts and effective horizon coverage.

Sharpe comparisons follow the Ledoit–Wolf framework on paired excess-return series. State
whether an interval is HAC asymptotic or studentized-bootstrap, together with its kernel,
bandwidth or block length. An asymptotic HAC interval is not the paper's preferred
studentized-bootstrap procedure. No significance claim is made without an interval.

## Gates and unresolved inputs

G3 requires the exact source extract/vintage, complete date coverage, a stated benchmark
construction and a table of all target differences. CRSP access is confirmed, but no extract
has been supplied. Before P6, freeze Nc, solver budgets and the final common evaluation
sample; establish G3 or record and approve a specific deviation. A bootstrap convention or
data vintage that fails the gate must be investigated rather than tuned to its target.

## Deviations from the supplied specification

| Decision | Reason |
| --- | --- |
| Distinguish the original monthly-redraw bootstrap from equal-weighted B4. | Bessembinder's Table 4 uses the former. |
| F4 omits transaction costs; confirmed by the owner on 2 October 2026. | The published cost-adjusted budget conflicts with the chosen common fully invested budget. |
| F5 on return scenarios from indtrack1 for the public probe. | port1 provides only first and second moments, which cannot identify a third moment. |
| Keep empirical phases blocked until the extract is supplied. | Synthetic or survivor-only universes cannot answer RQ1–RQ4. |
