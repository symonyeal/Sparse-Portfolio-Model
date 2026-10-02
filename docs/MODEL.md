# Model

## 1. Notation

| Symbol | Code | Meaning |
| --- | --- | --- |
| $N$, $i$ | `N` | number of assets; asset index |
| $S$, $s$ | `S` (scenarios) | estimation-window observations, each with probability $1/S$ |
| $r_s$ | `r[s]` | asset returns in scenario $s$ |
| $\mu$, $\Sigma$ | `mu`, `S` / `Sg` | mean and covariance of returns |
| $w$, $z$ | `w`, `z` | weights; inclusion indicators |
| $K$ | `K` | cardinality limit |
| $\varepsilon_i$, $\delta_i$ | `e`, `d` (`dl` where `d` is taken) | floor and ceiling on a held weight |
| $\lambda$ | `lam` | risk–return weight (Chang et al., 2000) |
| $\rho$ | `rho` | required mean return |
| $\beta$, $\alpha$, $u_s$ | `b`, `al`, `u` | CVaR level, VaR variable, tail excess |
| $y_s$ | `y` | centred portfolio return $(r_s-\mu)^\top w$ |
| $\gamma$, $\kappa$ | `g`, `k` | variance aversion, skewness preference |
| $Q$, $\pi$, $t_i$ | `Q`, `pi`, `t` | perspective split $\Sigma = Q + \operatorname{diag}(\pi)$ |
| $\mathcal{T}$ | `wc.top` | firms that together account for all net wealth creation |

## 2. The feasible set

$$\mathcal{X}(K) = \{(w,z) : \mathbf 1^\top w = 1,\ \varepsilon_i z_i \le w_i \le \delta_i z_i,\
\mathbf 1^\top z \ (\le \text{ or } =)\ K,\ z \in \{0,1\}^N\}$$

Chang et al. (2000), eqs. (10)–(13), impose $\mathbf 1^\top z = K$; the sense is the argument
`sn`. With $\varepsilon_i = 0$ and $=$, $z_i = 1$ marks an admitted name and $w_i$ may be $0$:
the indicator count is not the holding count (`rw.nz`). The return row $\mu^\top w \ (\ge
\text{ or } =)\ \rho$ is `rw.R`. Both are written once, in `pf/rw.py`.

## 3. Formulations

| ID | Model | Class | Source rows | Module |
| --- | --- | --- | --- | --- |
| F1 | $\min \lambda w^\top\Sigma w - (1-\lambda)\mu^\top w$ over $\mathcal X(K)$ | convex MIQP | Chang et al. (2000) eqs. (8)–(13), (16) | `mv.f1` |
| F1p | F1 with $\pi_i w_i^2 \to \pi_i t_i$, $w_i^2 \le t_i z_i$ | MISOCP | Günlük and Linderoth (2010) eqs. (17)–(23); Frangioni and Gentile (2006) | `mv.f1(p=True)` |
| F2 | $\min \frac1S\sum_s d_s$, $d_s \ge \pm y_s$, $\mu^\top w \ge \rho$ | MILP | Konno and Yamazaki (1991) eqs. (3.2)–(3.6) | `mad.mad` |
| F3 | $\min \alpha + \frac{1}{(1-\beta)S}\sum_s u_s$, $u_s \ge -r_s^\top w - \alpha$, $u \ge 0$, $\mu^\top w \ge \rho$ | MILP | Rockafellar and Uryasev (2000) eq. (17), rows (15)–(16) | `cv.cv` |
| F4 | lexicographic: $\min\lvert a^\top w\rvert$, then $\min\lvert b^\top w - 1\rvert$ | MILP ×2 | Canakgoz and Beasley (2009) eqs. (1)–(11), (16)–(26) | `it.it` |
| F5 | $\max \mu^\top w - \frac\gamma2 w^\top\Sigma w + \frac\kappa6\cdot\frac1S\sum_s y_s^3$ | nonconvex MINLP | Harvey et al. (2010) eqs. (3)–(6) | `sk.f5` |

**F1.** The quadratic enters through an epigraph $v \ge w^\top\Sigma w$; SCIP takes a linear
objective. The objective is multiplied by $1/\overline{\operatorname{diag}\Sigma}$ before the solve
and the certificate divided back: SCIP's tolerances are absolute below one, and at weekly-return
scale an unscaled epigraph left $9\times10^{-10}$ between the recomputed variance and the bound.

**F1p.** $\pi = \theta\,\lambda_{\min}(\Sigma)\mathbf 1$ with $\theta = 0.999$. Günlük and
Linderoth use $\pi = (\lambda_{\min} - 0.001)\mathbf 1$; at return scale $\lambda_{\min}
\approx 10^{-4}$ and that $\pi$ is negative.

**F2.** The source's epigraph form replaces the specification's split
$d^+_s - d^-_s = y_s$; the two are equivalent.

**F4.** $r_{it} = \log(V_{it}/V_{i,t-1})$, $R_t = \log(I_t/I_{t-1})$, and $(a_i, b_i)$ are OLS
coefficients of $r_i$ on $R$ with an intercept. Stage 2 holds $\lvert a^\top w - a^\ast\rvert \le
10^{-9}$, a project setting (the source says only that $\epsilon$ is small).

**F5.** $\frac1S\sum_s y_s^3$ is the sample third central moment of $r^\top w$. The default
$\kappa = \gamma(\gamma+1)$ is $U'''(1)/U'(1)$ for CRRA utility at unit wealth. Jondeau and
Rockinger (2006) expand about the expected wealth $1 + \mu^\top w$, so the coefficients agree
with theirs only to first order in $\mu^\top w$.

## 4. Restrictions relative to each source

| Model | Restriction |
| --- | --- |
| all | long only; $0 \le \varepsilon_i \le \delta_i \le 1$; fully invested |
| F1–F3 | no transaction costs |
| F1, F5 | $\Sigma$ by Ledoit and Wolf (2004) shrinkage (`mv.shrink`) out of sample; the sample covariance on OR-Library, whose files carry only moments |
| F2, F3 | the sample mean of the window in the return row |
| F4 | owner's restriction, 2 October 2026: the shared budget $\mathbf 1^\top w = 1$, hence $c = f_{is} = f_{ib} = 0$, $G = 0$, $w = xV/C$; the cost rows (3)–(6) and stage 3 are absent. Canakgoz and Beasley's weights sum to $(C - \sum G)/(C(1-c))$, generally not 1 |
| F5 | plug-in sample moments (Harvey et al.'s "parameter utility"), not their Bayesian predictive moments |

## 5. Wealth creation and the set $\mathcal T$

Bessembinder (2018), eq. (3), for a stock with beginning-of-month value $I_{t-1}$:

$$W_T - W_0\,FV_{0,T} = \sum_t I_{t-1}(R_t - r^f_t)\,FV_{t,T}, \qquad FV_{t,T} = \prod_{u=t+1}^{T}(1+r^f_u),$$

summed over the share classes of a PERMCO. $\mathcal T$ is the smallest set of top firms whose
sum reaches the net total (`wc.top`). Returns combine as $(1+\text{RET})(1+\text{DLRET})-1$.

## 6. Algorithms and independent checks

| Model | Reference route | Independent route |
| --- | --- | --- |
| F1 ($K=N$, $\varepsilon=0$, return row) | SCIP | active-set KKT solve `mv.qp`, certified by its residuals |
| F1 | SCIP | extended cutting planes, Westerlund and Pettersson (1995): HiGHS master over $v \ge 2w_k^\top\Sigma w - w_k^\top\Sigma w_k$; its dual bound is a lower bound |
| F2 | HiGHS | $\frac1S\sum_s\lvert y_s\rvert$ recomputed at $w$ |
| F3 | HiGHS | $\beta$-CVaR from the sorted losses, with the fractional atom, at $w$ |
| F4 | HiGHS | OLS of $r^\top w$ on $R$ reproduces $(a^\top w, b^\top w)$; tracking error from returns |
| F5 | SCIP spatial branch-and-bound | lattice of step $0.01$ on every support of size $\le K$, $N \le 12$, $K \le 3$ (`sk.grid`); SCIP's dual bound must not fall below its best value |
