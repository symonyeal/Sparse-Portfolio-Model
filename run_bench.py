"""Benchmark cases on public data (D1, D2): gates G1, G2, the Chang comparison, RQ5, F2-F5 checks.

    python run_bench.py                 tables only, from the stored bundles (docs/results/README.md)
    python run_bench.py --run           run every case, then the tables
        --missing                       only cases with no admitted bundle (stale ones rerun)
        --only PREFIX                   only cases whose key starts with PREFIX (repeatable)
        --time S                        per-solve time limit in seconds (default 60)
        --jobs N                        N cases at a time (times are then under shared load)

Each case writes docs/results/bench/<key>.json as it finishes, so a stopped run loses only the
cases in flight. Cases refuse to write into docs/results from a dirty tree.
"""

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np

from pf import bd, cv, dat, ecp, it, mad, mv, rw, sk

# Legend
# key, kd, pa : case key, kind and parameters; r : result record; ip : input hashes
# lam : Chang's weight; K, e : cardinality and floor; tl : per-solve time limit (s)

OUT = Path("docs/results/bench")
LAM = [(e - 1) / 49 for e in range(1, 51)]          # Chang et al. (2000): E = 50
CB = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 40, 7: 90, 8: 70}   # Canakgoz-Beasley Table 1 K
CH = {"GA": (1.2181, 1.0974), "TS": (1.2181, 1.1217), "SA": (1.2181, 1.0957)}   # Table 4, V


def cases():
    c = [(f"g1-port{k}", "g1", {"k": k}) for k in range(1, 6)]
    c += [("chang-port1", "chang", {"k": 1, "K": 10, "e": 0.01})]
    c += [(f"g2-port1-lam{l:g}", "g2", {"k": 1, "K": 10, "e": 0.01, "lam": l})
          for l in (0.0, 0.25, 0.5, 0.75, 1.0)]
    c += [(f"rq5-port{k}-K{K}-{f}", "rq5", {"k": k, "K": K, "e": 0.01, "lam": 0.5, "p": f == "f1p"})
          for k in range(1, 6) for K in (5, 10) for f in ("f1", "f1p")]
    c += [(f"f2-indtrack1-K{K}", "f2", {"k": 1, "K": K}) for K in (5, 10)]
    c += [(f"f3-indtrack1-K{K}", "f3", {"k": 1, "K": K, "b": 0.95}) for K in (5, 10)]
    c += [(f"f4-indtrack{k}", "f4", {"k": k, "K": CB[k]}) for k in range(1, 9)]
    c += [("f5-indtrack1-grid", "f5g", {"k": 1, "S": 36, "N": 6, "K": 2, "g": 3.0})]
    c += [(f"f5-indtrack1-K{K}", "f5", {"k": 1, "K": K, "g": 3.0}) for K in (3, 10)]
    return c


def crt(c):
    return {k: c.get(k) for k in ("st", "f", "bd", "g", "nd", "t", "rt", "it")}


def run(key, kd, pa, tl):
    p = dict(pa)
    if kd in ("g1", "chang", "g2", "rq5"):
        d = dat.port(p["k"])
        ip = {d["name"]: d["sha"]}
        mu, S = d["mu"], d["S"]
    else:
        d = dat.indtrack(p["k"], tail=p["k"] in (7, 8))
        ip = {d["name"]: d["sha"]}
        r, Ri = dat.lr(d["V"]).T, dat.lr(d["I"])
    if kd == "g1":
        e = dat.portef(p["k"])
        ip[e["name"]] = e["sha"]
        F0, dv, kt = np.ones(mu.size), [], []
        for x, v in e["ef"]:
            q = mv.qp(mu, S, x, F0, it=5 * mu.size)
            F0 = q["w"] + (q["w"] > 0)
            dv.append(q["f"] - v)
            kt.append(max(q["kt"].values()))
        dv = np.array(dv)
        return {"n": dv.size, "max_abs": float(np.abs(dv).max()), "max_pos": float(dv.max()),
                "min_neg": float(dv.min()), "half_unit": float(np.mean(np.abs(dv) <= 5e-11)),
                "max_rel": float(np.max(np.abs(dv) / e["ef"][:, 1])), "kkt": max(kt)}, ip
    if kd == "chang":
        e = dat.portef(p["k"])
        ip[e["name"]] = e["sha"]
        pts = []
        for l in LAM:
            s = mv.f1(mu, S, p["K"], l, e=p["e"], sn="==", tl=tl)
            w = s["w"]
            pts.append({"lam": l, "st": s["c"]["st"], "g": s["c"]["g"], "y": float(mu @ w),
                        "x": float(np.sqrt(w @ S @ w)),
                        "pd": float(mv.pdev(e["ef"], np.sqrt(w @ S @ w), mu @ w))})
        v = np.array([q["pd"] for q in pts])
        v = v[np.isfinite(v)]
        return {"pts": pts, "median": float(np.median(v)), "mean": float(np.mean(v)),
                "n": int(v.size), "all_opt": all(q["st"] == "opt" for q in pts),
                "chang_V": CH}, ip
    if kd == "g2":
        s = mv.f1(mu, S, p["K"], p["lam"], e=p["e"], sn="==", tl=tl)
        q = ecp.ecp(mu, S, p["K"], p["lam"], e=p["e"], sn="==", tl=tl)
        both = s["c"]["st"] == q["c"]["st"] == "opt"
        rel = abs(s["f"] - q["f"]) / max(abs(s["f"]), 1e-12)
        inb = q["c"]["bd"] - 1e-9 * abs(s["f"]) <= s["f"] <= q["c"]["f"] + 1e-9 * abs(s["f"])
        return {"scip": crt(s["c"]), "ecp": crt(q["c"]), "scip_f": s["f"], "ecp_f": q["f"],
                "rel": rel, "both_opt": both, "pass": bool(rel <= 1e-6) if both else None,
                "in_bracket": bool(inb),
                "ok": rw.ok(s["w"], s["z"], p["K"], p["e"], 1.0, "==") == []}, ip
    if kd == "rq5":
        s = mv.f1(mu, S, p["K"], p["lam"], e=p["e"], sn="==", p=p["p"], tl=tl)
        return {"c": crt(s["c"]), "f": s["f"], "N": int(mu.size)}, ip
    if kd == "f2":
        s = mad.mad(r, p["K"], tl=tl)
        return {"c": crt(s["c"]), "recomputed": s["f"],
                "diff": abs(s["c"]["f"] - s["f"]), "ok": rw.ok(s["w"], s["z"], p["K"]) == []}, ip
    if kd == "f3":
        s = cv.cv(r, p["K"], b=p["b"], tl=tl)
        return {"c": crt(s["c"]), "sorted": s["f"], "diff": abs(s["c"]["f"] - s["f"]),
                "ok": rw.ok(s["w"], s["z"], p["K"]) == []}, ip
    if kd == "f4":
        tr, ts = slice(0, 145), slice(145, None)       # Canakgoz-Beasley sec. 4.1
        s = it.it(r[tr], Ri[tr], p["K"], tl=tl)
        o = {"c1": crt(s["c1"]), "c2": crt(s["c2"]) if s["c2"] else None,
             "N": int(d["N"]), "drop": int(d["drop"])}
        if s.get("a") is not None:
            a2, b2 = it.ols(r[ts] @ s["w"], Ri[ts])
            o |= {"a_in": s["a"], "b_in": s["b"], "a_out": float(a2), "b_out": float(b2),
                  "te_in": it.te(r[tr], Ri[tr], s["w"]), "te_out": it.te(r[ts], Ri[ts], s["w"]),
                  "names": rw.nz(s["w"])}
        return o, ip
    if kd == "f5g":
        x = r[:p["S"], :p["N"]]
        s = sk.f5(x, p["K"], p["g"], tl=tl)
        gb, _ = sk.grid(x, p["K"], p["g"], h=0.01)
        return {"c": crt(s["c"]), "f": s["f"], "grid": gb,
                "bd_ge_grid": bool(s["c"]["bd"] >= gb - 1e-12)}, ip
    if kd == "f5":
        s = sk.f5(r, p["K"], p["g"], tl=tl)
        return {"c": crt(s["c"]), "f": s["f"], "N": int(r.shape[1]), "S": int(r.shape[0])}, ip
    raise ValueError(kd)


def one(a):
    key, kd, pa, tl, code = a
    t0 = time.perf_counter()
    r, ip = run(key, kd, pa, tl)
    r["wall"] = time.perf_counter() - t0
    b = bd.bundle(key, pa | {"tl": tl}, r, ip, sys.argv, code)
    bd.write(OUT, b)
    return key, r["wall"]


def table():
    B = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))}
    L = ["# Benchmark results", "",
         "Generated by `python run_bench.py` from the bundles in `docs/results/bench/`. "
         "Do not edit by hand.", ""]

    def sec(t, hd, rows):
        if rows:
            L.extend([f"## {t}", "", "| " + " | ".join(hd) + " |",
                      "|" + "---|" * len(hd)] + ["| " + " | ".join(x) + " |" for x in rows] + [""])

    f = lambda v, s=".3e": "n/a" if v is None else format(v, s)
    sec("G1: exact frontier against OR-Library portef", ["case", "points", "max abs diff",
        "range of diff", "max rel diff", "within 5e-11", "max KKT residual"],
        [[k, str(b["n"]), f(b["max_abs"]), f"{b['min_neg']:+.2e} .. {b['max_pos']:+.2e}",
          f(b["max_rel"]), f"{100 * b['half_unit']:.2f}%", f(b["kkt"], ".1e")]
         for k, b in B.items() if k.startswith("g1-")])
    if "chang-port1" in B:
        b = B["chang-port1"]
        sec("Exact CCEF against Chang et al. (2000) Table 4, Hang Seng (port1), K = 10",
            ["route", "median % error", "mean % error", "points"],
            [["exact (SCIP, 50 lambda)", f"{b['median']:.4f}", f"{b['mean']:.4f}",
              f"{b['n']} (all opt: {b['all_opt']})"]]
            + [[f"Chang {h} heuristic, V", f"{m:.4f}", f"{a:.4f}", "50"]
               for h, (m, a) in b["chang_V"].items()])
    sec("G2: SCIP against extended cutting planes, port1, K = 10, e = 0.01, 1'z = K",
        ["case", "SCIP", "ECP", "ECP masters", "rel diff", "SCIP in ECP bracket", "pass"],
        [[k, f"{b['scip']['st']} {b['scip_f']:.12e}",
          f"{b['ecp']['st']} [{b['ecp']['bd']:.10e}, {b['ecp']['f']:.10e}]",
          str(b["ecp"]["it"]), f(b["rel"], ".1e"), str(b["in_bracket"]),
          {True: "yes", False: "NO", None: "not both certified"}[b["pass"]]]
         for k, b in B.items() if k.startswith("g2-")])
    sec("RQ5: F1 and F1p by SCIP, lambda = 0.5, e = 0.01, 1'z = K",
        ["case", "status", "gap", "nodes", "root bound", "objective", "seconds"],
        [[k, b["c"]["st"], f(b["c"]["g"], ".1e"), str(b["c"]["nd"]), f(b["c"]["rt"], ".6e"),
          f(b["f"], ".10e"), f(b["c"]["t"], ".2f")] for k, b in B.items() if k.startswith("rq5")])
    sec("F2 and F3 on indtrack1 returns: solver value against independent recomputation",
        ["case", "status", "objective", "recomputed", "abs diff", "in X(K)"],
        [[k, b["c"]["st"], f(b["c"]["f"], ".10e"), f(b.get("recomputed", b.get("sorted")),
          ".10e"), f(b["diff"], ".1e"), str(b["ok"])]
         for k, b in B.items() if k[:2] in ("f2", "f3")])
    sec("F4 restricted (no transaction costs), Canakgoz-Beasley windows [1,145], [146,290]",
        ["case", "N", "K", "stage 1", "stage 2", "a in", "b in", "a out", "b out", "TE out"],
        [[k, str(b["N"]), str(B[k]["bundle"]["pa"]["K"]), b["c1"]["st"],
          b["c2"]["st"] if b["c2"] else "-", f(b.get("a_in"), ".2e"), f(b.get("b_in"), ".5f"),
          f(b.get("a_out"), ".5f"), f(b.get("b_out"), ".5f"), f(b.get("te_out"), ".3e")]
         for k, b in B.items() if k.startswith("f4-")])
    sec("F5 (g = 3, k = g(g+1)) by SCIP spatial branch-and-bound",
        ["case", "status", "objective", "dual bound", "gap", "nodes", "seconds", "grid best"],
        [[k, b["c"]["st"], f(b["f"], ".10e"), f(b["c"]["bd"], ".10e"), f(b["c"]["g"], ".1e"),
          str(b["c"]["nd"]), f(b["c"]["t"], ".1f"), f(b.get("grid"), ".10e")]
         for k, b in B.items() if k.startswith("f5-")])
    run_ = sorted({b["bundle"]["code"]["commit"][:10] for b in B.values()})
    hs = sorted({b["bundle"]["host"]["node"] for b in B.values()})
    L += [f"Revisions: {', '.join(run_) or 'none'}. Hosts: {', '.join(hs) or 'none'}.", ""]
    p = Path("docs/results/README.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes("\n".join(L).encode())


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--run", action="store_true")
    a.add_argument("--missing", action="store_true")
    a.add_argument("--only", action="append", default=[])
    a.add_argument("--time", type=float, default=60.0)
    a.add_argument("--jobs", type=int, default=1)
    x = a.parse_args(argv)
    if x.run:
        code = bd.identity()
        if code["dirty"]:
            raise SystemExit("refusing to write bundles from a dirty tree; commit first")
        todo = [c for c in cases() if not x.only or any(c[0].startswith(o) for o in x.only)]
        if x.missing:
            todo = [c for c in todo if bd.valid(OUT, c[0], c[2] | {"tl": x.time},
                                                code["source"]) is None]
        args = [(k, kd, pa, x.time, code) for k, kd, pa in todo]
        with cf.ProcessPoolExecutor(max_workers=x.jobs) as ex:
            for k, w in ex.map(one, args) if x.jobs > 1 else map(one, args):
                print(f"{k}: {w:.1f}s", flush=True)
    table()


if __name__ == "__main__":
    main()
