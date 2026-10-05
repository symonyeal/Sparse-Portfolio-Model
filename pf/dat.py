"""Strict loaders for the data tiers D1-D5; a field or record the loader does not read is refused.

D1  OR-Library port1-port5 and portef1-portef5 (Beasley, 1990; Chang et al., 2000)
D2  OR-Library indtrack1-indtrack8 (Beasley, Meade and Chang, 2003; Canakgoz and Beasley, 2009)
D3  CRSP monthly stock file with delisting returns, supplied by the user, never committed
D4  Kenneth French data library, one-month T-bill (RF of F-F_Research_Data_Factors), fetched
D5  synthetic universes, each labelled with the code property it tests

OR-Library files are committed under data/orlib with their MIT notice and pinned by SHA-256.
indtrack7 and indtrack8 carry one stock series beyond the N their headers declare; the loader
refuses it unless the caller passes tail=True, and then reports the dropped values.
"""

import csv
import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

# Legend
# N, T   : assets; time periods (indtrack: levels at t = 0..T)
# mu, s  : mean and standard deviation of each asset's return (port)
# Cr, S  : correlation and covariance matrices, S = s_i s_j Cr_ij
# I, V   : index levels (T+1) and stock levels (N, T+1) (indtrack)
# r, Ri  : stock and index log returns, r_it = log(V_it / V_i,t-1)
# ef     : frontier points (mean, variance) (portef)
# rf     : one-month T-bill return per month (decimal), mo : months as YYYYMM integers
# tail   : accept and drop a trailing series beyond the declared N
# sha    : SHA-256 of the raw file bytes
# kd     : property a synthetic universe is generated to test

ROOT = Path(__file__).resolve().parent.parent
ORLIB = ROOT / "data" / "orlib"
FETCH = ROOT / "data" / "fetched"
FF = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


class DataError(ValueError):
    """A data file that does not match its stated format."""


def sha256(p):
    """SHA-256 of the raw bytes of file p."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pins():
    """name -> pinned SHA-256 of the committed OR-Library files."""
    out = {}
    for ln in (ORLIB / "SHA256").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, n = ln.split()
            out[n.lstrip("*")] = h
    return out


def _orlib(name, path=None):
    """Tokens of an OR-Library file; the committed copy must match its pin."""
    p = Path(path) if path else ORLIB / f"{name}.txt"
    if path is None:
        h = sha256(p)
        if h != pins()[p.name]:
            raise DataError(f"{p.name}: SHA-256 {h} does not match the pin")
    return p.read_text(encoding="ascii").split(), p


def _num(tok, p, what):
    try:
        v = float(tok)
    except ValueError:
        raise DataError(f"{p}: {what} is not a number: {tok!r}") from None
    if not np.isfinite(v):
        raise DataError(f"{p}: {what} is not finite: {tok!r}")
    return v


def port(k, path=None):
    """D1 instance portk: N; N lines (mean, std); every pair i <= j once with its correlation."""
    t, p = _orlib(f"port{k}", path)
    if not t:
        raise DataError(f"{p}: empty")
    N = int(_num(t[0], p, "N"))
    if N < 1 or str(N) != t[0].strip():
        raise DataError(f"{p}: N must be a positive integer, got {t[0]!r}")
    P = N * (N + 1) // 2
    if len(t) != 1 + 2 * N + 3 * P:
        raise DataError(f"{p}: {len(t)} tokens, expected {1 + 2 * N + 3 * P} for N={N}")
    a = np.array([_num(x, p, "value") for x in t[1:1 + 2 * N]]).reshape(N, 2)
    mu, s = a[:, 0], a[:, 1]
    if np.any(s <= 0):
        raise DataError(f"{p}: nonpositive standard deviation")
    Cr = np.full((N, N), np.nan)
    q = np.array(t[1 + 2 * N:]).reshape(P, 3)
    for i, j, v in q:
        i, j, v = int(i) - 1, int(j) - 1, _num(v, p, "correlation")
        if not (0 <= i <= j < N):
            raise DataError(f"{p}: pair ({i + 1}, {j + 1}) outside 1 <= i <= j <= {N}")
        if not np.isnan(Cr[i, j]):
            raise DataError(f"{p}: pair ({i + 1}, {j + 1}) repeated")
        if abs(v) > 1:
            raise DataError(f"{p}: correlation {v} outside [-1, 1]")
        Cr[i, j] = Cr[j, i] = v
    if np.isnan(Cr).any():
        raise DataError(f"{p}: missing correlation pairs")
    if not np.allclose(np.diag(Cr), 1.0):
        raise DataError(f"{p}: diagonal correlation not 1")
    return {"N": N, "mu": mu, "s": s, "Cr": Cr, "S": np.outer(s, s) * Cr, "name": f"port{k}",
            "sha": sha256(p)}


def portef(k, path=None):
    """D1 frontier portefk: (mean, variance) per line; the unconstrained frontier, not a CCEF."""
    t, p = _orlib(f"portef{k}", path)
    if not t:
        raise DataError(f"{p}: empty frontier")
    if len(t) % 2:
        raise DataError(f"{p}: odd number of tokens")
    ef = np.array([_num(x, p, "value") for x in t]).reshape(-1, 2)
    if np.any(ef[:, 1] < 0):
        raise DataError(f"{p}: negative variance")
    return {"ef": ef, "name": f"portef{k}", "sha": sha256(p)}


def indtrack(k, path=None, tail=False):
    """D2 instance indtrackk: (N, T); index levels t=0..T; each stock's levels t=0..T."""
    t, p = _orlib(f"indtrack{k}", path)
    if len(t) < 2:
        raise DataError(f"{p}: missing header")
    N, T = int(_num(t[0], p, "N")), int(_num(t[1], p, "T"))
    if N != float(t[0]) or T != float(t[1]):
        raise DataError(f"{p}: N and T must be integers")
    if N < 1 or T < 1:
        raise DataError(f"{p}: N={N}, T={T}")
    n = (N + 1) * (T + 1)
    x = t[2:]
    if len(x) < n:
        raise DataError(f"{p}: {len(x)} values, expected {n}")
    dr = len(x) - n
    if dr and not tail:
        raise DataError(f"{p}: {dr} values after the {N} declared series "
                        f"({dr / (T + 1):g} series); pass tail=True to drop them")
    if dr and (k not in (7, 8) or dr != T + 1):
        raise DataError(f"{p}: only the documented extra series in indtrack7/8 may be dropped")
    if dr and any(_num(y, p, "trailing level") <= 0 for y in x[n:]):
        raise DataError(f"{p}: nonpositive trailing price level")
    v = np.array([_num(y, p, "level") for y in x[:n]]).reshape(N + 1, T + 1)
    if np.any(v <= 0):
        raise DataError(f"{p}: nonpositive price level")
    return {"N": N, "T": T, "I": v[0], "V": v[1:], "drop": dr, "name": f"indtrack{k}",
            "sha": sha256(p)}


def lr(L):
    """Log returns r_t = log(L_t / L_{t-1}) along the last axis (Canakgoz and Beasley, 2009)."""
    L = np.asarray(L, float)
    return np.log(L[..., 1:] / L[..., :-1])


def sr(L):
    """Simple returns L_t / L_{t-1} - 1 along the last axis."""
    L = np.asarray(L, float)
    return L[..., 1:] / L[..., :-1] - 1.0


# D4 -----------------------------------------------------------------------------------------

def rf_get(dst=None, url=FF):
    """Fetch the French factors zip with the system curl (Schannel); returns its path and SHA."""
    import subprocess
    import sys
    dst = Path(dst) if dst else FETCH / "F-F_Research_Data_Factors_CSV.zip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sSLf"]
    if sys.platform == "win32":
        cmd += ["--ssl-no-revoke"]
    subprocess.run(cmd + ["--max-time", "120", "-o", str(dst), url], check=True)
    return dst, sha256(dst)


def rf(path=None):
    """D4 monthly one-month T-bill returns (decimal) from the French factors file.

    Reads the monthly block between its header ',Mkt-RF,SMB,HML,RF' and the first blank line;
    months must be consecutive. Values are printed in percent.
    """
    p = Path(path) if path else FETCH / "F-F_Research_Data_Factors_CSV.zip"
    b = p.read_bytes()
    if p.suffix.lower() == ".zip":
        z = zipfile.ZipFile(io.BytesIO(b))
        if len(z.namelist()) != 1:
            raise DataError(f"{p}: expected one member, got {z.namelist()}")
        b = z.read(z.namelist()[0])
    L = b.decode("latin-1").splitlines()
    h = [i for i, ln in enumerate(L) if ln.replace(" ", "") == ",Mkt-RF,SMB,HML,RF"]
    if not h:
        raise DataError(f"{p}: monthly header not found")
    mo, v = [], []
    for ln in L[h[0] + 1:]:
        if not ln.strip():
            break
        c = [x.strip() for x in ln.split(",")]
        if len(c) != 5 or len(c[0]) != 6 or not c[0].isdigit():
            raise DataError(f"{p}: malformed monthly row {ln!r}")
        mo.append(int(c[0]))
        if not 1 <= mo[-1] % 100 <= 12:
            raise DataError(f"{p}: invalid calendar month")
        v.append(_num(c[4], p, "RF") / 100.0)
    if not mo:
        raise DataError(f"{p}: empty monthly block")
    mo = np.array(mo)
    if not np.array_equal(mo[1:], np.array([nx(m) for m in mo[:-1]])):
        raise DataError(f"{p}: months are not consecutive")
    vin = next((ln for ln in L[:h[0]] if "CRSP database" in ln), "").strip()
    return {"mo": mo, "rf": np.array(v), "sha": sha256(p), "vintage": vin}


def nx(m):
    """The month after YYYYMM."""
    y, k = divmod(int(m), 100)
    return (y + 1) * 100 + 1 if k == 12 else y * 100 + k + 1


def mseq(a, b):
    """Months a..b inclusive as YYYYMM integers."""
    out = [int(a)]
    while out[-1] < b:
        out.append(nx(out[-1]))
    return np.array(out)


# D3 -----------------------------------------------------------------------------------------

COLS = ("PERMNO", "PERMCO", "DATE", "SHRCD", "EXCHCD", "PRC", "SHROUT", "RET", "DLSTCD", "DLRET")
MISS = {"", ".", "A", "B", "C", "D", "E", "P", "S", "T", "-66", "-77", "-88", "-99", "-66.0",
        "-77.0", "-88.0", "-99.0"}


def _cell(x):
    """A CRSP numeric cell: (value, missing code or None). CRSP letter codes mark missing."""
    x = x.strip()
    if x.upper() in MISS:
        return np.nan, x.upper() or "blank"
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("nonfinite cell")
    return v, None


def _integer(x):
    """Parse an integer without silently truncating decimal cells."""
    v = float(x)
    if not np.isfinite(v) or not v.is_integer() or not -(2 ** 63) <= v < 2 ** 63:
        raise ValueError("invalid integer")
    return int(v)


def crsp(path, shr=(10, 11, 12)):
    """D3 rows of a CRSP monthly stock extract with delisting fields (legacy SIZ layout).

    The header must hold exactly COLS (any case, any order); any other field is refused. A
    missing RET or DLRET stays nan and its CRSP code is kept; nothing is read as zero. Share
    codes outside shr are dropped after reading (Bessembinder, 2018: 10, 11 and 12).
    """
    p = Path(path)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        if header is None:
            raise DataError(f"{p}: empty extract")
        hd = [c.strip().upper() for c in header]
        hd = ["DATE" if c in ("DATE", "MTHCALDT") else c for c in hd]
        if sorted(hd) != sorted(COLS) or len(set(hd)) != len(hd):
            ex = sorted(set(hd) - set(COLS))
            ms = sorted(set(COLS) - set(hd))
            raise DataError(f"{p}: fields must be exactly {COLS}; unread {ex}, missing {ms}")
        ix = {c: hd.index(c) for c in COLS}
        rows = list(rd)
    n = len(rows)
    o = {c: np.full(n, np.nan) for c in ("PRC", "SHROUT", "RET", "DLRET")}
    code = {"RET": np.full(n, "", dtype=object), "DLRET": np.full(n, "", dtype=object)}
    ii = {c: np.zeros(n, dtype=np.int64) for c in ("PERMNO", "PERMCO", "SHRCD", "EXCHCD")}
    dl = np.zeros(n, dtype=np.int64)
    mo = np.zeros(n, dtype=np.int64)
    for k, r in enumerate(rows):
        if len(r) != len(COLS):
            raise DataError(f"{p}: row {k + 2} has {len(r)} fields")
        for c, a in ii.items():
            try:
                a[k] = _integer(r[ix[c]])
                if c in ("PERMNO", "PERMCO") and a[k] <= 0:
                    raise ValueError("nonpositive identifier")
            except (ValueError, OverflowError):
                raise DataError(f"{p}: row {k + 2} invalid {c}") from None
        d = r[ix["DATE"]].strip().replace("-", "").replace("/", "")
        try:
            if len(d) != 8 or not d.isdigit():
                raise ValueError("invalid format")
            date(int(d[:4]), int(d[4:6]), int(d[6:]))
        except ValueError:
            raise DataError(f"{p}: row {k + 2} invalid DATE") from None
        mo[k] = int(d[:6])
        for c, a in o.items():
            try:
                v, cd = _cell(r[ix[c]])
                if (c in ("RET", "DLRET") and v < -1) or (c == "SHROUT" and v < 0):
                    raise ValueError("invalid value")
            except (ValueError, OverflowError):
                raise DataError(f"{p}: row {k + 2} invalid {c}") from None
            a[k] = v
            if c in code:
                code[c][k] = cd or ""
        x = r[ix["DLSTCD"]].strip()
        try:
            dl[k] = _integer(x) if x not in ("", ".") else 0
            if dl[k] < 0:
                raise ValueError("negative code")
        except (ValueError, OverflowError):
            raise DataError(f"{p}: row {k + 2} invalid DLSTCD") from None
    keep = np.isin(ii["SHRCD"], shr)
    out = dict(mo=mo, dl=dl, **ii, **o, RETc=code["RET"], DLRETc=code["DLRET"])
    out = {c: v[keep] for c, v in out.items()}
    if not out["mo"].size:
        raise DataError(f"{p}: no rows with the requested share codes")
    k = np.lexsort((out["PERMNO"], out["mo"]))
    out = {c: v[k] for c, v in out.items()}
    dup = (np.diff(out["mo"]) == 0) & (np.diff(out["PERMNO"]) == 0)
    if dup.any():
        raise DataError(f"{p}: {int(dup.sum())} repeated (PERMNO, month) rows")
    out["sha"] = sha256(p)
    out["dropped"] = int((~keep).sum())
    return out


# D5 -----------------------------------------------------------------------------------------

KD = ("moments", "skew")


def syn(N, S, kd, seed):
    """D5 synthetic returns (S, N), labelled with the code property kd they exercise.

    moments : multivariate normal draws; tests rows, certificates and moment identities only
    skew    : lognormal gross returns minus one; tests the cubic term of F5 only
    No empirical claim rests on these draws.
    """
    if kd not in KD:
        raise ValueError(f"kind {kd!r} not in {KD}")
    g = np.random.default_rng(seed)
    if kd == "moments":
        L = g.normal(size=(N, N)) / np.sqrt(N)
        return {"r": g.normal(0.005, 1.0, (S, N)) @ (0.04 * L) + 0.005, "kd": kd, "seed": seed}
    z = g.normal(size=(S, N)) * g.uniform(0.03, 0.15, N)
    return {"r": np.exp(z + 0.004) - 1.0, "kd": kd, "seed": seed}


def manifest(d):
    """JSON-safe identity of a loaded dataset (name, SHA-256, vintage)."""
    return json.loads(json.dumps({k: d[k] for k in ("name", "sha", "vintage") if k in d}))
