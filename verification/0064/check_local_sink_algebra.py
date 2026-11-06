"""Opt-in exact algebra for local_sink_asymptotics.md; no numerical spectra.

Source binding checks UTF-8 modelA.py with CRLF/CR normalized to LF.
It detects source drift; it does not mechanically derive equations from Python.
The holomorphic estimates, norm bounds and selector proof are not code-verified.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile

SOURCE_COMMIT = "6e2595be288f6e185a6ae4bd9d41a2965a9a296f"
SOURCE_SHA256_LF = "0e7264dbaf7b2968b38fd01905a778a05b31723f7ad7dd50326f5f5161d1a402"


class CheckFailure(Exception):
    pass


def verify_source(path):
    text = path.read_text(encoding="utf-8")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != SOURCE_SHA256_LF:
        raise CheckFailure("source_sha256_mismatch")
    return actual


class P:
    def __init__(self, terms=0):
        self.t = ({(0,)*6:terms} if isinstance(terms,int) and terms else
                  {} if isinstance(terms,int) else {k:v for k,v in terms.items() if v})
    def __add__(self, other):
        other=other if isinstance(other,P) else P(other)
        out=defaultdict(int,self.t)
        for k,v in other.t.items(): out[k]+=v
        return P(out)
    __radd__=__add__
    def __neg__(self): return P({k:-v for k,v in self.t.items()})
    def __sub__(self,other): return self+-coerce(other)
    def __rsub__(self,other): return coerce(other)+-self
    def __mul__(self,other):
        other=coerce(other); out=defaultdict(int)
        for a,x in self.t.items():
            for b,y in other.t.items(): out[tuple(u+v for u,v in zip(a,b))]+=x*y
        return P(out)
    __rmul__=__mul__
    def __pow__(self,k):
        if k<0: raise ValueError('Use explicit Laurent monomial')
        out=P(1)
        for _ in range(k): out=out*self
        return out
def coerce(x): return x if isinstance(x,P) else P(x)
def var(i,power=1):
    e=[0]*6;e[i]=power;return P({tuple(e):1})


def check_identities(mutation=None):
    z,mu,w,g,d,r=[var(i) for i in range(6)]
    ri=var(5,-1)
    Q=z*z+w*w+g*g
    B=mu*(z*z+w*w)+w*g*g
    H=z**4+z*z*(2*g*g+w*w+mu*mu)+(g*g+mu*w)**2
    x0=2*g*(z*z+mu*w+g*g)
    y0=2*z*g*(mu-w)
    x1=-2*B
    y1=2*z*Q
    x2=H
    bulk=2*mu*r*r+(z*z+mu*mu+(3 if mutation == "bulk" else 4))*r+2*mu
    vs=(2+mu*r,-z);vu=(2+mu*ri,-z)
    def det(a,b): return a[0]*b[1]-a[1]*b[0]
    delta=z*mu*(ri-r)
    hx=(z+d)**2+mu*mu;hy=2*(z+d)
    hs=hx*vs[0]+hy*vs[1];hu=hx*vu[0]+hy*vu[1]
    f=H+2*Q*(2+mu*r)
    identities={
     'left_probe_x':z*x0-w*y0+(g*y1 if mutation == 'probe-sign' else -g*y1),
     'left_probe_y':z*y0+w*x0+g*x1,
     'left_buffer_x':z*x1+mu*y1-g*y0,
     'left_buffer_y':z*y1-mu*x1-2*x2+g*x0,
     'left_H_factor':H-((z*z+mu*w+g*g)*Q-(w-mu)*B),
     'transfer_det_scaled':4*(z*z+mu*mu)-4*z*z-4*mu*mu,
     'transfer_eigen_x_mod_bulk':-(z*z+mu*mu)*vs[0]-2*z*vs[1]-2*mu*r*vs[0]+mu*bulk,
     'transfer_eigen_y':-2*z*vs[0]-4*vs[1]-2*mu*r*vs[1],
     'reciprocal_bulk':r*r*(2*mu*ri*ri+(z*z+mu*mu+4)*ri+2*mu)-bulk,
     'stable_unstable_det':det(vs,vu)-delta,
     'unstable_coefficient_numerator':det(vs,(H,2*z*Q))-z*f,
     'terminal_first_component':vs[0]*hu-hs*vu[0]-hy*delta,
     'zero_coupling_halfline_mod_bulk':(z*z+w*w)*(z*z+mu*mu+4+2*mu*r+2*mu*ri)-(z*z+w*w)*ri*bulk,
    }

    failed = [name for name, polynomial in identities.items() if polynomial.t]
    if failed:
        raise CheckFailure("nonzero_identities:" + ",".join(failed))
    return list(identities)


def rejection_checks(source):
    """Exercise real CLI failures, including under the current optimization mode."""
    prefix = [sys.executable] + (["-O"] if sys.flags.optimize else [])
    prefix += [str(Path(__file__).resolve())]
    checked = []
    with tempfile.TemporaryDirectory(prefix="local-sink-algebra-") as temp:
        changed = Path(temp)/"changed_modelA.py"
        changed.write_text(source.read_text(encoding="utf-8") + "\n# altered source\n", encoding="utf-8")
        cases = [
            (["--source", str(source), "--mutation", "bulk"], "nonzero_identities:"),
            (["--source", str(source), "--mutation", "probe-sign"], "nonzero_identities:left_probe_x"),
            (["--source", str(changed)], "source_sha256_mismatch"),
        ]
        for arguments, expected in cases:
            result = subprocess.run(prefix + arguments, capture_output=True, text=True, timeout=20)
            try:
                report = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise CheckFailure("invalid_rejection_report") from exc
            if result.returncode != 1 or report.get("ok") is not False or not report.get("error", "").startswith(expected):
                raise CheckFailure("rejection_control_failed:" + expected)
            checked.append(report["error"])
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("modelA.py"))
    parser.add_argument("--output", type=Path, help="Optional JSON output; stdout is always emitted")
    parser.add_argument("--mutation", choices=["bulk", "probe-sign"], help="Negative control: must fail")
    parser.add_argument("--self-test", action="store_true", help="Also test two mutations and altered source in temporary files")
    args = parser.parse_args(argv)
    try:
        source_hash = verify_source(args.source)
        identities = check_identities(args.mutation)
        report = {"ok": True, "ring": "Z[lambda,mu,omega0,g,d,r,r^-1]",
                  "source_commit": SOURCE_COMMIT, "source_sha256_lf": source_hash,
                  "exact_identities_passed": len(identities), "identities": identities,
                  "optimization_enabled": bool(sys.flags.optimize),
                  "limit_or_selector_proof_checked_by_code": False,
                  "numerical_samples": 0, "diagonalizations": 0, "root_solving": False}
        if args.self_test:
            report["rejections"] = rejection_checks(args.source.resolve())
    except (CheckFailure, OSError, UnicodeError, subprocess.SubprocessError) as exc:
        report = {"ok": False, "error": str(exc)}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
