"""Symbolic verification of the Seeley-DeWitt a1 coefficients and induced-Newton
sign table of 2512.0091 (heat kernel / induced gravity).

Convention (paper): Tr e^{-sP} ~ (4 pi s)^{-2} int sqrt(g) [a0 + a1 s + ...],
a1(x) = tr( R/6 - X ) for P = -nabla^2 + X.  W_R^total = -(A1^eff/(32 pi^2)) Lambda^2 int sqrt(g) R.
Bosonic field: W = +(1/2) ln det P.  Fermion: W = -ln det(D+m) = -(1/2) ln det P_D
(overall sign flip).  Ghost: enters as MINUS a complex boson.
"""
import sympy as sp
R, xi = sp.symbols('R xi', real=True)

def a1R_coeff(trace_one, X_R_coeff):
    # coefficient of R in tr(R/6 - X), with X's R-part = X_R_coeff, over 'trace_one' components
    return sp.Rational(1,6)*trace_one - X_R_coeff

results = {}
# --- real scalar, X = m^2 + xi R  -> X_R = xi, tr1 = 1 ; bosonic W=+ (1/2)ln det
a1_scalar = a1R_coeff(1, xi)               # = 1/6 - xi
A1_scalar = a1_scalar                       # bosonic: A1^eff = tr a1,R
results['real scalar (xi)'] = A1_scalar
results['real scalar minimal (xi=0)'] = A1_scalar.subs(xi,0)
results['complex scalar (xi=0)'] = 2*A1_scalar.subs(xi,0)

# --- Dirac fermion: P_D = -nabla^2 + m^2 + R/4  (Lichnerowicz) -> X_R = 1/4, tr1 = 4
a1_dirac = a1R_coeff(4, 4*sp.Rational(1,4))  # tr over 4 spinor comps of (1/6 - 1/4)R = 4*(-1/12)
# careful: X_R per-component = 1/4; tr(R/6 - X) = tr(1)*1/6 R - tr(X) ; tr(X_R) = 4*(1/4)=1
a1_dirac = 4*sp.Rational(1,6) - 4*sp.Rational(1,4)   # = 2/3 - 1 = -1/3
A1_dirac = -a1_dirac                          # fermion sign flip: A1^eff = -tr a1,R
results['Dirac fermion (4c)'] = A1_dirac
results['Weyl fermion (2c)'] = A1_dirac/2
results['Majorana fermion'] = A1_dirac/2

# --- gauge vector (1-form) P1 = -nabla^2 + Ric : X_R (trace) = R -> tr(X_R)=1, tr1=4
a1_vector = 4*sp.Rational(1,6) - 1           # = -1/3  (bosonic)
A1_vector = a1_vector                         # bosonic
# ghost: complex Grassmann scalar, enters as MINUS complex boson
a1_ghost_as_cplx_boson = 2*sp.Rational(1,6)  # = 1/3
A1_ghost = -a1_ghost_as_cplx_boson            # = -1/3
A1_gauge_total = A1_vector + A1_ghost         # = -2/3
results['gauge vector + ghosts'] = A1_gauge_total

paper = {
 'real scalar (xi)': sp.Rational(1,6)-xi,
 'real scalar minimal (xi=0)': sp.Rational(1,6),
 'complex scalar (xi=0)': sp.Rational(1,3),
 'Dirac fermion (4c)': sp.Rational(1,3),
 'Weyl fermion (2c)': sp.Rational(1,6),
 'Majorana fermion': sp.Rational(1,6),
 'gauge vector + ghosts': -sp.Rational(2,3),
}
print("species                         computed      paper        match")
allok = True
for k in paper:
    c = sp.simplify(results[k]); p = paper[k]
    ok = sp.simplify(c-p)==0
    allok &= ok
    print(f"{k:31s} {str(c):12s} {str(p):12s} {'OK' if ok else 'MISMATCH'}")
print(f"\nALL SEELEY-DEWITT COEFFICIENTS MATCH: {allok}")
# sign of induced G: Gind > 0 iff A1^eff > 0
print("\nInduced-Newton sign (Gind>0 iff A1^eff>0):")
print(f"  minimal real scalar A1=+1/6 -> Gind>0 : {results['real scalar minimal (xi=0)']>0}")
print(f"  non-minimal scalar xi=1/4 A1={sp.Rational(1,6)-sp.Rational(1,4)} -> Gind<0 : {(sp.Rational(1,6)-sp.Rational(1,4))<0}")
print(f"  conformal scalar (D=4) xi=1/6 A1={sp.Rational(1,6)-sp.Rational(1,6)} -> no EH contribution : {(sp.Rational(1,6)-sp.Rational(1,6))==0}")
print(f"  gauge vector+ghosts A1=-2/3 -> Gind<0 : {A1_gauge_total<0}")
