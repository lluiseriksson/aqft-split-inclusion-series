# Local-sink asymptotics for the 0064 probe branch

This technical note supplies a bounded analytic justification for the local-sink
model, with explicit hypotheses and a separate exact-algebra check. It has been
internally audited; it is not an independent review, a formal proof, a new paper,
or a revision of the archived v2 TEX/PDF. Finite-width spectra and fitted slopes
remain numerical evidence, not proofs of a subexponential prefactor at all widths.

The source is [modelA.py at 6e2595be](https://github.com/lluiseriksson/aqft-split-inclusion-series/blob/6e2595be288f6e185a6ae4bd9d41a2965a9a296f/verification/0064/modelA.py),
specifically `majorana_matrix`, `X_Y_from`, `build_AB`, `loss_vector` and `kappa_of`.
We use its defaults t=Delta_p=1, buffer chemical potential mu, real coupling g,
probe frequency w=omega_0, terminal loss gamma, and integer width N=eps>=2.
The [archived manuscript](../../papers/0064-heisenberg-cut/heisenberg_cut_v2.tex)
discusses the evanescent prediction and finite spectra; this note makes the
additional pole, normalization and selection arguments explicit.

## Statements and quantifiers

**A — fixed simple branch.** Fix mu>2, real g!=0, gamma>0 and real w. Suppose the
half-line function f defined below has a simple zero lambda_*=i omega_* with
0<omega_*<mu-2. There is a unique nearby finite eigenvalue lambda_N for all
sufficiently large N; it is algebraically simple and converges to lambda_*.
Writing q_*=q(omega_*), where

    cosh q(omega) = (mu^2+4-omega^2)/(4mu),   q(omega)>0,

its decay rate kappa_N=-2 Re lambda_N satisfies

    kappa_N = K exp(-2q_* N) [1+O(N exp(-2q_* N))],   K>0,
    log kappa_N + 2q(|Im lambda_N|)N
      = log K + O(N exp(-2q_* N)).

All constants are for fixed parameters, separated from the band edge. Thus the
estimate is bilateral for sufficiently large N. It is an asymptotic, not an
equality with a constant prefactor at every finite width. A alone does not say
this branch has maximum probe weight among all modes.

**B — nonempty regime for the actual selector.** For each fixed mu>2,
0<w<mu-2 and gamma>0, there exists g0>0 such that, for each fixed real g with
0<|g|<g0, A applies and there exists N0(g) such that for all N>=N0(g) the
conjugate pair of this branch is precisely the pair of highest probe weights in
`kappa_of`. Both eigenvalues have the same real part, so A gives the mathematical
exact-arithmetic output of that selector in this regime.

The thresholds are qualitative. This does not certify g=0.1 or 0.3, omega_0=0,
the tabulated parameter sets, floating-point eigensolver accuracy, strong
coupling, multiple poles, a band-edge limit, mu<2, arbitrary observables,
entropy relaxation, a global spectral gap, or interacting buffers. No uniform
N0 as g tends to zero is claimed.

## 1. Conventions and boundary equations

Let x_j,y_j be the two Majorana coordinates at site j; j=0 is the probe and
j=1,...,N the buffer. Put d=gamma/2. For real symmetric A and real antisymmetric
B the source gives Hmaj[a_p,b_q]=A_pq-B_pq, with the opposite entry its negative.
Indeed the coefficient accumulator has entries i(A_pq-B_pq)/4 and their
negatives; `-2j*(T-T.T)` gives A-B. Thus the buffer on-site entry is -mu,
the probe entry is w, the entry a_(j+1),b_j is -2 and the reverse bond
orientation a_j,b_(j+1) is zero. Both probe bonds have coefficient g.
The two terminal loss coefficients sqrt(gamma)/2 and i sqrt(gamma)/2 give
2 Re M=d on both terminal coordinates. Hence X=Hmaj-d Pi_N.

The left equations are

    lambda x0 - w y0 - g y1 = 0,
    lambda y0 + w x0 + g x1 = 0,
    lambda x1 + mu y1 - g y0 = 0,
    lambda y1 - mu x1 - 2x2 + g x0 = 0.

For 2<=j<=N-1 and at the right boundary, respectively,

    lambda x_j = -mu y_j - 2y_(j-1),
    lambda y_j = mu x_j + 2x_(j+1),
    (lambda+d)x_N = -mu y_N - 2y_(N-1),
    (lambda+d)y_N = mu x_N.

## 2. Polynomial seed and transfer

Define

    Q = lambda^2+w^2+g^2,
    B = mu(lambda^2+w^2)+w g^2,
    H = lambda^4+lambda^2(2g^2+w^2+mu^2)+(g^2+mu w)^2.

A polynomial seed satisfying all four left equations is

    x0 = 2g(lambda^2+mu w+g^2),    y0 = 2lambda g(mu-w),
    x1 = -2B,    y1 = 2lambda Q,    x2 = H.

For u_j=(x_j,y_(j-1))^T the interior recurrence is u_(j+1)=T u_j, with

    T = (1/(2mu)) [ -(lambda^2+mu^2)  -2lambda ]
                  [ -2lambda         -4       ],     det T=1.

Its multipliers obey

    p(r)=2mu r^2+(lambda^2+mu^2+4)r+2mu=0.

At lambda=i omega, 0<omega<mu-2, the stable root is r=-exp(-q(omega)) and
-1<r<-2/mu. The distinct roots r and r^(-1) extend holomorphically to a small
disc about lambda_*. Useful eigenvectors are

    v_s=(2+mu r,-lambda)^T,    v_u=(2+mu/r,-lambda)^T,
    Delta_v=det(v_s,v_u)=lambda mu(r^(-1)-r).

This determinant is nonzero on a sufficiently small disc. Decompose
l=(H,2lambda Q)^T=c_s v_s+c_u v_u. Then

    f(lambda)=H+2Q(2+mu r(lambda)),
    c_u=f/[mu(r^(-1)-r)],    c_s=det(l,v_u)/Delta_v.

The half-line condition is **f=0**, which removes the growing component. The
bulk dispersion alone is not the pole equation.

At a sub-gap zero f=0, Q cannot vanish. The identity
H=(lambda^2+mu w+g^2)Q-(w-mu)B gives, when Q=0,
H=-g^2(w-mu)^2. If also H=0, then w=mu and
omega_*^2=mu^2+g^2, outside the stipulated interval. Hence the seed is nonzero
at the zero. Locally the left four equations have rank four. More explicitly,
their solutions obey 2lambda Q x2=H y1, and projection onto (x2,y1) is
injective: a nonzero solution with x2=y1=0 would force w=mu and
lambda^2+mu^2+g^2=0. These exceptional points are outside a sufficiently small
disc about the sub-gap root. The relation has a nonzero coefficient and the
displayed seed spans its one-dimensional solution space.

## 3. Finite pole and algebraic multiplicity

Set m=N-2 and h=((lambda+d)^2+mu^2,2(lambda+d)). Near lambda_*, lambda+d!=0,
and the right conditions are equivalent to h u_N=0 and
y_N=mu x_N/(lambda+d). The exact finite pole equation is

    F_N(lambda)=h T^m l=0.

Write a=c_s h v_s and b=c_u h v_u. Since r!=0,

    F_N=r^m a+r^(-m)b=r^(-m)G_N,
    G_N=b+a r^(2m).

This equivalence also preserves algebraic multiplicity, not just zero sets.
Keep all rows of lambda I-X except the last terminal equation. The left four
rows have a one-dimensional kernel; each interior pair propagates it using
the constant nonzero pivots mu and 2. The first terminal row determines
y_N=-((lambda+d)x_N+2y_(N-1))/mu. These 2N+1 rows have full row rank and a
nonvanishing holomorphic kernel generator obtained from the seed. Locally choose
a nonzero coordinate of this generator and complete it to an invertible
holomorphic frame. Row and column elimination then leaves an identity block
and the last residual

    (lambda+d)y_N-mu x_N = -F_N/mu.

Thus det(lambda I-X) equals F_N times a holomorphic nonvanishing factor. Simple
zeros of G_N are algebraically simple eigenvalues. No global diagonalizability
is used. In the weak-coupling discs below this argument remains valid even at
Q=0: w!=mu, lambda!=0, H=-g^2(w-mu)^2!=0 there, and the seed has
y0=2lambda g(mu-w)!=0. No division by Q is made.

The right boundary is nonresonant under A. At lambda_*=i omega_* and for
z=r_* or r_*^(-1), mu z<-2 and

    Im [h(i omega_*) (2+mu z,-i omega_*)^T]
      =2d omega_*(1+mu z) != 0.

Both h v_s and h v_u are therefore nonzero. Also c_s*!=0, a*!=0, and f'!=0
implies b'_*!=0. This uses gamma>0 fixed, not a lossless boundary assumption.

## 4. Holomorphic remainder for the finite pole

Put rho=|r_*|<1. On the shrinking disc |lambda-lambda_*|<=R rho^(2m),
Taylor estimates from a fixed holomorphic neighborhood give

    b(lambda)=b'_*(lambda-lambda_*)+O(rho^(4m)),
    a(lambda)r(lambda)^(2m)=a_*r_*^(2m)+O(m rho^(4m)).

In particular r(lambda)/r_*=1+O(rho^(2m)), so raising this ratio to 2m gives
1+O(m rho^(2m)); a bound with a larger fixed rho is not being substituted for
this sharper remainder. Choose R>|a_*/b'_*|. On the boundary the affine
function b'_*(lambda-lambda_*)+a_*r_*^(2m) has modulus at least
(|b'_*|R-|a_*|)rho^(2m). Rouché's theorem therefore counts exactly one zero
for sufficiently large m. Its derivative is b'_*+O(m rho^(2m))!=0, and

    lambda_N=lambda_*-(a_*/b'_*)r_*^(2m)+O(m rho^(4m)).

## 5. Terminal weight, exact norm and positive prefactor

At the finite zero c_u=-c_s(hv_s/hv_u)r^(2m), hence

    u_N=r^m t(lambda_N),
    t=c_s[v_s-(hv_s/hv_u)v_u],
    t_x=c_s 2(lambda+d)Delta_v/(hv_u).

The last expression is nonzero at lambda_*. The unnormalized terminal weight is
exactly |r(lambda_N)|^(2m) S(lambda_N), where

    S(lambda)=|t_x|^2(1+mu^2/|lambda+d|^2),    S_*>0.

The half-line norm with this seed is

    Z_*=|x0_*|^2+|y0_*|^2+|x1_*|^2
         +|c_s*|^2 ||v_s*||^2/(1-rho^2)>0.

For alpha=c_s(lambda_N), beta=c_u(lambda_N), z=r(lambda_N), the **exact**
finite norm is

    Z_N=|x0|^2+|y0|^2+|x1|^2
        +sum_(k=0)^m ||alpha z^k v_s+beta z^(-k) v_u||^2 + |y_N|^2.

The sum contains x2,...,xN and y1,...,y_(N-1); no coordinate is counted twice.
Here beta=O(rho^(2m)) and |z|=rho(1+O(rho^(2m))). The reflected square is bounded
by C rho^(4m) sum_(k=0)^m |z|^(-2k)=O(rho^(2m)); the cross term is bounded by
C|beta|(m+1) because |z^k||z^(-k)|=1. The stable tail, the change of its infinite
geometric sum with lambda, the left components, and the final |y_N|^2 each
contribute O(rho^(2m)). Thus Z_N=Z_*+O((m+1)rho^(2m)).

For a right eigenvector, antisymmetry of Hmaj gives the exact dissipation
identity -2 Re lambda=gamma ||Pi_N v||^2/||v||^2. Applying it here yields

    kappa_N=gamma |r(lambda_N)|^(2m) S(lambda_N)/Z_N
           =gamma(S_*/Z_*)rho^(2m)[1+O(m rho^(2m))].

Therefore K=gamma(S_*/Z_*)rho^(-4)>0, proving A's bilateral estimate. Taking
the logarithm is justified by this positivity. Since q is smooth on the fixed
interior frequency interval and |Im lambda_N|-omega_*=O(rho^(2N)), the stated
logarithmic remainder follows. Its limit is log K, not necessarily zero.

## 6. Why the weak-coupling pair is selected

Fix the parameters of B. At g=0, the bulk equation gives

    f_0(lambda)=-2mu r(lambda)^(-1)(lambda^2+w^2).

Choose disjoint closed discs of radius delta about +/-iw, small enough to
avoid zero, -d and the bands, retain the analytic evanescent branches, and
keep h v_u nonzero. Each disc contains one simple zero of f_0. For sufficiently
small |g|, the argument principle and implicit function theorem retain exactly
one simple zero per disc. For real g the positive zero is purely imaginary:
f(i omega,g) is real and its real omega derivative at (w,0) is nonzero. Thus
omega_*(g)=w+O(g^2) remains strictly sub-gap.

For each such fixed nonzero g, on the disc boundaries |r|<=rho_bar<1 and b is
nonzero. Uniform convergence G_N->b and Rouché give exactly one finite zero
per disc for sufficiently large N. The multiplicity argument in section 3
applies across Q=0. The two eigenvalues are simple and conjugate since X is real.

The half-line probe weight tends to 1 as g tends to zero: at its positive pole,
x0/g->2w(mu-w), y0/g->2iw(mu-w), whereas x1, l and c_s are O(g^2).
The probe norm is therefore of order g^2 with a positive leading coefficient,
and the summed buffer norm is O(g^4). Shrink g0 so the half-line probe weight
exceeds 3/4; for each fixed g, norm convergence makes both finite weights exceed
1/2 for large N.

Every other finite eigenvalue lies outside both discs. The probe block
A0=((0,w),(-w,0)) is normal, so its resolvent and the probe equations give

    ||v_probe|| <= |g|/dist(lambda,{iw,-iw}) ||v_site1||
                <= |g|/delta ||v_buffer||.

Consequently every such right eigenvector has probe weight at most
g^2/(delta^2+g^2)<1/4, after a further reduction of g0. This does not assume
orthogonality of the eigenvectors of the full dissipative generator. The strict
weight separation proves B for the actual two-maximum-weight rule. Choose g0
first, then N0(g); no numerical threshold is inferred from a fitted slope.

## Reproducible algebra and its boundary

Run from the repository root, with Python 3.12 and no extra packages:

```sh
python verification/0064/check_local_sink_algebra.py --self-test
python -O verification/0064/check_local_sink_algebra.py --self-test
```

The checker verifies 13 identities over the integer Laurent-polynomial ring:
the four left equations, the H factorization, transfer determinant and
eigenvectors, reciprocal bulk equation, stable/unstable determinant,
coefficient numerator, terminal component, and zero-coupling simplification.
Each self-test also requires nonzero CLI exits for a changed bulk constant,
a flipped left probe sign, and altered model source. Explicit failures work
with assertions disabled. It prints JSON; `--output PATH` optionally saves it.
The explicit `--self-test` uses temporary files for the rejection controls.

Source binding is SHA256 of UTF-8 `modelA.py`, with CRLF and CR normalized to LF:
`0e7264dbaf7b2968b38fd01905a778a05b31723f7ad7dd50326f5f5161d1a402`.
No other whitespace, BOM or text changes are normalized. This detects drift of
the source used to derive the equations; it does not mechanically translate
that source into the equations. The source is never imported or executed.

The separate workflow checks algebra and rejection behavior only. It does not
verify Rouché's theorem applications, analytic estimates, norm limits, or the
selector proof by machine. It runs no historical suite, numerical spectrum,
optimizer or root solver and produces no new scientific data. The argument is
a local derivation of a standard mechanism; no novelty or independent review
is claimed.
