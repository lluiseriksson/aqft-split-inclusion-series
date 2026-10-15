# Spectral resource boundary verification

This directory verifies the finite examples in *Positive Spectral Measures as
Operational Resource Boundaries*.

Run from the repository root:

```bash
python verification/spectral_resource_boundary/generate_artifacts.py --check
python verification/spectral_resource_boundary/verify_artifact.py
```

The tracked JSON artifact is deterministic.  Its exact layer uses rational
arithmetic and contains:

- a Hausdorff moment matrix for a three-atom transfer spectrum;
- an exact positive atomic Gram decomposition for the correct support edge;
- an exact rational negative witness for a false support edge;
- an exact resolvent value and gap bound.

The floating-point layer checks the continuous-time exponential envelope, the
failure of an overstated gap, the doubled asymptotic slope after squaring an
amplitude, and the separation between impossibility and affordability
horizons.

These are finite certificates and toy-model checks.  They do not prove an
interacting rate-inheritance theorem, a Yang--Mills mass gap, or the Riemann
hypothesis.
