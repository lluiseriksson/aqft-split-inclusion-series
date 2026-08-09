# Positive Spectral Measures as Operational Resource Boundaries

This directory contains the source and compiled PDF for the standalone paper
*Positive Spectral Measures as Operational Resource Boundaries: Correlations,
Resolvents, Mass Gaps, and Rate Inheritance*.

Compile with:

```bash
pdflatex positive_spectral_resource_boundaries.tex
pdflatex positive_spectral_resource_boundaries.tex
```

The paper's deterministic verification suite lives in
[`../../verification/spectral_resource_boundary/`](../../verification/spectral_resource_boundary/).
It includes exact rational moment/support certificates, an independent
checker, and deterministic continuous-time stress tests.

Claim boundary: the paper proves an abstract spectral-measure framework and
finite certificates.  Its Yang--Mills and open-system applications are
explicitly conditional; its comparison with prime resolvents is structural
only.  It does not claim the Yang--Mills mass gap or the Riemann hypothesis.
