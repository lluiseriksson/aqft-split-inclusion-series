# results/0105 — data revision pendiente (JSONs no conservados; recuperables vía Job IDs desde la cuenta IBM)

Colocar aquí los JSON de análisis exportados de los runs en ibm_fez:
- rip_comparative_fit_v3_fixB.json   (3 cadenas: L, F_mean, F_sem)
- rip_strong_scaleup_v2.json         (18 cadenas: S_stat_corr, F_mean_by_L)
- rip_predictive_dynamic.json        (grupos LOW/HIGH: mu04 por cadena)

Esquema canónico documentado en verification/0105/reanalyze_0105.py (docstring).
Si los campos difieren, adaptar solo las tres funciones load_*.

Ejecutar después:
    python3 verification/0105/reanalyze_0105.py --data-dir results/0105 --output-dir papers/0105-prefix-path-bell-transport
y recompilar el tex.
