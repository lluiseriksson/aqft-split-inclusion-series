# 0105 v2 — paquete de integración

## Estado: v2-LITE, RELEASE-READY (JSONs no conservados)
Sin placeholders: la tabla de variantes on-device, el MDE numérico y µ_all
quedan explícitamente diferidos a una "data revision". Lo que sí entra:
- Sesgo de B_fix y estabilidad del ordering demostrados sobre ground truth sintético (--demo, ≥7σ, sesgo ~25%).
- Cota de potencia SOLO con números publicados: para que un efecto del tamaño curado (~0.3) escapara al test, σ_chain ≳ 0.19 (~60% de la media) — implausible ⇒ el null excluye ley fuerte device-wide.
- Sección de reproducibilidad honesta: JSONs no preservados; Job IDs recuperables desde la cuenta IBM (service.job(<id>).result() en qiskit-ibm-runtime) mientras el proveedor los retenga. RECOMENDACIÓN: intentar recuperarlos y hacer la data revision con raw counts por circuito.

## Cambios v1 → v2 (sin cambios en datos ni Job IDs)
1. Tabla de robustez de fit para µ (B_fix v1 / B libre / B=1/4): el titular pasa a ser "ordering + separación sobreviven variantes".
2. Análisis de potencia (MDE) del test prerregistrado negativo: el null se convierte en cota — descarta ley fuerte device-wide, compatible con ley débil/mecanismo-específica.
3. Proxy µ_all (los 4 L medidos) contrastando µ04.
4. Caveats: factores t para SEM con n_rep=3; no-clipping del estimador drift-corrected.
5. Reproducibilidad: JSONs archivados en repo + script offline (los Job IDs mueren si IBM purga datos).
6. Refs de contexto: Sarovar et al., Quantum 4 (2020) 321; Tripathi et al., Phys. Rev. Applied 18 (2022) 024068.

## Validación del script (--demo, sintético con ground truth; log en verification/0105/demo_reference_output.txt)
- Recupera los µ verdaderos y expone el sesgo de B_fix (0.284 vs 0.222 real) manteniendo el ordering 3/3 variantes (≥7.3σ).
- Asociación nula detectada como nula; test prerregistrado nulo; MDE = 1.57·σ_chain√(2/k) ✓.

## README del repo — fila propuesta
| **0105** — *Prefix-Path Bell Transport on IBM Quantum Hardware* | `papers/0105-prefix-path-bell-transport/` | `verification/0105/` (re-análisis offline de JSONs archivados; NO regenera medidas) | Pata experimental del programa RIP (0070/0072): dependencia geométrica >10σ robusta a variantes de fit; vínculo estático–dinámico negativo con potencia cuantificada |

Honesty statement del README: añadir que la suite 0105 re-analiza medidas
archivadas (no las regenera), a diferencia del resto de la serie.

## Al integrar
- results/0105/README.md queda como marcador de la data revision pendiente.
- Regenerar SHA256SUMS del repo al añadir 0105.
- Data revision futura: recuperar resultados vía Job IDs desde la cuenta IBM, exportar raw counts, correr reanalyze_0105.py, rellenar tabla/MDE/µ_all.
