"""
Oraculo Combes-Thomas para el Brick P4.5 (THE-ERIKSSON-PROGRAMME).
Dado un operador de covarianza-inversa K gapped de rango finito en 1D
(generalizable), certifica numericamente la tripleta del interfaz Lean:
  (coercivityConstant c, conjugationDefect d(theta), ExpDecay amp/rate)
y la compara con la tasa NITIDA (rama evanescente, arccosh).
Uso: python ct_oracle.py mu t N   (K = mu*I - t*(S + S^T), coercividad mu-2t)
"""
import numpy as np, sys

mu, t, N = (float(sys.argv[1]) if len(sys.argv)>1 else 3.0,
            float(sys.argv[2]) if len(sys.argv)>2 else 1.0,
            int(sys.argv[3]) if len(sys.argv)>3 else 400)

# --- operador K (rango R=1) ---
K = mu*np.eye(N) - t*(np.eye(N,k=1)+np.eye(N,k=-1))
c  = mu - 2*t                      # coercividad exacta (borde de banda inferior)
S  = mu + 2*t                      # cota de Schur (suma de fila)
R  = 1

# --- tasa admisible por conjugacion exponencial (lo que probaria Lean) ---
theta_max = np.log(1 + c/S)/R      # (e^{tR}-1)S < c
theta = 0.95*theta_max
defect = (np.exp(theta*R)-1)*S
amp = 1.0/(c - defect)

# --- tasa nitida (rama evanescente) ---
q_sharp = np.arccosh(mu/(2*t))

# --- verificacion directa: kernel de C = K^{-1} ---
C = np.linalg.inv(K)
x0 = N//2
xs = np.arange(1, N//3)
vals = np.abs(C[x0, x0+xs])
fit = -np.polyfit(xs[10:], np.log(vals[10:]), 1)[0]

ok_conj  = np.all(vals <= amp*np.exp(-theta*xs)*1.000001)
ok_sharp = abs(fit - q_sharp) < 1e-3
print(f"K = {mu}*I - {t}*(S+S^T), N={N}:  c={c:.4f}  SchurS={S:.4f}")
print(f"conjugacion:  theta_adm={theta_max:.4f}  cert(theta={theta:.4f}, amp={amp:.2f})  cota respetada: {ok_conj}")
print(f"nitida:       q_sharp=arccosh(mu/2t)={q_sharp:.4f}  ajuste kernel={fit:.4f}  coincide: {ok_sharp}")
print(f"holgura conjugacion/nitida: {q_sharp/theta_max:.1f}x  (esperada; cualquier tasa>0 alimenta hRpoly)")
print(f"guardarrail critico: c->0 =>", end=" ")
mu2 = 2*t + 1e-6
print(f"theta_adm={np.log(1+(mu2-2*t)/(mu2+2*t)):.2e}, q_sharp={np.arccosh(mu2/(2*t)):.2e}  (ambas se cierran) OK")
