import subprocess, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
# quick sensitivity: sigma_z floor row, varying N and dOm via sed-copies
base = (HERE / 'verify_0070.py').read_text()
runs = [("N5_dOm05",  base.replace("N = 6","N = 5")),
        ("N6_dOm10",  base.replace("dOm = 0.05","dOm = 0.1")),
        ("N6_thr6",   base.replace("keep = gw > 1e-10*gw.max()","keep = gw > 1e-6*gw.max()"))]
for name, src in runs:
    src = src.replace('for name,S in [("sigma_x (energy exchange)",op_at(sx,j0)),\n               ("sigma_z (near-zero freq)", op_at(sz,j0))]:',
                      'for name,S in [("sigma_z (near-zero freq)", op_at(sz,j0))]:')
    src = src.split('print("\\n=== C:')[0]
    (HERE / f'tmp_{name}.py').write_text(src)
    out = subprocess.run([sys.executable, str(HERE / f'tmp_{name}.py')], capture_output=True, text=True, timeout=120).stdout
    print(f"### {name}"); print(out.strip().split('---')[-1])
