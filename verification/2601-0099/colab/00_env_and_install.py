# 00_env_and_install.py
import os, sys, json, platform, subprocess, importlib.util, datetime

def sh(cmd):
    print("+", cmd)
    subprocess.check_call(cmd, shell=True)

PROJECT_DIR = "/content/drive/MyDrive/Ising_Project"
os.makedirs(PROJECT_DIR, exist_ok=True)

# Pin TeNPy to a known tag
if importlib.util.find_spec("tenpy") is None:
    sh("pip -q install git+https://github.com/tenpy/tenpy.git@v1.1.0")

# Ensure core deps exist (Colab already has these; keep minimal to avoid conflicts)
for pkg in ["numpy", "matplotlib"]:
    if importlib.util.find_spec(pkg) is None:
        sh(f"pip -q install {pkg}")

import numpy as np
import matplotlib
import tenpy

manifest = {
    "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "python": sys.version,
    "platform": platform.platform(),
    "tenpy_version": getattr(tenpy, "__version__", "unknown"),
    "matplotlib_version": matplotlib.__version__,
    "numpy_version": np.__version__,
}

# Save manifest header (more fields get appended later)
with open(os.path.join(PROJECT_DIR, "repro_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)

# Freeze full environment (best-effort; large but useful later)
sh(f"pip freeze > {os.path.join(PROJECT_DIR, 'pip_freeze.txt')}")
print("Wrote:", os.path.join(PROJECT_DIR, "repro_manifest.json"))
print("Wrote:", os.path.join(PROJECT_DIR, "pip_freeze.txt"))
