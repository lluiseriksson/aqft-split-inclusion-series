# 01_dmrg_ground_states.py
import os, time
import numpy as np

from tenpy.tools import hdf5_io
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

PROJECT_DIR = "/content/drive/MyDrive/Ising_Project"
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

H_GAPPED = 1.5
H_NEAR = 1.005

CHI_GAPPED = 128
CHI_BASE = 256
CHI_TARGET = 512

CKPT_GAPPED = os.path.join(CKPT_DIR, "GS_iDMRG_h1p5_chi128.h5")
CKPT_NEAR256 = os.path.join(CKPT_DIR, "GS_iDMRG_h1p005_chi256.h5")
CKPT_NEAR512 = os.path.join(CKPT_DIR, "GS_from256_h1p005_chi512.h5")

def build_model(h):
    return TFIChain({
        "L": 2,
        "J": 1.0,
        "g": float(h),
        "bc_MPS": "infinite",
        "conserve": "best",
    })

def init_psi(M):
    # deterministic product state
    return MPS.from_product_state(M.lat.mps_sites(), ["up"]*M.lat.N_sites,
                                  bc=M.lat.bc_MPS)

def run_dmrg(M, psi, dmrg_params):
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E0, psi = eng.run()
    psi.canonical_form()
    return float(E0), psi

def save_ckpt(path, psi, meta):
    hdf5_io.save({"psi": psi, **meta}, path)

def load_ckpt(path):
    obj = hdf5_io.load(path)
    psi = obj["psi"] if isinstance(obj, dict) and "psi" in obj else obj
    psi.canonical_form()
    return psi, obj if isinstance(obj, dict) else {}

def chi_eff(psi):
    chi = getattr(psi, "chi", None)
    return int(np.max(np.asarray(chi))) if chi is not None else None

def xi_corr(psi):
    xi = psi.correlation_length2()
    return float(np.max(np.asarray(xi, dtype=float)))

def make_gapped():
    if os.path.exists(CKPT_GAPPED):
        return
    M = build_model(H_GAPPED)
    psi = init_psi(M)
    params = {
        "mixer": True,
        "mixer_params": {"amplitude": 1e-4, "decay": 2.0,
                         "disable_after": 40},
        "trunc_params": {"chi_max": int(CHI_GAPPED), "svd_min": 1e-12},
        "max_sweeps": 80,
        "max_E_err": 1e-10,
    }
    t0 = time.time()
    E0, psi = run_dmrg(M, psi, params)
    meta = {
        "h": float(H_GAPPED),
        "chi_max": int(CHI_GAPPED),
        "E0": float(E0),
        "chi_eff": chi_eff(psi),
        "xi_corr": xi_corr(psi),
        "runtime_sec": float(time.time() - t0),
    }
    save_ckpt(CKPT_GAPPED, psi, meta)

def make_near_base_256():
    if os.path.exists(CKPT_NEAR256):
        return
    M = build_model(H_NEAR)
    psi = init_psi(M)
    params = {
        "mixer": True,
        "mixer_params": {"amplitude": 1e-4, "decay": 2.0,
                         "disable_after": 60},
        "trunc_params": {"chi_max": int(CHI_BASE), "svd_min": 1e-12},
        "max_sweeps": 140,
        "max_E_err": 1e-11,
    }
    t0 = time.time()
    E0, psi = run_dmrg(M, psi, params)
    meta = {
        "h": float(H_NEAR),
        "chi_max": int(CHI_BASE),
        "E0": float(E0),
        "chi_eff": chi_eff(psi),
        "xi_corr": xi_corr(psi),
        "runtime_sec": float(time.time() - t0),
    }
    save_ckpt(CKPT_NEAR256, psi, meta)

def ramp_near_to_512_from_256():
    if os.path.exists(CKPT_NEAR512):
        return
    psi256, meta256 = load_ckpt(CKPT_NEAR256)
    M = build_model(H_NEAR)
    params = {
        "mixer": True,
        "mixer_params": {"amplitude": 1e-3, "decay": 1.5,
                         "disable_after": 80},
        "trunc_params": {"chi_max": int(CHI_TARGET), "svd_min": 1e-12},
        "chi_list": {0: 128, 20: 256, 40: 384, 60: 512},
        "max_sweeps": 180,
        "max_E_err": 1e-12,
    }
    t0 = time.time()
    E0, psi = run_dmrg(M, psi256, params)
    meta = {
        "h": float(H_NEAR),
        "chi_max": int(CHI_TARGET),
        "E0": float(E0),
        "chi_eff": chi_eff(psi),
        "xi_corr": xi_corr(psi),
        "source_ckpt": CKPT_NEAR256,
        "source_meta": {k: meta256.get(k) for k in
                        ["E0", "chi_eff", "xi_corr", "chi_max", "h"]},
        "runtime_sec": float(time.time() - t0),
    }
    save_ckpt(CKPT_NEAR512, psi, meta)

make_gapped()
make_near_base_256()
ramp_near_to_512_from_256()

print("Ready checkpoints:")
print(" -", CKPT_GAPPED)
print(" -", CKPT_NEAR256)
print(" -", CKPT_NEAR512)
