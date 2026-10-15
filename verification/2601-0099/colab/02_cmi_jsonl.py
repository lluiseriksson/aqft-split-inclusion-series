# 02_cmi_jsonl.py
import os, json, time
import numpy as np
from tenpy.tools import hdf5_io

PROJECT_DIR = "/content/drive/MyDrive/Ising_Project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)

W_MAX = 12

CKPT_GAPPED = os.path.join(CKPT_DIR, "GS_iDMRG_h1p5_chi128.h5")
CKPT_NEAR512 = os.path.join(CKPT_DIR, "GS_from256_h1p005_chi512.h5")

OUT_GAPPED = os.path.join(DATA_DIR, "cmi_h1p5_chi128_semiinf_STREAM.jsonl")
OUT_NEAR = os.path.join(DATA_DIR,
                        "cmi_from256_h1p005_chi512_semiinf_STREAM.jsonl")

def load_psi(path):
    obj = hdf5_io.load(path)
    psi = obj["psi"] if isinstance(obj, dict) and "psi" in obj else obj
    psi.canonical_form()
    return psi

def S_cut_mean(psi):
    return float(np.mean(np.asarray(psi.entanglement_entropy(), dtype=float)))

def S_segment(psi, w, first_site):
    seg = list(range(w))
    out = psi.entanglement_entropy_segment(segment=seg,
                                           first_site=[int(first_site)])
    return float(out[0])

def CMI_semiinf(psi, w):
    # operational implementation of I(A:C|B)=2*S_cut - S(B),
    # averaged over the two inequivalent cuts for a 2-site unit cell.
    Sc = S_cut_mean(psi)
    I0 = 2.0*Sc - S_segment(psi, w, 0)
    I1 = 2.0*Sc - S_segment(psi, w, (1 % max(1, psi.L)))
    return float(0.5*(I0 + I1)), float(I0), float(I1), float(Sc)

def write_stream(out_path, ckpt_path, psi):
    if os.path.exists(out_path):
        os.remove(out_path)
    for w in range(1, W_MAX + 1):
        t0 = time.time()
        I, I0, I1, Sc = CMI_semiinf(psi, w)
        rec = {
            "type": "point",
            "mode": "semi_infinite",
            "ckpt": ckpt_path,
            "w": int(w),
            "I": float(I),
            "I0": float(I0),
            "I1": float(I1),
            "S_cut_mean": float(Sc),
            "dt_sec": float(time.time() - t0),
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print("Wrote:", out_path)

psi_g = load_psi(CKPT_GAPPED)
psi_n = load_psi(CKPT_NEAR512)

write_stream(OUT_GAPPED, CKPT_GAPPED, psi_g)
write_stream(OUT_NEAR, CKPT_NEAR512, psi_n)
