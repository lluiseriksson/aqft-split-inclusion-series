# 04_package_and_manifest.py
import os, json, hashlib, zipfile, datetime

PROJECT_DIR = "/content/drive/MyDrive/Ising_Project"

FILES = [
    ("moneyplot_CMI_xilocal.png",
     os.path.join(PROJECT_DIR, "moneyplot_CMI_xilocal.png")),
    ("summary.tex",
     os.path.join(PROJECT_DIR, "summary.tex")),
    ("cmi_h1p5_chi128_semiinf_STREAM.jsonl",
     os.path.join(PROJECT_DIR, "data/cmi_h1p5_chi128_semiinf_STREAM.jsonl")),
    ("cmi_from256_h1p005_chi512_semiinf_STREAM.jsonl",
     os.path.join(PROJECT_DIR,
                  "data/cmi_from256_h1p005_chi512_semiinf_STREAM.jsonl")),
]

ZIP_PATH = os.path.join(PROJECT_DIR, "overleaf_assets_exact.zip")
MANIFEST_PATH = os.path.join(PROJECT_DIR, "repro_manifest.json")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

# verify files
out = []
for arc, src in FILES:
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    size = os.path.getsize(src)
    if size == 0:
        raise RuntimeError(f"0-byte file: {src}")
    out.append({"arcname": arc, "src": src, "bytes": size,
                "sha256": sha256(src)})

# zip root-clean (exactly 4 files at root)
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for item in out:
        z.write(item["src"], arcname=item["arcname"])

print("Wrote zip:", ZIP_PATH, "bytes:", os.path.getsize(ZIP_PATH))

# append manifest info
manifest = {}
if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

manifest["outputs_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
manifest["overleaf_root_files"] = out
manifest["zip"] = {"path": ZIP_PATH, "bytes": os.path.getsize(ZIP_PATH),
                   "sha256": sha256(ZIP_PATH)}

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)

print("Updated manifest:", MANIFEST_PATH)
print("ZIP sha256:", manifest["zip"]["sha256"])
