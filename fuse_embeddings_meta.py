from pathlib import Path
import numpy as np
import pandas as pd
import json

EMB_DIR = Path("embeddings")

LANGS = ["fr", "en", "ja", "zh_tw"]

all_embs = []
all_meta = []

for lang in LANGS:
    emb_path = EMB_DIR / f"embeddings_{lang}.npy" 
    meta_path = EMB_DIR / f"meta_{lang}.parquet"
    if not emb_path.exists() or not meta_path.exists():
        print(f"[WARN] Fichiers manquants pour {lang}, je saute.")
        print(f"       emb:  {emb_path}")
        print(f"       meta: {meta_path}")
        continue

    print(f"[INFO] Chargement {lang} : {emb_path.name} + {meta_path.name}")
    E = np.load(emb_path).astype("float32")
    M = pd.read_parquet(meta_path)

    if len(M) != E.shape[0]:
        print(f"[WARN] {lang}: len(meta)={len(M)} != len(emb)={E.shape[0]}, on tronque au min.")
        n = min(len(M), E.shape[0])
        E = E[:n]
        M = M.iloc[:n].reset_index(drop=True)

    M = M.copy()
    M["lang"] = lang

    all_embs.append(E)
    all_meta.append(M)

if not all_embs:
    raise RuntimeError("Aucun embedding chargé, vérifie LANGS et les chemins dans EMB_DIR.")

E_all = np.concatenate(all_embs, axis=0)
M_all = pd.concat(all_meta, axis=0, ignore_index=True)

M_all = M_all.reset_index(drop=True)
M_all["id"] = M_all.index.astype(int)

print(f"[INFO] Fusion : N_total={E_all.shape[0]}, dim={E_all.shape[1]}")

norms = np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-8
E_all = E_all / norms

OUT_DIR = Path("assets/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

emb_bin_path = OUT_DIR / "embeddings_all.f32.bin"
print(f"[INFO] Écriture embeddings -> {emb_bin_path}")
E_all.astype("float32").tofile(emb_bin_path)

meta_parquet_path = OUT_DIR / "meta_all.parquet"

cols_str = ["lang", "set_id", "localId", "card_id", "name"]
for c in cols_str:
    if c in M_all.columns:
        M_all[c] = M_all[c].astype(str)

print(f"[INFO] Écriture meta parquet -> {meta_parquet_path}")
M_all.to_parquet(meta_parquet_path, index=False)

meta_jsonl_path = OUT_DIR / "meta_all.jsonl"
print(f"[INFO] Écriture meta JSONL -> {meta_jsonl_path}")

with open(meta_jsonl_path, "w", encoding="utf-8") as f:
    for _, row in M_all.iterrows():
        obj = {
            "id": int(row["id"]),
            "lang": str(row.get("lang", "")),
            "set_id": str(row.get("set_id", "")),
            "localId": str(row.get("localId", "")),
            "card_id": str(row.get("card_id", "")),
            "name": str(row.get("name", "")),
            "image_base": str(row.get("image_base", row.get("image", ""))),
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("[INFO] Fusion embeddings + meta terminée.")