import numpy as np
import pandas as pd

E = np.load("embeddings/embeddings_all.npy").astype("float32")
M = pd.read_parquet("index/meta_all.parquet")

n = min(E.shape[0], len(M))
E = E[:n]
M = M.iloc[:n].reset_index(drop=True)

E.tofile("index/embeddings_all.f32.bin")

with open("index/meta_all.jsonl", "w", encoding="utf-8") as f:
    for i, row in M.iterrows():
        obj = {
            "id": int(i),
            "lang": str(row.get("lang", "")),
            "set_id": str(row.get("set_id", "")),
            "localId": str(row.get("localId", "")),
            "card_id": str(row.get("card_id", "")),
            "name": str(row.get("name", "")),
            "image_base": str(row.get("image", "")),
        }
        import json
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")