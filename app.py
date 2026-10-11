# app_live_retrieval.py
# Webcam -> YOLO (conf=0.80) -> Click -> Rectify -> Retrieve FAISS -> Show candidates
import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple
import glob

import cv2
import numpy as np
import pandas as pd
import faiss
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel

CARD_RATIO = 88.0 / 63.0
TARGET_W = 480
TARGET_H = int(TARGET_W * CARD_RATIO)

DEFAULT_INDEX_DIR = Path("data/tcgdex/index")
DEFAULT_INDEX_NAME = "cards_all"
CANDIDATES_K = 6

MODEL_ID = "google/siglip-base-patch16-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray,
                         dst_w: int = TARGET_W, dst_h: int = TARGET_H) -> np.ndarray:
    rect = order_points(pts.astype(np.float32))
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC)

def find_card_quad(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    img = crop_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 15, 5)
    edges = cv2.Canny(thr, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = img.shape[:2]
    img_area = w * h

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.10:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(-1, 2).astype(np.float32)
            rect = order_points(pts)
            w_est = np.linalg.norm(rect[1] - rect[0])
            h_est = np.linalg.norm(rect[3] - rect[0])
            if w_est == 0 or h_est == 0:
                continue
            ratio = max(w_est, h_est) / max(1.0, min(w_est, h_est))
            if 1.2 <= ratio <= 1.7:
                return pts
    return None

def rectify_from_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    pad = int(0.04 * max(x2 - x1, y2 - y1))
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    crop = frame[y1p:y2p, x1p:x2p].copy()
    if crop.size == 0:
        return None

    quad = find_card_quad(crop)
    if quad is not None:
        return four_point_transform(crop, quad, dst_w=TARGET_W, dst_h=TARGET_H)

    Hc, Wc = crop.shape[:2]
    pts = np.array([[0, 0], [Wc - 1, 0], [Wc - 1, Hc - 1], [0, Hc - 1]], dtype=np.float32)
    return four_point_transform(crop, pts, dst_w=TARGET_W, dst_h=TARGET_H)

def load_embedding_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    model.eval().to(DEVICE)
    assert hasattr(model, "get_image_features"), "Le modèle doit exposer get_image_features"
    return processor, model

@torch.no_grad()
def embed_bgr_image(img_bgr: np.ndarray, processor, model) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    inputs = processor(images=[pil], return_tensors="pt").to(DEVICE, dtype=DTYPE)
    feat = model.get_image_features(**inputs)  # [1, D]
    feat = torch.nn.functional.normalize(feat.float(), dim=1)
    return feat.cpu().numpy()

def load_index(index_dir: Path, index_name: str):
    idx_path = index_dir / f"{index_name}.faiss"
    meta_path = index_dir / f"{index_name}.parquet"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index introuvable: {idx_path} / {meta_path}")
    index = faiss.read_index(str(idx_path))
    meta = pd.read_parquet(meta_path)
    for col in ["filepath", "lang", "set_id", "localId", "card_id", "name"]:
        if col not in meta.columns:
            meta[col] = ""
    return index, meta

def resolve_image_path(row: pd.Series, images_root: Optional[str | Path]) -> Optional[Path]:
    fp = str(row.get("filepath", "") or "")
    lang = str(row.get("lang", "") or "")
    card_id = str(row.get("card_id", "") or "")

    p = Path(fp)
    if p.exists() and p.stat().st_size > 0:
        return p

    if images_root:
        root = Path(images_root)

        if fp:
            p2 = root / lang / Path(fp).name
            if p2.exists() and p2.stat().st_size > 0:
                return p2

        if fp:
            p3 = root / Path(fp).name
            if p3.exists() and p3.stat().st_size > 0:
                return p3

        if card_id:
            hits = glob.glob(str(root / "**" / f"*{card_id}*.*"), recursive=True)
            for h in hits:
                ph = Path(h)
                if ph.exists() and ph.stat().st_size > 0:
                    return ph

    return None

def get_font(size=22) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

FONT = get_font(22)

def draw_utf8_text_bgr(img_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    pad_x, pad_y = 8, 6
    x, y = 8, 8

    h_total = 0
    w_max = 0
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=FONT)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        h_total += h + 4
        w_max = max(w_max, w)

    rect = (x - pad_x, y - pad_y, x + w_max + pad_x, y + h_total + pad_y)
    draw.rectangle(rect, fill=(0, 0, 0, 180))

    yy = y
    for ln in lines:
        draw.text((x, yy), ln, font=FONT, fill=(255, 255, 255))
        bbox = draw.textbbox((0, 0), ln, font=FONT)
        h = bbox[3] - bbox[1]
        yy += h + 4

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def tile_candidates(meta_rows: pd.DataFrame, images_root: str = "",
                    thumb_h: int = 260, margin: int = 10, cols: int = 3) -> np.ndarray:
    tiles = []
    for _, r in meta_rows.iterrows():
        p = resolve_image_path(r, images_root)
        if p is not None:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        else:
            img = None

        if img is None or img.size == 0:
            img = np.zeros((thumb_h, int(thumb_h * 0.7), 3), dtype=np.uint8)

        # resize
        h, w = img.shape[:2]
        scale = thumb_h / max(1, h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Lignes UTF-8
        label = f"{r.get('lang','?')} | {r.get('set_id','')} #{r.get('localId','')}"
        score = r.get("score", None)
        if score is not None:
            label += f" | {float(score):.3f}"
        name = str(r.get("name", ""))
        img = draw_utf8_text_bgr(img, [label, name])

        # Cadre
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 255, 255), 2)
        tiles.append(img)

    if not tiles:
        return np.zeros((thumb_h, int(thumb_h * 0.7), 3), dtype=np.uint8)

    rows = (len(tiles) + cols - 1) // cols
    maxw = max(i.shape[1] for i in tiles)
    tile_h = rows * thumb_h + (rows + 1) * margin
    tile_w = cols * maxw + (cols + 1) * margin
    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    y = margin
    k = 0
    for _ in range(rows):
        x = margin
        for _ in range(cols):
            if k >= len(tiles):
                break
            im = tiles[k]; k += 1
            hh, ww = im.shape[:2]
            canvas[y:y + hh, x:x + ww] = im
            x += maxw + margin
        y += thumb_h + margin
    return canvas

class ClickAndRetrieve:
    def __init__(self, yolo_conf: float, preview_name: str,
                 index, meta: pd.DataFrame, processor, model, images_root: str):
        self.boxes: List[Tuple[int, int, int, int, float]] = []
        self.frame_ref: Optional[np.ndarray] = None
        self.last_rect: Optional[np.ndarray] = None
        self.yolo_conf = yolo_conf
        self.preview_name = preview_name
        self.index = index
        self.meta = meta
        self.processor = processor
        self.model = model
        self.images_root = images_root

    def update_detections(self, frame: np.ndarray, r0):
        self.frame_ref = frame
        if r0.boxes is not None and len(r0.boxes) > 0:
            b = r0.boxes
            confs = b.conf.cpu().numpy()
            mask = confs >= self.yolo_conf
            boxes = b.xyxy.cpu().numpy()[mask]
            confs = confs[mask]
        else:
            boxes = np.empty((0, 4)); confs = np.empty((0,))
        self.boxes = [(int(x1), int(y1), int(x2), int(y2), float(c))
                      for (x1, y1, x2, y2), c in zip(boxes, confs)]

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.frame_ref is None or not self.boxes:
            return
        hit = None; best = -1.0
        for (x1, y1, x2, y2, conf) in self.boxes:
            if x1 <= x <= x2 and y1 <= y <= y2 and conf > best:
                best = conf; hit = (x1, y1, x2, y2)
        if hit is None:
            return

        rectified = rectify_from_bbox(self.frame_ref, hit)
        if rectified is None:
            print("[INFO] Rectification échouée.")
            return

        self.last_rect = rectified
        cv2.imshow(self.preview_name, rectified)
        self.retrieve_and_show(rectified)

    def retrieve_and_show(self, rectified_bgr: np.ndarray):
        if self.index is None or self.meta is None or self.processor is None or self.model is None:
            print("[WARN] Retrieval non initialisé.")
            return

        q = embed_bgr_image(rectified_bgr, self.processor, self.model)  # [1,D]
        D, I = self.index.search(q, CANDIDATES_K)
        hits = self.meta.iloc[I[0]].copy()
        hits["score"] = D[0]

        resolved_paths = []
        for _, row in hits.iterrows():
            rp = resolve_image_path(row, self.images_root)
            resolved_paths.append(str(rp) if rp else "")
        hits["filepath_resolved"] = resolved_paths

        grid = tile_candidates(hits, images_root=self.images_root, thumb_h=260, cols=3)
        cv2.imshow("Candidates", grid)

        cols = ["lang", "set_id", "localId", "card_id", "name", "score", "filepath_resolved"]
        print(hits[cols].to_string(index=False))

def is_int(s: str) -> bool:
    try:
        int(s); return True
    except Exception:
        return False

def parse_args():
    ap = argparse.ArgumentParser(description="Webcam -> Click -> Rectify -> Retrieve")
    ap.add_argument("--model", required=True, help="Poids YOLO (.pt/.onnx) sur 'card'")
    ap.add_argument("--source", default="0", help="Index webcam (0) ou chemin vidéo/URL")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.80, help="Seuil de confiance YOLO")
    ap.add_argument("--index_dir", default=str(DEFAULT_INDEX_DIR))
    ap.add_argument("--index_name", default=DEFAULT_INDEX_NAME, help="cards_all ou cards_fr, etc.")
    ap.add_argument("--images_root", default="", help="Dossier racine où se trouvent les images (si déplacées)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    return ap.parse_args()

def main():
    args = parse_args()

    print(f"[INFO] YOLO -> {args.model}")
    det_model = YOLO(args.model)

    print(f"[INFO] Index -> {args.index_dir}/{args.index_name}.faiss")
    index, meta = load_index(Path(args.index_dir), args.index_name)

    print("[INFO] SigLIP chargement…")
    processor, enc_model = load_embedding_model()

    source_arg = int(args.source) if is_int(args.source) else args.source
    cap = cv2.VideoCapture(source_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source: {args.source}")

    if args.show:
        cv2.namedWindow("YOLO Live", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    handler = ClickAndRetrieve(
        yolo_conf=args.conf,
        preview_name="Rectified",
        index=index,
        meta=meta,
        processor=processor,
        model=enc_model,
        images_root=args.images_root
    )
    if args.show:
        cv2.setMouseCallback("YOLO Live", handler.on_mouse)

    print("[INFO] Hotkeys: q=quit")
    frame_count = 0
    fps_smoothed = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Fin du flux."); break

            frame_count += 1
            if args.stride > 1 and (frame_count % args.stride != 0):
                if args.show:
                    cv2.imshow("YOLO Live", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue

            t0 = time.time()
            results = det_model.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False
            )
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            fps_smoothed = fps if fps_smoothed == 0 else 0.9 * fps_smoothed + 0.1 * fps

            r0 = results[0]
            annotated = r0.plot()
            handler.update_detections(frame, r0)

            cv2.putText(annotated, f"FPS: {fps_smoothed:.1f} conf={args.conf:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow("YOLO Live", annotated)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()