# detectar_bolhas_exportar.py
# Lê image.png, detecta somente bolhas PRETAS (preenchidas),
# grava bubbles.json (coordenadas em pixels da imagem original)
# e um preview opcional (bolinhas_preenchidas.png).
#
# Requisitos: pip install opencv-python numpy

import json
from pathlib import Path
import cv2
import numpy as np

IMG_IN   = "image.png"
PREVIEW  = "bolinhas_preenchidas.png"  # opcional, só para conferência
JSON_OUT = "bubbles.json"

# ===== Parâmetros de detecção =====
RESIZE_W      = 1600
HOUGH_PARAM1  = 100
HOUGH_PARAM2  = 20
MIN_DIST      = 18

# Classificação "preenchida" (interior bem mais escuro que o anel do entorno)
DARK_DROP_MIN = 25   # anel - interior (0–255)
INNER_MAX     = 170  # teto de luminância do interior (quanto menor, mais "preto")

R_IN_SCALE    = 0.80
R_RING_IN     = 1.05
R_RING_OUT    = 1.50
# ==================================

def resize_keep(img, target_w):
    if target_w <= 0: return img, 1.0
    h, w = img.shape[:2]
    s = target_w / float(w)
    return cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_CUBIC), s

def detect_circles(gray):
    H, W = gray.shape[:2]
    r_min = max(6,  int(min(H, W) * 0.010))
    r_max = max(r_min+6, int(min(H, W) * 0.028))
    g = cv2.GaussianBlur(gray, (5,5), 0)
    cs = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=r_min, maxRadius=r_max
    )
    return np.empty((0,3), dtype=np.float32) if cs is None else cs[0].astype(np.float32)

def circle_stats(gray, x, y, r):
    H, W = gray.shape[:2]
    r_in   = max(3, int(r * R_IN_SCALE))
    r_out1 = max(r_in+2, int(r * R_RING_IN))
    r_out2 = min(max(r_out1+2, int(r * R_RING_OUT)), int(min(H, W)*0.12))

    Y, X = np.ogrid[:H, :W]
    dist2 = (X - x)**2 + (Y - y)**2
    mask_in   = dist2 <= (r_in**2)
    mask_ring = (dist2 >= (r_out1**2)) & (dist2 <= (r_out2**2))
    if mask_in.sum() < 20 or mask_ring.sum() < 30:
        return None, None, -1e9
    inner = float(gray[mask_in].mean())
    ring  = float(gray[mask_ring].mean())
    score = (ring - inner)
    return inner, ring, score

def is_filled(inner, ring):
    return (ring - inner) >= DARK_DROP_MIN and inner <= INNER_MAX

def main():
    img_path = Path(IMG_IN)
    if not img_path.exists():
        raise FileNotFoundError(f"Não encontrei {IMG_IN} no diretório atual.")
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Falha ao abrir {IMG_IN}")

    work, scale = resize_keep(img, RESIZE_W)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    circles = detect_circles(gray)
    resultados = []
    out = img.copy()

    for (x, y, r) in circles:
        inner, ring, score = circle_stats(gray, int(round(x)), int(round(y)), int(round(r)))
        if inner is None:
            continue
        if is_filled(inner, ring):
            # volta para coordenadas da imagem ORIGINAL
            cx = float(x / scale)
            cy = float(y / scale)
            rr = float(r / scale)
            resultados.append({"x": cx, "y": cy, "r": rr, "score": float(score)})

    # (opcional) desenha preview
    for c in resultados:
        cv2.circle(out, (int(round(c["x"])), int(round(c["y"]))), max(4, int(round(c["r"]*0.75))), (0,0,255), -1)
        cv2.circle(out, (int(round(c["x"])), int(round(c["y"]))), max(5, int(round(c["r"]*0.90))), (0,0,255), 2)
    cv2.imwrite(PREVIEW, out)

    # salva JSON (apenas x,y,r; score é útil pra debug, remova se quiser)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump({"image": IMG_IN, "circles": resultados}, f, ensure_ascii=False, indent=2)

    print(f"✅ Detectadas {len(resultados)} bolhas preenchidas.")
    print(f"- Preview: {PREVIEW}")
    print(f"- JSON:    {JSON_OUT}")

if __name__ == "__main__":
    main()
