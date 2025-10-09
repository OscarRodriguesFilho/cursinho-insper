# marcar_bolinhas_preenchidas_v2.py
# Marca apenas as bolhas PRETAS e elimina duplicatas com NMS baseado no contraste.
# Requisitos: opencv-python numpy

import cv2
import numpy as np

IMG_IN  = "image.png"
IMG_OUT = "bolinhas_preenchidas.png"

# ---------- PARÂMETROS ----------
RESIZE_W      = 1600
HOUGH_PARAM1  = 100
HOUGH_PARAM2  = 20
MIN_DIST      = 18

DARK_DROP_MIN = 25   # anel - interior (quanto maior, mais “preenchida”)
INNER_MAX     = 170  # teto de luminância do interior
R_IN_SCALE    = 0.80
R_RING_IN     = 1.05
R_RING_OUT    = 1.50

# NMS: distância (em px da imagem de TRABALHO) abaixo da qual 2 detecções são consideradas duplicatas
NMS_DIST_FACTOR = 0.65   # multiplicado pelo raio médio estimado
# --------------------------------

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
    """Retorna (inner_mean, ring_mean, score)."""
    H, W = gray.shape[:2]
    r_in   = max(3, int(r * R_IN_SCALE))
    r_out1 = max(r_in+2, int(r * R_RING_IN))
    r_out2 = min(max(r_out1+2, int(r * R_RING_OUT)), int(min(H,W)*0.12))

    Y, X = np.ogrid[:H, :W]
    dist2 = (X - x)**2 + (Y - y)**2
    mask_in   = dist2 <= (r_in**2)
    mask_ring = (dist2 >= (r_out1**2)) & (dist2 <= (r_out2**2))
    if mask_in.sum() < 20 or mask_ring.sum() < 30:
        return None, None, -1e9
    inner = float(gray[mask_in].mean())
    ring  = float(gray[mask_ring].mean())
    score = (ring - inner)  # maior = mais “preto” dentro
    return inner, ring, score

def is_filled(inner, ring):
    return (ring - inner) >= DARK_DROP_MIN and inner <= INNER_MAX

def nms_keep_best(candidates, thr):
    """
    candidates: lista de dicts com keys x,y,r,score
    thr: distância para mesclagem
    Mantém apenas o de maior score em cada grupo de sobreposição.
    """
    if not candidates: return []
    # ordena por score desc
    candidates.sort(key=lambda d: d["score"], reverse=True)
    kept = []
    used = np.zeros(len(candidates), dtype=bool)
    for i, a in enumerate(candidates):
        if used[i]: continue
        kept.append(a); used[i] = True
        ax, ay = a["x"], a["y"]
        for j in range(i+1, len(candidates)):
            if used[j]: continue
            bx, by = candidates[j]["x"], candidates[j]["y"]
            if (ax-bx)**2 + (ay-by)**2 <= thr**2:
                used[j] = True  # suprime duplicata pior
    return kept

def main():
    img = cv2.imread(IMG_IN)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {IMG_IN}")

    work, scale = resize_keep(img, RESIZE_W)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    circles = detect_circles(gray)
    if circles.size == 0:
        print("Nenhuma bolha candidata. Tente diminuir HOUGH_PARAM2 ou aumentar RESIZE_W.")
        cv2.imwrite(IMG_OUT, img)
        return

    # calcula estatísticas e filtra por “preenchida”
    cand = []
    for (x, y, r) in circles:
        inner, ring, score = circle_stats(gray, int(round(x)), int(round(y)), int(round(r)))
        if inner is None: 
            continue
        if is_filled(inner, ring):
            cand.append({"x": float(x), "y": float(y), "r": float(r), "score": float(score)})

    if not cand:
        print("Não foram encontradas bolhas preenchidas com os thresholds atuais.")
        cv2.imwrite(IMG_OUT, img)
        return

    # NMS anti-duplicata
    r_med = np.median([c["r"] for c in cand])
    thr = max(6.0, float(r_med) * NMS_DIST_FACTOR)
    cand = nms_keep_best(cand, thr)

    # desenha no original
    out = img.copy()
    for c in cand:
        cx = int(round(c["x"] / scale))
        cy = int(round(c["y"] / scale))
        rr = int(round((c["r"] / scale) * 0.75))
        cv2.circle(out, (cx, cy), max(4, rr), (0,0,255), -1)
        cv2.circle(out, (cx, cy), max(5, rr+2), (0,0,255), 2)

    cv2.imwrite(IMG_OUT, out)
    print(f"✅ Bolhas marcadas (sem duplicatas): {len(cand)} | Salvo em: {IMG_OUT}")
    print("Se ainda ocorrer duplicata, aumente NMS_DIST_FACTOR (ex.: 0.8).")

if __name__ == "__main__":
    main()
