# marcar_bolinhas_preenchidas.py
# Lê image.png e salva bolinhas_preenchidas.png com somente as bolhas ESCURAS marcadas em vermelho.
# Requisitos: pip install opencv-python numpy

import cv2
import numpy as np

IMG_IN  = "image_cortada.png"
IMG_OUT = "bolinhas_preenchidas.png"

# ---------- PARÂMETROS AJUSTÁVEIS ----------
RESIZE_W      = 1600   # largura de trabalho p/ detecção (0 = sem redimensionar)
HOUGH_PARAM1  = 100    # Canny alta
HOUGH_PARAM2  = 20     # acumulação (aumente p/ menos círculos; diminua p/ mais)
MIN_DIST      = 18     # distância mínima entre centros
DARK_DROP_MIN = 25     # quão mais escuro o interior deve ser vs anel externo (0–255)
INNER_MAX     = 170    # teto de intensidade média do interior (quanto menor, mais “preto”)
R_IN_SCALE    = 0.80   # raio do interior (fração de r)
R_RING_IN     = 1.05   # anel externo interno  (r * R_RIN_IN)
R_RING_OUT    = 1.50   # anel externo externo  (r * R_RING_OUT)
# --------------------------------------------

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

def filled_filter(gray, cx, cy, r):
    """
    Retorna True se a bolha estiver preenchida:
    - média do interior (disco) *bem* menor que a média do anel externo
    - e também abaixo de um teto absoluto (INNER_MAX)
    """
    H, W = gray.shape[:2]
    # máscaras
    r_in   = max(3, int(r * R_IN_SCALE))
    r_out1 = max(r_in+2, int(r * R_RING_IN))
    r_out2 = min(max(r_out1+2, int(r * R_RING_OUT)), int(min(H, W)*0.12))

    Y, X = np.ogrid[:H, :W]
    dist2 = (X - cx)**2 + (Y - cy)**2

    mask_in   = dist2 <= (r_in**2)
    mask_ring = (dist2 >= (r_out1**2)) & (dist2 <= (r_out2**2))

    if mask_in.sum() < 20 or mask_ring.sum() < 30:
        return False

    inner_mean = float(gray[mask_in].mean())
    ring_mean  = float(gray[mask_ring].mean())

    return (ring_mean - inner_mean) >= DARK_DROP_MIN and inner_mean <= INNER_MAX

def main():
    img = cv2.imread(IMG_IN)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {IMG_IN}")

    work, scale = resize_keep(img, RESIZE_W)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    # melhora contraste local
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    circles = detect_circles(gray)
    if circles.size == 0:
        print("Nenhuma bolha candidata detectada. Tente diminuir HOUGH_PARAM2 ou aumentar RESIZE_W.")
    # desenhar somente as preenchidas
    out = img.copy()
    marcadas = 0
    for (x, y, r) in circles:
        cx = int(round(x / scale))
        cy = int(round(y / scale))
        rr = float(r / scale)

        # Avalia preenchimento na imagem de trabalho (coordenadas escaladas)
        if filled_filter(gray, int(round(x)), int(round(y)), int(round(r))):
            marcadas += 1
            # ponto vermelho preenchido + aro
            cv2.circle(out, (cx, cy), max(4, int(rr*0.65)), (0,0,255), thickness=-1)
            cv2.circle(out, (cx, cy), max(5, int(rr*0.85)), (0,0,255), thickness=2)

    cv2.imwrite(IMG_OUT, out)
    print(f"✅ Bolhas marcadas: {marcadas} | Arquivo salvo em: {IMG_OUT}")
    print("Ajustes úteis: DARK_DROP_MIN (diferença interior vs anel), INNER_MAX (teto de escuridão), HOUGH_PARAM2.")

if __name__ == "__main__":
    main()
