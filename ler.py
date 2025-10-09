# recortar_linhas_fix.py
# Entrada:  image_corrigida.png (mesma pasta)
# Saída:    rows_fix/Q01.png ... Q60.png
# Estratégia: acha os CENTROS das 15 linhas por bloco e corta uma janela de ALTURA FIXA.

from pathlib import Path
import cv2
import numpy as np

# ===== CONFIG =====
IMG_IN   = "image_corrigida.png"
OUT_DIR  = "rows_fix"

TOTAL_QUESTOES = 60

# 4 blocos (15 questões cada), em % do tamanho da imagem corrigida
BLOCOS = [
    {"q_ini": 1,  "q_fim": 15, "x1":0.060, "x2":0.260, "y1":0.300, "y2":0.920},
    {"q_ini": 16, "q_fim": 30, "x1":0.290, "x2":0.480, "y1":0.300, "y2":0.920},
    {"q_ini": 31, "q_fim": 45, "x1":0.520, "x2":0.725, "y1":0.300, "y2":0.920},
    {"q_ini": 46, "q_fim": 60, "x1":0.750, "x2":0.935, "y1":0.300, "y2":0.920},
]

# binarização simples para projeções
BLUR = 3
THRESH_BIN = 180

# detecção de picos VERTICAIS (centros das linhas)
# distância mínima entre picos ≈ 80% da altura teórica de cada linha
PEAK_DIST_FACTOR = 0.80
PEAK_VERT_THR_FRAC = 0.25  # fração do pico máximo

# janela de recorte vertical fixa por linha:
# half-height = WINDOW_HALF_FACTOR * (altura_teórica_da_linha)
WINDOW_HALF_FACTOR = 0.46   # 0.46 ≈ deixa uma gordurinha acima/abaixo

# refinamento horizontal por projeção
H_FRAC = 0.15   # fração do pico máximo do perfil horizontal
PAD_H  = 2      # “pelinho” extra nos lados

# (opcional) normalizar a altura final
RESIZE_HEIGHT = None  # ex.: 42 (None mantém)

# ==================
def to_abs(frac, total): 
    return int(round(frac * total))

def binarize(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if BLUR > 0:
        gray = cv2.GaussianBlur(gray, (BLUR, BLUR), 0)
    _, bw = cv2.threshold(gray, THRESH_BIN, 255, cv2.THRESH_BINARY_INV)
    return bw

def find_line_centers(bw_block, n_linhas, dist_min):
    """
    Acha os 15 centros de linha numa coluna usando projeção vertical e picos.
    Garante exatamente n_linhas centros: se faltar, completa por grade uniforme.
    """
    prof_v = bw_block.sum(axis=1).astype(np.float32)
    pv_max = prof_v.max()
    thr = pv_max * PEAK_VERT_THR_FRAC

    # picos simples
    cand = []
    for y in range(1, len(prof_v)-1):
        if prof_v[y] > prof_v[y-1] and prof_v[y] > prof_v[y+1] and prof_v[y] >= thr:
            cand.append((y, prof_v[y]))
    # ordena por altura e aplica supressão por distância
    cand.sort(key=lambda t: t[1], reverse=True)
    sel = []
    for y, val in cand:
        if all(abs(y - s) >= dist_min for s in sel):
            sel.append(y)
        if len(sel) >= n_linhas:
            break
    sel.sort()

    if len(sel) < n_linhas:
        # completa por grade uniforme (evita desalinhamento)
        Hb = bw_block.shape[0]
        passo = Hb / float(n_linhas)
        for i in range(n_linhas):
            y_uni = int(round((i + 1.5) * passo))
            sel.append(y_uni)
        sel = sorted(sel)[:n_linhas]

    return sel

def refine_horizontal(faixa_bgr):
    """Aperta horizontalmente (esq/dir) por projeção, sem mexer no y."""
    bw = binarize(faixa_bgr)
    prof_h = bw.sum(axis=0).astype(np.float32)
    ph_max = prof_h.max()
    if ph_max <= 0:
        return faixa_bgr

    thr = ph_max * H_FRAC
    cols = np.where(prof_h >= thr)[0]
    if cols.size == 0:
        return faixa_bgr

    left  = max(0, int(cols[0] - PAD_H))
    right = min(faixa_bgr.shape[1], int(cols[-1] + PAD_H + 1))
    return faixa_bgr[:, left:right].copy()

def recortar_linhas_fix():
    img_path = Path(IMG_IN)
    if not img_path.exists():
        print(f"[ERRO] Não achei {IMG_IN}.")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] Falha ao abrir {IMG_IN}.")
        return

    H, W = img.shape[:2]
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    for bloco in BLOCOS:
        # ROI do bloco
        x1 = to_abs(bloco["x1"], W); x2 = to_abs(bloco["x2"], W)
        y1 = to_abs(bloco["y1"], H); y2 = to_abs(bloco["y2"], H)
        col = img[y1:y2, x1:x2]
        Hb, Wb = col.shape[:2]

        n_linhas = bloco["q_fim"] - bloco["q_ini"] + 1
        # altura teórica da linha
        h_line = Hb / float(n_linhas)
        half = int(round(WINDOW_HALF_FACTOR * h_line))
        half = max(6, half)

        # centros por projeção vertical
        bw_col = binarize(col)
        dist_min = int(round(PEAK_DIST_FACTOR * h_line))
        centers = find_line_centers(bw_col, n_linhas, dist_min)

        for i, y_c_local in enumerate(centers):
            qnum = bloco["q_ini"] + i

            top = max(0, y_c_local - half)
            bot = min(Hb, y_c_local + half)

            faixa = col[top:bot, :].copy()
            faixa = refine_horizontal(faixa)

            if RESIZE_HEIGHT and faixa.size > 0:
                h, w = faixa.shape[:2]
                new_w = max(1, int(round(w * (RESIZE_HEIGHT / h))))
                faixa = cv2.resize(faixa, (new_w, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(Path(out_dir, f"Q{qnum:02d}.png")), faixa)

    print(f"[OK] {TOTAL_QUESTOES} faixas salvas em: {out_dir.resolve()}")

if __name__ == "__main__":
    recortar_linhas_fix()
