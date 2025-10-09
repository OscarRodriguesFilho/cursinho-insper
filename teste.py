# marcar_pontos_cartao_cursinho.py
# Uso:
#   python marcar_pontos_cartao_cursinho.py --img image.png \
#     --out pontos_detectados.png --corrigida image_corrigida.png --debug 1 --txt ocr.txt
#
# Requisitos: opencv-python, numpy, pytesseract (Tesseract OCR instalado)

import argparse, sys, unicodedata
from pathlib import Path
import cv2
import numpy as np
from math import atan2, degrees

# --- Tesseract ---
try:
    import pytesseract
    TESS_OK = True
    TESSERACT_DIR = r"C:\Users\USER\Documents\Projeto OCR\ocara\Tesseract-OCR"
    exe = Path(TESSERACT_DIR) / "tesseract.exe"
    if exe.exists():
        pytesseract.pytesseract.tesseract_cmd = str(exe)
except Exception:
    TESS_OK = False

def ler_imagem(p):
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {p}")
    return img

def sem_acentos(s: str) -> str:
    nf = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nf if not unicodedata.combining(ch))

def normaliza_token(s: str) -> str:
    s = sem_acentos((s or "").strip().upper())
    return s.replace("–","-").replace("—","-").replace("−","-")

def ocr_data(img, lang="por+eng"):
    cfg = r"--oem 3 --psm 6"
    return pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)

def ocr_text(img, lang="por+eng"):
    cfg = r"--oem 3 --psm 6"
    return pytesseract.image_to_string(img, lang=lang, config=cfg)

def encontrar_cartao_resposta_preciso(data):
    """Encontra a(s) caixa(s) apenas de 'CARTAO-RESPOSTA' (ou 'CARTAO RESPOSTA')."""
    n = len(data["text"])
    linhas = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        linhas.setdefault(key, []).append(i)

    padroes = [
        ["CARTAO-RESPOSTA"],
        ["CARTAO", "RESPOSTA"],
    ]

    for key, idxs in linhas.items():
        idxs.sort(key=lambda k: data["left"][k])
        toks = [normaliza_token(data["text"][k]) for k in idxs]

        # padrão de 1 token
        for j, t in enumerate(toks):
            if t == "CARTAO-RESPOSTA" or "CARTAO-RESPOSTA" in t:
                k = idxs[j]
                x, y, w, h = data["left"][k], data["top"][k], data["width"][k], data["height"][k]
                # centro da palavra
                cx = x + w/2.0
                cy = y + h/2.0
                return (cx, cy, (x, y, x+w, y+h))

        # padrão de 2 tokens
        for j in range(len(toks) - 1):
            if toks[j] == "CARTAO" and toks[j+1] == "RESPOSTA":
                k1, k2 = idxs[j], idxs[j+1]
                x1, y1, w1, h1 = data["left"][k1], data["top"][k1], data["width"][k1], data["height"][k1]
                x2, y2, w2, h2 = data["left"][k2], data["top"][k2], data["width"][k2], data["height"][k2]
                X1, Y1 = min(x1, x2), min(y1, y2)
                X2, Y2 = max(x1+w1, x2+w2), max(y1+h1, y2+h2)
                cx = (X1 + X2) / 2.0
                cy = (Y1 + Y2) / 2.0
                return (cx, cy, (X1, Y1, X2, Y2))

    return None

def encontrar_cursinho(data):
    n = len(data["text"])
    melhor = None
    for i in range(n):
        t = normaliza_token(data["text"][i])
        if not t: continue
        if t in ("CURSINHO","CURSINHE"):  # tolera OCR trocando 'O' por 'E'
            x,y,w,h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            cx, cy = x + w/2.0, y + h/2.0
            if melhor is None or cy < melhor[1][1]:
                melhor = ((x,y,w,h), (cx,cy))
    if melhor is None:
        return None
    (x,y,w,h),(cx,cy)=melhor
    return (cx,cy,(x,y,x+w,y+h))

def rotate_image_keep_bounds(image, angle_deg, border_value=(255,255,255)):
    """Retorna (rotated, M) — sem crop, com a matriz 2x3 usada."""
    (h, w) = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0,2] += (new_w / 2.0) - center[0]
    M[1,2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)
    return rotated, M

def aplicar_M(p, M):
    """Aplica M 2x3 em um ponto (x,y)."""
    x, y = p
    x2 = M[0,0]*x + M[0,1]*y + M[0,2]
    y2 = M[1,0]*x + M[1,1]*y + M[1,2]
    return (x2, y2)

def principal(img_path, out_points_path, out_corrigida_path, debug=False, txt_out=None):
    if not TESS_OK:
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError("pytesseract/Tesseract não configurado e não encontrado no PATH.")

    # 1) Lê imagem original
    img = ler_imagem(img_path)

    # 2) OCR ampliado para localizar pontos
    scale = 1.6
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    data = ocr_data(gray, lang="por+eng")
    ach_A = encontrar_cartao_resposta_preciso(data)
    ach_B = encontrar_cursinho(data)
    if ach_A is None: raise RuntimeError("Não encontrei 'CARTAO-RESPOSTA'.")
    if ach_B is None: raise RuntimeError("Não encontrei 'Cursinho'.")

    (cxA, cyA, boxA) = ach_A
    (cxB, cyB, boxB) = ach_B

    # Converte para coords da imagem ORIGINAL
    inv = 1.0/scale
    pA = (cxA*inv, cyA*inv)   # centro de 'CARTAO-RESPOSTA'
    pB = (cxB*inv, cyB*inv)   # centro de 'Cursinho'

    print(f"[PONTOS ORIG] A={pA}, B={pB}")
    dy0 = pB[1] - pA[1]
    dx0 = pB[0] - pA[0]
    angle0 = degrees(atan2(dy0, dx0))
    print(f"[ÂNGULO] antes={angle0:.3f}° (dy={dy0:.2f}, dx={dx0:.2f})")

    # 3) Desenha pontos na imagem ORIGINAL
    pontos_img = img.copy()
    r = max(4, int(0.004*max(pontos_img.shape[:2])))
    cv2.circle(pontos_img, (int(round(pA[0])), int(round(pA[1]))), r, (0,0,255), -1)
    cv2.circle(pontos_img, (int(round(pB[0])), int(round(pB[1]))), r, (0,0,255), -1)
    if not cv2.imwrite(out_points_path, pontos_img):
        raise RuntimeError("Falha ao salvar a imagem com pontos (original).")
    print(f"Imagem com pontos (ORIGINAL) salva em: {out_points_path}")

    # 4) Rotaciona para horizontalizar a reta AB
    rotate_deg = -angle0  # por convenção do OpenCV (CCW positivo)
    rotated, M = rotate_image_keep_bounds(img, rotate_deg, border_value=(255,255,255))

    # Verificação: transforma A e B e mede dy após a rotação
    pA_rot = aplicar_M(pA, M)
    pB_rot = aplicar_M(pB, M)
    dy1 = pB_rot[1] - pA_rot[1]
    print(f"[CHECK] pós-rotação dy={dy1:.3f} (esperado ~0)")

    # Se não melhorou (|dy1| > |dy0|), invertimos o sinal do ângulo
    if abs(dy1) > abs(dy0) * 0.9:  # tolerância
        print("[AJUSTE] Invertendo sinal do ângulo e rotacionando novamente.")
        rotate_deg = -rotate_deg
        rotated, M = rotate_image_keep_bounds(img, rotate_deg, border_value=(255,255,255))
        pA_rot = aplicar_M(pA, M)
        pB_rot = aplicar_M(pB, M)
        dy1 = pB_rot[1] - pA_rot[1]
        print(f"[CHECK2] pós-rotação dy={dy1:.3f}")

    # 5) Salva imagem ROTACIONADA "limpa"
    if not cv2.imwrite(out_corrigida_path, rotated):
        raise RuntimeError("Falha ao salvar a imagem corrigida.")
    print(f"Imagem corrigida (SEM pontos) salva em: {out_corrigida_path}")

    # 6) OCR na rotacionada (texto)
    rot_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    rot_gray = cv2.medianBlur(rot_gray, 3)
    texto = ocr_text(rot_gray, lang="por+eng")
    print("\n================= TEXTO OCR (IMAGEM ROTACIONADA) =================")
    print(texto.strip())
    print("==================================================================\n")
    if txt_out:
        try:
            Path(txt_out).write_text(texto, encoding="utf-8")
            print(f"Texto OCR salvo em: {txt_out}")
        except Exception as e:
            print(f"[Aviso] Falha ao salvar TXT ({e})")

    # 7) Debug opcional: mostra caixas na OCR ampliada
    if debug:
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        x1,y1,x2,y2 = map(int, map(round, boxA))
        cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(dbg, (int(round(cxA)), int(round(cyA))), 6, (0,255,0), -1)
        x1,y1,x2,y2 = map(int, map(round, boxB))
        cv2.rectangle(dbg, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.circle(dbg, (int(round(cxB)), int(round(cyB))), 6, (255,0,0), -1)
        dbg_path = Path(out_corrigida_path).with_suffix("").as_posix() + "_ocr_debug.png"
        cv2.imwrite(dbg_path, dbg)
        print(f"Debug (OCR ampliada) salvo em: {dbg_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="image.png", help="caminho da imagem")
    ap.add_argument("--out", default="pontos_detectados.png", help="imagem ORIGINAL com pontos pintados")
    ap.add_argument("--corrigida", default="image_corrigida.png", help="imagem ROTACIONADA sem pontos")
    ap.add_argument("--txt", default=None, help="(opcional) caminho para salvar o texto OCR (UTF-8)")
    ap.add_argument("--debug", type=int, default=0, help="1 para salvar arquivos extras de debug")
    args = ap.parse_args()
    try:
        principal(args.img, args.out, args.corrigida, debug=bool(args.debug), txt_out=args.txt)
    except Exception as e:
        print("ERRO:", e)
        sys.exit(1)
