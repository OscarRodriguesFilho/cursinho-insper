# ler_numeros_ocr.py
# Uso:
#   python ler_numeros_ocr.py --img image.png
#   (opcional) --lang por+eng  |  --minconf 30
#
# Requisitos: pip install opencv-python pytesseract

import argparse
import re
from pathlib import Path
import sys
import cv2
import numpy as np

# --- configura Tesseract no seu Windows ---
try:
    import pytesseract
    TESS_OK = True
    TESSERACT_DIR = r"C:\Users\USER\Documents\Projeto OCR\ocara\Tesseract-OCR"
    exe = Path(TESSERACT_DIR) / "tesseract.exe"
    if exe.exists():
        pytesseract.pytesseract.tesseract_cmd = str(exe)
    else:
        TESS_OK = False
except Exception:
    TESS_OK = False

def ler_imagem(p):
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir a imagem: {p}")
    return img

def preprocess(img):
    """Amplia e melhora contraste para OCR."""
    h, w = img.shape[:2]
    scale = 1.7 if max(h, w) < 2200 else 1.3
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    # duas versões: binária e original melhorada
    _, bin_ = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return big, clahe, bin_

def ocr_palavras(img, lang="por+eng", psm=6, whitelist=None):
    """Retorna lista de dicts: text, conf, x,y,w,h."""
    cfg = f"--oem 3 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf = float(data["conf"][i]) if data["conf"][i] not in (None, "", "-1") else -1.0
        if not txt:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append({"text": txt, "conf": conf, "x": x, "y": y, "w": w, "h": h})
    return out

def extrair_numeros(tokens, min_conf=30):
    """
    De uma lista de tokens OCR, extrai sequências numéricas (\d+) com confiança >= min_conf.
    Retorna lista de dicts: numero, conf, bbox (x,y,w,h) e ordem de leitura (y,x).
    """
    resultados = []
    for t in tokens:
        if t["conf"] < min_conf:
            continue
        # pega todos os grupos de dígitos dentro do token (ex: "15A" -> "15")
        for m in re.finditer(r"\d+", t["text"]):
            numero = m.group(0)
            # bbox do token; para simplificar, usamos a caixa toda
            resultados.append({
                "numero": numero,
                "conf": t["conf"],
                "x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"]
            })
    # ordena por linha (y) e depois por coluna (x)
    resultados.sort(key=lambda d: (d["y"], d["x"]))
    return resultados

def main():
    ap = argparse.ArgumentParser(description="Lê uma imagem e imprime todos os números encontrados (OCR).")
    ap.add_argument("--img", default="image.png", help="caminho da imagem (padrão: image.png)")
    ap.add_argument("--lang", default="por+eng", help="idiomas do Tesseract (ex.: 'por', 'eng', 'por+eng')")
    ap.add_argument("--minconf", type=float, default=30.0, help="confiança mínima para aceitar um número (0–100)")
    args = ap.parse_args()

    if not TESS_OK:
        print("❌ pytesseract/Tesseract não configurado no caminho informado.")
        print("Ajuste TESSERACT_DIR no topo do arquivo.")
        sys.exit(1)

    img = ler_imagem(args.img)
    big, clahe, bin_ = preprocess(img)

    # Duas passagens:
    # 1) OCR focado em dígitos (whitelist) — excelente para números “limpos”
    # 2) OCR geral (sem whitelist) — captura números mistos (“Sala 12”, “Q1”, etc.)
    tokens1 = ocr_palavras(bin_,  lang=args.lang, psm=6, whitelist="0123456789")
    tokens2 = ocr_palavras(clahe, lang=args.lang, psm=6, whitelist=None)

    # Mescla e remove duplicatas aproximadas (mesmo texto e bbox muito próximo)
    tokens = tokens1 + tokens2
    # (opcionalmente poderíamos fazer NMS por posição; aqui só juntamos)
    numeros = extrair_numeros(tokens, min_conf=args.minconf)

    # ---- Saída no terminal ----
    if not numeros:
        print("Nenhum número encontrado com os parâmetros atuais.")
        sys.exit(0)

    print("\n=== Ocorrências (na ordem de leitura) ===")
    for i, n in enumerate(numeros, 1):
        print(f"{i:3d}. {n['numero']:>10} | conf={n['conf']:5.1f} | bbox=({n['x']},{n['y']},{n['w']},{n['h']})")

    unicos = []
    seen = set()
    for n in numeros:
        if n["numero"] not in seen:
            unicos.append(n["numero"])
            seen.add(n["numero"])

    print("\n=== Números únicos (ordenados pelo 1º aparecimento) ===")
    print(", ".join(unicos))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERRO:", e)
        sys.exit(1)
