# app.py
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
import os
import json
import pandas as pd
from collections import Counter

# ====== OpenCV / NumPy para detecção ======
import cv2
import numpy as np
from pathlib import Path

# ===================== Configurações =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'xlsx'}
GABARITO_FILENAME = 'resultado_corrigido.xlsx'

# Parâmetros do detector (baseado no seu script)
DETECT_CFG = {
    "IMG_CANDIDATES": ("image.png", "image.jpg", "image.jpeg", "image.webp"),
    "RESIZE_W": 1600,
    "HOUGH_PARAM1": 100,
    "HOUGH_PARAM2": 20,
    "MIN_DIST": 18,
    "DARK_DROP_MIN": 25,  # anel - interior (0–255)
    "INNER_MAX": 170,     # teto luminância do interior
    "R_IN_SCALE": 0.80,
    "R_RING_IN": 1.05,
    "R_RING_OUT": 1.50,
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===================== Rotas utilitárias =====================

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve arquivos de /uploads (ex.: image.png, bubbles.json)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/root/<path:filename>')
def serve_root(filename):
    """Serve arquivos do diretório raiz do projeto (ex.: image.png no BASE_DIR)."""
    return send_from_directory(BASE_DIR, filename)

# (opcional) debug do conteúdo de uploads
@app.route('/debug/uploads')
def debug_uploads():
    files = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    return jsonify({"UPLOAD_FOLDER": app.config['UPLOAD_FOLDER'], "files": files})

# ===================== Fluxo principal (upload/correção) =====================

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Nenhum arquivo enviado.', 400
    file = request.files['file']
    if file.filename == '':
        return 'Nome do arquivo vazio.', 400

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_excel(filepath)
        if df.shape[0] < 2:
            return "Erro: a planilha deve conter ao menos um gabarito e um aluno."

        numero_questoes = df.shape[1]
        colunas_resposta = df.columns[:numero_questoes]
        gabarito = df.iloc[0][colunas_resposta]
        alunos = df.iloc[1:]

        resultados = []
        for i, aluno in alunos.iterrows():
            respostas = aluno[colunas_resposta]
            acertos = (respostas == gabarito).sum()
            erros = numero_questoes - acertos
            nota = round((acertos / numero_questoes) * 10, 2)
            nome = aluno.get('Nome') or aluno.get('nome') or f"Aluno {i}"
            resultados.append({'nome': nome, 'acertos': acertos, 'erros': erros, 'nota': nota})

        resultado_df = pd.DataFrame(resultados)
        resultado_path = os.path.join(app.config['UPLOAD_FOLDER'], GABARITO_FILENAME)
        resultado_df.to_excel(resultado_path, index=False)

        with open(os.path.join(UPLOAD_FOLDER, 'último_arquivo.txt'), 'w', encoding='utf-8') as f:
            f.write(filename)

        return (
            '✅ Correção concluída. '
            '<a href="/gabarito">Ver gabarito</a> | '
            '<a href="/conferir_gabarito">Conferir bolinhas</a>'
        )

    return '❌ Tipo de arquivo não permitido. Envie um .xlsx.', 400

@app.route('/gabarito')
def ver_gabarito():
    try:
        with open(os.path.join(UPLOAD_FOLDER, 'último_arquivo.txt'), 'r', encoding='utf-8') as f:
            nome_arquivo = f.read().strip()
        caminho = os.path.join(UPLOAD_FOLDER, nome_arquivo)
        df = pd.read_excel(caminho)
        gabarito = df.iloc[0]
        return render_template('gabarito.html', gabarito=gabarito)
    except Exception as e:
        return f"Erro ao tentar carregar o gabarito: {e}"

# ===================== Conferência de gabarito (bolinhas) =====================

def _find_image_path():
    """Procura image.* primeiro em /uploads e depois no diretório raiz."""
    # 1) nomes padrão em /uploads
    for c in DETECT_CFG["IMG_CANDIDATES"]:
        p = os.path.join(UPLOAD_FOLDER, c)
        if os.path.exists(p):
            return p, url_for('serve_upload', filename=c), c
    # 2) nomes padrão no raiz
    for c in DETECT_CFG["IMG_CANDIDATES"]:
        p = os.path.join(BASE_DIR, c)
        if os.path.exists(p):
            return p, url_for('serve_root', filename=c), c
    # 3) 1ª imagem encontrada (uploads → raiz)
    for f in sorted(os.listdir(UPLOAD_FOLDER)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            p = os.path.join(UPLOAD_FOLDER, f)
            return p, url_for('serve_upload', filename=f), f
    for f in sorted(os.listdir(BASE_DIR)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            p = os.path.join(BASE_DIR, f)
            return p, url_for('serve_root', filename=f), f
    return None, None, None

@app.route('/conferir_gabarito', methods=['GET'])
def conferir_gabarito():
    """
    Página interativa para visualizar/editar bolinhas sobre o gabarito.
    Agora com botão "Detectar bolinhas" que chama /api/detect_bubbles.
    """
    img_path, imagem_url, imagem_nome = _find_image_path()
    return render_template('conferir_gabarito_bolhas.html',
                           imagem_url=imagem_url,
                           imagem_nome=imagem_nome)

@app.route('/api/bubbles', methods=['GET', 'POST'])
def api_bubbles():
    """
    GET  -> retorna JSON com bolinhas (uploads/bubbles.json ou raiz/bubbles.json).
    POST -> salva SEMPRE em uploads/bubbles.json
    """
    uploads_json = os.path.join(app.config['UPLOAD_FOLDER'], 'bubbles.json')
    root_json = os.path.join(BASE_DIR, 'bubbles.json')

    if request.method == 'GET':
        path = uploads_json if os.path.exists(uploads_json) else (root_json if os.path.exists(root_json) else None)
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = {"image": "image.png", "circles": []}
        else:
            data = {"image": "image.png", "circles": []}
        return jsonify(data)

    # POST (salva sempre em uploads)
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict) or 'circles' not in payload or not isinstance(payload['circles'], list):
            return jsonify({"ok": False, "error": "Payload inválido. Esperado {'circles': [...]}"}), 400

        circles = []
        for c in payload['circles']:
            try:
                x = float(c.get('x', 0)); y = float(c.get('y', 0)); r = float(c.get('r', 0))
                if r > 0: circles.append({"x": x, "y": y, "r": r})
            except Exception:
                continue

        data = {"image": payload.get("image", "image.png"), "circles": circles}

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        with open(uploads_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({"ok": True, "count": len(circles)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ---- Nova rota: DETECÇÃO de bolinhas (equivalente ao seu script) ----
# ====== Helpers de detecção (robustos) ======
# --- coloque isto no seu app.py (substituindo a versão anterior do detect) ---

import cv2, numpy as np
from pathlib import Path

# Parâmetros mais restritos
RESIZE_W      = 1800
HOUGH_PARAM1  = 120
HOUGH_PARAM2  = 28     # mais alto => menos falsos positivos
MIN_DIST      = 22

# rádio (na imagem redimensionada). Bolhas de gabarito costumam ficar aqui:
R_MIN_FIX     = 11
R_MAX_FIX     = 20

# Classificação "preenchida"
DARK_DROP_MIN = 35     # anel - interior
INNER_MAX     = 140    # teto de luminância do interior
FILL_RATIO_MIN= 0.55   # % de pixels pretos dentro (após Otsu)

def _resize_keep(img, target_w):
    h, w = img.shape[:2]
    s = target_w / float(w)
    out = cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_CUBIC)
    return out, s

def _hough(gray):
    g = cv2.GaussianBlur(gray, (5,5), 0)
    cs = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=R_MIN_FIX, maxRadius=R_MAX_FIX
    )
    if cs is None:
        return np.empty((0,3), dtype=np.float32)
    return cs[0].astype(np.float32)

def _circle_measures(gray, x, y, r):
    H, W = gray.shape[:2]
    r_in   = max(4, int(round(r*0.80)))
    r_out1 = max(r_in+2, int(round(r*1.05)))
    r_out2 = min(max(r_out1+2, int(round(r*1.45))), int(min(H, W)*0.12))

    Y, X = np.ogrid[:H, :W]
    d2 = (X-x)**2 + (Y-y)**2
    mask_in   = d2 <= (r_in**2)
    mask_ring = (d2 >= (r_out1**2)) & (d2 <= (r_out2**2))
    if mask_in.sum() < 25 or mask_ring.sum() < 40:
        return None

    inner_mean = float(gray[mask_in].mean())
    ring_mean  = float(gray[mask_ring].mean())
    drop = ring_mean - inner_mean

    # Fração de “pretos” no interior (binarizando por Otsu)
    inner_vals = gray[mask_in].astype(np.uint8)
    _th, inner_bin = cv2.threshold(inner_vals, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fill_ratio = float((inner_bin == 0).sum()) / float(inner_bin.size)

    return inner_mean, ring_mean, drop, fill_ratio

def _is_filled(inner_mean, drop, fill_ratio):
    return (drop >= DARK_DROP_MIN) and (inner_mean <= INNER_MAX) and (fill_ratio >= FILL_RATIO_MIN)

def _nms_nearby(circles, thr=0.75):
    """junta círculos muito próximos; fica com o de maior 'r'."""
    if not circles: return circles
    circles = sorted(circles, key=lambda c: c["r"], reverse=True)
    kept = []
    for c in circles:
        ok = True
        for k in kept:
            dx = c["x"]-k["x"]; dy = c["y"]-k["y"]
            if (dx*dx + dy*dy)**0.5 <= thr*max(c["r"], k["r"]):
                ok = False; break
        if ok: kept.append(c)
    return kept

def detect_bubbles_opencv(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("Falha ao abrir a imagem.")

    work, s = _resize_keep(img, RESIZE_W)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    H, W = gray.shape[:2]

    # Limita à área de respostas (evita cabeçalho): de ~18% a 95% da altura
    y0 = int(0.18 * H)
    y1 = int(0.95 * H)
    roi = gray[y0:y1, :]

    det = _hough(roi)
    results = []
    for (x, y, r) in det:
        y_abs = y + y0
        m = _circle_measures(gray, int(round(x)), int(round(y_abs)), int(round(r)))
        if m is None: continue
        inner, ring, drop, fill_ratio = m
        if _is_filled(inner, drop, fill_ratio):
            # volta para coords da imagem ORIGINAL
            results.append({
                "x": float(x/s),
                "y": float(y_abs/s),
                "r": float(r/s),
                "score": float(drop),
                "fill": float(fill_ratio)
            })

    results = _nms_nearby(results, thr=0.8)
    return results

# --- endpoint (use seu nome de app/Blueprint) ---
@app.post("/api/detect_bubbles")
def api_detect_bubbles():
    try:
        data = request.get_json(force=True, silent=True) or {}
        image_name = data.get("image") or None
        # prioridade: /uploads/<image>, senão raiz
        cand = [UPLOAD_FOLDER, BASE_DIR]
        img_path = None
        for base in cand:
            p = Path(base) / (image_name or "image.png")
            if p.exists():
                img_path = p; break
        if img_path is None:
            return {"ok": False, "error": "Imagem não encontrada."}, 400

        circles = detect_bubbles_opencv(img_path)
        # salva também no bubbles.json (opcional)
        out = {"image": img_path.name, "circles": [{"x":c["x"],"y":c["y"],"r":c["r"]} for c in circles]}
        (Path(UPLOAD_FOLDER)/"bubbles.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"ok": True, "count": len(circles), "circles": out["circles"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

# ===================== Respostas por aluno =====================

@app.route('/respostas', methods=['GET', 'POST'])
def verificar_respostas():
    uploads = app.config['UPLOAD_FOLDER']
    arquivos_disponiveis = [
        f for f in os.listdir(uploads)
        if f.endswith('.xlsx') and f != 'resultado_corrigido.xlsx'
    ]

    if request.method == 'GET':
        return render_template('respostas_escolha.html', arquivos=arquivos_disponiveis)

    nome_arquivo = request.form.get('arquivo')
    if not nome_arquivo:
        return 'Nenhum arquivo selecionado.'

    caminho_excel = os.path.join(uploads, nome_arquivo)
    try:
        df = pd.read_excel(caminho_excel)
        if df.shape[0] < 2:
            return 'A planilha precisa ter ao menos um gabarito e um aluno.'

        gabarito = df.iloc[0]
        alunos = df.iloc[1:]

        col_nome = 'Nome completo:'
        col_email = 'Endereço de e-mail'
        colunas_questoes = [col for col in df.columns if col not in [col_nome, col_email]]

        tabela = []
        for _, aluno in alunos.iterrows():
            linha = {
                'nome': aluno.get(col_nome, '—'),
                'email': aluno.get(col_email, '—'),
                'respostas': [1 if aluno.get(col) == gabarito.get(col) else 0 for col in colunas_questoes]
            }
            tabela.append(linha)

        return render_template('respostas.html', tabela=tabela, questoes=colunas_questoes)

    except Exception as e:
        return f'Erro ao processar o arquivo: {e}'

# ===================== Dashboard =====================

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    uploads = app.config['UPLOAD_FOLDER']
    arquivos_disponiveis = [
        f for f in os.listdir(uploads)
        if f.endswith('.xlsx') and f != 'resultado_corrigido.xlsx'
    ]

    if request.method == 'GET':
        return render_template('dashboard_escolha.html', arquivos=arquivos_disponiveis)

    nome_arquivo = request.form.get('arquivo')
    modalidade_filtro = request.form.get('modalidade', 'Geral')

    if not nome_arquivo:
        return 'Nenhum arquivo selecionado.'

    caminho_excel = os.path.join(uploads, nome_arquivo)
    try:
        df = pd.read_excel(caminho_excel)
        if df.shape[0] < 2:
            return 'A planilha precisa ter ao menos um gabarito e um aluno.'

        COL_NOME = 'Nome completo:'
        COL_MODALIDADE = 'Modalidade:'
        possiveis_emails = ['Endereço de e-mail', 'Endereço de email', 'E-mail', 'Email', 'Endereço de Email']
        COL_EMAIL = next((c for c in possiveis_emails if c in df.columns), None)
        if COL_EMAIL is None:
            return 'Coluna de e-mail não encontrada. Esperado: "Endereço de e-mail" (ou variações).'

        COLS_EXCLUIR = [
            'Carimbo de data/hora','Endereço de e-mail','Nome completo:','Número de matrícula:','Modalidade:',
            'Trilha escolhida:','O quanto você sente que dominava o conteúdo da avaliação:','De 1 a 5, quantas perguntas você chutou?',
            'Matéria com maior dificuldade:','Matéria com mais facilidade:','Faça qualquer comentário sobre a prova (melhorias, impressões, autoavalição...)',
            'Pontuação'
        ]
        if COL_EMAIL not in COLS_EXCLUIR:
            COLS_EXCLUIR.append(COL_EMAIL)

        alvo_gabarito = 'cursinhoinsper@insper.edu.br'
        mask_gab = df[COL_EMAIL].astype(str).str.strip().str.lower() == alvo_gabarito
        if mask_gab.any():
            idx_gab = df.index[mask_gab][0]
        else:
            idx_gab = df.index[0]

        gabarito = df.loc[idx_gab]
        alunos_df = df.drop(index=idx_gab).copy()

        if modalidade_filtro != 'Geral':
            alunos_df = alunos_df[alunos_df[COL_MODALIDADE].astype(str).str.lower() == modalidade_filtro.lower()]

        if alunos_df.empty:
            return f'Nenhum aluno encontrado para a modalidade "{modalidade_filtro}".'

        quest_cols = [c for c in df.columns if c not in COLS_EXCLUIR]

        def conta_acertos(row):
            return sum(1 for c in quest_cols if pd.notna(row.get(c)) and row.get(c) == gabarito.get(c))

        alunos_df['acertos'] = alunos_df.apply(conta_acertos, axis=1)
        alunos_df['total'] = len(quest_cols)
        alunos_df['nota'] = (alunos_df['acertos'] / alunos_df['total'] * 10).round(2)

        COL_DOMINIO = 'O quanto você sente que dominava o conteúdo da avaliação:'
        COL_CHUTES = 'De 1 a 5, quantas perguntas você chutou?'
        COL_DIFICULDADE = 'Matéria com maior dificuldade:'
        campos_extra = [COL_DOMINIO, COL_CHUTES, COL_DIFICULDADE]

        top_melhores = alunos_df.sort_values(['acertos', COL_NOME], ascending=[False, True]).head(10)[
            [COL_NOME, COL_EMAIL, 'acertos', 'total', 'nota'] + campos_extra
        ]
        top_piores = alunos_df.sort_values(['acertos', COL_NOME], ascending=[True, True]).head(10)[
            [COL_NOME, COL_EMAIL, 'acertos', 'total', 'nota'] + campos_extra
        ]
        ranking_completo = alunos_df[[COL_NOME, COL_EMAIL, 'acertos', 'total', 'nota']].sort_values('acertos', ascending=False)

        acertos_por_questao, top3_por_questao = [], []
        for c in quest_cols:
            col_series = alunos_df[c]
            cont = Counter([str(v) for v in col_series.dropna().tolist()])
            correto = str(gabarito.get(c))
            acertos = int(sum(1 for v in col_series if str(v) == correto))
            acertos_por_questao.append(acertos)
            top3_por_questao.append({'questao': c, 'correta': correto, 'top3': cont.most_common(3)})

        Q = len(quest_cols)
        mapa_areas = [
            (1,10,"História"), (11,20,"Filosofia"), (21,30,"Geografia"), (31,40,"Gramática"),
            (41,50,"Interpretação"), (51,60,"Química"), (61,70,"Biologia"), (71,75,"Física"), (76,95,"Matemática"),
        ]

        grupos, perc_por_area = [], []
        for (ini_q, fim_q, nome_area) in mapa_areas:
            ini = ini_q - 1; fim = min(fim_q, Q)
            if ini >= fim: continue
            labels, valores = [], []
            soma_acertos = 0
            possivel = len(alunos_df) * (fim - ini)
            for j in range(ini, fim):
                qname = quest_cols[j]
                correto = str(gabarito.get(qname))
                labels.append(f"Q{j+1} ({correto})")
                valores.append(acertos_por_questao[j])
                soma_acertos += acertos_por_questao[j]
            grupos.append({'titulo': f'{nome_area} (Q{ini+1}–Q{fim})', 'labels': labels, 'valores': valores, 'intervalo': [ini, fim]})
            perc_area = 0.0 if possivel == 0 else round(100.0 * soma_acertos / possivel, 2)
            perc_por_area.append({'area': nome_area, 'perc': perc_area})

        media_nota = float(alunos_df['nota'].mean().round(2))
        mediana_nota = float(alunos_df['nota'].median().round(2))

        dificuldade = pd.DataFrame({
            'questao': quest_cols,
            'acertos': acertos_por_questao,
            'perc': [round(100 * a / max(1, len(alunos_df)), 2) for a in acertos_por_questao]
        }).sort_values('perc', ascending=False)

        return render_template(
            'dashboard.html',
            arquivo=nome_arquivo,
            top_melhores=top_melhores.to_dict(orient='records'),
            top_piores=top_piores.to_dict(orient='records'),
            ranking=ranking_completo.to_dict(orient='records'),
            grupos=grupos,
            total_alunos=len(alunos_df),
            total_questoes=Q,
            perc_por_area=perc_por_area,
            media_nota=media_nota,
            mediana_nota=mediana_nota,
            top3_por_questao=top3_por_questao,
            dificuldade=dificuldade.to_dict(orient='records'),
            modalidade_filtro=modalidade_filtro
        )

    except Exception as e:
        return f'Erro ao processar o dashboard: {e}'

# ===================== Run =====================


from flask import render_template, request
import pandas as pd, numpy as np, os

@app.route('/dashboard_marketing', methods=['GET', 'POST'])
def dashboard_marketing():
    uploads_dir = os.path.join(app.root_path, 'uploads_marketing')
    arquivos = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.xlsx', '.xls'))]
    arquivo = request.form.get('arquivo') or (arquivos[0] if arquivos else None)
    if not arquivo:
        return render_template('dashboard_marketing.html', arquivos=arquivos, arquivo=None,
                               erro="Nenhum arquivo encontrado em uploads_marketing.")

    # Carrega planilha
    df = pd.read_excel(os.path.join(uploads_dir, arquivo))

    # Normalização leve de nomes que usamos nos gráficos/filtros
    rename_map = {
        'Data do Post':'data','Plataforma':'plataforma','Tipo de Conteúdo':'tipo','Tema do Post':'tema',
        'Post Turbinado? (Instagram)':'turbinado','Público-alvo':'publico','Link do Post':'link_post',
        'link':'link','Impressões/Interações':'impressoes','Alcance (visualizações)':'views',
        'Curtidas/Reações':'curtidas','Comentários':'comentarios','Compartilhamentos':'compart',
        'Cliques no Link':'cliques_link','Salvamentos (Instagram)':'salvamentos','Cliques no Perfil':'cliques_perfil',
        'Índice de Engajamento':'engajamento','Indice de Engajamento':'engajamento','Eficiência':'eficiencia'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    # Converte números (vírgula → ponto) e garante colunas
    num_cols = ['impressoes','views','curtidas','comentarios','compart','cliques_link','salvamentos','cliques_perfil','eficiencia','engajamento']
    for c in num_cols:
        if c not in df.columns: df[c] = np.nan
        df[c] = (df[c].astype(str)
                    .str.replace('.', '', regex=False)
                    .str.replace(',', '.', regex=False))
        df[c] = pd.to_numeric(df[c], errors='coerce')

    for c in ['plataforma','tipo','tema','turbinado','publico','link','data']:
        if c not in df.columns: df[c] = ''

    # Label curto p/ eixo X
    def make_label(row):
        d = pd.to_datetime(row.get('data'), errors='coerce')
        ds = d.strftime('%d/%m') if not pd.isna(d) else str(row.get('data') or '')
        tema = str(row.get('tema') or '')
        if len(tema) > 24: tema = tema[:22] + '…'
        return f"{ds} • {str(row.get('tipo') or '').lower()} • {tema}"
    df['label'] = df.apply(make_label, axis=1)

    # Filtros dinâmicos
    sel = {
        'plataforma': request.form.get('plataforma', 'Todos'),
        'tipo_conteudo': request.form.get('tipo_conteudo', 'Todos'),
        'tema_post': request.form.get('tema_post', 'Todos'),
        'turbinado': request.form.get('turbinado', 'Todos'),
        'publico': request.form.get('publico', 'Todos'),
    }
    def uniques(col):
        x = df[col].dropna().astype(str).str.strip()
        return sorted([v for v in x.unique() if v])
    filtros_opts = {
        'plataforma': uniques('plataforma'),
        'tipo_conteudo': uniques('tipo'),
        'tema_post': uniques('tema'),
        'turbinado': uniques('turbinado'),
        'publico': uniques('publico'),
    }

    fdf = df.copy()
    if sel['plataforma']   != 'Todos': fdf = fdf[fdf['plataforma'].astype(str)==sel['plataforma']]
    if sel['tipo_conteudo']!= 'Todos': fdf = fdf[fdf['tipo'].astype(str)==sel['tipo_conteudo']]
    if sel['tema_post']    != 'Todos': fdf = fdf[fdf['tema'].astype(str)==sel['tema_post']]
    if sel['turbinado']    != 'Todos': fdf = fdf[fdf['turbinado'].astype(str)==sel['turbinado']]
    if sel['publico']      != 'Todos': fdf = fdf[fdf['publico'].astype(str)==sel['publico']]

    # --------- MÉTRICAS RESUMO (para a tabela de estatísticas) ---------
    metric_series = {
        'Visualizações': fdf['views'],
        'Curtidas': fdf['curtidas'],
        'Comentários': fdf['comentarios'],
        'Compartilhamentos': fdf['compart'],
        'Cliques no Perfil': fdf['cliques_perfil'],
        'Cliques no Link': fdf['cliques_link'],
        'Salvamentos': fdf['salvamentos'],
        'Eficiência': fdf['eficiencia'],
    }
    def resumo(s):
        s = s.dropna()
        if s.empty:
            return {k:0 for k in ['Média','Desvio Padrão','Variação Percentual','Máximo','Mínimo','Q1','Mediana','Q3','Variância']}
        return {
            'Média': float(s.mean()),
            'Desvio Padrão': float(s.std(ddof=1)) if len(s)>1 else 0.0,
            'Variação Percentual': float((s.max()-s.min())/s.min()*100) if s.min() not in [0,np.nan] else 0.0,
            'Máximo': float(s.max()),
            'Mínimo': float(s.min()),
            'Q1': float(s.quantile(0.25)),
            'Mediana': float(s.median()),
            'Q3': float(s.quantile(0.75)),
            'Variância': float(s.var(ddof=1)) if len(s)>1 else 0.0
        }
    metrics_table = {nome: resumo(serie) for nome, serie in metric_series.items()}

    # --------- DADOS DOS GRÁFICOS ---------
    def safe_list(vals):
        out = []
        for v in vals:
            try:
                fv = float(v)
                if not np.isfinite(fv): fv = 0.0
            except Exception:
                fv = 0.0
            out.append(fv)
        return out

    labels = fdf['label'].tolist()
    charts_payload = {
        'labels': labels,
        'views': safe_list(fdf['views'].fillna(0).tolist()),
        'curtidas': safe_list(fdf['curtidas'].fillna(0).tolist()),
        'comentarios': safe_list(fdf['comentarios'].fillna(0).tolist()),
        'cliques_perfil': safe_list(fdf['cliques_perfil'].fillna(0).tolist()),
        'cliques_link': safe_list(fdf['cliques_link'].fillna(0).tolist()),
        'compart': safe_list(fdf['compart'].fillna(0).tolist()),
        'salvamentos': safe_list(fdf['salvamentos'].fillna(0).tolist()),
        'top5_labels': safe_list([])  # placeholder; preenchido abaixo
    }
    top5 = fdf[['label','eficiencia']].dropna().sort_values('eficiencia', ascending=False).head(5)
    charts_payload['top5'] = {
        'labels': top5['label'].tolist(),
        'values': safe_list(top5['eficiencia'].tolist())
    }

    # --------- TABELA DE POSTS COMPLETA ---------
    # Mantém TODAS as colunas do arquivo; move 'link' para o fim e adiciona uma coluna "Imagem"
    display_cols = list(df.columns)
    # Se 'link' ou 'Link' existirem, deixamos como coluna normal; preview usa essa URL
    if 'link' not in display_cols: df['link'] = '' ; display_cols.append('link')
    # Constrói registros p/ template
    tabela = fdf[display_cols].fillna('').to_dict(orient='records')

    # data mais recente
    try:
        ate_data = pd.to_datetime(df['data'], errors='coerce').max()
        ate_data = ate_data.strftime('%d/%m/%Y') if pd.notna(ate_data) else ''
    except Exception:
        ate_data = ''

    return render_template('dashboard_marketing.html',
                           arquivos=arquivos, arquivo=arquivo,
                           filtros_opts=filtros_opts, sel=sel,
                           charts_payload=charts_payload,
                           metrics_table=metrics_table,
                           tabela=tabela,
                           ate_data=ate_data,
                           erro=None)

if __name__ == '__main__':
    app.run(debug=True)
