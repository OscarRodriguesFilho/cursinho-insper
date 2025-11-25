# app.py
from flask import (
    Flask, request, render_template, jsonify, send_from_directory,
    url_for, session, redirect, flash
)
import os, json, re, unicodedata
import pandas as pd
from collections import Counter

# ====== OpenCV / NumPy para detecção ======
import cv2
import numpy as np
from pathlib import Path

# ====== Segurança / .env ======
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename  # <<< para salvar upload com nome seguro

load_dotenv()  # carrega variáveis do .env

# ===================== Configurações =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'xlsx'}
GABARITO_FILENAME = 'resultado_corrigido.xlsx'

# Pasta de marketing (nova)
MARKETING_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_marketing')

# Pasta dos arquivos de formulário acadêmico
ARQUIVOS_FOLDER = os.path.join(BASE_DIR, 'arquivos')
os.makedirs(ARQUIVOS_FOLDER, exist_ok=True)

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

# ===================== App / Sessão =====================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MARKETING_UPLOAD_FOLDER, exist_ok=True)

# SECRET_KEY do .env (obrigatório em produção)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# ======== Credenciais (do .env) ========
AUTH_EMAIL = (os.getenv("AUTH_EMAIL") or "").strip().lower()
AUTH_PASSWORD_HASH = os.getenv("AUTH_PASSWORD_HASH")  # preferencial
AUTH_PASSWORD_PLAIN = os.getenv("AUTH_PASSWORD_PLAIN")  # fallback opcional

if not AUTH_EMAIL:
    raise RuntimeError("AUTH_EMAIL não definido no .env")

if not AUTH_PASSWORD_HASH:
    if not AUTH_PASSWORD_PLAIN:
        raise RuntimeError("Defina AUTH_PASSWORD_HASH (recomendado) ou AUTH_PASSWORD_PLAIN no .env")
    # Fallback seguro em memória: gera hash a partir da senha plana fornecida
    AUTH_PASSWORD_HASH = generate_password_hash(AUTH_PASSWORD_PLAIN)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------- Auth helpers ----------------------
def _is_api_request():
    return request.path.startswith("/api/") or "application/json" in (request.headers.get("Accept", "")).lower()


def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if session.get("user") == AUTH_EMAIL:
            return fn(*args, **kwargs)
        if _is_api_request():
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        nxt = request.full_path if request.query_string else request.path
        return redirect(url_for("login", next=nxt))
    return wrapper


# ===================== Normalização / Colunas =====================

def _norm_text(s: str) -> str:
    s = str(s or '')
    s = s.strip().lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')  # remove acentos
    s = re.sub(r'[\u2010-\u2015\u2212\u2043\uFE63\uFF0D]', '-', s)   # normaliza hífens para '-'
    s = re.sub(r'\s+', ' ', s)                                      # colapsa espaços
    return s


def find_email_column(columns) -> str | None:
    """Tenta achar a coluna de e-mail usando normalização e keywords."""
    norm_map = {col: _norm_text(col) for col in columns}

    # candidatos exatos (normalizados)
    candidates = {
        'endereco de e-mail',
        'endereco de email',
        'e-mail',
        'email',
        'endereco de email',
    }

    # 1) match exato
    for col, n in norm_map.items():
        if n in candidates:
            return col

    # 2) contém 'email'
    for col, n in norm_map.items():
        if 'email' in n:
            return col

    return None


# ===================== Rotas de Autenticação =====================

@app.route("/login", methods=["GET", "POST"])
def login():
    # JSON (ex.: axios/fetch)
    if request.method == "POST" and request.is_json:
        payload = request.get_json(silent=True) or {}
        email = str(payload.get("email", "")).strip().lower()
        password = str(payload.get("password", "")).strip()
        if email == AUTH_EMAIL and check_password_hash(AUTH_PASSWORD_HASH, password):
            session["user"] = AUTH_EMAIL
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "Credenciais inválidas"}), 401

    # Formulário HTML
    if request.method == "POST":
        email = str(request.form.get("email", "")).strip().lower()
        password = str(request.form.get("password", "")).strip()
        if email == AUTH_EMAIL and check_password_hash(AUTH_PASSWORD_HASH, password):
            session["user"] = AUTH_EMAIL
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        flash("Credenciais inválidas.", "error")

    return render_template("login.html")


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    if _is_api_request():
        return jsonify({"ok": True})
    return redirect(url_for("login"))


# ===================== Rotas utilitárias =====================

@app.route('/uploads/<path:filename>')
@login_required
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads_marketing/<path:filename>')
@login_required
def serve_upload_marketing(filename):
    return send_from_directory(MARKETING_UPLOAD_FOLDER, filename)


@app.route('/root/<path:filename>')
@login_required
def serve_root(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route('/debug/uploads')
@login_required
def debug_uploads():
    files = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    return jsonify({"UPLOAD_FOLDER": app.config['UPLOAD_FOLDER'], "files": files})


# ===================== Fluxo principal (upload/correção) =====================

@app.route('/')
@login_required
def index():
    # Se você quiser, pode redirecionar direto para o dashboard:
    # return redirect(url_for('dashboard'))
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """
    Recebe um .xlsx do formulário (campo 'file'),
    salva na pasta uploads/ e volta para o dashboard
    para permitir a seleção desse arquivo.
    """
    if 'file' not in request.files:
        flash('Nenhum arquivo enviado.', 'error')
        return redirect(url_for('dashboard'))

    file = request.files['file']

    if not file or file.filename == '':
        flash('Nome de arquivo vazio.', 'error')
        return redirect(url_for('dashboard'))

    if not allowed_file(file.filename):
        flash('❌ Tipo de arquivo não permitido. Envie um .xlsx.', 'error')
        return redirect(url_for('dashboard'))

    # Nome seguro
    filename = secure_filename(file.filename)

    # Garante que a pasta existe
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Registra o último arquivo enviado (se quiser usar em outra rota)
    with open(os.path.join(UPLOAD_FOLDER, 'último_arquivo.txt'), 'w', encoding='utf-8') as f:
        f.write(filename)

    flash(f'✅ Arquivo "{filename}" enviado com sucesso.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/gabarito')
@login_required
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
    for c in DETECT_CFG["IMG_CANDIDATES"]:
        p = os.path.join(UPLOAD_FOLDER, c)
        if os.path.exists(p):
            return p, url_for('serve_upload', filename=c), c
    for c in DETECT_CFG["IMG_CANDIDATES"]:
        p = os.path.join(BASE_DIR, c)
        if os.path.exists(p):
            return p, url_for('serve_root', filename=c), c
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
@login_required
def conferir_gabarito():
    img_path, imagem_url, imagem_nome = _find_image_path()
    return render_template('conferir_gabarito_bolhas.html',
                           imagem_url=imagem_url,
                           imagem_nome=imagem_nome)


@app.route('/api/bubbles', methods=['GET', 'POST'])
@login_required
def api_bubbles():
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

    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict) or 'circles' not in payload or not isinstance(payload['circles'], list):
            return jsonify({"ok": False, "error": "Payload inválido. Esperado {'circles': [...]}"}), 400

        circles = []
        for c in payload['circles']:
            try:
                x = float(c.get('x', 0)); y = float(c.get('y', 0)); r = float(c.get('r', 0))
                if r > 0:
                    circles.append({"x": x, "y": y, "r": r})
            except Exception:
                continue

        data = {"image": payload.get("image", "image.png"), "circles": circles}

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        with open(uploads_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({"ok": True, "count": len(circles)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---- Detecção de bolinhas (robusta) ----
RESIZE_W      = 1800
HOUGH_PARAM1  = 120
HOUGH_PARAM2  = 28
MIN_DIST      = 22
R_MIN_FIX     = 11
R_MAX_FIX     = 20
DARK_DROP_MIN = 35
INNER_MAX     = 140
FILL_RATIO_MIN= 0.55


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

    inner_vals = gray[mask_in].astype(np.uint8)
    _th, inner_bin = cv2.threshold(inner_vals, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fill_ratio = float((inner_bin == 0).sum()) / float(inner_bin.size)

    return inner_mean, ring_mean, drop, fill_ratio


def _is_filled(inner_mean, drop, fill_ratio):
    return (drop >= DARK_DROP_MIN) and (inner_mean <= INNER_MAX) and (fill_ratio >= FILL_RATIO_MIN)


def _nms_nearby(circles, thr=0.8):
    if not circles:
        return circles
    circles = sorted(circles, key=lambda c: c["r"], reverse=True)
    kept = []
    for c in circles:
        ok = True
        for k in kept:
            dx = c["x"]-k["x"]; dy = c["y"]-k["y"]
            if (dx*dx + dy*dy)**0.5 <= thr*max(c["r"], k["r"]):
                ok = False; break
        if ok:
            kept.append(c)
    return kept


def detect_bubbles_opencv(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("Falha ao abrir a imagem.")

    work, s = _resize_keep(img, RESIZE_W)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    H, W = gray.shape[:2]
    y0 = int(0.18 * H)
    y1 = int(0.95 * H)
    roi = gray[y0:y1, :]

    det = _hough(roi)
    results = []
    for (x, y, r) in det:
        y_abs = y + y0
        m = _circle_measures(gray, int(round(x)), int(round(y_abs)), int(round(r)))
        if m is None:
            continue
        inner, ring, drop, fill_ratio = m
        if _is_filled(inner, drop, fill_ratio):
            results.append({
                "x": float(x/s),
                "y": float(y_abs/s),
                "r": float(r/s),
                "score": float(drop),
                "fill": float(fill_ratio)
            })

    results = _nms_nearby(results, thr=0.8)
    return results


@app.post("/api/detect_bubbles")
@login_required
def api_detect_bubbles():
    try:
        data = request.get_json(force=True, silent=True) or {}
        image_name = data.get("image") or None
        cand = [UPLOAD_FOLDER, BASE_DIR]
        img_path = None
        for base in cand:
            p = Path(base) / (image_name or "image.png")
            if p.exists():
                img_path = p; break
        if img_path is None:
            return {"ok": False, "error": "Imagem não encontrada."}, 400

        circles = detect_bubbles_opencv(img_path)
        out = {"image": img_path.name, "circles": [{"x":c["x"],"y":c["y"],"r":c["r"]} for c in circles]}
        (Path(UPLOAD_FOLDER)/"bubbles.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"ok": True, "count": len(circles), "circles": out["circles"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


# ===================== Respostas por aluno =====================

@app.route('/respostas', methods=['GET', 'POST'])
@login_required
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
        col_email = find_email_column(df.columns) or 'Endereço de e-mail'
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
@login_required
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

    # Filtro "Tipo de aluno"
    tipo_aluno = request.form.get('tipo_aluno', 'Todos')  # 'Todos' | 'Remanescente' | 'Não remanescente'
    REMAN_COL_CANDIDATES = [
        'Você é aluno remanescente?',
        'Você é aluno remanscente?',
        'Você é aluno remancescente?',
        'Aluno remanescente?',
        'Remanescente?'
    ]

    if not nome_arquivo:
        return 'Nenhum arquivo selecionado.'

    caminho_excel = os.path.join(uploads, nome_arquivo)
    try:
        df = pd.read_excel(caminho_excel)
        if df.shape[0] < 2:
            return 'A planilha precisa ter ao menos um gabarito e um aluno.'

        COL_NOME = 'Nome completo:'
        COL_MODALIDADE = 'Modalidade:'

        # >>> detecção robusta do e-mail
        COL_EMAIL = find_email_column(df.columns)
        if COL_EMAIL is None:
            cols_list = ', '.join([str(c) for c in df.columns])
            return f'Coluna de e-mail não encontrada. Colunas vistas: [{cols_list}]'

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
        idx_gab = df.index[mask_gab][0] if mask_gab.any() else df.index[0]

        gabarito = df.loc[idx_gab]
        alunos_df = df.drop(index=idx_gab).copy()

        # Filtro por modalidade
        if modalidade_filtro != 'Geral' and COL_MODALIDADE in alunos_df.columns:
            alunos_df = alunos_df[alunos_df[COL_MODALIDADE].astype(str).str.lower() == modalidade_filtro.lower()]

        # Filtro por "Tipo de aluno"
        REMAN_COL = next((c for c in REMAN_COL_CANDIDATES if c in df.columns), None)
        has_reman_col = REMAN_COL is not None

        if has_reman_col:
            reman_norm = alunos_df[REMAN_COL].astype(str).str.strip().str.lower()
            if tipo_aluno == 'Remanescente':
                alunos_df = alunos_df[reman_norm == 'sim']
            elif tipo_aluno in ('Não remanescente', 'Nao remanescente', 'Não Remanescente'):
                alunos_df = alunos_df[~(reman_norm == 'sim')]

        if alunos_df.empty:
            msg_mod = f' e modalidade "{modalidade_filtro}"' if modalidade_filtro != 'Geral' else ''
            msg_tipo = f' e tipo "{tipo_aluno}"' if tipo_aluno != 'Todos' else ''
            return f'Nenhum aluno encontrado{msg_mod}{msg_tipo}.'

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
            if ini >= fim:
                continue
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

        tipo_aluno_opcoes = ['Todos', 'Remanescente', 'Não remanescente']

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
            modalidade_filtro=modalidade_filtro,

            # filtros novos
            has_reman_col=(REMAN_COL is not None),
            tipo_aluno=tipo_aluno,
            tipo_aluno_opcoes=tipo_aluno_opcoes,
            reman_col_name=REMAN_COL if REMAN_COL else None,

            # nome real da coluna de e-mail detectada (se quiser usar no template)
            email_col_name=COL_EMAIL
        )

    except Exception as e:
        return f'Erro ao processar o dashboard: {e}'


# ===================== Dashboard de Marketing =====================

def allowed_file_marketing(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls'}


@app.route('/upload_marketing', methods=['GET'])
@login_required
def upload_marketing_page():
    arquivos = [f for f in os.listdir(MARKETING_UPLOAD_FOLDER)
                if f.lower().endswith(('.xlsx', '.xls'))]
    return render_template('upload_marketing.html', arquivos=arquivos)


@app.route('/upload_marketing', methods=['POST'])
@login_required
def upload_marketing_post():
    if 'file' not in request.files:
        return 'Nenhum arquivo enviado.', 400
    file = request.files['file']
    if file.filename == '':
        return 'Nome do arquivo vazio.', 400
    if not allowed_file_marketing(file.filename):
        return '❌ Tipo de arquivo não permitido. Envie .xlsx ou .xls.', 400

    filename = secure_filename(file.filename)
    savepath = os.path.join(MARKETING_UPLOAD_FOLDER, filename)
    file.save(savepath)

    # Redireciona para o dashboard já focado no arquivo
    return redirect(url_for('dashboard_marketing', arquivo=filename))


@app.route('/dashboard_marketing', methods=['GET', 'POST'])
@login_required
def dashboard_marketing():
    uploads_dir = MARKETING_UPLOAD_FOLDER
    os.makedirs(uploads_dir, exist_ok=True)

    # GET sem arquivo selecionado -> vai para a página de upload/seleção
    if request.method == 'GET' and not request.args.get('arquivo'):
        return redirect(url_for('upload_marketing_page'))

    # Aceita arquivo via POST (form) ou via GET (?arquivo=...)
    arquivo = request.form.get('arquivo') or request.args.get('arquivo')

    # Se ainda não há arquivo definido (ou não existe), volta para a seleção
    if not arquivo:
        return redirect(url_for('upload_marketing_page'))
    fullpath = os.path.join(uploads_dir, arquivo)
    if not os.path.isfile(fullpath):
        # arquivo informado não existe mais; volta para escolha
        return redirect(url_for('upload_marketing_page'))

    # --------- lógica de análise de marketing ---------
    arquivos = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.xlsx', '.xls'))]
    df = pd.read_excel(fullpath)

    rename_map = {
        'Data do Post':'data','Plataforma':'plataforma','Tipo de Conteúdo':'tipo','Tema do Post':'tema',
        'Post Turbinado? (Instagram)':'turbinado','Público-alvo':'publico','Link do Post':'link_post',
        'link':'link','Impressões/Interações':'impressoes','Alcance (visualizações)':'views',
        'Curtidas/Reações':'curtidas','Comentários':'comentarios','Compartilhamentos':'compart',
        'Cliques no Link':'cliques_link','Salvamentos (Instagram)':'salvamentos','Cliques no Perfil':'cliques_perfil',
        'Índice de Engajamento':'engajamento','Indice de Engajamento':'engajamento','Eficiência':'eficiencia'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    num_cols = ['impressoes','views','curtidas','comentarios','compart','cliques_link','salvamentos','cliques_perfil','eficiencia','engajamento']
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = (df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False))
        df[c] = pd.to_numeric(df[c], errors='coerce')

    for c in ['plataforma','tipo','tema','turbinado','publico','link','data']:
        if c not in df.columns:
            df[c] = ''

    def make_label(row):
        d = pd.to_datetime(row.get('data'), errors='coerce')
        ds = d.strftime('%d/%m') if not pd.isna(d) else str(row.get('data') or '')
        tema = str(row.get('tema') or '')
        if len(tema) > 24:
            tema = tema[:22] + '…'
        return f"{ds} • {str(row.get('tipo') or '').lower()} • {tema}"
    df['label'] = df.apply(make_label, axis=1)

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
    if sel['plataforma']   != 'Todos':
        fdf = fdf[fdf['plataforma'].astype(str)==sel['plataforma']]
    if sel['tipo_conteudo']!= 'Todos':
        fdf = fdf[fdf['tipo'].astype(str)==sel['tipo_conteudo']]
    if sel['tema_post']    != 'Todos':
        fdf = fdf[fdf['tema'].astype(str)==sel['tema_post']]
    if sel['turbinado']    != 'Todos':
        fdf = fdf[fdf['turbinado'].astype(str)==sel['turbinado']]
    if sel['publico']      != 'Todos':
        fdf = fdf[fdf['publico'].astype(str)==sel['publico']]

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
            'Variação Percentual': float((s.max()-s.min())/max(s.min(), 1e-9)*100),
            'Máximo': float(s.max()),
            'Mínimo': float(s.min()),
            'Q1': float(s.quantile(0.25)),
            'Mediana': float(s.median()),
            'Q3': float(s.quantile(0.75)),
            'Variância': float(s.var(ddof=1)) if len(s)>1 else 0.0
        }
    metrics_table = {nome: resumo(serie) for nome, serie in metric_series.items()}

    def safe_list(vals):
        out = []
        for v in vals:
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    fv = 0.0
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
        'top5_labels': safe_list([])
    }
    top5 = fdf[['label','eficiencia']].dropna().sort_values('eficiencia', ascending=False).head(5)
    charts_payload['top5'] = {
        'labels': top5['label'].tolist(),
        'values': safe_list(top5['eficiencia'].tolist())
    }

    display_cols = list(df.columns)
    if 'link' not in display_cols:
        df['link'] = ''
        display_cols.append('link')
    tabela = fdf[display_cols].fillna('').to_dict(orient='records')

    try:
        ate_data = pd.to_datetime(df['data'], errors='coerce').max()
        ate_data = ate_data.strftime('%d/%m/%Y') if pd.notna(ate_data) else ''
    except Exception:
        ate_data = ''

    return render_template('dashboard_marketing.html',
                           arquivos=arquivos,
                           arquivo=arquivo,
                           filtros_opts=filtros_opts, sel=sel,
                           charts_payload=charts_payload,
                           metrics_table=metrics_table,
                           tabela=tabela,
                           ate_data=ate_data,
                           erro=None)



# ===================== PERFIL ACADÊMICO 
# =====================

def load_alumni_from_excel():
    """
    Lê o arquivo 'arquivos/Formulário Perfil Acadêmico.xlsx'
    e devolve uma lista de dicts no formato esperado pelo front:
    id, nome, email, whatsapp, score
    """
    # caminho: <pasta_do_app>/arquivos/Formulário Perfil Acadêmico.xlsx
    excel_path = Path(app.root_path) / "arquivos" / "Formulário Perfil Acadêmico.xlsx"

    if not excel_path.exists():
        print(f"[AVISO] Arquivo não encontrado: {excel_path}")
        return []

    # Lê o Excel
    df = pd.read_excel(excel_path)

    # Normaliza nomes das colunas (para achar por pedaço do nome)
    col_map = {c.strip().lower(): c for c in df.columns}

    def achar_coluna(fragmento):
        fragmento = fragmento.lower()
        for k, original in col_map.items():
            if fragmento in k:
                return original
        return None

    # tenta achar as colunas mais prováveis
    nome_col = achar_coluna("nome completo")
    email_col = achar_coluna("email")
    telefone_col = achar_coluna("telefone") or achar_coluna("whats")
    desempenho_col = achar_coluna("desempenho nos simulados")

    people = []

    for idx, row in df.iterrows():
        # id sequencial
        pid = int(idx) + 1

        # nome
        if nome_col and not pd.isna(row.get(nome_col, None)):
            nome = str(row[nome_col]).strip()
        else:
            nome = f"Aluno {pid}"

        # email
        if email_col and not pd.isna(row.get(email_col, None)):
            email = str(row[email_col]).strip()
        else:
            email = ""

        # whatsapp / telefone
        if telefone_col and not pd.isna(row.get(telefone_col, None)):
            whatsapp = str(row[telefone_col]).strip()
        else:
            whatsapp = ""

        # score: mapeia a resposta de desempenho para um número (bem simples)
        score = 70
        if desempenho_col and not pd.isna(row.get(desempenho_col, None)):
            txt = str(row[desempenho_col]).strip().lower()
            if "excelente" in txt or "ótimo" in txt:
                score = 95
            elif "bom" in txt:
                score = 85
            elif "regular" in txt or "mediano" in txt:
                score = 75
            elif "ruim" in txt or "baixo" in txt:
                score = 60

        people.append({
            "id": pid,
            "nome": nome,
            "email": email,
            "whatsapp": whatsapp,
            "score": score,
        })

    return people

@app.route("/ex-alunos")
def ex_alunos():
    people = load_alumni_from_excel()
    return render_template("ex-alunos.html", people=people)



def _simple_counts(series):
    """Retorna labels e counts limpos de uma Series categórica."""
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    vc = s.value_counts()
    return list(vc.index), [int(x) for x in vc.values]


def _map_media(series, mapping):
    """Converte categorias em número pela mapping e devolve média (ou None)."""
    s = series.dropna().astype(str).str.strip()
    nums = s.map(mapping)
    nums = nums.dropna()
    if nums.empty:
        return None
    return float(nums.mean())


@app.route("/perfil_academico")
@login_required
def perfil_academico():
    """
    Lê o arquivo 'Formulário Perfil Acadêmico.xlsx' em arquivos/
    e monta o dicionário perfil_data que o template 'perfil_academico.html'
    espera para desenhar os gráficos.
    """
    excel_name = "Formulário Perfil Acadêmico.xlsx"
    excel_path = os.path.join(ARQUIVOS_FOLDER, excel_name)

    if not os.path.isfile(excel_path):
        return f"Arquivo {excel_name} não encontrado na pasta 'arquivos/'.", 404

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        return f"Erro ao ler o arquivo de perfil acadêmico: {e}", 500

    # ==== Nomes de colunas usados ====
    COL_ENSINO_MEDIO = 'Você está atualmente no ensino médio ou já se formou?'
    COL_TURNO = 'Caso ainda esteja no ensino médio, em qual turno você estuda?'
    COL_CONDICOES = 'Você tem alguma condição ou dificuldade de aprendizado que possamos considerar para melhor te acompanhar? '
    COL_DESEMP_GERAL = 'Como você classificaria seu desempenho nos simulados do Cursinho? '
    COL_EVOLUCAO = 'Qual sua evolução nos simulados ao longo do semestre? '
    COL_AREA_MAT = 'Como você avalia seu desempenho nas seguintes áreas dos simulados?  [Matemática]'
    COL_AREA_PORT = 'Como você avalia seu desempenho nas seguintes áreas dos simulados?  [Português]'
    COL_AREA_CN = 'Como você avalia seu desempenho nas seguintes áreas dos simulados?  [Ciências da Natureza]'
    COL_AREA_CH = 'Como você avalia seu desempenho nas seguintes áreas dos simulados?  [Ciências Humanas]'
    COL_AREA_RED = 'Como você avalia seu desempenho nas seguintes áreas dos simulados?  [Redação]'
    COL_COMPLEMENTARES = 'Você participou de atividades complementares do Cursinho (monitorias, plantões, eventos, etc.)? '
    COL_SIMULADOS = 'Você participou dos simulados?'
    COL_DISPOSITIVO = 'Em que dispositivo você faz os simulados?'
    COL_ESTUDO = 'Quantas horas por dia você consegue estudar além das aulas da escola e/ou do cursinho?'
    COL_SONO = 'Quantas horas de sono por dia você possui em média?'
    COL_LAZER = 'Quanto tempo você dedica ao lazer por dia?'
    COL_TRABALHO = 'Você trabalha e/ou ajuda nas tarefas domésticas em casa? Se sim, quanto tempo em média essas atividades levam por dia?'
    COL_DESLOC = 'Quanto tempo você leva para chegar ao Insper para as aulas dos sábados letivos?'
    COL_SOCIO = 'Você utiliza do programa de apoio socioemocional do Cursinho?'

    total_respostas = int(len(df))

    # ==== Situação escolar e turno ====
    ensino_labels, ensino_counts = _simple_counts(df.get(COL_ENSINO_MEDIO, pd.Series(dtype=object)))
    turno_labels, turno_counts = _simple_counts(df.get(COL_TURNO, pd.Series(dtype=object)))

    # ==== Desempenho geral ====
    desempen_labels, desempen_counts = _simple_counts(df.get(COL_DESEMP_GERAL, pd.Series(dtype=object)))

    # ==== Desempenho por área (Likert 1–5) ====
    rating_map = {
        'muito ruim': 1,
        'ruim': 2,
        'razoável': 3,
        'razoavel': 3,
        'bom': 4,
        'excelente': 5,
    }

    def area_media(col):
        s = df.get(col, pd.Series(dtype=object)).dropna().astype(str).str.strip().str.lower()
        s = s.replace({'razoavel': 'razoável'})  # normalização leve
        nums = s.map(rating_map)
        nums = nums.dropna()
        if nums.empty:
            return None
        return float(nums.mean())

    areas_labels = ["Matemática", "Português", "Ciências da Natureza", "Ciências Humanas", "Redação"]
    areas_medias = [
        area_media(COL_AREA_MAT),
        area_media(COL_AREA_PORT),
        area_media(COL_AREA_CN),
        area_media(COL_AREA_CH),
        area_media(COL_AREA_RED),
    ]
    # substitui None por 0 para o gráfico não quebrar
    areas_medias = [0 if v is None else round(v, 2) for v in areas_medias]

    # ==== Participação, complementares, dispositivo ====
    simulados_labels, simulados_counts = _simple_counts(df.get(COL_SIMULADOS, pd.Series(dtype=object)))
    compl_labels, compl_counts = _simple_counts(df.get(COL_COMPLEMENTARES, pd.Series(dtype=object)))
    disp_labels, disp_counts = _simple_counts(df.get(COL_DISPOSITIVO, pd.Series(dtype=object)))

    # ==== Condições / dificuldades ====
    cond_series = df.get(COL_CONDICOES, pd.Series(dtype=object)).dropna().astype(str)

    def classifica_cond(v: str) -> str:
        t = v.strip().lower()
        if not t:
            return "Não informou"
        # normalizações de "não" / "nenhum"
        base_neg = ["não", "nao", "nenhum", "nenhuma", "não possuo", "nao possuo"]
        if any(b in t for b in ["não sei", "nao sei"]):
            return "Não sabe / não informou"
        # se menciona explicitamente alguma condição
        cond_palavras = ["ansied", "tdah", "depress", "autis", "dislex", "transtorno", "déficit", "deficit"]
        if any(c in t for c in cond_palavras):
            return "Relatou condição"
        # caso só diga "não", "nenhum" etc.
        if any(b in t for b in base_neg):
            return "Não relatou"
        # fallback: considera que relatou algo
        return "Relatou condição"

    cond_categ = cond_series.map(classifica_cond)
    cond_labels, cond_counts = _simple_counts(cond_categ)

    # ==== Apoio socioemocional ====
    socio_labels, socio_counts = _simple_counts(df.get(COL_SOCIO, pd.Series(dtype=object)))

    # ==== Hábitos (categorias + médias aproximadas em horas) ====
    # mapas para converter categorias em horas aproximadas
    estudo_map = {
        'menos de 1h': 0.5,
        '1 - 2 horas': 1.5,
        '2 - 4 horas': 3.0,
        'acima de 4 horas': 4.5,
    }
    sono_map = {
        '4 a 6h': 5.0,
        '7 a 8h': 7.5,
        '8h+': 8.5,
    }
    lazer_map = {
        '0 a 2h': 1.0,
        '2 a 4h': 3.0,
        '4 a 5h': 4.5,
    }
    trabalho_map = {
        '0-1h': 0.5,
        '1-3h': 2.0,
        '3 -4h': 3.5,
        '4 -6h': 5.0,
        '6+h': 7.0,
    }
    desloc_map = {
        'menos de 30 minutos': 0.3,
        '30min - 1h': 0.75,
        '1h30min - 2h': 1.75,
        '2h ou mais': 2.5,
    }

    def map_media_general(series, mapping):
        s = series.dropna().astype(str).str.strip().str.lower()
        nums = s.map(mapping)
        nums = nums.dropna()
        if nums.empty:
            return None
        return float(nums.mean())

    # distribuições categóricas
    estudo_labels, estudo_counts = _simple_counts(df.get(COL_ESTUDO, pd.Series(dtype=object)))
    sono_labels, sono_counts = _simple_counts(df.get(COL_SONO, pd.Series(dtype=object)))
    lazer_labels, lazer_counts = _simple_counts(df.get(COL_LAZER, pd.Series(dtype=object)))
    trab_labels, trab_counts = _simple_counts(df.get(COL_TRABALHO, pd.Series(dtype=object)))
    desloc_labels, desloc_counts = _simple_counts(df.get(COL_DESLOC, pd.Series(dtype=object)))

    media_estudo = map_media_general(df.get(COL_ESTUDO, pd.Series(dtype=object)), estudo_map)
    media_sono = map_media_general(df.get(COL_SONO, pd.Series(dtype=object)), sono_map)

    perfil_data = {
        "total_respostas": total_respostas,

        "ensino_medio": {
            "labels": ensino_labels,
            "counts": ensino_counts,
        },
        "turno": {
            "labels": turno_labels,
            "counts": turno_counts,
        },
        "desempenho_geral": {
            "labels": desempen_labels,
            "counts": desempen_counts,
        },
        "desempenho_areas": {
            "labels": areas_labels,
            "medias": areas_medias,
        },
        "simulados": {
            "labels": simulados_labels,
            "counts": simulados_counts,
        },
        "complementares": {
            "labels": compl_labels,
            "counts": compl_counts,
        },
        "dispositivo": {
            "labels": disp_labels,
            "counts": disp_counts,
        },
        "estudo": {
            "labels": estudo_labels,
            "counts": estudo_counts,
        },
        "sono": {
            "labels": sono_labels,
            "counts": sono_counts,
        },
        "lazer": {
            "labels": lazer_labels,
            "counts": lazer_counts,
        },
        "trabalho": {
            "labels": trab_labels,
            "counts": trab_counts,
        },
        "deslocamento": {
            "labels": desloc_labels,
            "counts": desloc_counts,
        },
        "condicoes": {
            "labels": cond_labels,
            "counts": cond_counts,
        },
        "socioemocional": {
            "labels": socio_labels,
            "counts": socio_counts,
        },
        "medias_resumo": {
            "sono": media_sono,
            "estudo": media_estudo,
        },
    }

    # Renderiza o template que você criou
    return render_template("perfil_academico.html", perfil_data=perfil_data)


# =========================================================

if __name__ == '__main__':
    app.run(debug=True)
