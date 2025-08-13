# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import uuid
import time
from datetime import datetime, timezone

st.set_page_config(page_title="Avaliação Humana do Modelo", layout="wide")

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")
ACCESS_CODE = st.secrets.get("ACCESS_CODE", "")  
DB_PATH = "avaliacao.db"

# ---------- Util: DB ----------
def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return c

def init_db():
    with conn() as c:
        cur = c.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            q_hash TEXT UNIQUE,
            pergunta TEXT NOT NULL,
            grupo TEXT NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            resposta TEXT NOT NULL,
            label_original INTEGER,
            label_modelo INTEGER,
            FOREIGN KEY(question_id) REFERENCES questions(id)
        )""")
        # Participantes
        cur.execute("""
        CREATE TABLE IF NOT EXISTS respondents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            respondent_uuid TEXT UNIQUE,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            profile TEXT NOT NULL,           -- 'comum' | 'saude'
            health_area TEXT,                -- se profile='saude'
            created_at TEXT,
            has_submitted INTEGER DEFAULT 0, -- 0/1
            submitted_at TEXT
        )""")
        # Amostragem congelada por participante
        cur.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            respondent_id INTEGER NOT NULL,
            answer_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            grupo TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            UNIQUE(respondent_id, answer_id),
            FOREIGN KEY(respondent_id) REFERENCES respondents(id),
            FOREIGN KEY(answer_id) REFERENCES answers(id),
            FOREIGN KEY(question_id) REFERENCES questions(id)
        )""")
        # Avaliações finais
        cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            respondent_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            answer_id INTEGER NOT NULL,
            is_useful INTEGER NOT NULL,      -- 0/1
            created_at TEXT,
            FOREIGN KEY(respondent_id) REFERENCES respondents(id),
            FOREIGN KEY(question_id) REFERENCES questions(id),
            FOREIGN KEY(answer_id) REFERENCES answers(id)
        )""")

        # Índices úteis
        cur.executescript("""
        CREATE INDEX IF NOT EXISTS idx_answers_qid ON answers(question_id);
        CREATE INDEX IF NOT EXISTS idx_assignments_resp ON assignments(respondent_id);
        CREATE INDEX IF NOT EXISTS idx_assignments_order ON assignments(respondent_id, order_index);
        CREATE INDEX IF NOT EXISTS idx_eval_resp ON evaluations(respondent_id);
        CREATE INDEX IF NOT EXISTS idx_eval_ans ON evaluations(answer_id);
        """)
        c.commit()

init_db()

# ---------- Estado ----------
if "respondent_uuid" not in st.session_state:
    st.session_state.respondent_uuid = str(uuid.uuid4())
if "respondent_id" not in st.session_state:
    st.session_state.respondent_id = None
if "assignment" not in st.session_state:
    st.session_state.assignment = None  # DataFrame com os 118 pares (congelado)
if "progress_idx" not in st.session_state:
    st.session_state.progress_idx = 0
if "answers_buffer" not in st.session_state:
    st.session_state.answers_buffer = {}  # answer_id -> 0/1 (só comita no final)
if "access_ok" not in st.session_state:
    st.session_state.access_ok = False if ACCESS_CODE else True
# persistência leve para retomar sessão
for k in ["user_name","user_email","user_profile","user_health_area"]:
    st.session_state.setdefault(k, "")

# ---------- Normalização de grupos e cotas ----------
GROUP_MAP = {
    "sintomas diagnósticos/consultas": "SDC",
    "sintomas e diagnóstico": "SDC",
    "sintomas e diagnostico": "SDC",
    "sintomas diagnósticos": "SDC",
    "sintomas diagnosticos/consultas": "SDC",
    "sintomas diagnosticos": "SDC",

    "sintomas físicos": "SF",
    "sintomas fisicos": "SF",

    "tratamento e medicação": "TM",
    "tratamento e medicacao": "TM",
}
QUOTAS = {"SDC": 66, "SF": 17, "TM": 35}
TOTAL_TARGET = sum(QUOTAS.values())  # 118

def norm_group(g: str) -> str:
    key = (g or "").strip().lower()
    return GROUP_MAP.get(key, "OUTROS")

# ---------- Corpus: importação ----------
def read_corpus(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(name):
        return cols_lower.get(name.lower())
    need = ["pergunta","resposta","label_original","label_modelo","grupo"]
    mapped = {k: pick(k) for k in need}
    missing = [k for k,v in mapped.items() if v is None]
    if missing:
        raise ValueError(f"CSV faltando colunas: {missing}. Esperado: {need}")

    df = df[[mapped[k] for k in need]].rename(columns={mapped[k]: k for k in need})
    df["pergunta"] = df["pergunta"].astype(str)
    df["resposta"] = df["resposta"].astype(str)
    df["grupo_raw"] = df["grupo"].astype(str)
    df["grupo"] = df["grupo_raw"].map(norm_group)
    for c in ["label_original","label_modelo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def import_corpus(df: pd.DataFrame):
    with conn() as c:
        cur = c.cursor()
        # Inserir perguntas (dedup por pergunta + grupo_raw)
        for per, grp_raw in df[["pergunta","grupo_raw"]].drop_duplicates().itertuples(index=False):
            q_hash = f"{hash(per)}::{hash(grp_raw)}"
            cur.execute("""
                INSERT OR IGNORE INTO questions (q_hash, pergunta, grupo)
                VALUES (?,?,?)
            """, (q_hash, per, grp_raw))
        c.commit()

        # Map pergunta->id
        cur.execute("SELECT id, pergunta, grupo FROM questions")
        qmap = {(row[1], row[2]): row[0] for row in cur.fetchall()}

        # Inserir respostas
        for row in df.itertuples(index=False):
            qid = qmap[(row.pergunta, row.grupo_raw)]
            cur.execute("""
                INSERT INTO answers (question_id, resposta, label_original, label_modelo)
                VALUES (?,?,?,?)
            """, (qid, row.resposta, int(row.label_original), int(row.label_modelo)))
        c.commit()

# ---------- Fetchers ----------
def fetch_grouped_answers_for_sampling():
    """Retorna dict por estrato normalizado {'SDC': df, 'SF': df, 'TM': df} + df total."""
    with conn() as c:
        cur = c.cursor()
        cur.execute("""
            SELECT a.id, a.question_id, q.pergunta, q.grupo, a.resposta
            FROM answers a
            JOIN questions q ON q.id = a.question_id
        """)
        rows = cur.fetchall()
    data = []
    for aid, qid, pergunta, grupo_raw, resp in rows:
        g_norm = norm_group(grupo_raw)
        data.append({"answer_id": aid, "question_id": qid, "pergunta": pergunta,
                     "grupo_norm": g_norm, "resposta": resp})
    df = pd.DataFrame(data)
    grouped = {k: df[df["grupo_norm"]==k].copy() for k in ["SDC","SF","TM"]}
    return grouped, df

def ensure_respondent(email: str, name: str, profile: str, health_area: str|None) -> int:
    with conn() as c:
        cur = c.cursor()
        cur.execute("SELECT id, has_submitted FROM respondents WHERE email = ?", (email,))
        row = cur.fetchone()
        if row:
            rid, _ = row
            cur.execute("""
                UPDATE respondents SET name=?, profile=?, health_area=? WHERE id=?
            """, (name, profile, health_area, rid))
            c.commit()
            return rid
        cur.execute("""
            INSERT INTO respondents (respondent_uuid, name, email, profile, health_area, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (st.session_state.respondent_uuid, name, email, profile, health_area,
              datetime.now(timezone.utc).isoformat()))
        c.commit()
        return cur.lastrowid

def respondent_status(email: str):
    with conn() as c:
        cur = c.cursor()
        cur.execute("SELECT id, has_submitted, name FROM respondents WHERE email = ?", (email,))
        return cur.fetchone()  # None | (id, has_submitted, name)

def has_assignments(respondent_id: int) -> bool:
    with conn() as c:
        cur = c.cursor()
        cur.execute("SELECT COUNT(*) FROM assignments WHERE respondent_id=?", (respondent_id,))
        return cur.fetchone()[0] > 0

def create_assignments_for_user(respondent_id: int):
    grouped, _ = fetch_grouped_answers_for_sampling()
    rng = np.random.default_rng()
    samples = []
    for key, quota in QUOTAS.items():
        pool = grouped.get(key, pd.DataFrame())
        if len(pool) < quota:
            raise ValueError(f"Corpus insuficiente para estrato {key}: precisa {quota}, tem {len(pool)}.")
        idx = rng.choice(pool.index, size=quota, replace=False)
        samples.append(pool.loc[idx])

    df_ass = pd.concat(samples, ignore_index=True)
    df_ass = df_ass.sample(frac=1.0, random_state=rng.integers(0, 1_000_000)).reset_index(drop=True)
    df_ass["order_index"] = np.arange(len(df_ass))

    with conn() as c:
        cur = c.cursor()
        for row in df_ass.itertuples(index=False):
            cur.execute("""
                INSERT OR IGNORE INTO assignments (respondent_id, answer_id, question_id, grupo, order_index)
                VALUES (?,?,?,?,?)
            """, (respondent_id, int(row.answer_id), int(row.question_id), row.grupo_norm, int(row.order_index)))
        c.commit()
    return df_ass

def load_assignments(respondent_id: int) -> pd.DataFrame:
    with conn() as c:
        cur = c.cursor()
        cur.execute("""
            SELECT asg.answer_id, asg.question_id, asg.grupo, asg.order_index,
                   q.pergunta, a.resposta
            FROM assignments asg
            JOIN questions q ON q.id = asg.question_id
            JOIN answers a ON a.id = asg.answer_id
            WHERE asg.respondent_id=?
            ORDER BY asg.order_index ASC
        """, (respondent_id,))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["answer_id","question_id","grupo","order_index","pergunta","resposta"])

def commit_final_evaluations(respondent_id: int, buffer: dict[int, int]):
    """
    Finaliza a participação:
      - Garante que todas as respostas da amostragem estão no banco (via UPSERT).
      - Não gera conflito com o índice UNIQUE(respondent_id, answer_id).
      - Marca o participante como submitted.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Descobre se a tabela tem updated_at para atualizar no UPSERT
    with conn() as c:
        cols_info = c.execute("PRAGMA table_info(evaluations)").fetchall()
        colnames = {row[1] for row in cols_info}
    has_updated = "updated_at" in colnames

    with conn() as c:
        cur = c.cursor()

        # Carrega a amostragem (precisamos do question_id por answer)
        cur.execute("""
            SELECT answer_id, question_id
            FROM assignments
            WHERE respondent_id=?
            ORDER BY order_index ASC
        """, (respondent_id,))
        ass = cur.fetchall()
        total = len(ass)

        if len(buffer) != total:
            raise ValueError(f"Você respondeu {len(buffer)} de {total} itens. Complete todos antes de enviar.")

        # UPSERT por item (caso algum clique tenha faltado ou para garantir consistência)
        for aid, qid in ass:
            aid = int(aid)
            qid = int(qid)
            val = int(buffer[aid])

            if has_updated:
                cur.execute("""
                    INSERT INTO evaluations (respondent_id, question_id, answer_id, is_useful, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(respondent_id, answer_id)
                    DO UPDATE SET
                        is_useful = excluded.is_useful,
                        question_id = excluded.question_id,
                        updated_at = CURRENT_TIMESTAMP
                """, (respondent_id, qid, aid, val, now))
            else:
                cur.execute("""
                    INSERT INTO evaluations (respondent_id, question_id, answer_id, is_useful, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(respondent_id, answer_id)
                    DO UPDATE SET
                        is_useful = excluded.is_useful,
                        question_id = excluded.question_id
                """, (respondent_id, qid, aid, val, now))

        # Marca o respondente como submetido
        cur.execute("""
            UPDATE respondents
               SET has_submitted=1,
                   submitted_at=?
             WHERE id=?
        """, (now, respondent_id))

        c.commit()


def clear_all_responses():
    with conn() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM evaluations")
        cur.execute("DELETE FROM assignments")
        cur.execute("UPDATE respondents SET has_submitted=0, submitted_at=NULL")
        c.commit()

# === NOVO: Helpers para inspecionar/remover respostas por usuário ===
def list_participants_for_admin() -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query("""
            SELECT
              r.id AS respondent_id,
              r.name,
              r.email,
              r.profile,
              COALESCE(r.health_area, '') AS health_area,
              r.created_at,
              r.has_submitted,
              r.submitted_at,
              (SELECT COUNT(*) FROM evaluations e WHERE e.respondent_id = r.id) AS n_avaliacoes,
              (SELECT COUNT(*) FROM assignments a WHERE a.respondent_id = r.id) AS n_assignments
            FROM respondents r
            ORDER BY datetime(r.created_at) DESC
        """, c)
    return df

def get_user_evaluations(respondent_id: int) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query("""
            SELECT
              e.id AS evaluation_id,
              e.created_at,
              e.is_useful,
              q.pergunta,
              a.resposta,
              q.grupo AS grupo_raw,
              a.label_modelo,
              a.label_original,
              e.answer_id,
              e.question_id
            FROM evaluations e
            JOIN answers a   ON a.id = e.answer_id
            JOIN questions q ON q.id = a.question_id
            WHERE e.respondent_id = ?
            ORDER BY datetime(e.created_at) DESC, e.id DESC
        """, c, params=(respondent_id,))
    if not df.empty:
        df["grupo_norm"] = df["grupo_raw"].map(norm_group)
    return df

def delete_user_evaluations(respondent_id: int):
    with conn() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM evaluations WHERE respondent_id = ?", (respondent_id,))
        c.commit()

def delete_user_evaluations_and_reopen(respondent_id: int):
    with conn() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM evaluations WHERE respondent_id = ?", (respondent_id,))
        cur.execute("DELETE FROM assignments WHERE respondent_id = ?", (respondent_id,))
        cur.execute("UPDATE respondents SET has_submitted=0, submitted_at=NULL WHERE id=?", (respondent_id,))
        c.commit()

# ---------- Helpers de sessão ----------
def participant_gate():
    if st.session_state.access_ok:
        return True
    st.info("Insira o código de acesso fornecido pelo organizador.")
    code = st.text_input("Código de acesso", type="password")
    if st.button("Validar código"):
        if code.strip() == ACCESS_CODE.strip():
            st.session_state.access_ok = True
            st.rerun()
        else:
            st.error("Código inválido.")
    return False

def bootstrap_user_session_by_email():
    """Recarrega respondent_id e assignments usando o e‑mail salvo (se existir)."""
    if st.session_state.get("respondent_id"):
        return
    email = st.session_state.get("user_email")
    if not email:
        return
    status = respondent_status(email)
    if not status:
        return
    rid, has_submitted, _ = status
    if has_submitted == 1:
        st.warning("Este e‑mail já enviou uma avaliação. Obrigado!")
        return
    st.session_state.respondent_id = rid
    if st.session_state.assignment is None or getattr(st.session_state.assignment, "empty", True):
        st.session_state.assignment = load_assignments(rid)

def ensure_assignment_loaded():
    """Garante DataFrame de amostragem carregado na sessão."""
    if st.session_state.assignment is None or getattr(st.session_state.assignment, "empty", True):
        st.session_state.assignment = load_assignments(st.session_state.respondent_id)
    return st.session_state.assignment

# ---------- Navegação ----------
st.sidebar.title("Avaliação Das Respostas do Modelo ")
page = st.sidebar.radio(
    "Navegação",
    ["Participar", "Admin ▸ Importar corpus", "Admin ▸ Resultados", "Admin ▸ Participantes", "Admin ▸ Limpar respostas"],
    index=0
)

# ---------- Página: Participar ----------
if page == "Participar":
    st.header("Participar da Avaliação")
    if ACCESS_CODE and not participant_gate():
        st.stop()

    # ---------- MIGRAÇÃO / SCHEMA (garante colunas/índice para UPSERT) ----------
    def ensure_evaluations_schema():
        """
        - Cria colunas created_at / updated_at se não existirem.
        - Garante índice único (respondent_id, answer_id) para permitir UPSERT.
        Retorna um set com os nomes das colunas atuais.
        """
        with conn() as c:
            cols_info = c.execute("PRAGMA table_info(evaluations)").fetchall()
            colnames = {row[1] for row in cols_info}

            # Timestamps (se faltar, adiciona)
            if "created_at" not in colnames:
                c.execute("ALTER TABLE evaluations ADD COLUMN created_at TEXT")
                colnames.add("created_at")
            if "updated_at" not in colnames:
                c.execute("ALTER TABLE evaluations ADD COLUMN updated_at TEXT")
                colnames.add("updated_at")

            # Índice único para (respondent_id, answer_id)
            idx_list = c.execute("PRAGMA index_list(evaluations)").fetchall()
            existing_idx = {row[1] for row in idx_list}
            if "ux_eval_resp_answer" not in existing_idx:
                c.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS ux_eval_resp_answer
                    ON evaluations(respondent_id, answer_id)
                """)
        return colnames

    # Garante schema OK para UPSERTs e guarda as colunas disponíveis
    EVAL_COLS = ensure_evaluations_schema()

    # ---------- HELPERS DE PERSISTÊNCIA / RETOMADA ----------
    def get_question_id_by_answer(answer_id: int) -> int | None:
        with conn() as c:
            row = c.execute("SELECT question_id FROM answers WHERE id=?", (answer_id,)).fetchone()
        return int(row[0]) if row else None

    def upsert_evaluation(respondent_id: int, answer_id: int, is_useful: int, question_id: int | None = None):
        """
        UPSERT resiliente:
          - inclui question_id (se existir no schema);
          - usa created_at/updated_at se existirem;
          - requer UNIQUE(respondent_id, answer_id).
        """
        has_created = "created_at" in EVAL_COLS
        has_updated = "updated_at" in EVAL_COLS
        has_qid     = "question_id" in EVAL_COLS

        if has_qid and question_id is None:
            question_id = get_question_id_by_answer(answer_id)
            if question_id is None:
                raise ValueError(f"Não foi possível resolver question_id para answer_id={answer_id}")

        cols   = ["respondent_id", "answer_id", "is_useful"]
        vals   = ["?", "?", "?"]
        params = [respondent_id, answer_id, int(bool(is_useful))]

        if has_qid:
            cols.append("question_id")
            vals.append("?")
            params.append(int(question_id))

        if has_created:
            cols.append("created_at")
            vals.append("CURRENT_TIMESTAMP")
        if has_updated:
            cols.append("updated_at")
            vals.append("CURRENT_TIMESTAMP")

        insert_cols = ", ".join(cols)
        insert_vals = ", ".join(vals)

        set_parts = ["is_useful=excluded.is_useful"]
        if has_qid:
            set_parts.append("question_id=excluded.question_id")
        if has_updated:
            set_parts.append("updated_at=CURRENT_TIMESTAMP")
        set_clause = ", ".join(set_parts)

        sql = f"""
            INSERT INTO evaluations ({insert_cols})
            VALUES ({insert_vals})
            ON CONFLICT(respondent_id, answer_id)
            DO UPDATE SET {set_clause}
        """
        with conn() as c:
            c.execute(sql, params)

    def load_answered_map(respondent_id: int, valid_answer_ids: list[int]) -> dict[int, int]:
        """Retorna {answer_id: is_useful} para as respostas da amostragem atual."""
        if not valid_answer_ids:
            return {}
        placeholders = ",".join(["?"] * len(valid_answer_ids))
        with conn() as c:
            rows = c.execute(f"""
                SELECT answer_id, CASE WHEN is_useful THEN 1 ELSE 0 END AS is_useful
                FROM evaluations
                WHERE respondent_id=? AND answer_id IN ({placeholders})
            """, (respondent_id, *valid_answer_ids)).fetchall()
        return {int(r[0]): int(r[1]) for r in rows}

    def first_unanswered_index(df_ass: pd.DataFrame, answered_ids: set[int]) -> int:
        """Índice do primeiro item não respondido (ou último, se todos respondidos)."""
        for i, aid in enumerate(df_ass["answer_id"].tolist()):
            if int(aid) not in answered_ids:
                return i
        return max(0, len(df_ass) - 1)

    def resume_from_db_if_possible():
        """Reconstrói answers_buffer a partir do banco e posiciona progress_idx."""
        rid = st.session_state.get("respondent_id")
        if not rid:
            return
        if st.session_state.get("assignment") is None:
            st.session_state.assignment = load_assignments(rid)

        df_ass = st.session_state.assignment
        if df_ass is None or df_ass.empty:
            return

        valid_ids = [int(x) for x in df_ass["answer_id"].tolist()]
        answered_map = load_answered_map(rid, valid_ids)

        buf = st.session_state.get("answers_buffer", {})
        buf.update(answered_map)
        st.session_state.answers_buffer = buf
        st.session_state.progress_idx = first_unanswered_index(df_ass, set(buf.keys()))

    def bootstrap_user_session_by_email_enhanced():
        """Se houver e-mail salvo e não houver respondent_id, reabre participação."""
        email = st.session_state.get("user_email")
        if not email or st.session_state.get("respondent_id"):
            return
        status = respondent_status(email)  # (respondent_id, has_submitted, ...)
        if status:
            rid = int(status[0])
            st.session_state.respondent_id = rid
            st.session_state.assignment = load_assignments(rid)
            if st.session_state.assignment is not None and not st.session_state.assignment.empty:
                resume_from_db_if_possible()

    # Retoma sessão básica (após reload)
    bootstrap_user_session_by_email_enhanced()

    # ---------- FORM DE IDENTIFICAÇÃO ----------
    with st.form("form-identificacao", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nome completo*", max_chars=120, value=st.session_state.get("user_name", ""))
        with col2:
            email = st.text_input("E-mail*", max_chars=180, value=st.session_state.get("user_email", ""))

        profile = st.radio(
            "Perfil*", ["Usuário comum", "Profissional da Saúde"], horizontal=True,
            index=1 if st.session_state.get("user_profile") == "saude" else 0
        )
        health_area = None
        if profile == "Profissional da Saúde":
            health_area = st.text_input(
                "Área de atuação (ex.: enfermagem, medicina, fisioterapia etc.)*",
                value=st.session_state.get("user_health_area", "")
            )
        start = st.form_submit_button("Iniciar / Continuar ▶️", type="primary")

    if start:
        if not name or not email:
            st.error("Preencha nome e e-mail.")
            st.stop()
        if profile == "Profissional da Saúde" and not health_area:
            st.error("Informe a área de atuação.")
            st.stop()

        # Salva dados básicos na sessão
        st.session_state.user_name = name.strip()
        st.session_state.user_email = email.strip()
        st.session_state.user_profile = "saude" if profile == "Profissional da Saúde" else "comum"
        st.session_state.user_health_area = (health_area or "").strip()

        status = respondent_status(st.session_state.user_email)
        if status and status[1] == 1:  # já submetido
            st.warning("Este e-mail já enviou uma avaliação. Obrigado!")
            st.stop()

        # Cria/atualiza respondent
        rid = ensure_respondent(
            email=st.session_state.user_email,
            name=st.session_state.user_name,
            profile=st.session_state.user_profile,
            health_area=st.session_state.user_health_area if st.session_state.user_profile == "saude" else None
        )
        st.session_state.respondent_id = rid

        # Cria amostragem se não existir
        if not has_assignments(rid):
            try:
                create_assignments_for_user(rid)
            except Exception as e:
                st.error(f"Não foi possível criar sua amostragem: {e}")
                st.stop()

        # Carrega amostragem e retoma progresso
        st.session_state.assignment = load_assignments(rid)
        st.session_state.answers_buffer = st.session_state.get("answers_buffer", {})
        resume_from_db_if_possible()

        if len(st.session_state.answers_buffer) == 0:
            st.success("Amostragem pronta. Você vai avaliar 118 pares pergunta-resposta.")

        st.rerun()

    # ---------- INTERFACE DE ROTULAGEM (com persistência contínua) ----------
    if st.session_state.get("respondent_id"):
        if st.session_state.get("assignment") is None:
            st.session_state.assignment = load_assignments(st.session_state["respondent_id"])

        df_ass = st.session_state.assignment
        if df_ass is None or df_ass.empty:
            st.info("Ainda não há amostragem. Clique em 'Iniciar / Continuar'.")
            st.stop()

        resume_from_db_if_possible()

        idx = int(st.session_state.get("progress_idx", 0))
        total = len(df_ass)
        answered_count = len(st.session_state.get("answers_buffer", {}))

        st.progress(answered_count / total if total else 0)
        st.caption(f"Progresso: {answered_count} / {total}")

        current = df_ass.iloc[idx]
        st.subheader("Pergunta")
        st.write(current["pergunta"])
        st.subheader("Resposta")
        st.write(current["resposta"])
        st.caption(f"Grupo (normalizado): **{current['grupo']}**")

        a_id = int(current["answer_id"])
        if "question_id" in current and pd.notna(current["question_id"]):
            q_id = int(current["question_id"])
        else:
            q_id = get_question_id_by_answer(a_id)

        rid = int(st.session_state["respondent_id"])

        cols = st.columns(2)
        with cols[0]:
            if st.button("Útil ✅", use_container_width=True, key=f"useful_{idx}"):
                upsert_evaluation(rid, a_id, 1, q_id)
                st.session_state.answers_buffer[a_id] = 1
                st.session_state.progress_idx = first_unanswered_index(df_ass, set(st.session_state.answers_buffer.keys()))
                st.rerun()
        with cols[1]:
            if st.button("Não útil / Pouco útil ❌", use_container_width=True, key=f"notuseful_{idx}"):
                upsert_evaluation(rid, a_id, 0, q_id)
                st.session_state.answers_buffer[a_id] = 0
                st.session_state.progress_idx = first_unanswered_index(df_ass, set(st.session_state.answers_buffer.keys()))
                st.rerun()

        st.divider()
        colA, colB = st.columns([1, 2])
        with colA:
            st.write("Resumo desta participação")
            st.write(f"Respondidos: {len(st.session_state.answers_buffer)} / {total}")
            st.write(f"Em aberto: {total - len(st.session_state.answers_buffer)}")
        with colB:
            if len(st.session_state.answers_buffer) == total:
                if st.button("Enviar tudo 🚀 (finalizar participação)", type="primary", key="submit_all"):
                    try:
                        commit_final_evaluations(st.session_state.respondent_id, st.session_state.answers_buffer)
                        st.success("Respostas enviadas com sucesso. Obrigado!")
                        st.balloons()
                        st.session_state.respondent_id = None
                        st.session_state.assignment = None
                        st.session_state.progress_idx = 0
                        st.session_state.answers_buffer = {}
                    except Exception as e:
                        st.error(f"Erro ao enviar: {e}")


# ---------- Página: Admin ▸ Importar corpus ----------
if page == "Admin ▸ Importar corpus":
    st.header("Admin ▸ Importar corpus (CSV)")
    pwd = st.text_input("Senha de admin", type="password")
    if st.button("Entrar", key="login_import"):
        if pwd == ADMIN_PASSWORD and ADMIN_PASSWORD:
            st.session_state["admin_ok_import"] = True
            st.rerun()
        else:
            st.error("Senha inválida.")
    if not st.session_state.get("admin_ok_import"):
        st.stop()

    st.success("Autenticado como admin.")
    up = st.file_uploader("Selecione o CSV", type=["csv"])
    if up:
        try:
            df = read_corpus(up)
            st.info("Prévia das primeiras linhas normalizadas:")
            st.dataframe(df.head(30), use_container_width=True)

            # Diagnóstico de cotas
            diag = df.groupby("grupo")["resposta"].count().rename("itens").reset_index()
            diag["cota_exigida"] = diag["grupo"].map(QUOTAS).fillna(0).astype(int)
            st.subheader("Diagnóstico de cotas (normalizado)")
            st.dataframe(diag, use_container_width=True)

            if st.button("Importar corpus para o banco", type="primary"):
                import_corpus(df)
                st.success(f"Importados {len(df)} pares pergunta‑resposta.")
        except Exception as e:
            st.error(f"Erro ao ler/importar CSV: {e}")

# ---------- Página: Admin ▸ Participantes ----------
if page == "Admin ▸ Participantes":
    st.header("Admin ▸ Participantes cadastrados")
    pwd = st.text_input("Senha de admin", type="password")
    if st.button("Entrar", key="login_participants"):
        if pwd == ADMIN_PASSWORD and ADMIN_PASSWORD:
            st.session_state["admin_ok_participants"] = True
            st.rerun()
        else:
            st.error("Senha inválida.")
    if not st.session_state.get("admin_ok_participants"):
        st.stop()

    st.success("Autenticado como admin.")

    # Filtros
    colf1, colf2, colf3 = st.columns([1,1,2])
    with colf1:
        f_profile = st.selectbox("Perfil", ["Todos", "Usuário comum", "Profissional da Saúde"], index=0)
    with colf2:
        f_status = st.selectbox("Status", ["Todos", "Enviaram", "Não enviaram"], index=0)
    with colf3:
        f_search = st.text_input("Buscar por nome ou e‑mail")

    with conn() as c:
        df_users = pd.read_sql_query("""
            SELECT
              name,
              email,
              CASE profile WHEN 'saude' THEN 'Profissional da Saúde' ELSE 'Usuário comum' END AS perfil,
              COALESCE(health_area, '') AS area_atuacao,
              created_at,
              has_submitted,
              submitted_at
            FROM respondents
            ORDER BY datetime(created_at) DESC
        """, c)

    # Aplica filtros
    if f_profile != "Todos":
        df_users = df_users[df_users["perfil"] == f_profile]
    if f_status != "Todos":
        if f_status == "Enviaram":
            df_users = df_users[df_users["has_submitted"] == 1]
        else:
            df_users = df_users[df_users["has_submitted"] == 0]
    if f_search:
        s = f_search.strip().lower()
        df_users = df_users[df_users["name"].str.lower().str.contains(s) | df_users["email"].str.lower().str.contains(s)]

    # KPIs
    total = len(df_users)
    enviados = int((df_users["has_submitted"] == 1).sum())
    saude = int((df_users["perfil"] == "Profissional da Saúde").sum())
    comuns = int((df_users["perfil"] == "Usuário comum").sum())

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("Total cadastrados", total)
    with k2: st.metric("Profissionais da Saúde", saude)
    with k3: st.metric("Usuários comuns", comuns)
    with k4: st.metric("Já enviaram", enviados)

    # Ajustes visuais/colunas
    df_view = df_users.rename(columns={
        "name": "Nome",
        "email": "E‑mail",
        "perfil": "Perfil",
        "area_atuacao": "Área de atuação",
        "created_at": "Cadastrado em",
        "submitted_at": "Enviado em",
        "has_submitted": "Enviou?"
    }).copy()

    df_view["Enviou?"] = df_view["Enviou?"].map({0: "Não", 1: "Sim"})

    st.dataframe(
        df_view[["Nome","E‑mail","Perfil","Área de atuação","Cadastrado em","Enviado em","Enviou?"]],
        use_container_width=True
    )

    #apagar usuarios
    if st.button("Apagar TODOS os participantes (irrevogável)", type="primary"):
        with conn() as c:
            cur = c.cursor()
            cur.execute("DELETE FROM respondents")
            cur.execute("DELETE FROM evaluations")
            cur.execute("DELETE FROM assignments")
            c.commit()
        st.success("Todos os participantes apagados com sucesso.")
        st.rerun()

    # Download
    st.download_button(
        "Baixar CSV dos participantes",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name=f"participantes_{int(time.time())}.csv",
        mime="text/csv"
    )

# ---------- Página: Admin ▸ Resultados ----------
if page == "Admin ▸ Resultados":
    st.header("Admin ▸ Resultados")

    # Login de admin
    pwd = st.text_input("Senha de admin", type="password", key="pwd_results")
    if st.button("Entrar", key="login_results"):
        if pwd == ADMIN_PASSWORD and ADMIN_PASSWORD:
            st.session_state["admin_ok_results"] = True
            st.rerun()
        else:
            st.error("Senha inválida.")
    if not st.session_state.get("admin_ok_results"):
        st.stop()

    st.success("Autenticado como admin.")

    # ===================== Consultas principais =====================
    with conn() as c:
        # Métricas agregadas por answer
        df_eval = pd.read_sql_query("""
            SELECT e.answer_id,
                   AVG(e.is_useful)*100.0 AS pct_util_humano,
                   COUNT(*) AS n_avaliacoes
            FROM evaluations e
            GROUP BY e.answer_id
        """, c)

        # Pares com metadados
        df_pairs = pd.read_sql_query("""
            SELECT a.id as answer_id, a.question_id, a.resposta, a.label_modelo, a.label_original,
                   q.pergunta, q.grupo as grupo_raw
            FROM answers a
            JOIN questions q ON q.id = a.question_id
        """, c)

        # Join principal
        df_all = df_pairs.merge(df_eval, on="answer_id", how="left")
        df_all["pct_util_humano"] = df_all["pct_util_humano"].fillna(0.0)
        df_all["n_avaliacoes"] = df_all["n_avaliacoes"].fillna(0).astype(int)
        df_all["grupo_norm"] = df_all["grupo_raw"].map(norm_group)

        # -------------------- KPIs gerais --------------------
        total_pairs = df_all["answer_id"].nunique()
        covered_pairs = (df_all["n_avaliacoes"] > 0).sum()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pares no corpus", total_pairs)
        with c2:
            st.metric("Pares já avaliados", covered_pairs)
        with c3:
            st.metric("Cobertura", f"{(covered_pairs/total_pairs*100):.1f}%")
        with c4:
            acc_model = (((df_all["pct_util_humano"] >= 50).astype(int) == df_all["label_modelo"]).mean() * 100)
            st.metric("Acurácia (modelo vs consenso humano)", f"{acc_model:.1f}%")

        # -------------------- Por grupo temático --------------------
        st.subheader("Distribuição por grupo temático (normalizado)")
        grp = (
            df_all.groupby("grupo_norm")
                  .agg(itens=("answer_id", "count"),
                       avaliados=("n_avaliacoes", lambda s: int((s > 0).sum())),
                       pct_util_medio=("pct_util_humano", "mean"))
                  .reset_index()
        )
        st.dataframe(grp, use_container_width=True)
        st.bar_chart(grp.set_index("grupo_norm")[["avaliados"]])

        # -------------------- Acurácia por grupo --------------------
        st.subheader("Acurácia do modelo vs consenso humano por grupo temático")
        df_evald = df_all.loc[df_all["n_avaliacoes"] > 0].copy()
        df_evald["consenso_humano"] = (df_evald["pct_util_humano"] >= 50).astype(int)
        df_evald["acertou"] = (df_evald["consenso_humano"] == df_evald["label_modelo"]).astype(int)

        acc_group = (
            df_evald.groupby("grupo_norm", dropna=False)
                    .agg(
                        total_avaliados=("acertou", "count"),
                        acertos=("acertou", "sum"),
                        pct_util_medio=("pct_util_humano", "mean")
                    )
                    .reset_index()
        )
        acc_group["acuracia_%"] = np.where(
            acc_group["total_avaliados"] > 0,
            acc_group["acertos"] / acc_group["total_avaliados"] * 100,
            0.0
        )
        acc_group["acuracia_%"] = acc_group["acuracia_%"].round(1)
        acc_group["pct_util_medio"] = acc_group["pct_util_medio"].round(1)
        acc_group = acc_group.sort_values("acuracia_%", ascending=False)

        st.dataframe(acc_group, use_container_width=True)
        st.bar_chart(acc_group.set_index("grupo_norm")[["acuracia_%"]])

        # ===================== Resultados por perfil =====================
        st.subheader("Resultados por perfil de avaliador")

        # % útil e n_avaliacoes por (answer_id, perfil)
        df_eval_prof = pd.read_sql_query("""
            SELECT e.answer_id,
                   r.profile,                                  -- 'comum' | 'saude'
                   AVG(e.is_useful)*100.0 AS pct_util_humano,  -- % útil dentro do perfil
                   COUNT(*) AS n_avaliacoes
            FROM evaluations e
            JOIN respondents r ON r.id = e.respondent_id
            GROUP BY e.answer_id, r.profile
        """, c)

        df_prof_all = df_eval_prof.merge(df_pairs, on="answer_id", how="left")
        df_prof_all["grupo_norm"] = df_prof_all["grupo_raw"].map(norm_group)

        def perfil_label(p):
            return "Usuários comuns" if p == "comum" else "Profissionais da Saúde"

        tabs = st.tabs(["Usuários comuns", "Profissionais da Saúde"])
        for tab, prof in zip(tabs, ["comum", "saude"]):
            with tab:
                sub = df_prof_all.loc[df_prof_all["profile"] == prof].copy()
                if sub.empty:
                    st.info("Ainda não há avaliações para este perfil.")
                    continue

                sub["consenso_humano"] = (sub["pct_util_humano"] >= 50).astype(int)
                sub["acertou"] = (sub["consenso_humano"] == sub["label_modelo"]).astype(int)

                total_pairs_prof = sub["answer_id"].nunique()
                acc_prof = (sub["acertou"].mean() * 100) if len(sub) else 0.0
                k1, k2 = st.columns(2)
                with k1:
                    st.metric("Pares avaliados (neste perfil)", total_pairs_prof)
                with k2:
                    st.metric(f"Acurácia do modelo vs consenso ({perfil_label(prof)})",
                              f"{acc_prof:.1f}%")

                acc_group_prof = (
                    sub.groupby("grupo_norm", dropna=False)
                       .agg(
                           total_avaliados=("acertou", "count"),
                           acertos=("acertou", "sum"),
                           pct_util_medio=("pct_util_humano", "mean"),
                       )
                       .reset_index()
                )
                acc_group_prof["acuracia_%"] = np.where(
                    acc_group_prof["total_avaliados"] > 0,
                    acc_group_prof["acertos"] / acc_group_prof["total_avaliados"] * 100,
                    0.0
                )
                acc_group_prof["acuracia_%"] = acc_group_prof["acuracia_%"].round(1)
                acc_group_prof["pct_util_medio"] = acc_group_prof["pct_util_medio"].round(1)
                acc_group_prof = acc_group_prof.sort_values("acuracia_%", ascending=False)

                st.dataframe(acc_group_prof, use_container_width=True)
                st.bar_chart(acc_group_prof.set_index("grupo_norm")[["acuracia_%"]])

                with st.expander("Ver pares avaliados neste perfil"):
                    view_cols = [
                        "pergunta","resposta","grupo_raw","grupo_norm",
                        "label_modelo","label_original","n_avaliacoes","pct_util_humano","consenso_humano"
                    ]
                    st.dataframe(sub[view_cols], use_container_width=True)

        # ===================== Concordância por perfil =====================
        st.subheader("Concordância (útil × não útil) por perfil")

        df_eval_full = pd.read_sql_query("""
            SELECT
                e.answer_id,
                e.respondent_id,
                CASE WHEN e.is_useful THEN 1 ELSE 0 END AS is_useful,
                r.profile
            FROM evaluations e
            JOIN respondents r ON r.id = e.respondent_id
        """, c)

        def compute_concordance(df_subset: pd.DataFrame):
            if df_subset.empty:
                per_answer = pd.DataFrame(columns=["answer_id","n_avaliacoes","n_em_acordo","concordancia_%"])
                metrics = {"total_avaliacoes": 0, "concordancia_micro_%": 0.0, "concordancia_macro_%": 0.0}
                return per_answer, metrics

            maj = (df_subset.groupby("answer_id")["is_useful"].mean() >= 0.5).astype(int).rename("maj")
            tmp = df_subset.merge(maj, on="answer_id", how="left")
            tmp["agree"] = (tmp["is_useful"] == tmp["maj"]).astype(int)

            per_answer = (
                tmp.groupby("answer_id")
                   .agg(n_avaliacoes=("agree","size"), n_em_acordo=("agree","sum"))
                   .reset_index()
            )
            per_answer["concordancia_%"] = (per_answer["n_em_acordo"] / per_answer["n_avaliacoes"] * 100).round(1)

            concord_micro = (tmp["agree"].mean() * 100.0)
            concord_macro = per_answer["concordancia_%"].mean() if len(per_answer) else 0.0

            metrics = {
                "total_avaliacoes": int(len(tmp)),
                "concordancia_micro_%": round(concord_micro, 1),
                "concordancia_macro_%": round(concord_macro, 1),
            }
            return per_answer, metrics

        perfis = {
            "Usuários comuns": df_eval_full[df_eval_full["profile"] == "comum"][["answer_id","respondent_id","is_useful"]],
            "Profissionais da Saúde": df_eval_full[df_eval_full["profile"] == "saude"][["answer_id","respondent_id","is_useful"]],
            "Geral (ambos perfis)": df_eval_full[["answer_id","respondent_id","is_useful"]],
        }

        tabs_conc = st.tabs(list(perfis.keys()))
        for tab, (nome, dfsub) in zip(tabs_conc, perfis.items()):
            with tab:
                per_answer, metrics = compute_concordance(dfsub)

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Total de avaliações consideradas", metrics["total_avaliacoes"])
                with k2:
                    st.metric("Concordância (micro)", f"{metrics['concordancia_micro_%']:.1f}%")
                with k3:
                    st.metric("Concordância (macro)", f"{metrics['concordancia_macro_%']:.1f}%")

                st.caption("Concordância: percentual de avaliações que coincidem com o rótulo majoritário (útil × não útil) em cada resposta.")
                st.dataframe(per_answer.sort_values("concordancia_%", ascending=True),
                            use_container_width=True, height=320)

                if not per_answer.empty:
                    st.bar_chart(per_answer.set_index("answer_id")[["concordancia_%"]])

        # -------------------- Cobertura ao longo do tempo --------------------
        st.subheader("Cobertura ao longo do tempo (submissões)")
        df_sub = pd.read_sql_query("""
            SELECT submitted_at FROM respondents
            WHERE has_submitted=1 AND submitted_at IS NOT NULL
        """, c)
        if not df_sub.empty:
            df_sub["date"] = pd.to_datetime(df_sub["submitted_at"]).dt.date
            ts = df_sub.groupby("date").size().reset_index(name="submissoes")
            st.line_chart(ts.set_index("date"))
        else:
            st.info("Ainda não há submissões concluídas.")

    # ===================== Tabela completa + Download =====================
    st.subheader("Tabela completa (pares + métricas)")
    st.dataframe(
        df_all[["pergunta", "resposta", "grupo_raw", "grupo_norm",
                "label_modelo", "label_original", "n_avaliacoes", "pct_util_humano"]],
        use_container_width=True
    )

    st.subheader("Baixar avaliações agregadas (CSV)")
    st.download_button(
        "Download CSV",
        data=df_all.to_csv(index=False).encode("utf-8"),
        file_name=f"analytics_{int(time.time())}.csv",
        mime="text/csv",
        key="download_results_csv"
    )

    # -------------------- Acurácia humana vs rótulo real --------------------
    st.subheader("Acurácia da avaliação humana com o rótulo real")
    df_evald2 = df_all.loc[df_all["n_avaliacoes"] > 0].copy()
    df_evald2["consenso_humano"] = (df_evald2["pct_util_humano"] >= 50).astype(int)
    df_evald2["acertou"] = (df_evald2["consenso_humano"] == df_evald2["label_original"]).astype(int)
    acc_human = (df_evald2["acertou"].mean() * 100) if len(df_evald2) else 0.0
    st.metric("Acurácia da avaliação humana com o rótulo real", f"{acc_human:.1f}%")

    st.subheader("Respostas por usuário (inspecionar e remover)")
    df_part = list_participants_for_admin()

    colf1, colf2, colf3 = st.columns([2,1,1])
    with colf1:
        search = st.text_input("Buscar por nome/e‑mail")
    with colf2:
        f_profile = st.selectbox("Perfil", ["Todos", "Usuário comum", "Profissional da Saúde"], index=0)
    with colf3:
        f_status = st.selectbox("Status", ["Todos", "Com avaliações", "Sem avaliações"], index=0)

    df_part_view = df_part.copy()
    # Mapeia perfil para rótulos amigáveis
    label_profile = {"comum": "Usuário comum", "saude": "Profissional da Saúde"}
    df_part_view["perfil_label"] = df_part_view["profile"].map(label_profile)

    if search:
        s = search.strip().lower()
        df_part_view = df_part_view[
            df_part_view["name"].str.lower().str.contains(s) |
            df_part_view["email"].str.lower().str.contains(s)
        ]
    if f_profile != "Todos":
        df_part_view = df_part_view[df_part_view["perfil_label"] == f_profile]
    if f_status != "Todos":
        if f_status == "Com avaliações":
            df_part_view = df_part_view[df_part_view["n_avaliacoes"] > 0]
        else:
            df_part_view = df_part_view[df_part_view["n_avaliacoes"] == 0]

    st.caption("Selecione um usuário para ver/remover suas respostas.")
    # Tabela resumida
    cols_show = ["respondent_id","name","email","perfil_label","health_area","created_at","n_assignments","n_avaliacoes","has_submitted","submitted_at"]
    df_show = df_part_view[cols_show].rename(columns={
        "respondent_id":"ID",
        "name":"Nome",
        "email":"E‑mail",
        "perfil_label":"Perfil",
        "health_area":"Área",
        "created_at":"Criado em",
        "n_assignments":"Amostra (itens)",
        "n_avaliacoes":"Avaliações",
        "has_submitted":"Enviou?",
        "submitted_at":"Enviado em",
    }).copy()
    if not df_show.empty:
        df_show["Enviou?"] = df_show["Enviou?"].map({0:"Não",1:"Sim"})
    st.dataframe(df_show, use_container_width=True, height=320)

    # Seleção do usuário
    options = df_part_view[["respondent_id","name","email"]].apply(
        lambda r: f"{r['respondent_id']} • {r['name']} • {r['email']}", axis=1
    ).tolist()
    selected_text = st.selectbox("Usuário", ["— selecione —"] + options, index=0)
    if selected_text != "— selecione —":
        sel_id = int(selected_text.split("•")[0].strip())

        # Carrega avaliações do usuário
        df_u = get_user_evaluations(sel_id)
        st.write(f"**Total de avaliações deste usuário:** {len(df_u)}")
        if df_u.empty:
            st.info("Este usuário ainda não possui avaliações registradas.")
        else:
            view_cols = ["created_at","is_useful","grupo_raw","grupo_norm","pergunta","resposta","label_modelo","label_original","answer_id","question_id","evaluation_id"]
            st.dataframe(df_u[view_cols], use_container_width=True, height=420)

            cdel1, cdel2 = st.columns([1,1])
            with cdel1:
                if st.button("Remover SOMENTE avaliações deste usuário", type="primary"):
                    delete_user_evaluations(sel_id)
                    st.success("Avaliações removidas. As métricas serão atualizadas com a próxima carga da página.")
                    st.rerun()
            with cdel2:
                if st.button("Remover avaliações + assignments e reabrir participação", type="secondary"):
                    delete_user_evaluations_and_reopen(sel_id)
                    st.success("Avaliações e amostragem removidas. Usuário agora pode participar novamente.")
                    st.rerun()

# ---------- Página: Admin ▸ Limpar respostas ----------
if page == "Admin ▸ Limpar respostas":
    st.header("Admin ▸ Limpar respostas e reabrir participação")
    pwd = st.text_input("Senha de admin", type="password")
    if st.button("Entrar", key="login_clear"):
        if pwd == ADMIN_PASSWORD and ADMIN_PASSWORD:
            st.session_state["admin_ok_clear"] = True
            st.rerun()
        else:
            st.error("Senha inválida.")
    if not st.session_state.get("admin_ok_clear"):
        st.stop()

    st.warning("Esta ação apaga TODAS as avaliações e amostragens, mas mantém o corpus.")
    if st.button("Apagar avaliações e amostragens (irrevogável)", type="primary"):
        clear_all_responses()
        st.success("Respostas e amostragens apagadas. Participantes podem reenviar usando o mesmo e‑mail.")
