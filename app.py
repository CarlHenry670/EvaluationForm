# app.py — Streamlit lendo exclusivamente de views do Supabase
import streamlit as st
import pandas as pd
import numpy as np
import uuid
import time
from datetime import datetime, timezone

from supabase_client import get_client

st.set_page_config(page_title="Avaliação Humana do Modelo", layout="wide")

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")
ACCESS_CODE    = st.secrets.get("ACCESS_CODE", "")

sb = get_client()

# ---------- Estado ----------
if "respondent_uuid" not in st.session_state:
    st.session_state.respondent_uuid = str(uuid.uuid4())
if "respondent_id" not in st.session_state:
    st.session_state.respondent_id = None
if "assignment" not in st.session_state:
    st.session_state.assignment = None
if "progress_idx" not in st.session_state:
    st.session_state.progress_idx = 0
if "answers_buffer" not in st.session_state:
    st.session_state.answers_buffer = {}
if "access_ok" not in st.session_state:
    st.session_state.access_ok = False if ACCESS_CODE else True
for k in ["user_name","user_email","user_profile","user_health_area"]:
    st.session_state.setdefault(k, "")

# ---------- Normalização (para amostragem) ----------
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
TOTAL_TARGET = sum(QUOTAS.values())

def norm_group(g: str) -> str:
    key = (g or "").strip().lower()
    return GROUP_MAP.get(key, "OUTROS")

# ---------- Corpus: importação ----------
def read_corpus(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(name): return cols_lower.get(name.lower())
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
    # 1) UPSERT de questions
    q_records = []
    for per, grp_raw in df[["pergunta","grupo_raw"]].drop_duplicates().itertuples(index=False):
        q_records.append({
            "q_hash": f"{hash(per)}::{hash(grp_raw)}",
            "pergunta": per,
            "grupo": grp_raw
        })
    if q_records:
        sb.table("questions").upsert(q_records, on_conflict="q_hash").execute()

    # map pergunta+grupo_raw -> id
    q_all = sb.table("questions").select("id, pergunta, grupo").execute().data or []
    qmap = {(q["pergunta"], q["grupo"]): q["id"] for q in q_all}

    # 2) INSERT de answers (em lotes)
    a_records = []
    for row in df.itertuples(index=False):
        a_records.append({
            "question_id": int(qmap[(row.pergunta, row.grupo_raw)]),
            "resposta": row.resposta,
            "label_original": int(row.label_original),
            "label_modelo": int(row.label_modelo),
        })
    for i in range(0, len(a_records), 1000):
        sb.table("answers").insert(a_records[i:i+1000]).execute()

# ---------- Fetchers para amostragem ----------
def fetch_grouped_answers_for_sampling():
    ans = sb.table("answers").select("*").execute().data or []
    qst = sb.table("questions").select("id, pergunta, grupo").execute().data or []
    if not ans or not qst:
        return {"SDC":pd.DataFrame(),"SF":pd.DataFrame(),"TM":pd.DataFrame()}, pd.DataFrame()

    df_a = pd.DataFrame(ans)
    df_q = pd.DataFrame(qst).rename(columns={"id":"question_id"})
    df = df_a.merge(df_q, on="question_id", how="left")
    df["grupo_norm"] = df["grupo"].map(norm_group)
    df = df.rename(columns={"id":"answer_id"})
    grouped = {k: df[df["grupo_norm"]==k][["answer_id","question_id","pergunta","grupo_norm","resposta"]].copy()
               for k in ["SDC","SF","TM"]}
    return grouped, df[["answer_id","question_id","pergunta","grupo_norm","resposta"]].copy()

def ensure_respondent(email: str, name: str, profile: str, health_area: str|None) -> int:
    res = sb.table("respondents").select("id").eq("email", email).limit(1).execute().data
    if res:
        rid = int(res[0]["id"])
        sb.table("respondents").update({
            "name": name, "profile": profile, "health_area": health_area
        }).eq("id", rid).execute()
        return rid
    r = sb.table("respondents").insert({
        "respondent_uuid": st.session_state.respondent_uuid,
        "name": name, "email": email, "profile": profile,
        "health_area": health_area, "created_at": datetime.now(timezone.utc).isoformat()
    }).execute().data[0]["id"]
    return int(r)

def respondent_status(email: str):
    res = sb.table("respondents").select("id, has_submitted, name").eq("email", email).limit(1).execute().data
    return (res[0]["id"], int(bool(res[0]["has_submitted"])), res[0]["name"]) if res else None

def has_assignments(respondent_id: int) -> bool:
    cnt = sb.table("assignments").select("id", count="exact").eq("respondent_id", respondent_id).execute().count or 0
    return cnt > 0

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
    df_ass = df_ass.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    df_ass["order_index"] = np.arange(len(df_ass))

    records = []
    for row in df_ass.itertuples(index=False):
        records.append({
            "respondent_id": int(respondent_id),
            "answer_id": int(row.answer_id),
            "question_id": int(row.question_id),
            "grupo": row.grupo_norm,
            "order_index": int(row.order_index),
        })
    for i in range(0, len(records), 1000):
        sb.table("assignments").upsert(records[i:i+1000], on_conflict="respondent_id,answer_id").execute()
    return df_ass

def load_assignments(respondent_id: int) -> pd.DataFrame:
    asg = (sb.table("assignments").select("*").eq("respondent_id", respondent_id)
           .order("order_index").execute().data or [])
    if not asg:
        return pd.DataFrame()
    df_asg = pd.DataFrame(asg)

    ans = sb.table("answers").select("id, question_id, resposta, label_modelo, label_original").execute().data or []
    qst = sb.table("questions").select("id, pergunta, grupo").execute().data or []

    df_a = pd.DataFrame(ans).rename(columns={"id": "answer_id"})
    df_q = pd.DataFrame(qst).rename(columns={"id": "question_id"})

    df = (df_asg.merge(df_a, on=["answer_id", "question_id"], how="left")
                 .merge(df_q, on="question_id", how="left", suffixes=("", "_q")))

    for cand in ["grupo", "grupo_x", "grupo_y", "grupo_q", "grupo_raw"]:
        if cand in df.columns:
            df["grupo"] = df[cand]; break
    else:
        df["grupo"] = ""

    cols = ["answer_id", "question_id", "grupo", "order_index", "pergunta", "resposta"]
    if "pergunta" not in df.columns and "pergunta_q" in df.columns:
        df["pergunta"] = df["pergunta_q"]

    return df[cols].copy()

def commit_final_evaluations(respondent_id: int, buffer: dict[int, int]):
    ass = sb.table("assignments").select("answer_id, question_id")\
           .eq("respondent_id", respondent_id).order("order_index").execute().data or []
    total = len(ass)
    if len(buffer) != total:
        raise ValueError(f"Você respondeu {len(buffer)} de {total} itens. Complete todos antes de enviar.")
    now = datetime.now(timezone.utc).isoformat()

    for row in ass:
        aid = int(row["answer_id"]); qid = int(row["question_id"])
        payload = {
            "respondent_id": respondent_id,
            "answer_id": aid,
            "question_id": qid,
            "is_useful": bool(int(buffer[aid])),
            "updated_at": now
        }
        upd = sb.table("evaluations").update(payload)\
            .eq("respondent_id", respondent_id).eq("answer_id", aid).execute().data
        if not upd:
            payload["created_at"] = now
            sb.table("evaluations").insert(payload).execute()

    sb.table("respondents").update({
        "has_submitted": True, "submitted_at": now
    }).eq("id", respondent_id).execute()

def clear_all_responses():
    sb.table("evaluations").delete().neq("id", 0).execute()
    sb.table("assignments").delete().neq("id", 0).execute()
    sb.table("respondents").update({"has_submitted": False, "submitted_at": None}).neq("id", 0).execute()

# === Admin: participantes direto da view ===
def list_participants_for_admin() -> pd.DataFrame:
    data = sb.table("v_participants").select("*").order("created_at", desc=True).execute().data or []
    return pd.DataFrame(data)

# ---------- Helpers de sessão ----------
def participant_gate():
    if st.session_state.access_ok:
        return True
    st.info("Insira o código de acesso fornecido pelo organizador.")
    code = st.text_input("Código de acesso", type="password")
    if st.button("Validar código"):
        if code.strip() == ACCESS_CODE.strip():
            st.session_state.access_ok = True; st.rerun()
        else:
            st.error("Código inválido.")
    return False
# ---------- Navegação ----------
st.sidebar.title("Avaliação Das Respostas do Modelo")
page = st.sidebar.radio(
    "Navegação",
    ["Participar", "Participantes", "Resultados", "Admin ▸ Importar corpus", "Admin ▸ Limpar respostas"],
    index=0
)

# ---------- Página: Participar ----------
if page == "Participar":
    st.header("Participar da Avaliação")
    if ACCESS_CODE and not participant_gate():
        st.stop()

    # FORM de identificação
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

        st.session_state.user_name = name.strip()
        st.session_state.user_email = email.strip()
        st.session_state.user_profile = "saude" if profile == "Profissional da Saúde" else "comum"
        st.session_state.user_health_area = (health_area or "").strip()

        status = respondent_status(st.session_state.user_email)
        if status and status[1] == 1:
            st.warning("Este e-mail já enviou uma avaliação. Obrigado!")
            st.stop()

        rid = ensure_respondent(
            email=st.session_state.user_email,
            name=st.session_state.user_name,
            profile=st.session_state.user_profile,
            health_area=st.session_state.user_health_area if st.session_state.user_profile == "saude" else None
        )
        st.session_state.respondent_id = rid

        if not has_assignments(rid):
            try:
                create_assignments_for_user(rid)
            except Exception as e:
                st.error(f"Não foi possível criar sua amostragem: {e}")
                st.stop()

        st.session_state.assignment = load_assignments(rid)
        st.session_state.answers_buffer = st.session_state.get("answers_buffer", {})

        # recuperar progresso salvo
        df_ass = st.session_state.assignment
        if not df_ass.empty:
            valid_ids = [int(x) for x in df_ass["answer_id"].tolist()]
            ev = (
                sb.table("evaluations")
                .select("answer_id, is_useful")
                .eq("respondent_id", rid)
                .in_("answer_id", valid_ids)
                .execute().data or []
            )
            answered_map = {int(r["answer_id"]): int(bool(r["is_useful"])) for r in ev}
            st.session_state.answers_buffer.update(answered_map)
            # primeiro não respondido
            answered_ids = set(st.session_state.answers_buffer.keys())
            idx = 0
            for i, aid in enumerate(df_ass["answer_id"].tolist()):
                if int(aid) not in answered_ids:
                    idx = i
                    break
            st.session_state.progress_idx = idx

        if len(st.session_state.answers_buffer) == 0:
            st.success("Amostragem pronta. Você vai avaliar 118 pares pergunta-resposta.")
        st.rerun()

    # UI de rotulagem
    if st.session_state.get("respondent_id"):
        if st.session_state.get("assignment") is None:
            st.session_state.assignment = load_assignments(st.session_state["respondent_id"])

        df_ass = st.session_state.assignment
        if df_ass is None or df_ass.empty:
            st.info("Ainda não há amostragem. Clique em 'Iniciar / Continuar'.")
            st.stop()

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
        q_id = int(current["question_id"])
        rid = int(st.session_state["respondent_id"])

        def upsert_eval(rid: int, a_id: int, q_id: int, is_useful: int):
            payload = {
                "respondent_id": rid,
                "answer_id": a_id,
                "question_id": q_id,
                "is_useful": bool(int(is_useful)),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            upd = (
                sb.table("evaluations")
                .update(payload)
                .eq("respondent_id", rid).eq("answer_id", a_id)
                .execute().data
            )
            if not upd:
                payload["created_at"] = datetime.now(timezone.utc).isoformat()
                sb.table("evaluations").insert(payload).execute()

        cols = st.columns(2)
        with cols[0]:
            if st.button("Útil ✅", use_container_width=True, key=f"useful_{idx}"):
                upsert_eval(rid, a_id, q_id, 1)
                st.session_state.answers_buffer[a_id] = 1
                # ir para o próximo não respondido
                answered_ids = set(st.session_state.answers_buffer.keys())
                next_idx = idx
                for i, aid in enumerate(df_ass["answer_id"].tolist()()):
                    if int(aid) not in answered_ids:
                        next_idx = i
                        break
                st.session_state.progress_idx = next_idx
                st.rerun()
        with cols[1]:
            if st.button("Não útil / Pouco útil ❌", use_container_width=True, key=f"notuseful_{idx}"):
                upsert_eval(rid, a_id, q_id, 0)
                st.session_state.answers_buffer[a_id] = 0
                answered_ids = set(st.session_state.answers_buffer.keys())
                next_idx = idx
                for i, aid in enumerate(df_ass["answer_id"].tolist()):
                    if int(aid) not in answered_ids:
                        next_idx = i
                        break
                st.session_state.progress_idx = next_idx
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

# ---------- Página: Participantes ----------
elif page == "Participantes":
    colf1, colf2, colf3 = st.columns([1, 1, 2])
    with colf1:
        f_profile = st.selectbox("Perfil", ["Todos", "Usuário comum", "Profissional da Saúde"], index=0)
    with colf2:
        f_status = st.selectbox("Status", ["Todos", "Enviaram", "Não enviaram"], index=0)
    with colf3:
        f_search = st.text_input("Buscar por nome ou e-mail")

    df_users = list_participants_for_admin()
    if df_users.empty:
        st.info("Sem participantes ainda.")
        st.stop()

    df_users["perfil"] = np.where(df_users["profile"] == "saude", "Profissional da Saúde", "Usuário comum")
    if f_profile != "Todos":
        df_users = df_users[df_users["perfil"] == f_profile]
    if f_status != "Todos":
        if f_status == "Enviaram":
            df_users = df_users[df_users["has_submitted"] == True]
        else:
            df_users = df_users[df_users["has_submitted"] == False]
    if f_search:
        s = f_search.strip().lower()
        df_users = df_users[df_users["name"].str.lower().str.contains(s) | df_users["email"].str.lower().str.contains(s)]

    total = len(df_users)
    enviados = int((df_users["has_submitted"] == True).sum())
    saude = int((df_users["perfil"] == "Profissional da Saúde").sum())
    comuns = int((df_users["perfil"] == "Usuário comum").sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total cadastrados", total)
    with k2:
        st.metric("Profissionais da Saúde", saude)
    with k3:
        st.metric("Usuários comuns", comuns)
    with k4:
        st.metric("Já enviaram", enviados)

    df_view = df_users.rename(columns={
        "name": "Nome", "email": "E-mail", "perfil": "Perfil",
        "health_area": "Área de atuação", "created_at": "Cadastrado em",
        "submitted_at": "Enviado em", "has_submitted": "Enviou?",
        "n_assignments": "Amostra (itens)", "n_avaliacoes": "Avaliações"
    })[["Nome", "E-mail", "Perfil", "Área de atuação", "Cadastrado em", "Enviado em", "Enviou?", "Amostra (itens)", "Avaliações"]]
    df_view["Enviou?"] = df_view["Enviou?"].map({False: "Não", True: "Sim"})
    st.dataframe(df_view, use_container_width=True)

    st.download_button(
        "Baixar CSV dos participantes",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name=f"participantes_{int(time.time())}.csv",
        mime="text/csv"
    )

# ---------- Página: Resultados ----------
elif page == "Resultados":
    st.header("Resultados")

    # Visão geral vindo de v_overview
    overview = sb.table("v_overview").select("*").execute().data or []
    if overview:
        o = overview[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pares no corpus", o.get("total_pairs", 0))
        with c2:
            st.metric("Pares já avaliados", o.get("covered_pairs", 0))
        with c3:
            st.metric("Cobertura", f"{o.get('cobertura_pct', 0):.1f}%")
        with c4:
            st.metric("Acurácia (modelo vs consenso humano)", f"{o.get('acc_model_pct', 0):.1f}%")
    else:
        st.info("Sem dados em v_overview (verifique as views).")

    # Distribuição por grupo (v_groups)
    st.subheader("Distribuição por grupo temático (normalizado)")
    df_groups = pd.DataFrame(sb.table("v_groups").select("*").execute().data or [])
    if not df_groups.empty:
        st.dataframe(df_groups, use_container_width=True)
        try:
            st.bar_chart(df_groups.set_index("grupo_norm")[["avaliados"]])
        except Exception:
            pass
    else:
        st.info("Sem dados em v_groups.")

    # Acurácia por grupo (v_acc_group)
    st.subheader("Acurácia do modelo vs consenso humano por grupo temático")
    acc_group = pd.DataFrame(sb.table("v_acc_group").select("*").execute().data or [])
    if not acc_group.empty:
        st.dataframe(acc_group, use_container_width=True)
        try:
            st.bar_chart(acc_group.set_index("grupo_norm")[["acuracia_pct"]])
        except Exception:
            pass
    else:
        st.info("Sem dados em v_acc_group.")

    # === Concordância com o modelo (micro e macro) ===
    st.subheader("Concordância com o modelo")

    mi = pd.DataFrame(sb.table("v_model_agreement_micro").select("*").execute().data or [])
    ma = pd.DataFrame(sb.table("v_model_agreement_macro").select("*").execute().data or [])
    bg = pd.DataFrame(sb.table("v_model_agreement_by_group").select("*").execute().data or [])

    label_map = {"comum": "Usuários comuns", "saude": "Profissionais da Saúde", "geral": "Geral"}

    def metric_row(title: str, df: pd.DataFrame, count_col: str):
        cols = st.columns(3)
        for col, key in zip(cols, ["comum", "saude", "geral"]):
            row = df[df["profile"] == key]
            pct = float(row.iloc[0]["agree_pct"]) if not row.empty else 0.0
            cnt = int(row.iloc[0][count_col]) if not row.empty else 0
            col.metric(f"{title} • {label_map[key]}", f"{pct:.1f}%", help=f"Itens considerados: {cnt}")

    # MICRO: acordo item-a-item (ponderado por nº de respostas do perfil)
    metric_row("Micro", mi, "eligible_n")
    # MACRO: média do acordo por eixo (cada eixo vale 1)
    metric_row("Macro", ma, "groups_with_overlap")

    # Detalhamento por eixo (cada perfil)
    st.markdown("**Detalhamento por eixo temático (concordância com o modelo)**")
    if bg.empty:
        st.info("Sem dados por eixo ainda.")
    else:
        tabs = st.tabs([label_map[k] for k in ["comum", "saude", "geral"]])
        for tab, key in zip(tabs, ["comum", "saude", "geral"]):
            with tab:
                sub = bg[bg["profile"] == key].copy()
                sub = sub.rename(columns={
                    "grupo_norm": "Eixo",
                    "eligible_n": "Itens no eixo",
                    "agree_n": "Acordos com o modelo",
                    "agree_pct": "Concordância %"
                }).sort_values("Concordância %", ascending=False)
                if sub.empty:
                    st.info("Sem sobreposição para este perfil.")
                else:
                    st.dataframe(sub, use_container_width=True)
                    try:
                        st.bar_chart(sub.set_index("Eixo")[["Concordância %"]])
                    except Exception:
                        pass

    # Resultados por perfil (v_profile)
    st.subheader("Resultados por perfil de avaliador")
    df_profile = pd.DataFrame(sb.table("v_profile").select("*").execute().data or [])
    if not df_profile.empty:
        tabs = st.tabs(["Usuários comuns", "Profissionais da Saúde"])
        for tab, prof, label_txt in zip(tabs, ["comum", "saude"], ["Usuários comuns", "Profissionais da Saúde"]):
            with tab:
                sub = df_profile[df_profile["profile"] == prof].copy()
                if sub.empty:
                    st.info("Ainda não há avaliações para este perfil.")
                    continue
                total_pairs_prof = int(sub["total_avaliados"].sum())
                acc_prof = float(sub["acuracia_pct"].mean() if len(sub) else 0.0)
                k1, k2 = st.columns(2)
                with k1:
                    st.metric("Pares avaliados (neste perfil)", total_pairs_prof)
                with k2:
                    st.metric(f"Acurácia (modelo vs consenso • {label_txt})", f"{acc_prof:.1f}%")
                st.dataframe(sub.sort_values("acuracia_pct", ascending=False), use_container_width=True)

                # Detalhes por perfil (opcional: v_profile_pairs)
                with st.expander("Ver pares avaliados neste perfil"):
                    sub_pairs = pd.DataFrame(
                        sb.table("v_profile_pairs").select("*").eq("profile", prof).limit(500).execute().data or []
                    )
                    if sub_pairs.empty:
                        st.info("Sem detalhes disponíveis.")
                    else:
                        view_cols = [
                            "pergunta", "resposta", "grupo_raw", "grupo_norm",
                            "label_modelo", "label_original", "n_avaliacoes", "pct_util_humano"
                        ]
                        st.dataframe(sub_pairs[view_cols], use_container_width=True)

    # Submissões por dia (v_submissions_daily)
    st.subheader("Cobertura ao longo do tempo (submissões)")
    df_sub = pd.DataFrame(sb.table("v_submissions_daily").select("*").execute().data or [])
    if not df_sub.empty:
        try:
            df_sub["date"] = pd.to_datetime(df_sub["date"])
            st.line_chart(df_sub.set_index("date")[["submissoes"]])
        except Exception:
            st.dataframe(df_sub, use_container_width=True)
    else:
        st.info("Ainda não há submissões concluídas.")

    # Tabela completa (v_all) + download
    st.subheader("Tabela completa (pares + métricas)")
    df_all = pd.DataFrame(
        sb.table("v_all")
        .select("pergunta,resposta,grupo_raw,grupo_norm,label_modelo,label_original,n_avaliacoes,pct_util_humano")
        .limit(5000).execute().data or []
    )
    st.dataframe(df_all, use_container_width=True)

    st.subheader("Baixar avaliações agregadas (CSV)")
    st.download_button(
        "Download CSV",
        data=df_all.to_csv(index=False).encode("utf-8"),
        file_name=f"analytics_{int(time.time())}.csv",
        mime="text/csv",
        key="download_results_csv"
    )

# ---------- Página: Admin ▸ Importar corpus ----------
elif page == "Admin ▸ Importar corpus":
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

            diag = df.groupby("grupo")["resposta"].count().rename("itens").reset_index()
            diag["cota_exigida"] = diag["grupo"].map(QUOTAS).fillna(0).astype(int)
            st.subheader("Diagnóstico de cotas (normalizado)")
            st.dataframe(diag, use_container_width=True)

            if st.button("Importar corpus para o banco", type="primary"):
                import_corpus(df)
                st.success(f"Importados {len(df)} pares pergunta-resposta.")
        except Exception as e:
            st.error(f"Erro ao ler/importar CSV: {e}")

# ---------- Página: Admin ▸ Limpar respostas ----------
elif page == "Admin ▸ Limpar respostas":
    st.header("Admin ▸ Limpar respostas e reabrir participação")
    pwd = st.text_input("Senha de admin", type="password")
    if st.button("Entrar", key="login_clear"):
        if pwd == ADMIN_PASSWORD and ADMIN_PASSWORD:  # atenção ao 'and' minúsculo (correto)
            st.session_state["admin_ok_clear"] = True
            st.rerun()
        else:
            st.error("Senha inválida.")
    if not st.session_state.get("admin_ok_clear"):
        st.stop()

    st.warning("Esta ação apaga TODAS as avaliações e amostragens, mas mantém o corpus.")
    if st.button("Apagar avaliações e amostragens (irrevogável)", type="primary"):
        clear_all_responses()
        st.success("Respostas e amostragens apagadas. Participantes podem reenviar usando o mesmo e-mail.")
