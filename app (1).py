# -*- coding: utf-8 -*-
# =============================================================================
# PL01 - Machine Learning | Visualização de dados
# Dashboard Formula 1 (Streamlit + Pandas + Plotly + Matplotlib)
#
# Autor(es): NOME 1 (nº XXXX) e NOME 2 (nº XXXX)   <-- substituir
#
# Executar:  streamlit run app.py
# Os CSV do pl01_dataset.zip devem estar na mesma pasta do app.py
# (ou numa subpasta chamada "pl01_dataset" ou "data").
# =============================================================================

from pathlib import Path

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

st.set_page_config(page_title="F1 Dashboard", page_icon="🌸", layout="wide")

# -----------------------------------------------------------------------------
# ESTILO (paleta pastel). Tema base em .streamlit/config.toml; aqui o CSS e os gráficos.
# -----------------------------------------------------------------------------
ROSA, ROSA_ESC, LILAS, PESSEGO, MENTA = "#E58BB1", "#C2487F", "#B39DDB", "#F4A38C", "#7FC8C0"
ESCALA_ROSA = ["#FFF0F5", "#F8C8DC", "#E58BB1", "#B5457A"]

_tpl = go.layout.Template(pio.templates["plotly_white"])
_tpl.layout.update(
    colorway=[ROSA, LILAS, PESSEGO, MENTA, "#F2C14E", "#8EA7E9", ROSA_ESC, "#A5D6A7"],
    font=dict(family="Quicksand, sans-serif", color="#5B3A4A"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)",
    colorscale=dict(sequential=ESCALA_ROSA),
)
pio.templates["rosa"] = _tpl
pio.templates.default = "rosa"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Quicksand:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, label, p { font-family: 'Quicksand', sans-serif !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #B5457A !important; letter-spacing: .3px; }
.stApp { background: linear-gradient(160deg, #FFF7FA 0%, #FDEFF5 45%, #F3ECFA 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #FCE4EE 0%, #EFE6F8 100%); border-right: 1px solid #F3C9DB; }
[data-testid="stMetric"] { background: rgba(255,255,255,.75); border: 1px solid #F3C9DB; border-radius: 20px;
    padding: 14px 18px; box-shadow: 0 4px 14px rgba(229,139,177,.18); }
[data-testid="stMetricValue"] { color: #B5457A; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #B5457A; }
.stButton > button, .stLinkButton > a { border-radius: 999px; border: 1px solid #E58BB1; }
div[data-testid="stAlert"] { border-radius: 16px; }
div[data-testid="stDataFrame"], .js-plotly-plot { border-radius: 16px; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 1. CARREGAMENTO E TRATAMENTO DE DADOS
#    (apenas os ficheiros necessários às análises)
# -----------------------------------------------------------------------------
FICHEIROS = [
    "results", "races", "drivers", "constructors", "status", "circuits",
    "driver_standings", "constructor_standings", "constructor_results",
    "pit_stops", "lap_times", "qualifying",
]


def encontrar_pasta():
    base = Path(__file__).resolve().parent
    for p in [base, base / "pl01_dataset", base / "data", Path.cwd(), Path.cwd() / "pl01_dataset"]:
        if (p / "races.csv").exists():
            return p
    return None


def tempo_para_segundos(s):
    """'1:26.572' -> 86.572 ; valores nulos continuam nulos."""
    if pd.isna(s):
        return np.nan
    try:
        partes = str(s).split(":")
        if len(partes) == 2:
            return int(partes[0]) * 60 + float(partes[1])
        return float(partes[0])
    except ValueError:
        return np.nan


@st.cache_data(show_spinner="A carregar e tratar os dados...")
def carregar_dados():
    pasta = encontrar_pasta()
    if pasta is None:
        return None
    # No dataset os nulos aparecem como \N
    d = {n: pd.read_csv(pasta / f"{n}.csv", na_values=["\\N"]) for n in FICHEIROS}

    # --- tabelas de apoio, com colunas renomeadas para evitar conflitos nos merges
    races = d["races"][["raceId", "year", "round", "name", "date", "circuitId"]].rename(
        columns={"name": "gp"})
    drivers = d["drivers"].copy()
    drivers["piloto"] = drivers["forename"] + " " + drivers["surname"]
    drivers["dob"] = pd.to_datetime(drivers["dob"])
    drivers = drivers[["driverId", "piloto", "code", "nationality", "dob"]].rename(
        columns={"nationality": "nac_piloto"})
    teams = d["constructors"][["constructorId", "name", "nationality"]].rename(
        columns={"name": "equipa", "nationality": "nac_equipa"})
    circuits = d["circuits"][["circuitId", "name", "location", "country", "lat", "lng"]].rename(
        columns={"name": "circuito"})
    status = d["status"]

    # --- tabela principal: resultados + corridas + pilotos + equipas + estado
    res = (d["results"]
           .merge(races, on="raceId")
           .merge(drivers, on="driverId")
           .merge(teams, on="constructorId")
           .merge(status, on="statusId")
           .merge(circuits, on="circuitId"))
    res["date"] = pd.to_datetime(res["date"])
    res["idade"] = (res["date"] - res["dob"]).dt.days / 365.25
    res["vitoria"] = (res["positionOrder"] == 1).astype(int)
    res["podio"] = (res["positionOrder"] <= 3).astype(int)
    res["terminou"] = (res["status"] == "Finished") | res["status"].str.startswith("+")

    # --- classificações finais por época (última corrida de cada ano)
    ronda_final = races.groupby("year")["round"].max().rename("ronda_final").reset_index()

    ds = (d["driver_standings"].merge(races[["raceId", "year", "round"]], on="raceId")
          .merge(ronda_final, on="year"))
    ds = ds[ds["round"] == ds["ronda_final"]].merge(drivers, on="driverId")
    ds_final = ds[["year", "piloto", "position", "points", "wins"]].sort_values(["year", "position"])

    cs = (d["constructor_standings"].merge(races[["raceId", "year", "round"]], on="raceId")
          .merge(ronda_final, on="year"))
    cs = cs[cs["round"] == cs["ronda_final"]].merge(teams, on="constructorId")
    cs_final = cs[["year", "equipa", "position", "points", "wins"]].sort_values(["year", "position"])

    # --- pontos por equipa e corrida
    cr = (d["constructor_results"].merge(races, on="raceId").merge(teams, on="constructorId"))

    # --- pit stops (segundos; removemos outliers > 60s, ex.: bandeiras vermelhas)
    pit = d["pit_stops"].merge(races[["raceId", "year", "gp"]], on="raceId")
    pit = pit.merge(d["results"][["raceId", "driverId", "constructorId"]], on=["raceId", "driverId"])
    pit = pit.merge(teams[["constructorId", "equipa"]], on="constructorId").merge(
        drivers[["driverId", "piloto"]], on="driverId")
    pit["segundos"] = pit["milliseconds"] / 1000
    pit = pit[pit["segundos"] < 60]

    # --- voltas
    laps = d["lap_times"].merge(drivers[["driverId", "piloto", "code"]], on="driverId")
    laps["segundos"] = laps["milliseconds"] / 1000

    # --- qualificação: tempos em segundos
    q = d["qualifying"].merge(races, on="raceId").merge(circuits, on="circuitId").merge(
        drivers[["driverId", "piloto"]], on="driverId")
    for c in ["q1", "q2", "q3"]:
        q[c + "_s"] = q[c].apply(tempo_para_segundos)
    q["melhor_s"] = q[["q1_s", "q2_s", "q3_s"]].min(axis=1)

    return dict(res=res, races=races, circuits=circuits, ds_final=ds_final, cs_final=cs_final,
                cr=cr, pit=pit, laps=laps, q=q, brutos=d)


dados = carregar_dados()
if dados is None:
    st.error("Não encontrei os ficheiros CSV. Coloca o conteúdo do pl01_dataset.zip "
             "na mesma pasta do app.py (ou numa pasta chamada 'pl01_dataset').")
    st.stop()

res, races, circuits = dados["res"], dados["races"], dados["circuits"]
ds_final, cs_final, cr = dados["ds_final"], dados["cs_final"], dados["cr"]
pit, laps, q = dados["pit"], dados["laps"], dados["q"]


# -----------------------------------------------------------------------------
# Funções utilitárias de apresentação
# -----------------------------------------------------------------------------
def mostrar_grafico(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # versões mais antigas do Streamlit
        st.plotly_chart(fig, use_container_width=True)


def mostrar_tabela(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def bandeira_axadrezada():
    """Multimédia gerada em código: bandeira axadrezada (matplotlib)."""
    fig, ax = plt.subplots(figsize=(12, 1.6))
    linhas, colunas = 4, 48
    mapa = np.indices((linhas, colunas)).sum(axis=0) % 2
    from matplotlib.colors import ListedColormap
    ax.imshow(mapa, cmap=ListedColormap(["#F8C8DC", "#FFF7FA"]), aspect="auto", interpolation="nearest")
    ax.text(colunas / 2, (linhas - 1) / 2, "Formula 1  ·  1950 – 2024", ha="center", va="center",
            fontsize=26, fontweight="bold", color="white", family="serif",
            bbox=dict(facecolor="#C2487F", edgecolor="none", boxstyle="round,pad=0.5"))
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf


# -----------------------------------------------------------------------------
# 2. NAVEGAÇÃO (sidebar) + FILTRO GLOBAL DE ÉPOCAS
# -----------------------------------------------------------------------------
PAGINAS = [
    "🏠 Início",
    "🔎 Exploração dos dados",
    "🥇 1. Vitórias por piloto",
    "🏆 2. Campeões do mundo",
    "🏎️ 3. Construtores",
    "🚦 4. Grelha vs resultado",
    "🔧 5. Pit stops",
    "⏱️ 6. Ritmo de corrida (voltas)",
    "🅿️ 7. Qualificação e pole position",
    "🌍 8. Circuitos e mapa",
    "💥 9. Abandonos e fiabilidade",
    "🎂 10. Idade e nacionalidade dos pilotos",
    "🎬 Multimédia",
]
st.sidebar.title("🎀 F1 Dashboard")
pagina = st.sidebar.radio("Navegação", PAGINAS)
st.sidebar.divider()
ano_min, ano_max = int(races["year"].min()), int(races["year"].max())
anos = st.sidebar.slider("Épocas (filtro global)", ano_min, ano_max, (1990, ano_max))
st.sidebar.caption("O filtro aplica-se a todas as análises.")

res_f = res[res["year"].between(*anos)]
st.title(pagina.split(" ", 1)[1])

# -----------------------------------------------------------------------------
# PÁGINAS
# -----------------------------------------------------------------------------
if pagina == PAGINAS[0]:
    st.image(bandeira_axadrezada())
    st.write(f"Dashboard interativo sobre a história da Fórmula 1, com dados de "
             f"**{anos[0]}** a **{anos[1]}** (ajustável na barra lateral).")
    c = st.columns(5)
    c[0].metric("Épocas", res_f["year"].nunique())
    c[1].metric("Grandes Prémios", res_f["raceId"].nunique())
    c[2].metric("Pilotos", res_f["driverId"].nunique())
    c[3].metric("Equipas", res_f["constructorId"].nunique())
    c[4].metric("Circuitos", res_f["circuitId"].nunique())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Últimos Grandes Prémios do período")
        ult = (res_f[res_f["positionOrder"] == 1]
               .sort_values("date", ascending=False)
               [["year", "gp", "piloto", "equipa", "circuito"]].head(10))
        mostrar_tabela(ult.rename(columns={"year": "Ano", "gp": "Grande Prémio", "piloto": "Vencedor",
                                          "equipa": "Equipa", "circuito": "Circuito"}))
    with col2:
        st.subheader("Grandes Prémios por época")
        n = res_f.groupby("year")["raceId"].nunique().reset_index(name="corridas")
        mostrar_grafico(px.bar(n, x="year", y="corridas", color_discrete_sequence=[ROSA],
                               labels={"year": "Ano", "corridas": "Nº de corridas"}))
    st.info("Usa o menu lateral para navegar pelas 10 análises.")

elif pagina == PAGINAS[1]:
    st.write("Visão geral dos ficheiros usados, valores em falta e estatísticas descritivas.")
    brutos = dados["brutos"]
    tab1, tab2, tab3 = st.tabs(["📁 Ficheiros", "❓ Valores nulos", "📈 Estatísticas"])
    with tab1:
        resumo = pd.DataFrame({"ficheiro": list(brutos), "linhas": [len(v) for v in brutos.values()],
                               "colunas": [v.shape[1] for v in brutos.values()]})
        mostrar_tabela(resumo)
        nome = st.selectbox("Pré-visualizar ficheiro", list(brutos))
        st.dataframe(brutos[nome].head(20))
        st.caption("Tabela principal após merges (results + races + drivers + constructors + status + circuits):")
        st.dataframe(res.head(20))
    with tab2:
        nome = st.selectbox("Ficheiro", list(brutos), key="nulos")
        nulos = brutos[nome].isna().sum().reset_index()
        nulos.columns = ["coluna", "nulos"]
        nulos["% nulos"] = (100 * nulos["nulos"] / len(brutos[nome])).round(1)
        col1, col2 = st.columns([1, 2])
        col1.dataframe(nulos, hide_index=True)
        mostrar_grafico(px.bar(nulos, x="coluna", y="% nulos", title=f"% de nulos em {nome}.csv",
                               color_discrete_sequence=[ROSA]))
        st.caption("Os nulos (\\N) fazem sentido: p.ex. 'position' é nulo quando o piloto não classificou, "
                   "e os tempos de volta mais rápida só existem a partir de 2004.")
    with tab3:
        st.dataframe(res_f[["grid", "positionOrder", "points", "laps", "idade"]].describe().round(2))
        col1, col2 = st.columns(2)
        with col1:
            mostrar_grafico(px.histogram(res_f, x="grid", nbins=30, title="Distribuição da posição na grelha"))
        with col2:
            mostrar_grafico(px.histogram(res_f, x="points", nbins=30, title="Distribuição dos pontos por corrida"))

elif pagina == PAGINAS[2]:
    top = st.slider("Nº de pilotos a mostrar", 5, 25, 10)
    vit = (res_f.groupby("piloto")["vitoria"].sum().sort_values(ascending=False).head(top)
           .reset_index(name="vitórias"))
    col1, col2 = st.columns([1, 2])
    with col1:
        mostrar_tabela(vit)
    with col2:
        fig = px.bar(vit.sort_values("vitórias"), x="vitórias", y="piloto", orientation="h",
                     color="vitórias", color_continuous_scale=ESCALA_ROSA, title="Vitórias no período")
        mostrar_grafico(fig)
    st.subheader("Evolução acumulada das vitórias (top 5)")
    top5 = vit["piloto"].head(5)
    ev = (res_f[res_f["piloto"].isin(top5)].groupby(["year", "piloto"])["vitoria"].sum()
          .groupby(level="piloto").cumsum().reset_index())
    mostrar_grafico(px.line(ev, x="year", y="vitoria", color="piloto", markers=True,
                            labels={"year": "Ano", "vitoria": "Vitórias acumuladas"}))

elif pagina == PAGINAS[3]:
    f = ds_final[ds_final["year"].between(*anos)]
    campeoes = f[f["position"] == 1]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Campeões por época")
        mostrar_tabela(campeoes[["year", "piloto", "points", "wins"]].rename(
            columns={"year": "Ano", "points": "Pontos", "wins": "Vitórias"}).sort_values("Ano", ascending=False))
    with col2:
        st.subheader("Títulos por piloto")
        titulos = campeoes["piloto"].value_counts().reset_index()
        titulos.columns = ["piloto", "títulos"]
        mostrar_grafico(px.bar(titulos.head(12), x="piloto", y="títulos",
                               color_discrete_sequence=[ROSA]))
    st.subheader("Pontos e vitórias do campeão ao longo dos anos")
    fig = go.Figure()
    fig.add_bar(x=campeoes["year"], y=campeoes["points"], name="Pontos", marker_color=LILAS)
    fig.add_scatter(x=campeoes["year"], y=campeoes["wins"], name="Vitórias", yaxis="y2",
                    line=dict(color=ROSA_ESC))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Vitórias"),
                      yaxis_title="Pontos", xaxis_title="Ano")
    mostrar_grafico(fig)
    ano = st.selectbox("Ver classificação final de", sorted(f["year"].unique(), reverse=True))
    mostrar_tabela(f[f["year"] == ano][["position", "piloto", "points", "wins"]]
                   .rename(columns={"position": "Pos", "points": "Pontos", "wins": "Vitórias"}))

elif pagina == PAGINAS[4]:
    f = cr[cr["year"].between(*anos)]
    top = st.slider("Nº de equipas", 5, 20, 10)
    pts = f.groupby("equipa")["points"].sum().sort_values(ascending=False).head(top).reset_index()
    vit = res_f.groupby("equipa")["vitoria"].sum().sort_values(ascending=False).head(top).reset_index()
    col1, col2 = st.columns(2)
    with col1:
        mostrar_grafico(px.bar(pts.sort_values("points"), x="points", y="equipa", orientation="h",
                               title="Pontos totais", color_discrete_sequence=[LILAS]))
    with col2:
        mostrar_grafico(px.bar(vit.sort_values("vitoria"), x="vitoria", y="equipa", orientation="h",
                               title="Vitórias totais", color_discrete_sequence=[ROSA]))
    st.subheader("Pontos por época - comparar equipas")
    escolha = st.multiselect("Equipas", sorted(f["equipa"].unique()),
                             default=list(pts["equipa"].head(4)))
    por_ano = f[f["equipa"].isin(escolha)].groupby(["year", "equipa"])["points"].sum().reset_index()
    mostrar_grafico(px.line(por_ano, x="year", y="points", color="equipa", markers=True))
    st.subheader("Classificação final de construtores")
    cs = cs_final[cs_final["year"].between(*anos)]
    if cs.empty:
        st.info("O campeonato de construtores só existe a partir de 1958.")
    else:
        ano = st.selectbox("Época", sorted(cs["year"].unique(), reverse=True))
        mostrar_tabela(cs[cs["year"] == ano][["position", "equipa", "points", "wins"]])

elif pagina == PAGINAS[5]:
    st.write("A posição de partida determina o resultado? Analisamos a relação entre a "
             "posição na grelha e a posição final.")
    g = res_f[(res_f["grid"] > 0) & (res_f["grid"] <= 20) & res_f["position"].notna()]
    media = g.groupby("grid")["position"].agg(["mean", "median", "count"]).round(2).reset_index()
    col1, col2 = st.columns(2)
    with col1:
        mostrar_grafico(px.bar(media, x="grid", y="mean", title="Posição final média por posição na grelha",
                               labels={"grid": "Grelha", "mean": "Posição final média"},
                               color_discrete_sequence=[ROSA]))
    with col2:
        mostrar_grafico(px.density_heatmap(g, x="grid", y="position", nbinsx=20, nbinsy=20,
                                           color_continuous_scale=ESCALA_ROSA,
                                           title="Grelha vs posição final",
                                           labels={"grid": "Grelha", "position": "Posição final"}))
    corr = g["grid"].corr(g["position"])
    st.metric("Correlação grelha ↔ posição final", f"{corr:.2f}")
    pole = res_f[res_f["grid"] == 1]
    st.metric("Taxa de vitória de quem parte da pole", f"{100 * pole['vitoria'].mean():.1f}%")
    mostrar_tabela(media.rename(columns={"grid": "Grelha", "mean": "Média", "median": "Mediana",
                                         "count": "Nº casos"}))

elif pagina == PAGINAS[6]:
    p = pit[pit["year"].between(*anos)]
    if p.empty:
        st.info("Dados de pit stops existem apenas a partir de 2011. Ajusta o filtro de épocas.")
    else:
        por_ano = p.groupby("year")["segundos"].agg(["mean", "median", "count"]).round(2).reset_index()
        col1, col2 = st.columns(2)
        with col1:
            mostrar_grafico(px.line(por_ano, x="year", y=["mean", "median"], markers=True,
                                    title="Duração média/mediana do pit stop (s)",
                                    labels={"value": "Segundos", "year": "Ano"}))
        with col2:
            mostrar_grafico(px.bar(por_ano, x="year", y="count", title="Nº de pit stops por época",
                                   color_discrete_sequence=[LILAS]))
        ano = st.selectbox("Época para análise por equipa", sorted(p["year"].unique(), reverse=True))
        pa = p[p["year"] == ano]
        mostrar_grafico(px.box(pa, x="equipa", y="segundos", color="equipa",
                               title=f"Distribuição dos pit stops por equipa ({ano})"))
        st.subheader("10 pit stops mais rápidos do período")
        mostrar_tabela(p.nsmallest(10, "segundos")[["year", "gp", "piloto", "equipa", "segundos"]])

elif pagina == PAGINAS[7]:
    anos_laps = sorted(res[res["raceId"].isin(laps["raceId"].unique())]["year"].unique(), reverse=True)
    anos_laps = [a for a in anos_laps if anos[0] <= a <= anos[1]]
    if not anos_laps:
        st.info("Tempos por volta existem a partir de 1996. Ajusta o filtro de épocas.")
    else:
        c1, c2, c3 = st.columns(3)
        ano = c1.selectbox("Época", anos_laps)
        corridas = races[(races["year"] == ano) & races["raceId"].isin(laps["raceId"].unique())].sort_values("round")
        gp = c2.selectbox("Grande Prémio", corridas["gp"])
        race_id = int(corridas[corridas["gp"] == gp]["raceId"].iloc[0])
        n = c3.slider("Nº de pilotos (top da classificação)", 2, 10, 5)
        top_ids = res[res["raceId"] == race_id].nsmallest(n, "positionOrder")["driverId"]
        l = laps[(laps["raceId"] == race_id) & laps["driverId"].isin(top_ids)]
        mediana = l["segundos"].median()
        l = l[l["segundos"] < mediana * 1.25]  # remove voltas de safety car / pit stop
        col1, col2 = st.columns(2)
        with col1:
            mostrar_grafico(px.line(l, x="lap", y="segundos", color="piloto",
                                    title="Tempo por volta (voltas lentas removidas)",
                                    labels={"lap": "Volta", "segundos": "Segundos"}))
        with col2:
            lp = laps[(laps["raceId"] == race_id) & laps["driverId"].isin(top_ids)]
            fig = px.line(lp, x="lap", y="position", color="piloto", title="Posição ao longo da corrida")
            fig.update_yaxes(autorange="reversed")
            mostrar_grafico(fig)
        st.subheader("Volta mais rápida e ritmo médio")
        resumo = (laps[(laps["raceId"] == race_id) & laps["driverId"].isin(top_ids)]
                  .groupby("piloto")["segundos"].agg(["min", "mean", "std"]).round(3)
                  .sort_values("min").reset_index())
        mostrar_tabela(resumo.rename(columns={"min": "Melhor (s)", "mean": "Média (s)", "std": "Desvio-padrão"}))

elif pagina == PAGINAS[8]:
    st.subheader("Quanto vale partir da pole position?")
    pole = res_f[res_f["grid"] == 1].groupby("year")["vitoria"].mean().mul(100).reset_index(name="% vitórias")
    fig = px.line(pole, x="year", y="% vitórias", markers=True,
                  title="% de poles convertidas em vitória por época")
    fig.update_yaxes(range=[0, 100])
    mostrar_grafico(fig)

    st.subheader("Evolução do tempo da pole num circuito")
    qq = q[q["year"].between(*anos)]
    if qq.empty:
        st.info("Dados de qualificação existem a partir de 1994. Ajusta o filtro de épocas.")
    else:
        circ = st.selectbox("Circuito", sorted(qq["circuito"].unique()))
        poles = (qq[(qq["circuito"] == circ) & (qq["position"] == 1)]
                 .sort_values("year")[["year", "piloto", "melhor_s"]].dropna())
        col1, col2 = st.columns([2, 1])
        with col1:
            mostrar_grafico(px.line(poles, x="year", y="melhor_s", markers=True, hover_data=["piloto"],
                                    labels={"melhor_s": "Tempo da pole (s)", "year": "Ano"}))
        with col2:
            mostrar_tabela(poles.rename(columns={"melhor_s": "Tempo (s)", "year": "Ano"}))
        st.subheader("Pilotos com mais poles (período)")
        top = (qq[qq["position"] == 1]["piloto"].value_counts().head(10).reset_index())
        top.columns = ["piloto", "poles"]
        mostrar_grafico(px.bar(top, x="piloto", y="poles", color_discrete_sequence=[ROSA]))

elif pagina == PAGINAS[9]:
    corridas_circ = (res_f.groupby(["circuito", "country", "lat", "lng"])["raceId"].nunique()
                     .reset_index(name="corridas"))
    fig = px.scatter_geo(corridas_circ, lat="lat", lon="lng", size="corridas", hover_name="circuito",
                         hover_data=["country", "corridas"], projection="natural earth",
                         color="corridas", color_continuous_scale=ESCALA_ROSA, title="Circuitos do período")
    mostrar_grafico(fig)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Circuitos com mais corridas")
        mostrar_tabela(corridas_circ.sort_values("corridas", ascending=False)
                       [["circuito", "country", "corridas"]].head(15))
    with col2:
        st.subheader("Corridas por país")
        pais = corridas_circ.groupby("country")["corridas"].sum().sort_values(ascending=False).head(15).reset_index()
        mostrar_grafico(px.bar(pais, x="corridas", y="country", orientation="h",
                               color_discrete_sequence=[LILAS]).update_yaxes(autorange="reversed"))
    st.subheader("Mapa nativo do Streamlit")
    st.map(corridas_circ.rename(columns={"lng": "lon"}), latitude="lat", longitude="lon", size="corridas")

elif pagina == PAGINAS[10]:
    st.write("Porque é que os pilotos não terminam as corridas? Consideramos 'terminou' quem viu a "
             "bandeira de xadrez (Finished ou +N voltas).")
    por_ano = res_f.groupby("year")["terminou"].mean().mul(100).reset_index(name="% terminaram")
    por_ano["% abandonos"] = 100 - por_ano["% terminaram"]
    col1, col2 = st.columns(2)
    with col1:
        mostrar_grafico(px.area(por_ano, x="year", y="% abandonos", title="% de abandonos por época",
                                color_discrete_sequence=[ROSA]))
    with col2:
        motivos = (res_f[~res_f["terminou"]]["status"].value_counts().head(12).reset_index())
        motivos.columns = ["motivo", "ocorrências"]
        mostrar_grafico(px.pie(motivos, names="motivo", values="ocorrências", hole=0.4,
                               title="Principais motivos de abandono"))
    st.subheader("Equipas mais fiáveis (mín. 100 participações)")
    fi = res_f.groupby("equipa").agg(participações=("resultId", "count"), taxa=("terminou", "mean"))
    fi = fi[fi["participações"] >= 100]
    fi["% terminou"] = (fi["taxa"] * 100).round(1)
    mostrar_tabela(fi.drop(columns="taxa").sort_values("% terminou", ascending=False).reset_index().head(15))

elif pagina == PAGINAS[11]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Idade dos pilotos por época")
        idade = res_f.dropna(subset=["idade"]).groupby("year")["idade"].mean().reset_index()
        mostrar_grafico(px.line(idade, x="year", y="idade", markers=True,
                                labels={"idade": "Idade média (anos)", "year": "Ano"}))
    with c2:
        st.subheader("Idade dos vencedores")
        v = res_f[res_f["vitoria"] == 1].dropna(subset=["idade"])
        mostrar_grafico(px.histogram(v, x="idade", nbins=25, color_discrete_sequence=[ROSA]))
    st.subheader("Nacionalidades com mais vitórias")
    nac = (res_f[res_f["vitoria"] == 1]["nac_piloto"].value_counts().head(12).reset_index())
    nac.columns = ["nacionalidade", "vitórias"]
    col1, col2 = st.columns([1, 2])
    with col1:
        mostrar_tabela(nac)
    with col2:
        mostrar_grafico(px.bar(nac, x="nacionalidade", y="vitórias", color="vitórias",
                               color_continuous_scale=ESCALA_ROSA))
    col1, col2 = st.columns(2)
    novo = res_f.dropna(subset=["idade"]).nsmallest(5, "idade")[["piloto", "gp", "year", "idade"]]
    velho = res_f.dropna(subset=["idade"]).nlargest(5, "idade")[["piloto", "gp", "year", "idade"]]
    col1.write("**Pilotos mais novos a competir**")
    col1.dataframe(novo.round(1), hide_index=True)
    col2.write("**Pilotos mais velhos a competir**")
    col2.dataframe(velho.round(1), hide_index=True)

else:  # Multimédia
    st.write("Elementos multimédia do dashboard: imagens geradas em código, mapas interativos e vídeo.")
    st.image(bandeira_axadrezada())
    tab1, tab2, tab3 = st.tabs(["🗺️ Mapa do circuito", "🎥 Vídeo", "🖼️ Imagens"])
    with tab1:
        circ = st.selectbox("Escolhe um circuito", sorted(circuits["circuito"]))
        linha = circuits[circuits["circuito"] == circ].iloc[0]
        st.write(f"**{linha['circuito']}** · {linha['location']}, {linha['country']}")
        st.map(pd.DataFrame({"lat": [linha["lat"]], "lon": [linha["lng"]]}), zoom=11)
        url = dados["brutos"]["circuits"].query("name == @circ")["url"].iloc[0]
        st.link_button("Ver na Wikipedia", url)
    with tab2:
        video = st.text_input("Cola um link de vídeo (ex.: YouTube)", "")
        if video:
            st.video(video)
        else:
            st.caption("Cola acima o link de um vídeo de F1 para o reproduzir aqui.")
    with tab3:
        ficheiro = st.file_uploader("Carrega uma imagem (carro, circuito, piloto...)", type=["png", "jpg", "jpeg"])
        if ficheiro:
            st.image(ficheiro, caption=ficheiro.name)
        else:
            st.caption("Carrega uma imagem para a mostrar aqui.")
