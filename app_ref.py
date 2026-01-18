# %% STREAMLIT CONFIG DEVE VIR PRIMEIRO!
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import sys
from sql_queries import imdb_query
from functions import clean_list_items, ensure_list_format, filter_df_by_list_contains,get_filtered_actors,get_filtered_genres,update_selected_genres ,update_selected_actors


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from connection import read_database

st.set_page_config(
    page_title="IMDB Dashboard", layout="wide", initial_sidebar_state="expanded"
)

# --- Load data
df_0 = read_database(imdb_query)

# Drop linhas com valores ausentes em colunas importantes
df = df_0.dropna(
    subset=[
        "Rating",
        "Votes",
        "RevenueMillions",
        "Metascore",
        "Genre",
        "Actors",
        "Year",
    ]
)

df["Actors"] = ensure_list_format(df["Actors"])
df["Genre"] = ensure_list_format(df["Genre"])
df["Actors"] = clean_list_items(df["Actors"])
df["Genre"] = clean_list_items(df["Genre"])


# ------------------------------------------------------------

# Sidebar filtros comuns
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())


st.sidebar.title("Filtros")

selected_range = st.sidebar.slider(
    "Years",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

# Inicializar listas vazias (Streamlit reseta a execução toda vez)
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []

if "selected_actors" not in st.session_state:
    st.session_state.selected_actors = []


# Primeiro, opções dinâmicas baseadas no estado atual
genres_options = get_filtered_genres(df, st.session_state.selected_actors)
actors_options = get_filtered_actors(df, st.session_state.selected_genres)


# aplicar filtro de anos
df_filtered = df[(df["Year"] >= selected_range[0]) & (df["Year"] <= selected_range[1])]

# Aplicar filtro dinâmico para atores e gêneros, se houver seleção
filters = {}
if st.session_state.selected_actors:
    filters["Actors"] = st.session_state.selected_actors
if st.session_state.selected_genres:
    filters["Genre"] = st.session_state.selected_genres

if filters:
    df_filtered = filter_df_by_list_contains(df_filtered, filters)

# ------------------------------------------------------------
# Dashboard e visualizações

st.sidebar.multiselect(
    label="Genres",
    options=genres_options,
    default=st.session_state.selected_genres,
    key="tmp_genres",
    on_change=update_selected_genres,
)


st.sidebar.multiselect(
    label="Actors",
    options=actors_options,
    default=st.session_state.selected_actors,
    key="tmp_actors",
    on_change=update_selected_actors,
)

st.title("🎬 IMDB Movies Dashboard")
st.markdown(
    "*Análise de 1000 filmes do IMDB com base em Rating, Receita e Gênero e outros KPI's.*"
)


col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🎞️ Total de Filmes", len(df_filtered), border=True)
col2.metric("⭐ Média de Rating", round(df_filtered["Rating"].mean(), 2), border=True)
col3.metric(
    "📏 Receita Média (M)",
    f"${round(df_filtered['RevenueMillions'].mean(), 2)}M",
    border=True,
)
col4.metric(
    "💰 Receita Total (M)",
    f"${round(df_filtered['RevenueMillions'].sum(), 5)}M",
    border=True,
)
col5.metric("🧮 Média de Votos", round(df_filtered["Votes"].mean()), border=True)

col_y = {"RevenueMillions", "Rating", "Runtime"}
col_x = {"Year", "Genre", "Actors", "Title"}

col_y_explanation = """
RevenueMillions = Receita do filme em milhões de dólares
• Rating = Avaliação média do filme (ex: IMDb)
• Runtime = Duração do filme em minutos
"""

col_x_explanation = """
Year = Ano de lançamento do filme
• Genre = Gênero do filme (ex: Ação, Comédia, Drama)
• Actors = Atores principais que participaram do filme
• Title = Título do filme
"""


# Faz a interseção com as colunas do dataframe
colunas_filtradas_y = list(set(df.columns) & col_y)
colunas_filtradas_x = list(set(df.columns) & col_x)

# Define o índice padrão de forma segura
default_index_y = (
    colunas_filtradas_y.index("RevenueMillions")
    if "RevenueMillions" in colunas_filtradas_y
    else 0
)

default_index_x = (
    colunas_filtradas_x.index("Genre") if "Genre" in colunas_filtradas_x else 0
)

col1, col2 = st.columns(2)

with col1:
    y_selected = st.selectbox(
        "Selecione a métrica Y:",
        options=colunas_filtradas_y,
        index=default_index_y,
        help=col_y_explanation,
    )

with col2:
    x_selected = st.selectbox(
        "Selecione a métrica X:",
        options=colunas_filtradas_x,
        index=default_index_x,
        help=col_x_explanation,
    )


# Explodir 'Genre' para o agrupamento funcionar
df_for_group = df_filtered.explode(x_selected)

genre_rating = (
    df_for_group.groupby(x_selected)[y_selected]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig1 = px.bar(
    genre_rating,
    x=x_selected,
    y=y_selected,
    color=x_selected,
    title=f"Top 10 {x_selected} por {y_selected}",
)
st.plotly_chart(fig1, use_container_width=True)


st.dataframe(df_filtered)
