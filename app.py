import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os

# -----------------------------------------------------------------------------
# 1. Data Cleaning / Preparation Functions
# -----------------------------------------------------------------------------

def parse_money(x):
    """Extracts numeric values from strings like '$208.12' or '₹100'."""
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

def parse_date(s):
    """Handles inconsistent date formats (YYYY/MM/DD, DD-MM-YYYY, etc.)."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.to_datetime(s, errors="coerce")

@st.cache_data
def load_and_clean_data(path: str):
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    
    # Date conversion
    df['date'] = df['Ad_Date'].apply(parse_date)
    df = df.dropna(subset=['date'])
    
    # Numeric conversions
    df['Cost_num'] = df['Cost'].apply(parse_money)
    df['Sale_num'] = df['Sale_Amount'].apply(parse_money)
    
    numeric_cols = ['Clicks', 'Impressions', 'Leads', 'Conversions']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Standardize campaign names for cleaner filters
    df['Campaign_Name'] = df['Campaign_Name'].fillna('Unknown').str.strip()
    
    return df

# -----------------------------------------------------------------------------
# 2. Main Dashboard Setup
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Paid Media Performance Dashboard", layout="wide")

# Custom CSS for Premium Design
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
          background: #ffffff; 
          border: 1px solid #eef1f6; 
          padding: 14px 14px 10px 14px; 
          border-radius: 14px;
          box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
          transition: all 0.3s ease;
      }
      div[data-testid="stMetric"]:hover {
          border-color: #d1d5db;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
          transform: translateY(-2px);
      }
      .section-title {font-size: 1.1rem; font-weight: 700; margin: 0.25rem 0 0.5rem 0;}
      .subtle {color:#6b7280; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# Load Data
DATA_PATH = "data/GoogleAds_Performance_Standardized_v2.csv"
if not os.path.exists(DATA_PATH):
    # Try fallback to original if standardized doesn't exist
    DATA_PATH = "GoogleAds_DataAnalytics_Sales_Uncleaned.csv"

df = load_and_clean_data(DATA_PATH)

if df is None:
    st.error("Dataset not found. Please ensure 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv' is in the directory.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar Filters
# -----------------------------------------------------------------------------

st.sidebar.header("Dashboard Filters")

# Date range
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Fallback for date_range selection
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Multi-select filters
all_campaigns = sorted(df['Campaign_Name'].unique())
selected_campaigns = st.sidebar.multiselect("Campaign", all_campaigns, default=all_campaigns)

all_devices = sorted(df['Device'].dropna().unique())
selected_devices = st.sidebar.multiselect("Device", all_devices, default=all_devices)

all_locations = sorted(df['Location'].dropna().unique())
selected_locations = st.sidebar.multiselect("Location", all_locations, default=all_locations)

# Apply Filtered DataFrame (dff)
mask = (
    (df['date'].dt.date >= start_date) &
    (df['date'].dt.date <= end_date) &
    (df['Campaign_Name'].isin(selected_campaigns if selected_campaigns else all_campaigns)) &
    (df['Device'].isin(selected_devices if selected_devices else all_devices)) &
    (df['Location'].isin(selected_locations if selected_locations else all_locations))
)
dff = df.loc[mask].copy()

# -----------------------------------------------------------------------------
# 4. KPI Aggregation & Layout
# -----------------------------------------------------------------------------

st.title("Paid Media Performance Dashboard")

total_spend = dff['Cost_num'].sum()
total_revenue = dff['Sale_num'].sum()
total_conversions = dff['Conversions'].sum()
total_clicks = dff['Clicks'].sum()

# Calculated Metrics
total_impressions = dff['Impressions'].sum()
roas = total_revenue / total_spend if total_spend > 0 else 0
cpa = total_spend / total_conversions if total_conversions > 0 else 0
conv_rate = total_conversions / total_clicks if total_clicks > 0 else 0
cpm = (total_spend / total_impressions) * 1000 if total_impressions > 0 else 0
ctr = total_clicks / total_impressions if total_impressions > 0 else 0
cpc = total_spend / total_clicks if total_clicks > 0 else 0

# -----------------------------
# KPI Cards (Narrative Sectors)
# -----------------------------

st.markdown('<div class="section-title">1. Visibilidade & Alcance (Awareness)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Como o público está visualizando seus anúncios?</div>', unsafe_allow_html=True)
row1_1, row1_2, row1_3 = st.columns(3)
row1_1.metric("Impressions", f"{total_impressions:,.0f}", help="Número total de vezes que seus anúncios foram exibidos.")
row1_2.metric("CPM", f"${cpm:,.2f}", help="Custo médio por mil impressões.")
row1_3.metric("CTR", f"{ctr:.2%}", help="Click-Through Rate: Porcentagem de impressões que resultaram em cliques.")

st.markdown('<div class="section-title">2. Engajamento & Eficiência (Engagement)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">O público está interagindo e o tráfego é qualificado?</div>', unsafe_allow_html=True)
row2_1, row2_2, row2_3 = st.columns(3)
row2_1.metric("Clicks", f"{total_clicks:,.0f}", help="Número total de cliques em seus anúncios.")
row2_2.metric("CPC", f"${cpc:,.2f}", help="Custo médio pago por cada clique.")
row2_3.metric("Conversion Rate", f"{conv_rate:.2%}", help="Porcentagem de cliques que resultaram em conversões.")

st.markdown('<div class="section-title">3. Conversão & Retorno (Financial Impact)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Qual o custo por resultado e o retorno sobre o investimento?</div>', unsafe_allow_html=True)
row3_1, row3_2, row3_3, row3_4, row3_5 = st.columns(5)
row3_1.metric("Conversions", f"{total_conversions:,.0f}", help="Número total de ações valiosas (vendas/leads).")
row3_2.metric("CPA (Cost/Conv)", f"${cpa:,.2f}", help="Custo por Aquisição: Valor médio gasto para gerar uma conversão.")
row3_3.metric("Total Spend", f"${total_spend:,.2f}", help="Valor total investido nas campanhas.")
row3_4.metric("Total Revenue", f"${total_revenue:,.2f}", help="Valor total de receita gerada.")
row3_5.metric("ROAS", f"{roas:.2f}x", help="Return on Ad Spend: Vezes que o valor investido retornou em receita.")

st.divider()

# -----------------------------------------------------------------------------
# 5. Charts (Plotly Dual Axis)
# -----------------------------------------------------------------------------

# Daily Aggregation for Charts
daily = dff.groupby(dff['date'].dt.date).agg({
    'Impressions': 'sum',
    'Clicks': 'sum',
    'Cost_num': 'sum',
    'Conversions': 'sum',
    'Sale_num': 'sum'
}).reset_index().rename(columns={'index': 'date'})

daily['CPC'] = np.where(daily['Clicks'] > 0, daily['Cost_num'] / daily['Clicks'], 0)
daily['ConvRate'] = np.where(daily['Clicks'] > 0, daily['Conversions'] / daily['Clicks'], 0)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Chart 1 — Impressions vs Clicks
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x=daily['date'], y=daily['Impressions'], name="Impressions", fill='tozeroy', line=dict(color='#1f77b4')), secondary_y=False)
    fig1.add_trace(go.Scatter(x=daily['date'], y=daily['Clicks'], name="Clicks", line=dict(color='#ff7f0e')), secondary_y=True)
    fig1.update_layout(title_text="Impressions vs Clicks", height=400, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig1.update_yaxes(title_text="Impressions", secondary_y=False)
    fig1.update_yaxes(title_text="Clicks", secondary_y=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 3 — Spend vs Revenue (Stacked)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=daily['date'], y=daily['Cost_num'], name="Spend ($)", marker_color='#e377c2'))
    fig3.add_trace(go.Bar(x=daily['date'], y=daily['Sale_num'], name="Revenue ($)", marker_color='#2196f3'))

    fig3.update_layout(
        title_text="Spend vs Revenue (Stacked)", 
        height=400, 
        barmode='stack',
        template="plotly_white", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Amount ($)")
    )
    st.plotly_chart(fig3, use_container_width=True)

with chart_col2:
    # Chart 2 — Clicks vs CPC
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=daily['date'], y=daily['Clicks'], name="Clicks", marker_color='#ff7f0e', opacity=0.7), secondary_y=False)
    fig2.add_trace(go.Scatter(x=daily['date'], y=daily['CPC'], name="CPC ($)", mode="lines+markers", line=dict(color='#2ca02c')), secondary_y=True)
    fig2.update_layout(title_text="Clicks vs CPC", height=400, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig2.update_yaxes(title_text="Clicks", secondary_y=False)
    fig2.update_yaxes(title_text="CPC ($)", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 4 — Conversions vs Conversion Rate
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Bar(x=daily['date'], y=daily['Conversions'], name="Conversions", marker_color='#9467bd', opacity=0.7), secondary_y=False)
    fig4.add_trace(go.Scatter(x=daily['date'], y=daily['ConvRate'], name="Conv. Rate (%)", mode="lines+markers", line=dict(color='#d62728')), secondary_y=True)
    fig4.update_layout(title_text="Conversions vs Conv. Rate", height=400, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig4.update_yaxes(title_text="Conversions", secondary_y=False)
    fig4.update_yaxes(title_text="Conv. Rate (%)", tickformat=".1%", secondary_y=True)
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. Extra Table (Top Campaigns)
# -----------------------------------------------------------------------------

st.subheader("Top Campaigns Comparison")

campaign_stats = dff.groupby("Campaign_Name").agg({
    'Cost_num': 'sum',
    'Sale_num': 'sum',
    'Clicks': 'sum',
    'Impressions': 'sum',
    'Conversions': 'sum'
}).reset_index()

campaign_stats['ROAS'] = np.where(campaign_stats['Cost_num'] > 0, campaign_stats['Sale_num'] / campaign_stats['Cost_num'], 0)
campaign_stats['CPA'] = np.where(campaign_stats['Conversions'] > 0, campaign_stats['Cost_num'] / campaign_stats['Conversions'], 0)

# Sort by ROAS desc then Revenue desc
campaign_stats = campaign_stats.sort_values(by=['ROAS', 'Sale_num'], ascending=[False, False])

# Final formatting for display
campaign_stats_display = campaign_stats.rename(columns={
    'Campaign_Name': 'Campaign',
    'Cost_num': 'Spend ($)',
    'Sale_num': 'Revenue ($)',
    'Clicks': 'Clicks',
    'Impressions': 'Impressions',
    'Conversions': 'Conversions',
    'ROAS': 'ROAS (x)',
    'CPA': 'CPA ($)'
})

st.dataframe(campaign_stats_display.style.format({
    'Spend ($)': '${:,.2f}',
    'Revenue ($)': '${:,.2f}',
    'ROAS (x)': '{:.2f}x',
    'CPA ($)': '${:,.2f}',
    'Clicks': '{:,.0f}',
    'Impressions': '{:,.0f}',
    'Conversions': '{:,.0f}'
}), use_container_width=True, hide_index=True)

st.caption("Generated by Senior Paid Media Performance System")
