import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import re
import os
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# 0. Forecasting Engine (Robust Baseline)
# -----------------------------------------------------------------------------

def baseline_forecast(daily_df: pd.DataFrame, horizon: int = 7, cap: float = 0.50):
    """
    daily_df: DataFrame with colunas ['date', 'revenue', 'spend', 'conversions']
    horizon: dias a prever
    cap: limite de ajuste de tendência (ex: 0.50 = +/-50% para mais sensibilidade)
    """
    d = daily_df.sort_values("date").copy()

    # Garantir colunas numéricas
    for col in ["revenue", "spend", "conversions"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)

    if len(d) < 7:
        return None

    def _forecast_metric(series: pd.Series):
        # base = MA7
        base = series.tail(7).mean()

        # tendência controlada (se tiver 14 dias)
        if len(series) >= 14:
            prev_mean = series.tail(14).head(7).mean()
            if prev_mean > 0:
                ratio = base / prev_mean
            else:
                ratio = 1.0
            ratio = float(np.clip(ratio, 1 - cap, 1 + cap))
        else:
            ratio = 1.0

        daily_pred = base * ratio
        daily_pred = max(daily_pred, 0)

        preds = np.repeat(daily_pred, horizon)
        return preds, daily_pred

    # Forecasts
    rev_preds, rev_daily = _forecast_metric(d["revenue"])
    spend_preds, spend_daily = _forecast_metric(d["spend"])
    conv_preds, conv_daily = _forecast_metric(d["conversions"])

    # Datas futuras
    last_date = pd.to_datetime(d["date"].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Confidence (baseado em variação recente)
    recent = d["revenue"].tail(14) if len(d) >= 14 else d["revenue"].tail(7)
    mean = recent.mean()
    std = recent.std()
    cv = (std / (mean + 1e-9)) if mean > 0 else 1.0

    if cv < 0.25:
        confidence = "High"
    elif cv < 0.50:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Range Low/High simples
    low_factor = 0.90
    high_factor = 1.10

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "pred_revenue": rev_preds,
        "pred_spend": spend_preds,
        "pred_conversions": conv_preds,
    })

    # ROAS previsto
    forecast_df["pred_roas"] = np.where(
        forecast_df["pred_spend"] > 0,
        forecast_df["pred_revenue"] / forecast_df["pred_spend"],
        0
    )

    summary = {
        "revenue_total": float(forecast_df["pred_revenue"].sum()),
        "spend_total": float(forecast_df["pred_spend"].sum()),
        "conversions_total": float(forecast_df["pred_conversions"].sum()),
        "roas_total": float(
            (forecast_df["pred_revenue"].sum() / forecast_df["pred_spend"].sum())
            if forecast_df["pred_spend"].sum() > 0 else 0
        ),
        "revenue_low": float(forecast_df["pred_revenue"].sum() * low_factor),
        "revenue_high": float(forecast_df["pred_revenue"].sum() * high_factor),
        "confidence": confidence,
    }

    return forecast_df, summary

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
    df["date"] = df["Ad_Date"].apply(parse_date)
    df = df.dropna(subset=["date"])

    # Numeric conversions
    df["Cost_num"] = df["Cost"].apply(parse_money)
    df["Sale_num"] = df["Sale_Amount"].apply(parse_money)

    numeric_cols = [
        "Clicks",
        "Impressions",
        "Leads",
        "Conversions",
        "latitude",
        "longitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Standardize campaign names for cleaner filters
    df["Campaign_Name"] = df["Campaign_Name"].fillna("Unknown").str.strip()

    return df


# -----------------------------------------------------------------------------
# 2. Main Dashboard Setup & Theme Variables
# -----------------------------------------------------------------------------

# Theme Colors
COLOR_SURFACE = "#444748"
COLOR_BORDER = "#929292"
COLOR_TEXT_PRIMARY = "#FBFBFB"
COLOR_TEXT_SECONDARY = "#D2D1D2"
COLOR_ACCENT = "#38d5ea"  # Keeping the neon cyan for accents

st.set_page_config(
    page_title="Paid Media Performance Dashboard of NITIT Courses", layout="wide"
)

# Custom CSS for Premium Design
st.markdown(
    f"""
    <style>
      .stApp {{
          background-color: #181818;
          color: {COLOR_TEXT_PRIMARY};
      }}
      .block-container {{padding-top: 1.2rem; padding-bottom: 2rem;}}
      
      /* Dark Sidebar Styling */
      section[data-testid="stSidebar"] {{
          background-color: #111111;
      }}
      section[data-testid="stSidebar"] .stMarkdown p, 
      section[data-testid="stSidebar"] h1, 
      section[data-testid="stSidebar"] h2, 
      section[data-testid="stSidebar"] label {{
          color: #ffffff !important;
      }}

      /* Filter Button/Tag Colors */
      .stMultiSelect div[data-baseweb="tag"] {{
          background-color: {COLOR_ACCENT} !important;
          color: #181818 !important;
      }}
      
      div[data-testid="stMetric"] {{
          background: {COLOR_SURFACE}; 
          border: 1px solid {COLOR_BORDER}; 
          padding: 14px 14px 10px 14px; 
          border-radius: 14px;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          transition: all 0.3s ease;
      }}
      div[data-testid="stMetric"] label {{
          color: {COLOR_TEXT_SECONDARY} !important;
      }}
      div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
          color: {COLOR_TEXT_PRIMARY} !important;
      }}
      
      div[data-testid="stMetric"]:hover {{
          border-color: {COLOR_ACCENT};
          transform: translateY(-2px);
      }}
      .section-title {{font-size: 1.1rem; font-weight: 700; margin: 0.25rem 0 0.5rem 0; color: {COLOR_TEXT_PRIMARY};}}
      .subtle {{color:{COLOR_TEXT_SECONDARY}; font-size:0.9rem;}}
      
      /* Tabs Styling */
      .stTabs [data-baseweb="tab-list"] {{
          gap: 10px;
      }}
      .stTabs [data-baseweb="tab"] {{
          color: {COLOR_TEXT_SECONDARY};
      }}
      .stTabs [aria-selected="true"] {{
          color: {COLOR_ACCENT} !important;
          border-bottom-color: {COLOR_ACCENT} !important;
      }}
      
      /* Table/DataFrame Styling (Dark) */
      .stDataFrame {{
          background-color: {COLOR_SURFACE};
          border: 1px solid {COLOR_BORDER};
          border-radius: 10px;
      }}
      [data-testid="stHeader"] {{
          background-color: rgba(0,0,0,0);
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Data
DATA_PATH = "data/GoogleAds_Performance_Standardized_v2.csv"
if not os.path.exists(DATA_PATH):
    # Try fallback to original if standardized doesn't exist
    DATA_PATH = "GoogleAds_DataAnalytics_Sales_Uncleaned.csv"

df = load_and_clean_data(DATA_PATH)

if df is None:
    st.error(
        "Dataset not found. Please ensure 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv' is in the directory."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar Filters
# -----------------------------------------------------------------------------

# Sidebar Logo/Header Image
st.sidebar.image("assets/nitit_courses_header.png", use_container_width=True)

st.sidebar.header("Dashboard Filters")

# Date range
min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
)

# Fallback for date_range selection
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Multi-select filters
all_campaigns = sorted(df["Campaign_Name"].unique())
selected_campaigns = st.sidebar.multiselect(
    "Campaign", all_campaigns, default=all_campaigns
)

all_devices = sorted(df["Device"].dropna().unique())
selected_devices = st.sidebar.multiselect("Device", all_devices, default=all_devices)

all_locations = sorted(df["Location"].dropna().unique())
selected_locations = st.sidebar.multiselect(
    "Location", all_locations, default=all_locations
)

# Apply Filtered DataFrame (dff)
mask = (
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (
        df["Campaign_Name"].isin(
            selected_campaigns if selected_campaigns else all_campaigns
        )
    )
    & (df["Device"].isin(selected_devices if selected_devices else all_devices))
    & (df["Location"].isin(selected_locations if selected_locations else all_locations))
)
dff = df.loc[mask].copy()

# -----------------------------------------------------------------------------
# 4. KPI Aggregation & Layout
# -----------------------------------------------------------------------------

st.title("Paid Media Performance Dashboard")
st.markdown(
    '<div class="subtle">This dashboard shows the performance of paid media campaigns for NITIT Courses.</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 4. Navigation Tabs
# -----------------------------------------------------------------------------
tab_dashboard, tab_prediction, tab_about = st.tabs(["Dashboard", "Prediction", "About"])

with tab_dashboard:
    # -----------------------------------------------------------------------------
    # 5. Dashboard Data Aggregation & Prediction Logic
    # -----------------------------------------------------------------------------
    
    # Daily Aggregation for Charts and Predictions
    daily = (
        dff.groupby(dff["date"].dt.date)
        .agg(
            revenue=("Sale_num", "sum"),
            spend=("Cost_num", "sum"),
            conversions=("Conversions", "sum"),
            impressions=("Impressions", "sum"),
            clicks=("Clicks", "sum"),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # Base KPI Definitions (Using dff for consistency with filter logic)
    total_spend = dff["Cost_num"].sum()
    total_revenue = dff["Sale_num"].sum()
    total_conversions = dff["Conversions"].sum()
    total_clicks = dff["Clicks"].sum()

    daily["cpc"] = np.where(daily["clicks"] > 0, daily["spend"] / daily["clicks"], 0)
    daily["conv_rate"] = np.where(
        daily["clicks"] > 0, daily["conversions"] / daily["clicks"], 0
    )

    # Forecast Calculation (Using the new robust engine with higher sensitivity)
    result_forecast = baseline_forecast(daily, horizon=7, cap=0.50)
    
    if result_forecast:
        forecast_df_dash, summary_dash = result_forecast
        week_forecast = summary_dash["revenue_total"]
        forecast_info = (
            f"Expected Revenue (7d): **${week_forecast:,.2f}** | "
            f"Confidence: **{summary_dash['confidence']}** | "
            "Based on recent MA7 trends with optimized growth sensitivity (up to 50%)."
        )
    else:
        week_forecast = 0
        forecast_info = "Insufficient data (at least 7 days required) for an accurate forecast."

    # Calculated Metrics
    total_impressions = dff["Impressions"].sum()
    roas = total_revenue / total_spend if total_spend > 0 else 0
    cpa = total_spend / total_conversions if total_conversions > 0 else 0
    conv_rate = total_conversions / total_clicks if total_clicks > 0 else 0
    cpm = (total_spend / total_impressions) * 1000 if total_impressions > 0 else 0
    ctr = total_clicks / total_impressions if total_impressions > 0 else 0
    cpc = total_spend / total_clicks if total_clicks > 0 else 0

    # -----------------------------
    # KPI Cards (Narrative Sectors)
    # -----------------------------

    st.markdown(
        '<div class="section-title">1. Visibility & Reach (Awareness)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtle">How is the audience viewing your ads?</div>',
        unsafe_allow_html=True,
    )
    row1_1, row1_2, row1_3 = st.columns(3)
    row1_1.metric(
        "Impressions",
        f"{total_impressions:,.0f}",
        help="Total number of times your ads were displayed.",
    )
    row1_2.metric("CPM", f"${cpm:,.2f}", help="Average cost per thousand impressions.")
    row1_3.metric(
        "CTR",
        f"{ctr:.2%}",
        help="Click-Through Rate: Percentage of impressions that resulted in clicks.",
    )

    st.markdown(
        '<div class="section-title">2. Engagement & Efficiency (Engagement)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtle">Is the audience interacting and is the traffic qualified?</div>',
        unsafe_allow_html=True,
    )
    row2_1, row2_2, row2_3 = st.columns(3)
    row2_1.metric(
        "Clicks", f"{total_clicks:,.0f}", help="Total number of clicks on your ads."
    )
    row2_2.metric("CPC", f"${cpc:,.2f}", help="Average cost paid for each click.")
    row2_3.metric(
        "Conversion Rate",
        f"{conv_rate:.2%}",
        help="Percentage of clicks that resulted in conversions.",
    )

    st.markdown(
        '<div class="section-title">3. Conversion & Return (Financial Impact)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtle">What is the cost per result and the return on investment?</div>',
        unsafe_allow_html=True,
    )
    
    # Financial Row 1
    row3_1, row3_2, row3_3 = st.columns(3)
    row3_1.metric(
        "Conversions",
        f"{total_conversions:,.0f}",
        help="Total number of valuable actions (sales/leads).",
    )
    row3_2.metric(
        "CPA (Cost/Conv)",
        f"${cpa:,.2f}",
        help="Cost per Acquisition: Average amount spent to generate a conversion.",
    )
    row3_3.metric(
        "Total Spend",
        f"${total_spend:,.2f}",
        help="Total amount invested in campaigns.",
    )
    
    # Financial Row 2
    row4_1, row4_2, row4_3 = st.columns(3)
    row4_1.metric(
        "Total Revenue",
        f"${total_revenue:,.2f}",
        help="Total amount of revenue generated.",
    )
    row4_2.metric(
        "ROAS",
        f"{roas:.2f}x",
        help="Return on Ad Spend: Number of times the invested amount returned as revenue.",
    )
    row4_3.metric(
        "Exp. Revenue (7d) 🔮",
        f"${week_forecast:,.2f}",
        help=forecast_info
    )

    st.divider()

    # -----------------------------------------------------------------------------
    # 6. Charts (Plotly Dual Axis)
    # -----------------------------------------------------------------------------

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Chart 1 — Impressions vs Clicks
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["impressions"],
                name="Impressions",
                fill="tozeroy",
                line=dict(color="#1f77b4"),
            ),
            secondary_y=False,
        )
        fig1.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["clicks"],
                name="Clicks",
                line=dict(color="#ff7f0e"),
            ),
            secondary_y=True,
        )
        fig1.update_layout(
            title_text="Impressions vs Clicks",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_PRIMARY),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=COLOR_TEXT_PRIMARY),
            ),
            title_font=dict(color=COLOR_TEXT_PRIMARY),
        )
        fig1.update_xaxes(showgrid=False, color=COLOR_TEXT_SECONDARY)
        fig1.update_yaxes(
            title_text="Impressions",
            secondary_y=False,
            gridcolor=COLOR_SURFACE,
            color=COLOR_TEXT_SECONDARY,
        )
        fig1.update_yaxes(
            title_text="Clicks",
            secondary_y=True,
            showgrid=False,
            color=COLOR_TEXT_SECONDARY,
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 3 — Spend vs Revenue (Stacked)
        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(
                x=daily["date"],
                y=daily["spend"],
                name="Spend ($)",
                marker_color="#e377c2",
            )
        )
        fig3.add_trace(
            go.Bar(
                x=daily["date"],
                y=daily["revenue"],
                name="Revenue ($)",
                marker_color=COLOR_ACCENT,
            )
        )

        fig3.update_layout(
            title_text="Spend vs Revenue (Stacked)",
            height=400,
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_PRIMARY),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=COLOR_TEXT_PRIMARY),
            ),
            yaxis=dict(title="Amount ($)", gridcolor=COLOR_SURFACE),
            title_font=dict(color=COLOR_TEXT_PRIMARY),
        )
        fig3.update_xaxes(showgrid=False, color=COLOR_TEXT_SECONDARY)
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col2:
        # Chart 2 — Clicks vs CPC
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(
            go.Bar(
                x=daily["date"],
                y=daily["clicks"],
                name="Clicks",
                marker_color="#ff7f0e",
                opacity=0.7,
            ),
            secondary_y=False,
        )
        fig2.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["cpc"],
                name="CPC ($)",
                mode="lines+markers",
                line=dict(color="#2ca02c"),
            ),
            secondary_y=True,
        )
        fig2.update_layout(
            title_text="Clicks vs CPC",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_PRIMARY),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=COLOR_TEXT_PRIMARY),
            ),
            title_font=dict(color=COLOR_TEXT_PRIMARY),
        )
        fig2.update_xaxes(showgrid=False, color=COLOR_TEXT_SECONDARY)
        fig2.update_yaxes(
            title_text="Clicks",
            secondary_y=False,
            gridcolor=COLOR_SURFACE,
            color=COLOR_TEXT_SECONDARY,
        )
        fig2.update_yaxes(
            title_text="CPC ($)",
            secondary_y=True,
            showgrid=False,
            color=COLOR_TEXT_SECONDARY,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 4 — Conversions vs Conversion Rate
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(
            go.Bar(
                x=daily["date"],
                y=daily["conversions"],
                name="Conversions",
                marker_color="#9467bd",
                opacity=0.7,
            ),
            secondary_y=False,
        )
        fig4.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["conv_rate"],
                name="Conv. Rate (%)",
                mode="lines+markers",
                line=dict(color="#d62728"),
            ),
            secondary_y=True,
        )
        fig4.update_layout(
            title_text="Conversions vs Conv. Rate",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_PRIMARY),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=COLOR_TEXT_PRIMARY),
            ),
            title_font=dict(color=COLOR_TEXT_PRIMARY),
        )
        fig4.update_xaxes(showgrid=False, color=COLOR_TEXT_SECONDARY)
        fig4.update_yaxes(
            title_text="Conversions",
            secondary_y=False,
            gridcolor=COLOR_SURFACE,
            color=COLOR_TEXT_SECONDARY,
        )
        fig4.update_yaxes(
            title_text="Conv. Rate (%)",
            tickformat=".1%",
            secondary_y=True,
            showgrid=False,
            color=COLOR_TEXT_SECONDARY,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # st.markdown('<div class="section-title">4. Map Analytics</div>', unsafe_allow_html=True)
    # st.markdown('<div class="subtle">Geographic distribution of revenue generated by location.</div>', unsafe_allow_html=True)

    # # Aggregate revenue by coordinates
    # map_data = dff.groupby(['latitude', 'longitude', 'Location']).agg({'Sale_num': 'sum'}).reset_index()

    # # Pre-format revenue for the tooltip (Deck.gl template fix)
    # map_data['revenue_formatted'] = map_data['Sale_num'].apply(lambda x: f"${x:,.2f}")

    # if not map_data.empty:
    #     st.pydeck_chart(pdk.Deck(
    #         map_style='light',
    #         initial_view_state=pdk.ViewState(
    #             latitude=map_data['latitude'].mean(),
    #             longitude=map_data['longitude'].mean(),
    #             zoom=10, # Zoom mais próximo para ver a granularidade
    #             pitch=45,
    #         ),
    #         layers=[
    #             pdk.Layer(
    #                 'HeatmapLayer',
    #                 data=map_data,
    #                 get_position='[longitude, latitude]',
    #                 get_weight='Sale_num', # Heat weight is revenue
    #                 radius_pixels=60,      # Heat granularity
    #                 intensity=1,
    #                 threshold=0.05,
    #                 aggregation='"SUM"',
    #             ),
    #         ],
    #         tooltip={
    #             "html": "<b>Location:</b> {Location}<br><b>Revenue:</b> {revenue_formatted}",
    #             "style": {"color": "white"}
    #         }
    #     ))
    # else:
    #     st.info("No location data available for the current filters.")

    st.divider()

    # -----------------------------------------------------------------------------
    # 7. Extra Table (Top Campaigns)
    # -----------------------------------------------------------------------------

    st.subheader("Top Campaigns Comparison")

    campaign_stats = (
        dff.groupby("Campaign_Name")
        .agg(
            {
                "Cost_num": "sum",
                "Sale_num": "sum",
                "Clicks": "sum",
                "Impressions": "sum",
                "Conversions": "sum",
            }
        )
        .reset_index()
    )

    campaign_stats["ROAS"] = np.where(
        campaign_stats["Cost_num"] > 0,
        campaign_stats["Sale_num"] / campaign_stats["Cost_num"],
        0,
    )
    campaign_stats["CPA"] = np.where(
        campaign_stats["Conversions"] > 0,
        campaign_stats["Cost_num"] / campaign_stats["Conversions"],
        0,
    )

    # Sort by ROAS desc then Revenue desc
    campaign_stats = campaign_stats.sort_values(
        by=["ROAS", "Sale_num"], ascending=[False, False]
    )

    # Final formatting for display
    campaign_stats_display = campaign_stats.rename(
        columns={
            "Campaign_Name": "Campaign",
            "Cost_num": "Spend ($)",
            "Sale_num": "Revenue ($)",
            "Clicks": "Clicks",
            "Impressions": "Impressions",
            "Conversions": "Conversions",
            "ROAS": "ROAS (x)",
            "CPA": "CPA ($)",
        }
    )

    # Export Button
    csv = campaign_stats_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Campaign Comparison to CSV",
        data=csv,
        file_name="campaign_comparison.csv",
        mime="text/csv",
    )

    st.dataframe(
        campaign_stats_display.style.format(
            {
                "Spend ($)": "${:,.2f}",
                "Revenue ($)": "${:,.2f}",
                "ROAS (x)": "{:.2f}x",
                "CPA ($)": "${:,.2f}",
                "Clicks": "{:,.0f}",
                "Impressions": "{:,.0f}",
                "Conversions": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_prediction:
    st.markdown('<div class="section-title">Revenue Forecast Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">This model uses <b>Linear Trend Analysis</b> combined with a robust MA7 baseline for scaling. '
        '<b>Simulation (What-if):</b> Enter your planned budget below to see how the trend line and projected revenue react based on your simulated investment.</div>', 
        unsafe_allow_html=True
    )
    
    if len(daily) >= 7:
        # 1. Base Setup & Controls
        col_setup1, col_setup2 = st.columns([1, 1])
        
        with col_setup1:
            horizon = st.slider(
                "Forecast horizon (days)", 
                7, 30, 14, 
                help="Select how many days into the future to project."
            )
        
        # Run Robust Forecast for Summary Stats (with 50% trend cap for sensitivity)
        result_base = baseline_forecast(daily, horizon=horizon, cap=0.50)
        
        if result_base:
            forecast_df_base, summary = result_base
            
            with col_setup2:
                # Defaulting to the baseline trend spend instead of 0 to show the full potential immediately
                planned_spend = st.number_input(
                    "Planned Campaign Spend ($)",
                    value=float(summary['spend_total']),
                    step=100.0,
                    help="Simulation: This defaults to the current trend spend. Increase it to see growth potential!"
                )
            
            st.divider()
            
            # 2. Linear Regression for Visually Dynamic Slope
            daily_reg = daily.copy()
            daily_reg['t'] = (daily_reg['date'] - daily_reg['date'].min()).dt.days
            X = daily_reg[['t']].values
            y = daily_reg['revenue'].values
            
            model = LinearRegression().fit(X, y)
            
            # Future Dates for Charting
            last_date = pd.to_datetime(daily_reg['date'].max())
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
            future_t = (future_dates - pd.to_datetime(daily_reg['date'].min())).days.values.reshape(-1, 1)
            
            # Base Regression Projection (the points on the line)
            y_trend_base = np.maximum(model.predict(future_t), 0)
            
            # Scaling Factor: Based on Planned Spend vs Baseline Trend Spend
            # We want the SUM of the plotted line to match the Simulated Revenue
            simulated_total_revenue = planned_spend * summary['roas_total']
            
            # If the user enters 0, we show 0. Otherwise we scale the trend line.
            if planned_spend > 0:
                current_trend_sum = y_trend_base.sum()
                scaling_factor = simulated_total_revenue / current_trend_sum if current_trend_sum > 0 else 1.0
                y_plot = y_trend_base * scaling_factor
            else:
                y_plot = np.zeros(len(future_dates))
            
            # 3. KPI Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Simulated Revenue 💰", f"${simulated_total_revenue:,.2f}")
            with m2:
                st.metric("Planned Spend 🎯", f"${planned_spend:,.2f}")
    
            
            # Context Info
            st.write(f"**Confidence:** {summary['confidence']} | **Projected ROAS:** {summary['roas_total']:.2f}x | **Range (Baseline):** ${summary['revenue_low']:,.0f} - ${summary['revenue_high']:,.0f}")
            
            # 4. Final Chart (Dynamic Slope)
            fig_p = go.Figure()
            
            fig_p.add_trace(go.Scatter(
                x=daily['date'], y=daily['revenue'],
                mode="lines+markers",
                name="Actual Revenue",
                line=dict(color=COLOR_ACCENT)
            ))
            
            fig_p.add_trace(go.Scatter(
                x=future_dates, y=y_plot,
                mode="lines+markers",
                name="Simulated Forecast",
                line=dict(color="#e377c2", dash='dot')
            ))
            
            fig_p.update_layout(
                title=f"Forecast Simulation ({horizon} days) — Dynamic Trend",
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLOR_TEXT_PRIMARY),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=COLOR_TEXT_PRIMARY)
                ),
                xaxis=dict(gridcolor=COLOR_SURFACE, color=COLOR_TEXT_SECONDARY),
                yaxis=dict(title="Revenue ($)", gridcolor=COLOR_SURFACE, color=COLOR_TEXT_SECONDARY),
                height=450
            )
            
            st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.warning("Insufficient historical data. At least 7 days of daily history are required for a trend forecast.")

with tab_about:
    st.markdown(
        '<div class="section-title">About the Creator</div>', unsafe_allow_html=True
    )

    col_img, col_text = st.columns([1, 3])

    with col_img:
        st.image(
            "https://avatars.githubusercontent.com/u/38968344?v=4",
            use_container_width=True,
        )

    with col_text:
        st.write("### Mateus Nitzsche")
        st.write("""
        **Senior Marketing Data Analyst**  
        Focused on building high-impact data solutions for paid media and business intelligence.
        """)

        st.markdown("""
        - **LinkedIn:** [linkedin.com/in/mateusnit](https://www.linkedin.com/in/mateusnit/)  
        - **GitHub Repository:** [github.com/mmnitzsche/paid-media-insights-dashboard](https://github.com/mmnitzsche/paid-media-insights-dashboard)
        - **Portfolio:** [mateusnitzsche.vercel.app](mateusnitzsche.vercel.app)
        """)

    st.divider()

    st.markdown(
        '<div class="section-title">Project Overview</div>', unsafe_allow_html=True
    )
    st.write("""
    The **Paid Media Performance Dashboard** was designed specifically for **NITIT Courses** to provide a clear, 
    narrative-driven view of advertising results. 
    
    By grouping metrics into specialized sectors (Visibility, Engagement, and Financial Impact), this tool 
    goes beyond just displaying numbers—it tells the story of the customer journey from the first impression 
    to the final return on investment.
    
    **Key Features:**
    - **Narrative Flow:** Optimized layout for decision-making.
    - **Geographic Insights:** Heatmap-based value tracking.
    - **Advanced Data Cleaning:** Automatic standardization of campaign names, dates, and currency.
    - **Integrated Branding:** Custom aesthetic aligned with the NITIT Courses identity.
    """)
