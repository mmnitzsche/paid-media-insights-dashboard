import pandas as pd
import numpy as np
import re
import os

def parse_money(x):
    """Converts money strings (e.g., '$208.12') to floats."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    # Remove everything except numbers, decimal points, and minus signs
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan

def parse_date_multi(s):
    """Parses dates in multiple formats including YYYY-MM-DD, DD-MM-YYYY, etc."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.to_datetime(s, errors="coerce")

def pct_change(curr, prev):
    """Calculates percentage change between current and previous values."""
    if prev is None or np.isnan(prev) or prev == 0:
        return None
    return (curr - prev) / prev

def clean_campaign_name(name):
    """Fixes common typos in campaign names."""
    if pd.isna(name):
        return name
    s = str(name).strip()
    # Fix typos like "Corse" -> "Course", "Analytcis" -> "Analytics", etc.
    s = s.replace("Corse", "Course")
    s = s.replace("Analytcis", "Analytics")
    s = s.replace("Anlytics", "Analytics")
    return s

def clean_location(loc):
    """Fixes common typos in locations."""
    if pd.isna(loc):
        return loc
    s = str(loc).strip().title()
    # Fix Hyderabad typos
    if any(typo in s for typo in ["Hydrebad", "Hyderbad", "Hyderabadh"]):
        return "Hyderabad"
    return s

def clean_dataframe(df):
    """Performs a complete cleaning and normalization of the dataframe."""
    df = df.copy()

    # Normalize numeric columns
    df["Cost_num"] = df["Cost"].apply(parse_money)
    df["Sale_num"] = df["Sale_Amount"].apply(parse_money)

    for c in ["Clicks", "Impressions", "Leads", "Conversions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize dates
    df["Ad_Date"] = df["Ad_Date"].apply(parse_date_multi)

    # Clean text dimensions
    df["Location"] = df["Location"].apply(clean_location)
    df["Device"] = df["Device"].astype(str).str.strip().str.title()
    df["Campaign_Name"] = df["Campaign_Name"].apply(clean_campaign_name)
    df["Keyword"] = df["Keyword"].astype(str).str.strip()

    # Derived metrics (row-level)
    df["CTR"] = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], np.nan)
    df["CPC"] = np.where(df["Clicks"] > 0, df["Cost_num"] / df["Clicks"], np.nan)
    df["Conversion_Rate_calc"] = np.where(df["Clicks"] > 0, df["Conversions"] / df["Clicks"], np.nan)
    df["CPA"] = np.where(df["Conversions"] > 0, df["Cost_num"] / df["Conversions"], np.nan)
    df["ROAS"] = np.where(df["Cost_num"] > 0, df["Sale_num"] / df["Cost_num"], np.nan)
    df["CPM"] = np.where(df["Impressions"] > 0, (df["Cost_num"] / df["Impressions"]) * 1000, np.nan)

    return df

def load_data(path: str):
    """Loads and cleans data from a CSV file."""
    if not os.path.exists(path):
        return None
    
    df_raw = pd.read_csv(path)
    return clean_dataframe(df_raw)
