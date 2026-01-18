import pandas as pd
import numpy as np

# Load original uncleaned data
df = pd.read_csv("GoogleAds_DataAnalytics_Sales_Uncleaned.csv")

# 1. Normalize Column "Campaign"
campaign_map = {
    "Data Analytics Corse": "Software Engineer Course",
    "Data Anlytics Corse": "Machine Learning Course",
    "DataAnalyticsCourse": "DevOps Course"
}
# Keep original if not in map, but apply requested changes
df['Campaign_Name'] = df['Campaign_Name'].replace(campaign_map)

# 2. Normalize Column "Device"
# Normalize to lowercase as requested
df['Device'] = df['Device'].astype(str).str.lower()

# 3. Normalize Column "Location"
# Replace specific Hyderabad variations with 3 European cities
# Since there are many variations, we'll map them to [London, Paris, Berlin]
locations_to_replace = ["HYDERABAD", "Hyderbad", "hyderabad", "hydrebad"]
euro_cities = ["London", "Paris", "Berlin"]

def map_location(loc):
    if str(loc).strip() in locations_to_replace:
        # Distribute randomly among the 3 cities for variety
        return np.random.choice(euro_cities)
    return loc

df['Location'] = df['Location'].apply(map_location)

# Export to a new cleaned CSV
CLEANED_FILE = "GoogleAds_Performance_Standardized.csv"
df.to_csv(CLEANED_FILE, index=False)

print(f"Data exported successfully to {CLEANED_FILE}")