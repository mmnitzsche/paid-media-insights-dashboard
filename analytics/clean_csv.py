import pandas as pd
from data_processing import clean_dataframe

def main():
    input_file = "GoogleAds_DataAnalytics_Sales_Uncleaned.csv"
    output_file = "GoogleAds_DataAnalytics_Sales_Cleaned.csv"

    print(f"Reading from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print("Cleaning data...")
        df_cleaned = clean_dataframe(df)
        
        print(f"Saving to {output_file}...")
        df_cleaned.to_csv(output_file, index=False)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
