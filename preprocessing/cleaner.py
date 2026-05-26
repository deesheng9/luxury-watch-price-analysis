import pandas as pd
import logging

def clean_data(df):
    """
    Cleans the luxury watch dataset by converting prices to numeric
    and removing rows with missing values.
    """
    logging.info("Starting data cleaning process...")
    data = df.copy()
    
    # Clean the 'Price (USD)' column: remove '$' and ',' then convert to float
    if 'Price (USD)' in data.columns:
        data['Price (USD)'] = data['Price (USD)'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    
    # Drop rows with missing values to ensure model stability
    data = data.dropna()
    
    logging.info(f"Cleaning complete. Remaining records: {len(data)}")
    return data