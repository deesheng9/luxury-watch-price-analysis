import pandas as pd
from preprocessing.cleaner import clean_data

def main():
    print("--- Luxury Watch Price Analysis System ---")
    
    # Define the data file path
    file_path = 'data/Luxury watch.csv'
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Original row count: {len(df)}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}. Please check the directory.")
        return

    # Process and clean the data
    print("Processing data...")
    df_cleaned = clean_data(df)
    
    # Display the result preview
    print("\n--- Cleaned Data Preview (Top 5 rows) ---")
    print(df_cleaned.head())
    
    # Export the processed data for modeling
    output_path = 'data/cleaned_watch_data.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")

if __name__ == "__main__":
    main()