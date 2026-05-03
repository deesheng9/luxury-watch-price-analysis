import pandas as pd
from preprocessing.cleaner import clean_data
from preprocessing.encoder import encode_features
from models.ml_models import train_model
from evaluation.evaluator import evaluate_model

def main():
    print("--- Luxury Watch Price Analysis System ---")
    
    # Phase 1: Data Loading
    file_path = 'data/Luxury watch.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}.")
        return

    # Phase 2: Preprocessing and Cleaning
    print("\n[Phase 2] Data Cleaning")
    df_cleaned = clean_data(df)
    
    # Phase 3: Feature Engineering
    print("\n[Phase 3] Feature Engineering")
    df_encoded = encode_features(df_cleaned)
    
    # Phase 4: Model Training
    print("\n[Phase 4] Model Training")
    model, X_test, y_test = train_model(df_encoded)
    
    # Phase 5: Evaluation
    print("\n[Phase 5] Model Evaluation")
    evaluate_model(model, X_test, y_test)
    
    print("\n--- Pipeline Execution Completed Successfully ---")

if __name__ == "__main__":
    main()