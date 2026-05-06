import pandas as pd 
import logging
from utils.helpers import setup_environment
from preprocessing.cleaner import clean_data
from preprocessing.encoder import encode_features
from models.ml_models import train_model
from evaluation.evaluator import evaluate_model
from visualization.plotter import generate_all_plots, generate_feature_importance_plot

def main():
    # Phase 0: Utility Setup (This starts the logger)
    setup_environment()
    logging.info("--- Luxury Watch Price Analysis System Started ---")
    
    # Phase 1: Data Loading
    file_path = 'data/Luxury watch.csv'
    try:
        logging.info(f"[Phase 1] Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"CRITICAL: CSV file not found at {file_path}. Pipeline aborted.")
        return

    # Phase 2: Preprocessing and Cleaning
    logging.info("[Phase 2] Executing Data Cleaning...")
    df_cleaned = clean_data(df)
    
    # Phase 3: Feature Engineering
    logging.info("[Phase 3] Executing Feature Engineering...")
    df_encoded = encode_features(df_cleaned)
    
    # Phase 4: EDA Visualization
    logging.info("[Phase 4] Generating Exploratory Data Analysis (EDA) Visualizations...")
    generate_all_plots(df_cleaned, df_encoded)
    
    # Phase 5: Model Training
    logging.info("[Phase 5] Initializing Model Training...")
    model, X_test, y_test = train_model(df_encoded)
    
    # Generate the new Feature Importance plot!
    feature_names = df_encoded.drop(columns=['Price (USD)']).columns
    generate_feature_importance_plot(model, feature_names)
    
    # Phase 6: Evaluation and Validation
    logging.info("[Phase 6] Starting Model Evaluation...")
    evaluate_model(model, X_test, y_test)
    
    logging.info("--- Pipeline Execution Completed Successfully ---")

if __name__ == "__main__":
    main()