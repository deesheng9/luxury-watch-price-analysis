import matplotlib.pyplot as plt 
import seaborn as sns 
import os
import logging
import pandas as pd

def _save_current_plot(output_path):
    """
    Saves the current matplotlib figure and closes it.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Visualization saved successfully to: {output_path}")

def generate_correlation_heatmap(df, output_path='outputs/plots/correlation.png'):
    """
    Generates and saves a correlation heatmap for the numerical features.
    """
    print("Generating visualization: Correlation Heatmap...")
    
    plt.figure(figsize=(10, 8))
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap - Luxury Watches')
    _save_current_plot(output_path)

def generate_price_histogram(df, output_path='outputs/plots/price_histogram.png'):
    """
    Generates and saves a histogram showing the distribution of watch prices.
    """
    print("Generating visualization: Price Histogram...")
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with a density curve (KDE)
    sns.histplot(df['Price (USD)'], bins=30, kde=True, color='skyblue')
    
    plt.title('Distribution of Luxury Watch Prices')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency (Number of Watches)')
    _save_current_plot(output_path)

def generate_average_price_by_brand(df, output_path='outputs/plots/average_price_by_brand.png'):
    """
    Generates and saves a bar chart of average watch price by brand.
    """
    print("Generating visualization: Average Price by Brand...")
    brand_prices = (
        df.groupby('Brand', as_index=False)['Price (USD)']
        .mean()
        .sort_values('Price (USD)', ascending=False)
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(data=brand_prices, x='Price (USD)', y='Brand', hue='Brand', palette='viridis', legend=False)
    plt.title('Average Luxury Watch Price by Brand')
    plt.xlabel('Average Price (USD)')
    plt.ylabel('Brand')
    _save_current_plot(output_path)

def generate_case_diameter_vs_price(df, output_path='outputs/plots/case_diameter_vs_price.png'):
    """
    Generates and saves a scatter plot comparing case diameter and price.
    """
    print("Generating visualization: Case Diameter vs Price...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='Case Diameter (mm)',
        y='Price (USD)',
        hue='Movement Type',
        alpha=0.8
    )
    plt.title('Case Diameter vs Luxury Watch Price')
    plt.xlabel('Case Diameter (mm)')
    plt.ylabel('Price (USD)')
    _save_current_plot(output_path)

def generate_price_by_movement_type(df, output_path='outputs/plots/price_by_movement_type.png'):
    """
    Generates and saves a box plot showing price ranges by movement type.
    """
    print("Generating visualization: Price by Movement Type...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Movement Type', y='Price (USD)', hue='Movement Type', palette='Set2', legend=False)
    plt.title('Luxury Watch Price by Movement Type')
    plt.xlabel('Movement Type')
    plt.ylabel('Price (USD)')
    _save_current_plot(output_path)

def generate_case_material_counts(df, output_path='outputs/plots/case_material_counts.png'):
    """
    Generates and saves a count plot for case materials in the dataset.
    """
    print("Generating visualization: Case Material Counts...")
    material_counts = df['Case Material'].value_counts().reset_index()
    material_counts.columns = ['Case Material', 'Count']

    plt.figure(figsize=(12, 7))
    sns.barplot(data=material_counts, x='Count', y='Case Material', hue='Case Material', palette='mako', legend=False)
    plt.title('Number of Watches by Case Material')
    plt.xlabel('Count')
    plt.ylabel('Case Material')
    _save_current_plot(output_path)

def generate_all_plots(df_cleaned, df_encoded):
    """
    Generates all visualization output files for the analysis pipeline.
    """
    generate_correlation_heatmap(df_encoded)
    generate_price_histogram(df_cleaned)
    generate_average_price_by_brand(df_cleaned)
    generate_case_diameter_vs_price(df_cleaned)
    generate_price_by_movement_type(df_cleaned)
    generate_case_material_counts(df_cleaned)

def generate_feature_importance_plot(model, feature_names, output_path='outputs/plots/feature_importance.png'):
    """
    Generates a bar chart showing the most important features in the Random Forest model.
    """
    logging.info("Generating visualization: Feature Importance...")
    
    # Extract importances from the model
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette='magma', legend=False)
    plt.title('Top 10 Feature Importances for Price Prediction')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    _save_current_plot(output_path)

def generate_actual_vs_predicted_plot(y_true, y_pred, output_path='outputs/plots/actual_vs_predicted.png'):
    """
    Generates a scatter plot comparing actual watch prices against predicted prices.
    """
    logging.info("Generating visualization: Actual vs Predicted Prices...")
    plt.figure(figsize=(10, 6))
    
    # Plot the points
    plt.scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', edgecolor='k')
    
    # Plot the line of perfect prediction
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.title('Actual vs. Predicted Luxury Watch Prices')
    plt.xlabel('Actual Price (USD)')
    plt.ylabel('Predicted Price (USD)')
    plt.legend()
    _save_current_plot(output_path)