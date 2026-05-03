import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualization saved successfully to: {output_path}")