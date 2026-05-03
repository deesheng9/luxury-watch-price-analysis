import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    """
    Encodes categorical features (text strings) into numerical values.
    """
    print("Starting feature engineering (label encoding)...")
    data = df.copy()
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col].astype(str))
        
    print(f"Encoding complete. Processed columns: {list(categorical_cols)}")
    return data