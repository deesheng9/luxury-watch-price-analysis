from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(df, target_col='Price (USD)'):
    """
    Splits the data into training and testing sets, then trains a Random Forest Regressor.
    """
    print("Splitting data into 80% train and 20% test sets...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model training completed successfully.")
    return model, X_test, y_test