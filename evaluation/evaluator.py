from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using standard metrics (MSE and R-squared).
    """
    print("Evaluating model performance on test data...")
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n========================================")
    print("        MODEL EVALUATION RESULTS        ")
    print("========================================")
    print(f"- Mean Squared Error (MSE): {mse:.2f}")
    print(f"- R-squared (R2 Score):     {r2:.4f}")
    print("========================================")