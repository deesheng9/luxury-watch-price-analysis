import logging
from sklearn.metrics import mean_squared_error, r2_score
from visualization.plotter import generate_actual_vs_predicted_plot

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using standard metrics and generates an evaluation plot.
    """
    logging.info("Evaluating model performance on test data...")
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    logging.info("\n=======================================")
    logging.info("        MODEL EVALUATION RESULTS       ")
    logging.info("=======================================")
    logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
    logging.info(f"R-squared (R2 Score):     {r2:.4f}")
    logging.info("=======================================\n")
    
    # Trigger the new evaluation plot
    generate_actual_vs_predicted_plot(y_test, predictions)