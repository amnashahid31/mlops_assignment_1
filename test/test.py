import joblib
import pandas as pd
from model import preprocess_data, prepare_features
from sklearn.metrics import mean_absolute_error as mae


def test_model(test_data_path):
    # Load the saved model and scaler
    model = joblib.load("model/box_office_revenue_prediction_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    # Load and preprocess test data
    test_df = pd.read_csv(test_data_path, encoding='latin-1')
    test_df_processed = preprocess_data(test_df)
    
    # Prepare features
    test_features, test_target = prepare_features(test_df_processed)
    
    # Scale features
    test_features_scaled = scaler.transform(test_features)
    
    # Make predictions
    predictions = model.predict(test_features_scaled)
    
    # Calculate error
    error = mae(test_target, predictions)
    print(f"Test Mean Absolute Error: {error}")
    
    return predictions


if __name__ == "__main__":
    test_data_path = 'model/boxoffice.csv'
    predictions = test_model(test_data_path)
