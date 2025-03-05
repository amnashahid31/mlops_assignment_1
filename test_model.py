# import unittest
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler

# class TestBoxOfficeModel(unittest.TestCase):
#     def setUp(self):
#         """Set up test fixtures before each test method"""
#         # Create sample test data
#         self.X_test = pd.DataFrame({
#             'budget': [1000000, 2000000, 3000000],
#             'popularity': [10, 20, 30],
#             'runtime': [90, 120, 150],
#             'vote_average': [7.5, 8.0, 6.5],
#             'vote_count': [1000, 2000, 1500]
#         })
#         self.y_test = np.array([2000000, 4000000, 6000000])  # Sample revenues
        
#         # Initialize model and scaler
#         self.model = LinearRegression()
#         self.scaler = StandardScaler()
        
#     def test_model_initialization(self):
#         """Test if model is properly initialized"""
#         self.assertIsInstance(self.model, LinearRegression)
        
#     def test_input_shape(self):
#         """Test if input data has correct shape and features"""
#         expected_features = ['budget', 'popularity', 'runtime', 
#                            'vote_average', 'vote_count']
#         self.assertEqual(list(self.X_test.columns), expected_features)
#         self.assertEqual(len(self.X_test.columns), 5)
        
#     def test_data_preprocessing(self):
#         """Test if data preprocessing works correctly"""
#         # Test scaling
#         X_scaled = self.scaler.fit_transform(self.X_test)
        
#         # Check if scaled data has mean close to 0 and std close to 1
#         self.assertTrue(np.abs(X_scaled.mean()) < 0.1)
#         self.assertTrue(np.abs(X_scaled.std() - 1) < 0.1)
        
#     def test_predictions_shape(self):
#         """Test if model predictions have correct shape"""
#         # Fit model with sample data
#         X_scaled = self.scaler.fit_transform(self.X_test)
#         self.model.fit(X_scaled, self.y_test)
        
#         # Make predictions
#         predictions = self.model.predict(X_scaled)
        
#         # Check prediction shape
#         self.assertEqual(len(predictions), len(self.X_test))
        
#     def test_predictions_range(self):
#         """Test if predictions are within reasonable range"""
#         # Fit model
#         X_scaled = self.scaler.fit_transform(self.X_test)
#         self.model.fit(X_scaled, self.y_test)
        
#         # Make predictions
#         predictions = self.model.predict(X_scaled)
        
#         # Check if predictions are positive (revenue can't be negative)
#         self.assertTrue(np.all(predictions >= 0))
        
#     def test_model_coefficients(self):
#         """Test if model coefficients are present and reasonable"""
#         # Fit model
#         X_scaled = self.scaler.fit_transform(self.X_test)
#         self.model.fit(X_scaled, self.y_test)
        
#         # Check if coefficients exist
#         self.assertEqual(len(self.model.coef_), len(self.X_test.columns))
        
#     def test_model_score(self):
#         """Test if model score is within reasonable range"""
#         # Fit model
#         X_scaled = self.scaler.fit_transform(self.X_test)
#         self.model.fit(X_scaled, self.y_test)
        
#         # Get model score (R²)
#         score = self.model.score(X_scaled, self.y_test)
        
#         # Check if score is between 0 and 1
#         self.assertTrue(0 <= score <= 1)

# if __name__ == '__main__':
#     unittest.main()