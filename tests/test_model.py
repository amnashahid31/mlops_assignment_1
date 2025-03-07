import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd

class TestBoxOfficeRevenueModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load dataset and train model once for all tests."""
        df = pd.read_csv('boxoffice.csv', encoding='latin-1')
        df.drop(['world_revenue', 'opening_revenue'], axis=1, inplace=True)
        df.drop('budget', axis=1, inplace=True)
        for col in ['MPAA', 'genres']:
            df[col] = df[col].fillna(df[col].mode()[0])
        df.dropna(inplace=True)
        df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:].str.replace(',', '').astype(float)
        features = df.drop(['title', 'domestic_revenue'], axis=1)
        target = df['domestic_revenue'].values
        X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(X_train)
        cls.X_val = scaler.transform(X_val)
        cls.Y_train = Y_train
        cls.Y_val = Y_val
        cls.model = XGBRegressor()
        cls.model.fit(cls.X_train, cls.Y_train)
    
    def test_model_training(self):
        """Check if model is trained without errors."""
        self.assertIsNotNone(self.model)
    
    def test_model_prediction_shape(self):
        """Ensure predictions have correct shape."""
        preds = self.model.predict(self.X_val)
        self.assertEqual(preds.shape, self.Y_val.shape)
    
    def test_model_performance(self):
        """Check if model error is within an acceptable range."""
        preds = self.model.predict(self.X_val)
        error = mae(self.Y_val, preds)
        self.assertLess(error, 10000000, "Model error is too high!")

if __name__ == '__main__':
    unittest.main()
