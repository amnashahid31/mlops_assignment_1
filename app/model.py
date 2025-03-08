import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor
import joblib
import warnings

warnings.filterwarnings('ignore')

def preprocess_data(df):
    # Remove unnecessary columns
    to_remove = ['world_revenue', 'opening_revenue']
    df = df.drop(to_remove, axis=1)
    
    # Handle null values
    df = df.drop('budget', axis=1)
    for col in ['MPAA', 'genres']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna()
    
    # Clean revenue and numeric columns
    df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:]
    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        df[col] = df[col].astype(str).str.replace(',', '')
        temp = (~df[col].isnull())
        df[temp][col] = df[temp][col].convert_dtypes(float)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Log transform numeric features
    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        df[col] = df[col].apply(lambda x: np.log10(x))
    
    # Process genres using CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(df['genres'])
    features = vectorizer.transform(df['genres']).toarray()
    genres = vectorizer.get_feature_names_out()
    for i, name in enumerate(genres):
        df[name] = features[:, i]
    df = df.drop('genres', axis=1)
    
    # Remove sparse genre columns
    if 'action' in df.columns and 'western' in df.columns:
        for col in df.loc[:, 'action':'western'].columns:
            if (df[col] == 0).mean() > 0.95:
                df = df.drop(col, axis=1)
    
    # Encode categorical variables
    for col in ['distributor', 'MPAA']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df

def prepare_features(df):
    features = df.drop(['title', 'domestic_revenue'], axis=1)
    target = df['domestic_revenue'].values
    return features, target

def train_model():
    # Load data
    df = pd.read_csv('data/boxoffice.csv', encoding='latin-1')
    # Preprocess data
    df_processed = preprocess_data(df)
    # Prepare features and target
    features, target = prepare_features(df_processed)
    # Split data
    X_train, X_val, Y_train, Y_val = train_test_split(
        features, target,
        test_size=0.1,
        random_state=22
    )
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    # Train model
    model = XGBRegressor()
    model.fit(X_train_scaled, Y_train)
    # Save model and scaler
    joblib.dump(model, "model/box_office_revenue_prediction_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    
    return model, scaler, X_val_scaled, Y_val

if __name__ == "__main__":
    model, scaler, X_val, Y_val = train_model()
