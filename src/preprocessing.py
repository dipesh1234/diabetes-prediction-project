# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.copy()
    
    # Replace zeroes with NaN for certain columns
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, pd.NA)
    
    # Fill NaNs with median values
    df.fillna(df.median(), inplace=True)

    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
