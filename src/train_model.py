# src/train_model.py
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X_train, y_train, save_path='models/logreg_model.pkl'):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model
