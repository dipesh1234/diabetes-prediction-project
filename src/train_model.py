from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, '../models/logreg_model.pkl')
