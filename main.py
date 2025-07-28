# main.py

from src.preprocessing import load_data, preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model

def main():
    print("📥 Step 1: Loading Data...")
    df = load_data('data/diabetes.csv')

    print("🧹 Step 2: Preprocessing Data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("🧠 Step 3: Training Model...")
    model = train_model(X_train, y_train)

    print("📈 Step 4: Evaluating Model...")
    evaluate_model(model, X_test, y_test)

    print("✅ Pipeline Completed!")

if __name__ == "__main__":
    main()
