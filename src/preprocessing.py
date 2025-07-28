# Handling missing values with mean/median
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(df):
    df = df.copy()
    df[[ 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[[ 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, pd.NA)
    df.fillna(df.median(), inplace=True)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
