import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Sample dataset (replace with heart.csv if available)
data = {
    'age': [52, 53, 70, 61, 62, 58, 45, 63, 39, 54],
    'sex': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'cp': [0, 0, 0, 0, 0, 1, 2, 1, 2, 0],
    'trestbps': [125, 140, 145, 148, 138, 130, 120, 150, 110, 135],
    'chol': [212, 203, 174, 203, 294, 240, 180, 260, 190, 220],
    'target': [0, 0, 0, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))