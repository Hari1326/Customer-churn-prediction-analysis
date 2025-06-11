import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n = 100

# Generate data
df = pd.DataFrame({
    'gender': np.random.choice(['Male', 'Female'], n),
    'SeniorCitizen': np.random.choice([0, 1], n),
    'Partner': np.random.choice(['Yes', 'No'], n),
    'Dependents': np.random.choice(['Yes', 'No'], n),
    'tenure': np.random.randint(0, 72, n),
    'PhoneService': np.random.choice(['Yes', 'No'], n),
    'MonthlyCharges': np.round(np.random.uniform(20, 120, n), 2),
    'Churn': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
})

df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']

# Encode categorical columns
le = LabelEncoder()
for col in ['gender', 'Partner', 'Dependents', 'PhoneService']:
    df[col] = le.fit_transform(df[col])
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Train/test split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))