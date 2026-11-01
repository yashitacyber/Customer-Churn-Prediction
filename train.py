import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Preprocessing
# TotalCharges mein kuch blank spaces ho sakte hain, unhe handle karein
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# customerID ki zaroorat nahi hai model ke liye
df.drop('customerID', axis=1, inplace=True)

# Sabhi categorical columns ko label encode karna
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# 3. Features (X) aur Target (y) alag karein
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Train karein
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy check
accuracy = model.score(X_test, y_test)
print(f"Model Training Complete! Test Accuracy: {accuracy * 100:.2f}%")

# 5. Model ko save karein
with open('model.pkl', 'wb') as f:
    pickle.dump((model, X.columns), f)

print("Model saved successfully as model.pkl!")