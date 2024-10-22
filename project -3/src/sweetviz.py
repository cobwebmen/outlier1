# Import necessary libraries
from sklearn.datasets import load_diabetes
import pandas as pd
import sweetviz as sv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# 1. Load the diabetes dataset from scikit-learn
diabetes = load_diabetes()

# Convert to Pandas DataFrame for easy handling
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target  # Add the target variable

# 2. Perform Sweetviz analysis for descriptive statistics, histograms, scatter plots, heatmaps
report = sv.analyze(df)
report.show_html("diabetes_sweetviz_report.html")

# 3. Check for any null values and fill them with mean values
df.fillna(df.mean(), inplace=True)

# 4. Feature scaling using MinMaxScaler
scaler = MinMaxScaler()
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target

X_scaled = scaler.fit_transform(X)

# 5. Train and model the data using Logistic Regression

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Instantiate the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
