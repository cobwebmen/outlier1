import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()

# Create a DataFrame
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Generate descriptive statistics
print("Descriptive Statistics:")
print(df.describe())

# Create histograms for each feature
print("\nHistograms for each feature:")
df.hist(figsize=(10, 8))
plt.show()

# Create scatter plots for each feature
print("\nScatter plots for each feature:")
for column in df.columns[:-1]:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[column], df['target'])
    plt.title(f"Scatter plot of {column} vs target")
    plt.xlabel(column)
    plt.ylabel('target')
    plt.show()

# Create heatmaps for each feature
print("\nHeatmaps for each feature:")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', square=True)
plt.title("Heatmap of correlation between features")
plt.show()

# Fill null values with mean values (if any)
print("\nNull values count before filling:")
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)
print("\nNull values count after filling:")
print(df.isnull().sum())

# Perform feature scaling using MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('target', axis=1)), columns=df.columns[:-1])
df_scaled['target'] = df['target']

# Convert the target variable into binary (0 and 1) for logistic regression
df_scaled['target'] = np.where(df_scaled['target'] > df_scaled['target'].mean(), 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_scaled.drop('target', axis=1), df_scaled['target'], test_size=0.2, random_state=42)

# Train and model the data with Logistic Regression algorithm
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
