
# 8. Complete Python Code


# Import Libr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load Dataset

df = pd.read_csv('Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')

print(df.head())
print(df.shape)
print(df.info())


#Remove Unnecessary Columns

unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
df.drop(columns=unnamed_cols, inplace=True)

print(df.columns)


# Check Missing Values

print(df.isnull().sum())


# Data Cleaning

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values if any
df.fillna(method='ffill', inplace=True)


# Basic Statistics

print(df.describe())





# 9. Exploratory Data Analysis (EDA)

# Screen Time Distribution

plt.figure(figsize=(10,5))
sns.histplot(df['daily_screen_time_hours'], bins=30, kde=True)
plt.title('Daily Screen Time Distribution')
plt.show()


# Gender-wise Screen Time

plt.figure(figsize=(8,5))
sns.boxplot(x='gender', y='daily_screen_time_hours', data=df)
plt.title('Gender vs Screen Time')
plt.show()


# Addiction Level Count

plt.figure(figsize=(8,5))
sns.countplot(x='addiction_level', data=df)
plt.title('Addiction Level Distribution')
plt.show()


# Correlation Heatmap

plt.figure(figsize=(12,8))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Sleep vs Screen Time

plt.figure(figsize=(8,5))
sns.scatterplot(x='daily_screen_time_hours', y='sleep_hours', data=df)
plt.title('Screen Time vs Sleep Hours')
plt.show()





# 10. Feature Engineering


# Encode Categorical Variables

encoder = LabelEncoder()

categorical_cols = ['gender', 'academic_work_impact', 'addicted_label']

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])






# 11. Machine Learning Model


# Define Features & Target

X = df.drop(['addiction_level'], axis=1)
y = df['addiction_level']


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Random Forest Model

model = RandomForestClassifier()
model.fit(X_train, y_train)


# Predictions

y_pred = model.predict(X_test)


# Accuracy Score

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# Classification Report

print(classification_report(y_test, y_pred))