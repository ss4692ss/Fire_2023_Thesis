import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("fire_ps.csv")
dataset = dataset.sample(frac=1)  # Shuffle dataset

# One-hot encode categorical variable
dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)

# Drop unnecessary columns
dataset.drop(['landcover', 'MeanMinimumTemperature'], axis=1, inplace=True)

# Split data into features and target variable
target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values
feature_names = dataset.columns[1:]

# List of continuous columns
continuous_columns = ['Aspect', 'Droads', 'Moisture', 'NDVI', 'Precipitation', 'Runoff', 'Slope', 'Dset', 'Vapour',
                      'DEM', 'MeanMaximumTemperature', 'WindSpeed', 'Drivers', 'Dlake']

# Standardize continuous features
scaler = StandardScaler()
data[:, :len(continuous_columns)] = scaler.fit_transform(data[:, :len(continuous_columns)])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier
rf_classifier = RandomForestClassifier(criterion="gini", max_depth=4, n_estimators=100, bootstrap=True)
rf_classifier.fit(X_train, y_train)

# Get feature importance for Random Forest
rf_feature_importance = rf_classifier.feature_importances_

# Sort feature importance indices
sorted_indices_rf = np.argsort(rf_feature_importance)[::-1]

# Plot feature importance for Random Forest
plt.figure(figsize=(10, 6))
plt.bar(range(len(rf_feature_importance)), rf_feature_importance[sorted_indices_rf], color='blue', alpha=0.5)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Feature Importance for Random Forest')
plt.xticks(range(len(feature_names)), [f"{feature_names[i]}: {rf_feature_importance[i]:.4f}" for i in sorted_indices_rf], rotation=90)  # Set feature names and importance values as x-axis ticks
plt.show()

# Initialize and train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(criterion="gini", max_depth=5)
dt_classifier.fit(X_train, y_train)

# Get feature importance for Decision Tree
dt_feature_importance = dt_classifier.feature_importances_

# Sort feature importance indices
sorted_indices_dt = np.argsort(dt_feature_importance)[::-1]

# Plot feature importance for Decision Tree
plt.figure(figsize=(10, 6))
plt.bar(range(len(dt_feature_importance)), dt_feature_importance[sorted_indices_dt], color='green', alpha=0.5)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Feature Importance for Decision Tree')
plt.xticks(range(len(feature_names)), [f"{feature_names[i]}: {dt_feature_importance[i]:.4f}" for i in sorted_indices_dt], rotation=90)  # Set feature names and importance values as x-axis ticks
plt.show()

# Initialize and train Logistic Regression Classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

# Get feature importance for Logistic Regression
lr_feature_importance = np.abs(lr_classifier.coef_[0])  # Use absolute values for logistic regression coefficients

# Sort feature importance indices
sorted_indices_lr = np.argsort(lr_feature_importance)[::-1]

# Plot feature importance for Logistic Regression
plt.figure(figsize=(10, 6))
plt.bar(range(len(lr_feature_importance)), lr_feature_importance[sorted_indices_lr], color='red', alpha=0.5)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Feature Importance for Logistic Regression')
plt.xticks(range(len(feature_names)), [f"{feature_names[i]}: {lr_feature_importance[i]:.4f}" for i in sorted_indices_lr], rotation=90)  # Set feature names and importance values as x-axis ticks
plt.show()
