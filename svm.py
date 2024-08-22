import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


dataset = pd.read_csv("fire_ps.csv")


dataset = dataset.sample(frac=1)


dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)
dataset.drop(['landcover'], axis=1, inplace=True)


target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values


continuous_columns = ['Aspect', 'DEM', 'MeanMaximumTemperature', 
                      'Moisture', 'NDVI', 'Precipitation', 'Runoff', 'Slope', 
                      'WindSpeed', 'Drivers', 'Dset','Vapour', 'Droads', 'Dlake','MeanMinimumTemperature']


scaler = StandardScaler()
data[:, :len(continuous_columns)] = scaler.fit_transform(data[:, :len(continuous_columns)])


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=100)


svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_error = 1 - accuracy


print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Error:", classification_error)


print("Classification Report:")
print(classification_report(y_test, y_pred))


coefficients = svm.coef_


absolute_coefficients = np.abs(coefficients)


feature_importance_dict = dict(zip(dataset.columns[1:], absolute_coefficients[0]))


sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)


print("Feature Importance Scores:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")



