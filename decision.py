import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn import tree
import matplotlib.pyplot as plt
import kfold_template


dataset = pd.read_csv("fire_points.csv")


dataset = dataset.sample(frac=1)


dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)
dataset.drop(['landcover', 'Mean_Minimum_Temperature'], axis=1, inplace=True)
target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values
continuous_columns = ['Aspect', 'Droads',
                      'Moisture', 'NDVI', 'Precipitation', 'Runoff', 'Slope', 'Dset', 'Vapour','DEM', 'Mean_Maximum_Temperature',
                      'Wind_Speed', 'Drivers', 'Dlake']

scaler = StandardScaler()
data[:, :len(continuous_columns)] = scaler.fit_transform(data[:, :len(continuous_columns)])

machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
return_values = kfold_template.run_kfold(machine, data, target, 4, False)
print(return_values)

machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
machine.fit(data, target)


predictions = machine.predict(data)


accuracy = accuracy_score(target, predictions)
print("Accuracy:", accuracy)


conf_matrix = confusion_matrix(target, predictions)
print("Confusion Matrix:")
print(conf_matrix)


classification_error = 1 - accuracy
print("Classification Error:", classification_error)


feature_importance = machine.feature_importances_

feature_importance_dict = dict(zip(dataset.columns[1:], feature_importance))


sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)


print("Feature Importance Scores:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


auc_score = roc_auc_score(y_test, machine.predict_proba(X_test)[:, 1])
print("AUC Score:", auc_score)


fpr, tpr, thresholds = roc_curve(y_test, machine.predict_proba(X_test)[:, 1])


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Decision Tree')
plt.legend(loc="lower right")
plt.show()

