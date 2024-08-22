import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
machine = RandomForestClassifier(criterion="gini", max_depth=4, n_estimators=100, bootstrap=True)
machine.fit(X_train, y_train)


predictions = machine.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)
error = 1 - accuracy
print("Classification Error:", error)
feature_names = dataset.columns[1:]
feature_importances = pd.DataFrame({"Feature": feature_names, "Importance": machine.feature_importances_})
feature_importances.sort_values(by="Importance", ascending=False, inplace=True)
print("Feature Importances:")
print(feature_importances)



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
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend(loc="lower right")
plt.show()






