import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


dataset = pd.read_csv("fire_points.csv")
dataset = dataset.sample(frac=1)

dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)


dataset.drop(['landcover', 'MeanMinimumTemperature'], axis=1, inplace=True)


target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values


continuous_columns = ['Aspect', 'Droads', 'Moisture', 'NDVI', 'Precipitation', 'Runoff', 
                      'Slope', 'Dset', 'Vapour', 'DEM', 'MeanMaximumTemperature', 
                      'WindSpeed', 'Drivers', 'Dlake']


scaler = StandardScaler()
data[:, :len(continuous_columns)] = scaler.fit_transform(data[:, :len(continuous_columns)])


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


machine = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                        alpha=0.0001, batch_size='auto', learning_rate='constant', 
                        learning_rate_init=0.001, max_iter=200, shuffle=True, random_state=42)
machine.fit(X_train, y_train)

predictions = machine.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
result = permutation_importance(machine, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)


feature_importances = result.importances_mean
feature_names = dataset.columns[1:]
sorted_idx = feature_importances.argsort()


plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in MLP using Permutation Importance')
plt.show()
# conf_matrix = confusion_matrix(y_test, predictions)
# print("Confusion Matrix:")
# print(conf_matrix)


# error = 1 - accuracy
# print("Classification Error:", error)


# auc_score = roc_auc_score(y_test, machine.predict_proba(X_test)[:, 1])
# print("AUC Score:", auc_score)


# fpr, tpr, thresholds = roc_curve(y_test, machine.predict_proba(X_test)[:, 1])
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
# plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve for Multilayer Perceptron')
# plt.legend(loc="lower right")
# plt.show()
