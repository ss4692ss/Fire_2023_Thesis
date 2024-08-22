import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


dataset = pd.read_csv("Fire_Social.csv")


dataset = dataset.sample(frac=1)

# # Drop the column 'IMR_ProjectRaster'
# dataset.drop('IMR_ProjectRaster', axis=1, inplace=True)


target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)


oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)


machine = RandomForestClassifier(criterion="gini", max_depth=4, n_estimators=100, bootstrap=True, random_state=42)
machine.fit(X_train_resampled, y_train_resampled)


predictions = machine.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)


conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)


print("Classification Report:")
print(classification_report(y_test, predictions))


error = 1 - accuracy
print("Classification Error:", error)


feature_names = dataset.columns[1:]


feature_importances = machine.feature_importances_


normalized_importances = feature_importances / feature_importances.sum()


print("Normalized Feature Importances:")
for feature, importance in zip(feature_names, normalized_importances):
    print(f"{feature}: {importance}")





