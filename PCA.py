
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



dataset = pd.read_csv("fire_points.csv")


dataset = dataset.sample(frac=1)


dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)
dataset.drop(['landcover', 'MeanMinimumTemperature'], axis=1, inplace=True)


target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values


continuous_columns = ['Aspect', 'DEM', 'MeanMaximumTemperature',
                      'Moisture', 'NDVI', 'Precipitation', 'Runoff', 'Slope', 'Vapour', 
                      'WindSpeed', 'Drivers','Droads', 'Dlake', 'Dset']


scaler = StandardScaler()
data[:, :len(continuous_columns)] = scaler.fit_transform(data[:, :len(continuous_columns)])


pca = PCA(n_components=4)  
data_pca = pca.fit_transform(data)


explained_variance_ratio = pca.explained_variance_ratio_
components = pca.components_



num_top_features = 5
top_features = {}
for i, component in enumerate(components):
    top_feature_indices = component.argsort()[-num_top_features:][::-1]
    top_feature_names = [str(dataset.columns[1:][index]) for index in top_feature_indices]
    top_features[f'Principal Component {i+1}'] = top_feature_names


for component, features in top_features.items():
    print(f"{component}: {', '.join(features)}")



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Prepare data with top principal components
top_components_data = data_pca[:, :4]  # Select top 4 principal components

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(top_components_data, target, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

