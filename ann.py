import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix


dataset = pd.read_csv("fire_ps.csv")


dataset = dataset.sample(frac=1)


dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)
dataset.drop(['landcover', 'MeanMinimumTemperature', 'Droads'], axis=1, inplace=True)


target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values


continuous_columns = ['Aspect', 'DEM', 'MeanMaximumTemperature', 
                      'Moisture', 'NDVI', 'Precipitation', 'Runoff', 'Slope', 
                      'WindSpeed', 'Drivers', 'Dset','Vapour', 'Dlake']



scaler = StandardScaler()
data[:, :len(continuous_columns)] = scaler.fit_transform(data[:, :len(continuous_columns)])


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(np.unique(target)), activation='softmax')  # Output layer with softmax activation for classification
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=64, validation_split=0.1)


loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Accuracy on Test Data:", accuracy)


probabilities = model.predict(X_test_scaled)
predictions = np.argmax(probabilities, axis=1)


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)


classification_error = 1 - accuracy
print("Classification Error:", classification_error)





