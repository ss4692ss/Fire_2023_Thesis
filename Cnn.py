import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


dataset = pd.read_csv("fire_ps.csv")


dataset = dataset.sample(frac=1)


dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)
dataset.drop(['landcover','MeanMinimumTemperature'], axis=1, inplace=True)


target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:].values


continuous_columns = ['Aspect', 'DEM', 'MeanMaximumTemperature', 
                      'Moisture', 'NDVI', 'Precipitation', 'Runoff', 'Slope', 
                      'WindSpeed', 'Drivers', 'Dset','Vapour', 'Droads', 'Dlake']




scaler = StandardScaler()
data[:, :len(continuous_columns)] = scaler.fit_transform(data[:, :len(continuous_columns)])


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


X_train = X_train.reshape(-1, 1, X_train.shape[1], 1).astype(np.float32)
X_test = X_test.reshape(-1, 1, X_test.shape[1], 1).astype(np.float32)


def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (1, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((1, 2)),
        layers.Conv2D(64, (1, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') 
    ])
    return model


model = create_model(input_shape=(1, X_train.shape[2], 1))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=30, batch_size=32)  


test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


predictions = model.predict(X_test)


threshold = 0.5
predictions = np.where(predictions > threshold, 1, 0)


accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)


conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)





