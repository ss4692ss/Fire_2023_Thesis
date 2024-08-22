import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn import metrics 


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


kfold_object = KFold(n_splits=4)


coefficients_list = []


for i, (train_index, test_index) in enumerate(kfold_object.split(data)):
    print("Round:", str(i+1))
    

    data_train, data_test = data[train_index], data[test_index]
    target_train, target_test = target[train_index], target[test_index]


    machine = LogisticRegression()
    machine.fit(data_train, target_train)


    prediction = machine.predict(data_test)
    

    accuracy_score = metrics.accuracy_score(target_test, prediction)
    print("Accuracy score:", accuracy_score)


    confusion_matrix = metrics.confusion_matrix(target_test, prediction)
    print("Confusion matrix:")
    print(confusion_matrix)
    

    error = 1 - accuracy_score
    print("Classification Error:", error)
    

    coefficients_list.append(machine.coef_[0])
    
    print(coefficients_list)


average_coefficients = np.mean(coefficients_list, axis=0)


sum_abs_coefficients = np.sum(np.abs(average_coefficients))
normalized_coefficients = average_coefficients / sum_abs_coefficients


feature_names = dataset.columns[1:]


feature_coefficients = dict(zip(feature_names, normalized_coefficients))


print("Feature Coefficients (Normalized to 1):")
for feature, coefficient in feature_coefficients.items():
    print(f"{feature}: {coefficient}")









    