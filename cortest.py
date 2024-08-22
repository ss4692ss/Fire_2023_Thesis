import pandas as pd
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("fire_points.csv")


dataset = dataset.sample(frac=1, random_state=42)


dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)
dataset.drop(['landcover'], axis=1, inplace=True)


target = dataset.iloc[:, 0]
data = dataset.iloc[:, 1:]


continuous_columns = ['Aspect', 'Droads', 'DEM', 'Mean_Maximum_Temperature', 'Mean_Minimum_Temperature',
                      'Moisture', 'NDVI', 'Precipitation', 'Runoff', 'Slope', 'Vapour', 
                      'Wind_Speed', 'Drivers', 'Dlake', 'Dset']


scaler = StandardScaler()
data.loc[:, continuous_columns] = scaler.fit_transform(data.loc[:, continuous_columns])


data_with_target = pd.concat([target, data], axis=1)


corr_matrix = data_with_target.corr()


target_corr = corr_matrix.iloc[:, 0]


print("Correlation coefficients with the target variable:")
print(target_corr)