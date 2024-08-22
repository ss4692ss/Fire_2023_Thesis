import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression



dataset = pd.read_csv("fire_points.csv")


dataset = dataset.sample(frac=1)


dummy_variables = pd.get_dummies(dataset['landcover'], prefix='landcover')
dataset = pd.concat([dataset, dummy_variables], axis=1)
dataset.drop(['landcover'], axis=1, inplace=True)

target = dataset.iloc[:, 0]
data = dataset.iloc[:, 1:]


continuous_columns = ['Aspect', 'Droads', 'DEM', 'Max_Temp', 'Min_Temp',
                      'Moisture', 'NDVI', 'Precipitation', 'Runoff', 'Slope', 'Vapour', 
                      'WindSpeed', 'Drivers', 'Dlake', 'Dset']


scaler = StandardScaler()
data.loc[:, continuous_columns] = scaler.fit_transform(data.loc[:, continuous_columns])


correlation_matrix = data.corrwith(target)

print("Correlation coefficients:")
print(correlation_matrix)


matrix = data.corr()

print("\nCorrelation Matrix:")
print(matrix)


plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap='YlGnBu')
plt.title("Correlation Matrix", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.35)
plt.savefig("correlation_matrix.png")
plt.show()


selector = SelectKBest(score_func=f_regression, k=10)
selected_data = selector.fit_transform(data, target)


selected_indices = selector.get_support(indices=True)


selected_features = data.columns[selected_indices]

print("Selected Features:")
print(selected_features)
