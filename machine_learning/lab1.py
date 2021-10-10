import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

HouseDF = pd.read_csv (r'D:\py\test\machine_learning\Datasets\USA_Housing.csv')

print(HouseDF.head())
print("-----------------------------------------")
print(HouseDF.info())
print("-----------------------------------------")
print(HouseDF.describe())
print("-----------------------------------------")


print(sns.distplot(HouseDF['Price']))

sns.heatmap(HouseDF.corr(), annot=True)

X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = HouseDF['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) coeff_df

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

sns.distplot((y_test-predictions),bins=50);

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))