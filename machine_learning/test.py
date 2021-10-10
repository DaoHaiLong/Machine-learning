import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv(r'D:\py\test\machine_learning\Datasets\student_scores.csv')
dataset1 = pd.read_csv (r'D:\py\test\machine_learning\Datasets\kc_house_data.csv')


print("--------------------Dataset.shape----------------------------------")
print(dataset.shape)

print("--------------------Dataset.head-----------------------------")
print(dataset.head())

print("--------------------Dataset.describe----------------------")
print(dataset.describe())


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

print("-------------------dataset.iloc(X)------------------")
print(X)

print("-------------------dataset.iloc(Y)--------------------")
print(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("-------------------regressor.intercept---------------------")
print(regressor.intercept_)

print("-------------------regressor.coef---------------------")
print(regressor.coef_)

y_pred = regressor.predict(X_test)

print("-------------------y_pred---------------------")
print(y_pred)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("-------------------df---------------------")
print(df)


print("----------------------------------------")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))