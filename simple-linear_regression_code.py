# importing numpy ,pandas and matplotlib librabries from python 
# pandas for importing datasets ,matplotlib for ploting points on a scatterplot 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# impoting dataset x will take all the datas(all rows and all columns) except the last one , which is going to be predicted that will be in y

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# dividing the dataset into training and test set where training data is more then testing data so that we can reduce the error but we should not overfit the data set which will make my model work for only a specific dataset not on other dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# training Simple Linear Regression model on training set to minimize the error and check my model efficiency 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting a values from test case to check the difference between my model output value and given value in dataset 

y_pred = regressor.predict(X_test)

# ploting the result of training set on a scatterplot

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# ploting the result of test set on a scatterplot

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# predicting a random value

print(regressor.predict([[12]]))

# getting the coefficient

print(regressor.coef_)

# getting the intercept

print(regressor.intercept_)


# both the intercept and coefficient will together give the salary of a person
# Salary=coef×YearsExperience+intercept
# I m sandeep i tried to explain each and every line of code ,i hope you have understood it properly if any doubt just let me known in the comment!!!  
