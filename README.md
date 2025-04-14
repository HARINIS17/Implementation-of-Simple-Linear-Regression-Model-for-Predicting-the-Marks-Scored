# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import Libraries – Load necessary Python libraries.

2.Load Dataset – Read the dataset containing study hours and marks.

3.Preprocess Data – Check for missing values and clean the data if needed.

4.Split Data – Divide the dataset into training and testing sets.

5.Train Model – Fit a Simple Linear Regression model to the training data.

6.Make Predictions – Use the trained model to predict marks on the test data.

7.Evaluate Model – Calculate Mean Absolute Error (MAE) and R² score.

8.Visualize Results – Plot the regression line along with actual data points

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: HARINI S

RegisterNumber: 212224230083

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error,r2_score

data=pd.read_csv("/content/student_scores.csv")

print("Dataset Preview:\n",data.head())

print("\nMissing Values:\n",data.isnull().sum())

x=data[['Hours']]

y=data['Scores']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("\nIntercept:",model.intercept_)

print("Slope:",model.coef_[0])

mae=mean_absolute_error(y_test,y_pred)

r2=r2_score(y_test,y_pred)

print("\nMean Absolute Error:",mae)

print("R^2 Score:",r2)

X = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1, 1)

y = np.array([1,3,5,7,9,2,4,6,8,0])

model = LinearRegression()

model.fit(X, y)

m = model.coef_[0]

b = model.intercept_

print(f"Slope (m): {m}")

print(f"Y-Intercept (b): {b}")

y_pred = model.predict(X)

print("Predicted y values:", y_pred)

plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel("X-axis")

plt.ylabel("Y-axis")

plt.title("Linear Regression")

plt.legend()

plt.show()


## Output:
![Screenshot 2025-04-14 213234](https://github.com/user-attachments/assets/4cc911ac-ad73-4ab7-b6fa-f5b7067b7352)
![Screenshot 2025-04-14 213255](https://github.com/user-attachments/assets/bef3fb8c-ff28-42bd-89bb-dbaa04a457a1)
![Screenshot 2025-04-14 213308](https://github.com/user-attachments/assets/5b94d647-e98e-4d48-9335-bf1ee59bf2da)
![Screenshot 2025-04-14 213324](https://github.com/user-attachments/assets/d8762ac8-9d08-45f0-bff7-ddc753d307fb)
![Screenshot 2025-04-14 213336](https://github.com/user-attachments/assets/c8bf3f30-ddc6-4426-ba75-c9849e5cd3a2)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
