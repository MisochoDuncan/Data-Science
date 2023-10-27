# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Creation of a DataFrame you can also add data fron a csv file by importation
data = pd.DataFrame({
    'WeatherConditions': [1, 2, 3, 4, 3, 2, 1, 4, 3, 2],
    'RoadType': [1, 2, 3, 2, 1, 3, 1, 2, 3, 2],
    'SpeedLimit': [30, 40, 50, 60, 70, 40, 50, 60, 30, 70],
    'TimeofDay': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    'VehicleType': [1, 2, 2, 3, 1, 2, 1, 3, 3, 2],
    'DriverAge': [25, 35, 45, 28, 50, 32, 22, 39, 57, 41],
    'AlcoholInvolvement': [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'SeatbeltUsage': [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    'RoadCondition': [1, 2, 3, 2, 1, 3, 2, 3, 2, 1],
    'AccidentSeverity': [2, 3, 4, 1, 5, 3, 2, 4, 1, 5]
})

# Splitting the data into training and test sets
X = data[['WeatherConditions', 'RoadType', 'SpeedLimit', 'TimeofDay', 'VehicleType', 'DriverAge', 'AlcoholInvolvement', 'SeatbeltUsage', 'RoadCondition']]
y = data['AccidentSeverity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creation and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions for a hypothetical accident
new_data = pd.DataFrame({
    'WeatherConditions': [3],
    'RoadType': [2],
    'SpeedLimit': [50],
    'TimeofDay': [2],
    'VehicleType': [1],
    'DriverAge': [30],
    'AlcoholInvolvement': [0],
    'SeatbeltUsage': [1],
    'RoadCondition': [3]
})

predicted_severity = model.predict(new_data)
print("Predicted Severity for an Accident:", predicted_severity[0])

