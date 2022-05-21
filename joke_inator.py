# data handling
import os
import pandas as pd
# model
from sklearn import linear_model
# cross-validation, training and splitting
from sklearn.model_selection import KFold
# measuring metrics
from sklearn.metrics import mean_squared_error

# get path to prev dir which contains this file
dir = os.path.abspath(os.path.dirname(__file__))
data_file_address = os.path.join(dir, "Responses.csv")

""" FEATURES:
1. length of characters (number)
2. punctuation (number)
3. presence of numbers (0 = no presence, 1 = presence made the joke better)
4. use of emoticons ( 0 = no emoticons, 1 = presence of emoticons made the joke better)"""

charLengthInput = 0
puncNumInput = 0
numberPresenceInput = 0
emoPresenceInput = 0
funnyRating = 0

#df = pd.read_csv(data_file_address)

Stock_Market = {
    'Year': [2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2016, 2016, 2016, 2016, 2016, 2016,
             2016, 2016, 2016, 2016, 2016, 2016],
    'Month': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'Interest_Rate': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                      1.75, 1.75, 1.75, 1.75, 1.75],
    'Unemployment_Rate': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1,
                          6.1, 5.9, 6.2, 6.2, 6.1],
    'Stock_Index_Price': [10, 15, 44, 77, 77, 90, 33, 22, 55, 55, 43, 43, 76, 54, 89, 98,
                          91, 49, 84, 86, 86, 22, 74, 19]
    }

df = pd.DataFrame(Stock_Market, columns=['Year', 'Month', 'Interest_Rate', 'Unemployment_Rate', 'Stock_Index_Price'])

# Independent variables
X = df[['Month','Interest_Rate']]
# dependent variable
y = df['Stock_Index_Price']

# Utilizing sklearn
model = linear_model.LinearRegression()

# Cross-validation, split into training and testing data
kf = KFold(n_splits=3, shuffle=True, random_state=None)
for train_index, test_index in kf.split(X):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    model.fit(X_train, y_train)

# Process to see what model was fit as:
# print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)

# metrics on model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(y_test, y_pred)
print(mse)

# Prediction with sklearn
New_Interest_Rate = 2.75
New_Unemployment_Rate = 5.3
print('Predicted Stock Index Price: \n', model.predict([[New_Interest_Rate, New_Unemployment_Rate]]))
