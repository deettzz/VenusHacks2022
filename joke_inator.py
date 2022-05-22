# data handling
import os
from turtle import color
from matplotlib.ft2font import BOLD
import pandas as pd
from regex import E
# model
from sklearn import linear_model
# cross-validation, training and splitting
from sklearn.model_selection import KFold
# measuring metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# GUI
import tkinter as tk


def round_rectangle(x1, y1, x2, y2, radius=25, **kwargs):
        
    points = [x1+radius, y1,
              x1+radius, y1,
              x2-radius, y1,
              x2-radius, y1,
              x2, y1,
              x2, y1+radius,
              x2, y1+radius,
              x2, y2-radius,
              x2, y2-radius,
              x2, y2,
              x2-radius, y2,
              x2-radius, y2,
              x1+radius, y2,
              x1+radius, y2,
              x1, y2,
              x1, y2-radius,
              x1, y2-radius,
              x1, y1+radius,
              x1, y1+radius,
              x1, y1]

    return canvas1.create_polygon(points, **kwargs, smooth=True)

# Path to prev dir contains this file
dir = os.path.abspath(os.path.dirname(__file__))
data_file_address = os.path.join(dir, "Responses.csv")

# open slang file for reading
slangDir = os.path.abspath(os.path.dirname(__file__))
slang_file_address = os.path.join(slangDir, "slang_dict.txt")
slang_file = open(slang_file_address, 'r')

""" FEATURES:
1. length of characters (number)
2. punctuation (number)
3. presence of numbers (0 = no presence, 1 = presence made the joke better)
4. use of emoticons ( 0 = no emoticons, 1 = presence of emoticons made the joke better)"""

## PARSING
# Features, Input:
charLengthInput = 0
puncNumInput = 0
numberPresenceInput = 0
emoPresenceInput = 0
# Output:
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
X = df[['Interest_Rate', 'Unemployment_Rate']]
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

## GUI
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 600, height = 400)
root.resizable(False, False)
canvas1.pack()

background=tk.PhotoImage(file='gradient.PNG')
canvas1.create_image(300, 200, image = background)

rect = round_rectangle(100, 105, 500, 225, fill="#cda5df")

img=tk.PhotoImage(file='testImage1.PNG')
canvas1.create_image(495, 250, image = img)
img2=tk.PhotoImage(file='testImage2.PNG')
canvas1.create_image(125, 250, image = img2)
titleArt=tk.PhotoImage(file='wordArt.PNG')
titleArt=titleArt.subsample(2,2)
canvas1.create_image(300, 55, image = titleArt)

root.title("Joke-inator")


# Final equation of model
# intercept
print_intercept = ('Model Intercept: ', model.intercept_) # sklearn function to derive intercept
canvas1.create_text(300, 270, fill="white", text=print_intercept, justify='center')
# coefficients
print_coefs = ('Coefficients: ', model.coef_) # sklearn function to derive intercept
canvas1.create_text(300, 290, fill="white", text=print_coefs, justify='center')

# Create entry boxes
# First ind variable
canvas1.create_text(270,130, anchor=tk.E, width=200, fill="white", justify='right',font=('futura'), text='Type Interest Rate: ')
entry1 = tk.Entry(root,bd=0) # create 1st entry box
canvas1.create_window(370, 130, window=entry1)

# Second ind variable
canvas1.create_text(270,155, anchor=tk.E, width=200, fill="white", justify='right',font=('futura'), text='Type Unemployment Rate: ')
entry2 = tk.Entry(root,bd=0)  # create 2nd entry box
canvas1.create_window(370, 155, window=entry2)

def values():
    # first input variable from GUI
    global New_Interest_Rate
    New_Interest_Rate = float(entry1.get())
    # 2nd input variable from GUI
    global New_Unemployment_Rate
    New_Unemployment_Rate = float(entry2.get())

    y_predicted = ('Predicted Stock Index Price: ', model.predict([[New_Interest_Rate, New_Unemployment_Rate]]))
    canvas1.create_text(300,350, fill="white", text=y_predicted)

# button inputs datapoint to model and displays output
model_output_button = tk.Button(root, text='Predict Stock Index Price', bd=0,command=values)
canvas1.create_window(300, 200, window=model_output_button)
# Continue looping over script with GUI input
root.mainloop()

# returns the number of characterss in the joke
def getNumChars(joke):
    return len(joke)

# returns the number of punctuation characters in the joke
def getNumPunc(joke):
    puncList = ['!', '@', '#', '$', '%', '&', '(', ')', ':', ';', '"', ',', '.','?', '/'] 
    numPunc
    for char in joke:
        if char in puncList:
            numPunc += 1
    return numPunc

# returns True if the joke contains numbers, False otherwise
def hasNums(joke):
    for char in joke:
        if char.isdigit():
            return True
    return False

# returns True if the joke contains emoticons/slang from the slang_file, False otherwise
def hasEmo(joke):
    lines = slang_file.readlines()
    for line in lines:
        emoticon = line.split('`')[0]
        if emoticon in joke:
            return True
    return False