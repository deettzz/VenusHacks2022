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

""" DOCUMENTATION:
GUI INPUT: A JOKE!

MODEL FEATURES:
1. length of characters (number)
2. punctuation (number)
3. presence of numbers (0 = no presence, 1 = presence made the joke better)
4. use of emoticons ( 0 = no emoticons, 1 = presence of emoticons made the joke better)

MODEL OUTPUT:
Funny Rating
"""

## PARSING FUNCTIONS ##
# returns the number of characters in the joke
def getNumChars(joke):
    return len(joke)

# returns the number of punctuation characters in the joke
def getNumPunc(joke):
    puncList = ['!', '@', '#', '$', '%', '&', '(', ')', ':', ';', '"', ',', '.','?', '/']
    numPunc=0
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

## GUI FUNCTIONS ##
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

predicted_label = tk.Label
def values():
    # first input variable from GUI
    global jokeInput
    jokeInput = entry1.get()
    ## PARSING
    # Features, Input:
    charLengthInput = getNumChars(jokeInput)
    puncNumInput = getNumPunc(jokeInput)
    numberPresenceInput = hasNums(jokeInput)
    emoPresenceInput = hasNums(jokeInput)
    resultsBox = round_rectangle(130, 250, 470, 300, fill="#d9a5b1")
    y_predicted = str(round(model.predict([[charLengthInput, puncNumInput, numberPresenceInput, emoPresenceInput]])[0], 4))
    y_predicted = 'Funny Rating: '+y_predicted+'% Funny'
    predicted_label = tk.Label(root, fg="white", font=('futura', 12), text=y_predicted, bg="#d9a5b1")
    canvas1.create_window(300, 275, window=predicted_label)

## OPEN NECESSARY FILES
# Path to prev dir contains this file
dir = os.path.abspath(os.path.dirname(__file__))
# -> open data file with this directory
data_file_address = os.path.join(dir, "Responses.csv")
# -> open slang_dict.txt file with this directory
slang_file_address = os.path.join(dir, "slang_dict.txt")
slang_file = open(slang_file_address, 'r')
# -> open gradient2.PNG file with this directory
gradient_address = os.path.join(dir, "Pictures/gradient2.PNG")
# -> open wordArt.PNG file with this directory
wordArt_address = os.path.join(dir, "Pictures/wordArt.PNG")

## READ AND DEFINE DATA VARIABLES
df = pd.read_csv(data_file_address)
# Independent variables
X = df[['charLengthScale', 'puncNumScale','numberPresenceScale','emoPresenceScale']]
# Dependent variable
y = df['funnyRatingResponse']

## CREATE MODEL
# Utilize sklearn to define model
model = linear_model.LinearRegression()
# Cross-validation, split into training and testing data as well
kf = KFold(n_splits=3, shuffle=True, random_state=None)
for train_index, test_index in kf.split(X):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    model.fit(X_train, y_train)
# Additional Information on Fit of Model
# print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)

## MODEL METRICS
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
#print(y_test, y_pred)
print('MSE OF MODEL:', mse)

## GUI
# Initialize GUI
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 600, height = 400)
root.resizable(False, False)
canvas1.pack()
# Size and background color of output GUI
background=tk.PhotoImage(file=gradient_address)
background=background.subsample(2,2)
canvas1.create_image(300, 200, image = background)
# Round edges of output GUI
rect = round_rectangle(100, 105, 500, 225, fill="#d9a5b1")
# Add title to output GUI
root.title("Joke-inator")
titleArt=tk.PhotoImage(file=wordArt_address)
titleArt=titleArt.subsample(2,2)
canvas1.create_image(300, 55, image = titleArt)
# Add final equation of model to output GUI
# Intercept of equation
print_intercept = 'Model Intercept: '+str(round(model.intercept_, 4)) # sklearn function to derive intercept
canvas1.create_text(300, 360, fill="white", font=('futura', 10),text=print_intercept, justify='center')
# Coefficients of equation
coefs_rounded = [round(num, 4) for num in list(model.coef_)]
print_coefs = 'Coefficients: '+ str(coefs_rounded) # sklearn function to derive intercept
canvas1.create_text(300, 380, fill="white", font=('futura', 10),text=print_coefs, justify='center')
# Create entry box to collect input joke
# First ind variable
canvas1.create_text(300, 130, fill="white",font=('futura', 22), text='Enter Your Joke: ')
entry1 = tk.Entry(root, bd=0) # create 1st entry box
canvas1.create_window(300, 160, width=350, window=entry1)
# Button inputs datapoint (joke and parsed input from joke) to model and displays output
model_output_button = tk.Button(root, font=('futura'),text='Get Funny Rating!', bd=0,command=values)
canvas1.create_window(300, 200, window=model_output_button)
# Continue looping over script with GUI input
root.mainloop()
