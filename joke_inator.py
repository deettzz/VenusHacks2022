import os
import pandas as pd
import matplotlib.pyplot as plt


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

df = pd.read_csv(data_file_address)
print(df)