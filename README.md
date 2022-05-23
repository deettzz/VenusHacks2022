# VenusHacks2022

## Inspiration
Four Computer Science and Engineering students, years of preparation, a number of lectures, and hours spent ‘learning’ later (us, not our computers of course!), embark on a mission to determine what IS and IS NOT funny. Our trusty sidekick you might ask? None other than machine learning of course.

We hope our project not only brings you many laughs, but also helps you determine how and what will bring joy to others! <3

## What it does
Tell the computer a joke! Once your joke is parsed and fed into the machine learning model, it’ll rate how funny it is ! :)

## How we built it
We first gathered data from anonymous users using a google form that had a compilation of ten jokes with varying levels of humor. Users responded with their opinion about how funny these jokes were and the data was collected in a google sheet. The data was parsed and ranked on differing factors that impacted how funny the users found each joke. The parsed data was split in order to train the linear regression model and test the effectiveness of it. A GUI was built after the model was trained to rate how funny a joke is.

## Challenges we ran into
Creating a machine learning model is a difficult task, but creating a model with little to no data is nearly impossible!! The solution? Collect data. Using Google Forms, the team compiled ten jokes and asked users how differing factors impacted how funny they found each joke was.

The next obstacle? Determining what factors make a joke funny. Another challenge? Determining which machine learning model fits the data and situation best. The solution? Comparing the mean squared error (MSE) across differing regression models (Linear, Lasso, Ridge, etc) and differing parameters; that is having already split the training and testing data and also accounting for a cross-validation of 3. With more tuning, and more research (other than the 400+ data points collected), the team plans to perfect the model based on adding, removing, and better scaling input features to provide more accurate responses in the future.

## What we learned
Survey participants to collect data for our model
Parse input to extract specific features
Create a machine learning model using the collected data
Design a user-friendly UI
What's next for The Joke-Inator
First, UCI. Next, Orange County, Los Angeles, the United States, North America, THE WORLD!! Laughs, laughs everywhere. >:) → :^)

## Built With
laughter, love, git, github, Google Forms, Google Sheets, Microsoft Excel, pandas,
pycharm, vscode, python, sklearn, tkinter.

## Important Links
Devpost Submission: https://devpost.com/software/the-joke_inator

Additional Information: https://docs.google.com/presentation/d/1dZKOrZiOmKRuPZUUaUGN_C8HBlmoBXrcMVnrhrHWzAI/edit?usp=sharing