# march_madness_data_project
Used the Tournament Game Data set from the March Madness Data page on Kaggle (https://www.kaggle.com/datasets/nishaanamin/march-madness-data?resource=download&amp;select=Tournament+Game+Data.csv).
My goal was to understand which attributes about a specific team made it more likely for that team to advance through the tournament and eventually win the tournament. 
I cleaned the data by dropping unwanted or unnecessary variables (block %s, rebound %s, etc..) and only used data from before the first round. 
Then, I created dummy variables for each of the x variables that I wanted to analyze by determining the average of each variable and determining if a team was above or below average in that category (3 point percentage, KenPom Adjusted Efficiency). I also created dummies for the y variable (whether or not a team won the championship) and for their seeding in the tournament. 
Finally, I concatted the dummy variables into a new dataset, ran a logistic regression using sklearn, and discovered which coefficients were good predictors of whether a team would win march madness, which can be found in the ModelOutput excel file.
