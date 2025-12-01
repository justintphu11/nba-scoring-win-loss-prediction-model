import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score
import matplotlib.patches as mpatches
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_excel("/Users/justinphu/Desktop/Dataset.xlsx")
df.to_csv("nba_stats.csv", index = False)

#Transforming the data
#Converting the W/L column to a binary column
df["Result"] = df["W/L"].apply(lambda x: 1 if x == "W" else 0)
# Convert FT% to numeric, handling missing values (dashes) as NaN
df["FT%"] = pd.to_numeric(df["FT%"], errors='coerce')
#Fill any NaN values with 0
df.fillna(0, inplace = True)
df['Team_code'] = df["Team"].astype("category").cat.codes
#Using regex to look into the "Match up" column. determining which team is the
# home team (represented by 1), and creating a new column that states Home or Away
df["Home_Away"] = df["Match Up"].str.contains(r" vs\. ").map({True: 1, False: 0})
df["date"] = pd.to_datetime(df["Game Date"])

# Get last 3 characters from 'Match Up' into new column 'Opponent'
df['Opponent'] = df['Match Up'].str.extract(r'(.{3})$')

#Setting input and output features
input_stats = ["FG%", "FT%", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF", "Home_Away"]
rolling_avg_stats = input_stats[:-1]
new_cols = [f"{c}_rolling" for c in rolling_avg_stats]
new_features = new_cols + ["Home_Away"]
print(new_features)
output = 'Result'

###Rolling Averages Version
def rolling_averages(group, stat_cols, assign_new_cols):
    group = group.sort_values("date")
    #Compute the rolling averages using rolling() where 3 represents the number of previous
    #instances to refer back to, while the closed parameter is saying to not include
    #the current week with the 3 previous instances that we are referencing
    rolling_stats = group[stat_cols].rolling(3, min_periods = 1, closed = 'left').mean()

    #Assign the computed rolling stats back to the given group but in new columns
    group[assign_new_cols] = rolling_stats

    #We make sure to drop NA data elements found in the assign_new_cols columns
    #in the case that there aren't 3 previous instances to refer back to.
    group.dropna(subset = assign_new_cols)

    return group

#Splitting the overall dataframe into individual dataframes based on team
grouped_matches = df.groupby("Team")
#Can access specific groups by team name using the get_group() function
current_group = grouped_matches.get_group("LAL")

#For example, we call the rolling_averages() function on LAL and see the rolling averages for each included stat category
#This will help improve accuracy when we pass this into our machine learning models
rolling_averages(current_group, rolling_avg_stats, new_cols)

#We can apply this rolling_averages() function to each team in the dataset
#This gives us a new dataframe where each instance is grouped based on team 
#and has its rolling averages calculated from the previous 3 games
rolling_avg_df = (grouped_matches.apply(lambda x: rolling_averages(x, rolling_avg_stats, new_cols))).droplevel("Team")
#However, we see that the labeled indices of each instance is out of order, as it is based on the initial index assignment
#Therefore, we reassign the indices so that each instance is in numerical order
rolling_avg_df.index = range(df.shape[0])

#Now the data is fixed in a way so that each instance consists of the 
#given team's rolling average stats from their previous 3 games
#Now we can correctly proceed with making an actual predictive model that uses
#past stats to predict outcomes of upcoming games



#Random Forest Classifier using Rolling Averages
#Getting the training and testing dataset from our rolling averages dataset
train = rolling_avg_df[rolling_avg_df["date"] < "3/14/24"]
test = rolling_avg_df[rolling_avg_df["date"] >= "3/14/24"]

#Make sure to drop the first instance from the training set since it has NaN values for its rolling averages
train= train.dropna()

#Creating and fitting our new Random Forest Classifier Model
rf_rolling = RandomForestClassifier(n_estimators=50, min_samples_split=10, max_features = 8, random_state = 42)
rf_rolling.fit(train[new_cols], train[output])
rf_preds = rf_rolling.predict(test[new_cols])

#The precision score (before GridSearchCV) is 0.5680933852140078
#The precision score (after GridSearchCV) is 0.5852713178294574
#which is higher than the random forest classifier model with data leakage
rf_precision = precision_score(test[output], rf_preds)
##print(rf_precision)

#The accuracy score (before GridSearchCV) is 0.5708502024291497
#The accuracy score (after GridSearchCV) is 0.5890688259109311
#which is higher than the random forest classifier model with data leakage
rf_accuracy = accuracy_score(test[output], rf_preds)
##print(rf_accuracy)

#We can create a combined dataframe that compares the actual label to the predicted label
rand_forest_combined = pd.DataFrame(dict(actual = test[output], prediction = rf_preds), index=test.index)
##print(pd.crosstab(index=rand_forest_combined["actual"], columns=rand_forest_combined["prediction"]))

#Additionally, I performed a GridSearchCV in order to find any possible improvements in the hyperparameter values
#that I can make in order to increase the accuracy. This was the resulting set of hyperparameters
#{'max_features': 8, 'min_samples_split': 10, 'n_estimators': 50}
#param_grid = [{'n_estimators': [50, 100, 1000], 'max_features': [5, 8, 10], 'min_samples_split': [2, 5, 10]}]
#gridsearch = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=7,scoring='accuracy', return_train_score=True)
#gridsearch.fit(train[new_cols], train[output])

#Note that this dataframe is not separated by individual teams, which can be fixed as follows:
rand_forest_combined = rand_forest_combined.merge(rolling_avg_df[["date", "Team", "Opponent", "Result"]], left_index=True, right_index=True)
#This looks at the dataframe rand_forest_combined, takes a given instance, finds its corresponding instance in the
#rolling_avg_df based on index, and merges each instance with the specified attributes in the merge() function

#Now we will look at how our model performs when it comes to performing both sides of a given game
#This involves checking to see if a given game correctly identifies the winner and the loser
#To perform this check, we will combine game instances from both teams into one individual row
#and it will show the predictions when each team is the current team being predicted for
#print(rolling_avg_df["Team"], rolling_avg_df["Opponent"])
merged = rand_forest_combined.merge(rand_forest_combined, left_on=["date", "Team"], right_on=["date", "Opponent"])
##print(merged)

#For example, this will obtain all instances where the first team was predicted to win and the second team was predicted
#to lose, and with these instances, we look at the actual outcome that the first team resulted in and see how many
#were correctly predicted as a win (since we are looking at instances in which the first team wins)
merged[(merged["prediction_x"] == 1) & (merged["prediction_y"] == 0)]["actual_x"].value_counts()
#The result of this shows that for all instances where the model predicted that the first team would win while the second
#team loses, 83 of those games were correctly predicted to win while 48 were incorrectly predicted to lose. This gives a 
#rate of 63.3587786%, which is relatively good


#Now I will create a logistic regression model to see if we can get an improvement on the accuracy and precision scores
log_reg = LogisticRegression(C = 100, penalty = 'l1', solver = "liblinear", max_iter = 100)
log_reg.fit(train[new_cols], train[output])
#We use predict_proba() to get the percentage probability of each instance resulting in a "W"
#Each instance is represented by a two element array, where the first element represents the probability
#of 0 (loss) while the second element represents the probability of 1 (win)
y_proba = log_reg.predict_proba(test[new_cols])
log_pred = log_reg.predict(test[new_cols])

#The accuracy score is 0.5850202429149798, which is better than the accuracy score of the 
#Random Forest Classifier
log_acc = accuracy_score(test[output], log_pred)
#print(log_acc)

log_reg_combined = pd.DataFrame(dict(actual = test[output], prediction = log_pred))
##print(pd.crosstab(index=log_reg_combined["actual"], columns=log_reg_combined["prediction"]))

#The precision score is 0.5801526717557252
log_precision = precision_score(test[output], log_pred)
#print(log_precision)

#Additionally, I performed a GridSearchCV in order to find any possible improvements in the hyperparameter values
#that I can make in order to increase the accuracy. This was the resulting set of hyperparameters:
#{'C': 100, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}
#This led to the same accuracy score as before, so no changes are needed
#param_grid = {'penalty': ['l1', 'l2'], 
#              'C': [0.001, 0.01, 0.1, 1, 10, 100], 
#              'solver': ['liblinear', 'saga'], 
#             'max_iter': [50, 100, 300, 500, 1000]}
#gridsearch = GridSearchCV(LogisticRegression(), param_grid, cv=7,scoring='accuracy', return_train_score=True)
#gridsearch.fit(train[new_cols], train[output])
#print(gridsearch.best_params_)
