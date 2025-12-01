import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score
import matplotlib.patches as mpatches

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
output = 'Result'

#Setting training data to be the upper 80% of the dataset while test data is the lower 20% of the dataset
train = df[df["date"] < "3/14/24"]
test = df[df["date"] >= "3/14/24"]

#Creating our first model: Random Forest Classifier
#n_estimators is the number of decision trees that we want to train
#min_samples_split is the number of samples we want in a leaf of a decision tree before we split the node
#The higher that min_samples_split is, the less likely it is to overfit, but the lower the accuracy on the training data
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state = 42)
rf.fit(train[input_stats], train[output])

#Obtain the initial predictions
init_preds = rf.predict(test[input_stats])

#Measure of accuracy, where we see out of all the positive predictions, how many were actually positive
init_acc = accuracy_score(test[output], init_preds)
#The accuracy score is 0.8215767634854771

#We can create a confusion matrix where the rows are actual labels while columns are prediction labels
rand_forest_combined = pd.DataFrame(dict(actual = test[output], prediction = init_preds))
pd.crosstab(index=rand_forest_combined["actual"], columns=rand_forest_combined["prediction"])

#The precision score is 0.8259109311740891
init_precision = precision_score(test[output], init_preds)
#print(init_precision)

##########
#Now I will create a logistic regression model to see if we can get an improvement on the accuracy and precision scores
log_reg = LogisticRegression(max_iter = 100)
log_reg.fit(train[input_stats], train[output])
#We use predict_proba() to get the percentage probability of each instance resulting in a "W"
#Each instance is represented by a two element array, where the first element represents the probability
#of 0 (loss) while the second element represents the probability of 1 (win)
y_proba = log_reg.predict_proba(test[input_stats])
log_pred = log_reg.predict(test[input_stats])

#The accuracy score is 0.8441295546558705, which is better than the accuracy score of the 
#Random Forest Classifier
log_acc = accuracy_score(test[output], log_pred)

log_reg_combined = pd.DataFrame(dict(actual = test[output], prediction = log_pred))
pd.crosstab(index=log_reg_combined["actual"], columns=log_reg_combined["prediction"])

#The precision score is 0.8346456692913385
log_precision = precision_score(test[output], log_pred)
#print(log_precision)

#The code below focuses on one stat in the input features array and sees how the predicted 
#probability changes depending on the given stat's value

for i in range(10):
    feature1 = input_stats[i]  # pick one feature
    plt.scatter(test[feature1], y_proba[:,1], c=test[output], cmap="bwr", alpha=0.6)
    plt.xlabel(feature1)
    plt.ylabel("Predicted Probability of Win")
    win_patch = mpatches.Patch(color="red", label="Actual Win (1)")
    loss_patch = mpatches.Patch(color="blue", label="Actual Loss (0)")
    plt.legend(handles=[loss_patch, win_patch])
    #plt.show()

##############################
#Rolling Averages model