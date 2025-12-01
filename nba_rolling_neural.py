import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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

train = rolling_avg_df[rolling_avg_df["date"] < "3/14/24"]
test = rolling_avg_df[rolling_avg_df["date"] >= "3/14/24"]

#Make sure to drop the first instance from the training set since it has NaN values for its rolling averages
train= train.dropna()

#Now the data is fixed in a way so that each instance consists of the 
#given team's rolling average stats from their previous 3 games
#Now we can correctly proceed with making an actual predictive model that uses
#past stats to predict outcomes of upcoming games

class NeuralNetwork(nn.Module):
    def __init__(self, in_features = 9):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
torch.manual_seed(42)
model = NeuralNetwork()

X_train = train[new_cols].to_numpy(dtype=np.float32)
y_train = train[output].to_numpy(dtype=np.float32)

X_test = test[new_cols].to_numpy(dtype=np.float32)
y_test = test[output].to_numpy(dtype=np.float32)

# Normalize using training set stats
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Convert to tensors
x_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
x_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Create datasets & dataloaders
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#Basic training loop to find the most optimal parameters for the model
epochs = 1000
losses = []
#We run the model through the dataset 100 times (epochs)
for i in range(epochs):
    #Obtain y predictions for each instance in the training data by running it through the forward method
    #y_pred = model.forward(x_train)
    y_pred = model(x_train)

    #measure the loss (error) between the predictions and the actual y values
    loss = criterion(y_pred, y_train)

    #Keep track of our losses by adding each of these losses into our losses array
    losses.append(loss.detach().numpy())

    #Print every 10 epochs to check in with the model
    #if i % 10 == 0:
    #   print(f'Epoch: {i} and loss: {loss}')

    #Now we perform backpropagation where we take the error rate of forward propagation and feed it
    #back through the network to fine tune the weights
    #Zero out the optimizer
    optimizer.zero_grad()
    #Backpropagate
    loss.backward()
    optimizer.step()


#Alternate training loop to test out different learning rates and epoch totals.
#In order to decrease the error, we can increase the learning rate or increase the number of epochs
#Testing different learning rates and number of epochs to see which variation would result in 
#lower loss and better convergence. Created three individual models for each learning rate
#to analyze its overall performance over the span of the training loop
'''
learning_rates = [0.01, 0.001, 0.0001]
epochs = 1000

loss_curves = {}

for lr in learning_rates:
    print(f"\nTraining with learning rate = {lr}")
    
    # fresh model each time
    torch.manual_seed(42)
    model = NeuralNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    
    for epoch in range(epochs):
        for xb, yb in train_loader:
            y_pred = model(xb)
            loss = criterion(y_pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    loss_curves[lr] = losses

# Plot all curves
plt.figure(figsize=(8,6))
for lr, losses in loss_curves.items():
    plt.plot(losses, label=f"LR={lr}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs for Different Learning Rates")
plt.legend()
plt.show()
'''


#Model testing to see how accurate the model is on test data
#Testing the model and evaluating its performance
#Turn off backpropagation to save memory and computational power
with torch.no_grad():
    #Predictions from the x_test dataset
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)

#Checking the accuracy of the model on the test data by comparing the predicted value to the actual value
correct = 0
with torch.no_grad():
    #Set val to a random value to initialize it
    val = 99
    #Goes through each instance in the test data
    for i, data in enumerate(x_test):
        #Predicts the value for the given instance
        y_val = model.forward(data)
        #Once we obtain the predicted value for the current index, we round it to 1 or 0 to represent a win or loss
        if y_val.item() >= 0.5:
            y_val = 1.0
        else:
            y_val = 0.0
        #Checks if the predicted (rounded) value is equal to the actual label for that instance
        if y_val == y_test[i].item():
            #If the predicted value is equal to the actual label, we increment the correct counter
            correct += 1
        #Used to check specific instances (save the predicted value of the instance)
        if i == 329:
            val = y_val
    #Used to check specific instances (print True or False if 
    #the predicted value is equal to the actual label for that instance)
    #Returns True if the predicted rounded value is equal to the actual label for that instance
    #Returns False if there is a mismatch
    print(val == y_test[329].item())

#Obtained an accuracy of 0.8191056910569106
print(f'Accuracy: {correct / len(x_test)}')