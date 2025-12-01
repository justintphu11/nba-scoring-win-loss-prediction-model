import pandas as pd
import numpy as np
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
output = 'Result'

class NeuralNetwork(nn.Module):
    def __init__(self, in_features = 10):
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

X = df[input_stats].to_numpy(dtype=np.float32)
y = df[output].to_numpy(dtype=np.float32)

#Normalize the data features before inputting into the neural network
X = (X - X.mean(axis=0)) / X.std(axis=0)

#Obtain the training and test data splits
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Convert the data splits into tensors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

#Set the criterion to analyze the difference in prediction vs actual results
criterion = nn.BCELoss()
#Set the model optimizer as the Adam optimizer with learning rate that decreases the error (brings us closer to the actual)
#Note that model.parameters returns the hidden layers in the neural network model that we created
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=False)

#Training the model
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

#Plotting the decrease in error as number of epochs increases
#plt.plot(range(epochs), losses)
#plt.ylabel("loss/error")
#plt.xlabel("epoch")
#plt.show()
'''
# Hyperparameters
epochs = 500
learning_rate = 0.0005

# Initialize model, criterion, optimizer
model = NeuralNetwork()  # your model class
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Scheduler: reduce LR by half every 100 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Lists to track progress
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for xb, yb in train_dataloader:
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * xb.size(0)  # sum over batch
        preds = (torch.sigmoid(y_pred) > 0.5).float()
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    
    train_losses.append(epoch_loss / total)
    train_accuracies.append(correct / total)
    
    # Evaluate on test set
    model.eval()
    epoch_loss_test = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for xb, yb in test_dataloader:
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            
            epoch_loss_test += loss.item() * xb.size(0)
            preds = (torch.sigmoid(y_pred) > 0.5).float()
            correct_test += (preds == yb).sum().item()
            total_test += xb.size(0)
    
    test_losses.append(epoch_loss_test / total_test)
    test_accuracies.append(correct_test / total_test)
    
    # Step the scheduler
    scheduler.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, "
              f"Train Acc={train_accuracies[-1]:.4f}, "
              f"Test Loss={test_losses[-1]:.4f}, "
              f"Test Acc={test_accuracies[-1]:.4f}")

# Plot loss
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()

# Plot accuracy
plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(test_accuracies, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()

plt.show()
'''

#In order to decrease the error, we can increase the learning rate or increase the number of epochs

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
