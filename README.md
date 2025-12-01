# NBA Scoring and Win/Loss Prediction Model
Created two different prediction models. The first one is a multiple linear regression model that predicted team scoring output, with EDA and stepwise regression to identify influential predictors. 

The second version is an implementation of an ETL pipeline where I trained classification models using cross-validation, grid search and early stopping. Here is the workflow and observations:

Step 1. ETL (Extract → Transform → Load)

• Extract:
In this case I am using a dataset of all regular season games during the 2023-24 NBA season.

• Transform:
First I will clean the data. Since machine learning models can only work with floats and ints, I converted some of the main columns/features into ints. For example, the “W/L” column had data in the form of an object type, so I used a lambda function to convert “W” values into 1 and “L” values into 0. Also, I found that in the FT% column, there was a game where no free throws were shot throughout the entire game. Therefore, it left a “-” to mark that there was no value. To fix this, I used the pd.to_numeric() function to convert the entire “FT% column into floats, while handling missing values (dashes) as NaN. Thereafter, I used the fillna() function to convert any NaN values into 0. I also decided to add a new feature called “Home_Away” that states whether the current team’s row is playing at home or away. I believe this is a major factor when it comes to winning more games, as home court advantage is extremely advantageous to a player’s performance. Therefore, I used a regular expression to filter through the elements in the “Match Up” column and input this result into a new column called “Home_Away”. Lastly, I set a “Team_code” feature that attaches a team ID number to each team so that it is easier to identify.

Some useful features I included are:

Effective FG% (shooting efficiency)

FT %
    
OREB
    
DREB
    
AST
    
STL
    
BLK
    
TOV
    
PF
    
Note that I did not include any stat differentials between the two competing teams, as this is a form of label leakage where we are using differentials from the final box score of the game that we are trying to predict. This is basically seeing the stat results of the given game and predicting based on that, as the game has already been played out. For now, I will keep the current set of features and run a few models on it to see how accurate it is, just as a set of baseline models. Later on, I will create a new feature set consisting of rolling averages from a team’s last 3 to 5 games, so that it uses stats that are known before the game in order to predict the result of the current game. 

• Load: Save into a clean CSV or SQLite DB.

Step 2. Machine Learning Models (Comparison)

Models to Train:

1. Random Forest Classifier (scikit-learn)

When using the Random Forest Classifier with the following parameters: n_estimators=50, min_samples_split=10, random_state = 42, we get an accuracy score of 0.8215767634854771, which says that out of all the total predictions, how many of them were correct. I also created a confusion matrix based on the predictions and actual labels of the test set, and it showed the following:
   
prediction    0    1
actual
0           204   43
1            43  204

But since we want to focus more on predicting wins, we actually prioritize the precision score, which states that out of all the positive predictions, how many were actually correct. In doing so, I obtained a precision score of 0.8259109311740891, which showcases a small improvement from the accuracy score.

Additional to the model created, I tried to use GridSearchCV and cross validation individually to see whether the precision score of the modified RandomForestClassifier would increase. This included using GridSeachCV to find the set of hyperparameters that made the model perform the best (measured by precision score). Even so, it showed that the original model actually performed with a higher precision score than the updated model with the hyperparameters obtained from the GridSearchCV. As a result, I chose to remain with the original model. 

2. Logistic Regression (scikit-learn)
   
     For the logistic regression model, we performed the same procedures as for the Random Forest Classifier. I set the max_iters parameter for the logistic regression model to 1000, and when I computed its accuracy score, I obtained a score of 0.8441295546558705. As for the confusion matrix, I got the following:
   
prediction    0    1
actual
0           205   42
1            35  212

Thereafter, I obtained the precision score, which ended up resulting in a score of 0.8346456692913385, which is slightly lower than the accuracy score obtained above.

Evaluation Metrics:

• Classification: Accuracy, Precision, F1 score.

• In addition to computing these metric scores above, I also computed individual graphs that show how the prediction probability changes with response to varying values of a specific stat, specifically on the logistic regression model. I performed this comparison for each stat in the “input_stats” list, and I highlighted the relationship between each stat and the predicted probabilities.

1. FG%
   
Looking at the graph for FG% vs predicted probability, we see that near the lower field goal percentage end, there are more occurrences of actual losses that end up being classified as “L”. As we trend towards higher field goal percentages, we begin to see more actual win instances being predicted as “W”. This highlights the relationship between field goal percentage and game outcome, as teams that have higher field goal percentages usually end up winning the game (showcases a positive correlation).

2. FT%
   
Looking at the graph for FT% vs predicted probability, we see that the majority of “Actual Loss” instances are clustered at the bottom of the graph, with the majority of instances being under the 0.5 predicted probability of win. Meanwhile, the majority of “Actual Win” instances are clustered near the top of the graph, where the predicted probability of win for these instances are above 0.5. However, this makes logical sense, as actual loss instances should be near the bottom where they would be predicted as having a lower probability of winning, and vice versa for actual win instances. The main thing to focus on is that all instances are located on the right side of the graph with no clear relationship between FT% and game outcome. In fact, the graph consists of both actual win and actual loss instances for any FT%. As a result, we can see that FT% does not have a huge impact on the game outcome, and it has no correlation with W/L.

3. OREB
   
Looking at the graph for OREB vs predicted probability, we see that a majority of the actual loss instances are below the 0.5 predicted probability while most of the actual win instances are above the 0.5 predicted probability range. Once again, we see that there is no clear correlation between the game outcome and the number of offensive rebounds grabbed, as there exists both actual win and loss instances for any number of OREB. Thus, this shows that the number of offensive rebounds grabbed by a team does not play a clear role in the outcome of a given NBA game.

4. DREB
   
Looking at the graph for DREB vs predicted probability, we see that as we move towards higher amounts of defensive rebounds grabbed by a given NBA team, more actual win instances are evident. In addition, the model is able to correctly predict the majority of these actual win instances, thus showing that the model recognizes this stat when it comes to predicting game outcomes. With this in mind, we can say that there exists a positive relationship between the number of defensive rebounds grabbed and the outcome of the given NBA game, which makes logical sense since the more rebounds grabbed by a team, the more offensive possessions that they have and the more shots that their opponents have missed.

5. AST
   
Looking at the graph for assists vs predicted probability, we see that as the number of assists increases, the more actual win instances start to appear. This trend is directly noticeable just by observing the dataset in general, but the key thing to highlight is that the model is generally making the correct prediction on these instances, meaning that the model is also taking into account this stat when it comes to game outcome. Therefore, this shows that game outcomes are positively correlated with number of assists in a game.

6. STL
   
Looking at the graph for STL vs predicted probability, we see the general trend of actual losses being below the 0.5 predicted probability threshold and actual wins being above the 0.5 predicted probability, thus showing that the model is making good predictions. However, we see that there exists actual win and actual loss instances for all values in the range of STL totals. This shows that the number of steals made by a given team do not show a clear indication of game outcome, and that there is no correlation between the number of steals and the game outcome. This makes sense since a team can have a large amount of steals, but it does not mean that they are producing enough offensive output to win them the game.

7. BLK
   
Looking at the graph for BLK vs predicted probability, we see the general trend of actual losses being below the 0.5 predicted probability threshold and actual wins being above the 0.5 predicted probability, thus showing that the model is making good predictions. However, we see that there exists actual win and actual loss instances for all values in the range of BLK totals. This shows that the number of blocks made by a given team do not show a clear indication of game outcome, and that there is no correlation between the number of blocks and the game outcome. This makes sense since a team can have a large amount of steals, but it does not mean that they are producing enough offensive output to win them the game.

8. TOV
    
Looking at the graph for TOV vs predicted probability, we see the general trend of actual losses being below the 0.5 predicted probability threshold and actual wins being above the 0.5 predicted probability, thus showing that the model is making good predictions. However, we see that there exists actual win and actual loss instances for all values in the range of TOV totals. This shows that the number of turnovers made by a given team do not show a clear indication of game outcome, and that there is no correlation between the number of turnovers and the game outcome. This makes sense since a team can have a large amount of turnovers, but it does not mean that they are not still producing high offensive output, even with having lower amounts of possessions.

9. PF
    
Looking at the graph for PF vs predicted probability, we see the general trend of actual losses being below the 0.5 predicted probability threshold and actual wins being above the 0.5 predicted probability, thus showing that the model is making good predictions. However, we see that there exists actual win and actual loss instances for all values in the range of PF totals. This shows that the number of personal fouls made by a given team do not show a clear indication of game outcome, and that there is no correlation between the number of personal fouls and the game outcome. This makes sense since a team can have a large amount of personal fouls, but it should not have a real impact on the offensive or defensive output of a given team other than giving more free throws to the opponent, which does not necessarily imply a win or loss.

10. Home_Away
    
Looking at the graph for Home_Away vs predicted probability, we see the general trend of actual losses being below the 0.5 predicted probability threshold and actual wins being above the 0.5 predicted probability, thus showing that the model is making good predictions. However, we see that there exists actual win and actual loss instances for all values in the range of Home_Away totals.  Note that the graph is hard to visually distinguish any trends since the only two possible values for this category is home or away, which makes this a binary variable. But based on the results, this shows that the location of a game does not show a clear indication of game outcome, and that there is no correlation between the home court and the game outcome. This does not makes sense due to the concept of home court advantage, where a team usually performs better with a home crowd cheering them on. But yet again, it is a general belief that does not automatically enhance a team’s performance, offensively or defensively.

Rolling Averages Improved Model

Now we will split the dataset up by team so that we can look at each team’s past games and predict how they will perform in future games. Using the same predictors as data leakage version, we create a rolling average version for each of these predictors except for the “Home_Away” predictor since its rolling average would not contribute any value when it comes to predicting team performance. 

*** Note that in this new version, we will not use any of the total game stats in our models in order to predict the outcome of future games, as only the rolling average stats from the previous three games will be used for prediction. This makes it a truly predictive model that does not contain any data leakage.

Rolling Average Computation:

Before computing the rolling averages, I grouped the dataset based on “Team” so that I can create a function that can specifically compute rolling averages on an exact team. To that end, I created a function called rolling_averages, which takes a specific team name, the stats that we want to compute the rolling averages for, and the new rolling average columns that will be filled as a result of this function.

In this function, we take a specific team’s game logs, organize them by date in ascending order, compute the rolling averages of our desired stats by using the rolling() function, assign these new stats for each game to the new rolling average columns, and returns the newly modified game logs of the given team. Note that since we take the rolling averages of the last three games, that means for the first three games, there does not exist enough games to compute the average of the past three games. For that reason, I used the min_periods parameter and set it to 1. It controls the minimum number of non-NaN observations required inside the rolling window for Pandas to compute a result. This parameter defines the threshold for how soon rolling values start showing up, so since it is equal to 1 in this case, that means it needs at least one prior instance to compute the rolling average stat. This makes sure that the previous one and two games get their rolling average stats filled, but it won’t fill the third previous game (the first game) since there are no previous games that it can compute averages of. To account for the NaN values in the rolling average columns for the first games of each team, I will  not include the first game in the training set since the model cannot accept NaN values and it may add biases to the model if I fill it with 0. As a result, the original 80:20 split of training:testing data will be a split of 64:17 games, which was originally 65:17 but I removed the first instance from the training set to account for this issue.

Using this new function, I applied it to the dataset with team based grouping, and by using a lambda function, I managed to apply this function to each team grouping in the dataset, thus creating the new dataset “rolling_avg_df”.

Now that we have our dataset containing the rolling averages of each game based on the prior three games, along with the original stats that we had, we can now create our models that can predict game outcomes based solely on these rolling averages.

1. Random Forest Classifier Model
   
After separating the data into training and testing datasets with a ratio of about 80:20, I created a Random Forest Classifier with n_estimators = 50, min_samples_split=10, and random_state=42. Recall that I am only using the rolling average stats from the previous three games as the means of predicting the outcome of the current game, as I want to ensure that this model is actually predictive. For that reason, when fitting/training this model on the training data, I am only using the rolling average stats to do so. 

*** Performance Metrics

After obtaining the predictions for the Random Forest Classifier, I computed the precision score to get 0.5680933852140078 and the accuracy score to get 0.5708502024291497. In addition, I created a confusion matrix to gauge how well the model performed:

prediction    0    1
actual
0           136  111
1           101  146

I also did another variation where I included the “Home_Away” predictor to provide some context for each game, as home court could be advantageous and possibly lead to the home team winning. However, the precision score I obtained was 0.5287356321839081 and the accuracy score was 0.5303643724696356. As for the confusion matrix, I got the following:

prediction    0    1
actual
0           124  123
1           109  138

This actually shows that including the “Home_Away” feature decreased the accuracy and precision of the model. This was surprising due to the idea that home court advantage usually acts as a boost to the home team. However, if we recall the graph of the predicted probability vs “Home_Away” relationship, it showed no relationship as there was an equal split of actual wins and actual losses of varying predicted probabilities in both home and away categories, which supports the outcome of the precision and accuracy scores.

Comments:

The low precision and accuracy scores were somewhat surprising, since I expected the model to perform much better due to the use of previous game stats to predict future performance. However, it performed a bit better than random predictions, which is known to be 50% since a team either wins or loses. But the one thing to note is that this model is truly predictive, as it does not use any total game stats to predict the outcome of a game. Some possible explanations as to why the performance is too low could be because the 3 game window (3 prior games) that we used is very limited and is not enough to capture a true idea of how well the team is performing. Another possibility could be due to the lack of contextual features such as opponent strength, rest days, injuries/roster changes, etc. The main thing to note is that the NBA is a high variance sport, as NBA games are noisy and upsets happen from time to time. Even Vegas betting models have an accuracy score of around 65% at best, which is considered relatively good. For that reason, the scores that I obtained are not necessarily horrible, but there is definitely room for improvement.

2. Logistic Regression Model
   
The same logistic regression model process (with max_iter=100) used in the data leakage version was reused but we replaced the training features with only the rolling average stats. After obtaining the predicted results and probabilities for each instance in the training set, I computed the accuracy score to get 0.5850202429149798 and a precision score of 0.5801526717557252. As for the confusion matrix, I got the following:

prediction    0    1
actual
0           137  110
1            95  152

This shows that the logistic regression model performed slightly better than the random forest classifier (both versions). However, this could be due to the hyperparameters that were set. For example, this logistic regression model has a max_iter parameter equal to 100, thus allowing 100 iterations of the optimization algorithm in order to find the optimal model parameters. With more iterations, there usually comes higher accuracy since the algorithm has more time to converge to the most minimal value of the cost function. In the case of the random forest classifier above, the hyperparameter was the window size, or the number of previous games that we use to compute the rolling average for. After testing out various values, I found that a window size of 13 gave the higher accuracy and precision score at around 0.58 (even though our whole objective was to look at the past three games as a reference to predict future game outcomes). However, I should highlight that it is possible that this hyperparameter value is completely random, as it just happens that looking back at the 13 previous games would result in a rolling average that would help the model produce the most accurate predictions.

___________________________________________________________________________________________

Neural Network Models

After creating these two basic models for both the data leakage version and the rolling averages version, I will now implement a neural network model for each version. The general structure of the neural network would consist of an input layer that takes the input features, runs it through 5 hidden layers (and a sigmoid function at the end), and obtains the output that classifies each instance as a win or loss. Since this is a binary classification task, I will use binary cross entropy as the loss function criterion that will help us classify the instances.

As for the layers of the neural network, there will be three linear layers, two ReLU layers (introduce non-linearity to the model), and an ending sigmoid function. The model takes an input of 10 input features (since each instance has 10 input features that will be used to classify the instance), and it will output a final decision that specifies if the instance is a win (1) or a loss (0). 

Leakage Neural Network

To begin, I created an instance of the Neural Network class that we just created, along with the training and test data splits, after having made sure that we normalized the data features so that they were all being measured on the same scale. Furthermore, I used an Adam optimizer to help us find the best parameters and decrease our loss/error.

___________________________________________________________________________________________

To train the model, I initially ran 100 epochs of training, where I obtained predictions from the x features of the training data, and I computed the loss between this prediction and the actual y label. During each epoch, I made sure to zero out the optimizer to clear space for new gradients, and I would then perform backpropagation in order to fine tune the weights/parameters of the neural network. Note that I made sure to append the loss obtained during each epoch into an array called losses, and I would eventually use this to create a line graph against the epoch number. Also, at every 10 epochs, I made the for loop return a status check showcasing the loss value at each epoch checkmark. By the end of the 100 epochs, I found that the loss was at 0.20273065567016602, with its line graph attached below.

However, the loss value was still relatively high, and I knew that if I increased the number of epochs, the model would have more time to refine its predictions and obtain a lower loss value. Therefore, I increased the loss to 1000 epochs and found the loss value of 0.0009382374119013548, with its corresponding graph attached below. Another possibility could be to increase the learning rate of the optimizer, as this will result in more rapid changes to the weights/parameters, thus decreasing the loss/error at a faster rate.

___________________________________________________________________________________________

After training the model, I began to test the data by turning off backpropagation using no_grad(), obtaining predictions using forward(), and computing the total loss. In general, we should want our loss to lie between 0.0 and 0.7, which shows that the model is learning properly.

Based on our initial training setup with 1000 epochs and an optimizer learning rate of 0.01, I ran the test data through the model with no back-propagation and obtained a test loss of 3.13. This possibly implies that the model’s predictions are very poor or that there is a data mismatch that causes instability. 

To reduce the loss, I decreased the optimizer learning rate to 0.001, which means that any updates to the weight parameters of the model will be smaller and less impactful (The optimizer takes very small, cautious steps. This avoids overshooting, but it can take many more epochs to reach a good solution). As a result, I obtained a test loss of 1.58, which is still relatively high compared to the desired range. 

Finally, I decreased my learning rate even more to 0.0001, and I obtained a test loss of 0.39, which shows that the model is actually learning and predicting much better than random guessing. 

The only downside of having such a low learning rate is that the training of the model will be very slow and would require more epochs to converge. Therefore, I decided to graph the number of epochs against the test loss results for different learning rates to see if I could find an optimal learning rate and number of epochs, thus finding a balance between the speed of convergence and the test set prediction accuracy. Setting the number of epochs to 100 as a starting point, it is evident that for the learning rates that were used (0.01, 0.001, and 0.0001), there was a large amount of fluctuation for all three learning rates. For the line representing the learning rate of 0.01, the loss starts at 0.587 but does not stabilize at any point in the 500 epochs, as it remained in the range of 0.49-0.64. This shows that the learning rate is too high for the dataset and model, and that the optimizer is trying to take big steps, overshooting the minimum repeatedly, thus causing the loss to bounce instead of steadily decreasing. As for the line representing 0.001, the loss starts at 0.66 but gradually stabilizes between 0.49-0.6. Even so, there still remained high oscillation and no clear down trend in the test loss. This shows that the learning steps are still too big for smooth convergence. Lastly, the line for 0.0001 starts at 0.71 and decreases over time, causing the loss to fluctuate between 0.53-0.63. Though the decreasing trend is much smoother, the training rate is much slower as well due to the small learning rate. Although this is beneficial in that it avoids overshooting, it would require a higher number of epochs in order to converge to a good minimum (good set of parameters/weights that would result in a good minimum test loss)

___________________________________________________________________________________________

(Note that in these initial models, I am not using the dataloader, so we are not feeding small batches of the dataset to the model. Instead, I am feeding the entire dataset at once into the model, which may crash the GPU if I am working with a large dataset. This is equivalent to batch gradient descent where I make the model compute gradients once per epoch of the entire dataset. This can make training slow to converge and possibly less stable. Also there is no shuffling or randomization in the training data, so the model might remember the order of the data and overfit to it. If I were to use a dataloader to provide the model with multiple small batches of data during each epoch, it would be much more memory efficient since only one batch is loaded into the GPU at a time, while also computing gradients for each batch and updating the weights many times per epoch. This also implements shuffling to the training data so that the model will not remember any patterns or order of the data, thus preventing overfitting. As a result of using dataloaders, it will result in faster convergence since more weight updates are performed during each epoch).

To that end, I implemented into the training loop a learning rate scheduler that halves the learning rate after every 100 epochs, while also making use of the dataloader in order to utilize the batching and shuffling tools in the training loop. Additionally, I implemented the calculation for the accuracy of the predictions for each epoch as another means of measuring the model’s performance. This time, I set the number of epochs to 500, and altered the learning rate to 0.0005. 

In the resulting loss vs epoch graph, both the training and test loss experience a major decrease in the loss value at the very start of the epochs, as they both start with a loss of 0.7 and rapidly decrease below 0.6. However, the test loss plateaus at around the 0.58 loss range, and it remains relatively steady for the remaining 480 epochs. As for the train loss, it decreases at a more gradual rate, as it continues decreasing as the number of epochs increase. By the time it reaches 500 epochs, it has a loss value of about 0.55. Looking at both loss curves, we can conclude that the model is learning and fitting to the training data better due to its lower loss value. The early plateau in the test loss serves as an indication that the model is starting to overfit since the model keeps improving on training data but the test loss does not decrease further. 

In the resulting accuracy vs epoch graph, the training accuracy increases from 50% to 80%, while the test accuracy increases from 50% to 78%, and stabilizes. Once again, we see that the training accuracy ends up being higher than the test accuracy over the span of the 500 epochs, as it is around the 150 epoch range where the disparity between the training and testing accuracy begins to grow. This is a slight indication of overfitting but is not severe enough to require changes to the model. Overall, the model generalizes well with a 78% test accuracy. 

Some general observations that can be made based on the models is that early rapid improvement in the loss for both the training and test curves shows that the model quickly learns general patterns. Also, with gradual slow improvement later one, this either indicates that the model capacity is limited or that the learning rate is very small, which very much could be the case. Last of all, the plateaus in the testing metrics show that increasing the number of epochs will not help improve the model performance, as other strategies must be implemented if improvements are desired.

Rolling Averages Neural Network

In order to implement the neural network model for the rolling averages data, I utilized a similar neural network structure to the one that I implemented for the data leakage model. After having computed the rolling averages for each game and splitting the data into training and test data, I began constructing the structure of the neural network by defining its init() and forward() functions. In this case, I am only using the rolling averages to make predictions on the game outcome, so there will only be 9 input features since I am not including the “Home/Away” attribute. Once again, I am using binary cross entropy loss as the criterion for computing the error between the predictions and the actual labels, and I am using the Adam optimizer to ensure that we can obtain the best parameters that result in the lowest loss.

___________________________________________________________________________________________

To train the model, I inputted the rolling average attributes (x values) into the model in order to obtain the corresponding y predictions. Thereafter, I inputted the y predictions and the actual y labels into the BCE loss criterion. I performed this training loop 1000 times (epochs), and I made sure I kept track of the loss result for each epoch in order to see how the loss varies over the span of the entire training process. Note that I made sure to use backpropagation using the backward() function so that the weight parameters would be updated during each epoch and so that better parameters can be obtained.

Upon the initial training attempt, I set the number of epochs to 1000 and the learning rate to 0.0001. With the training loop printing the current loss at every 10 epochs, it showed that the loss began at 0.6992089152336121, and by the end of the 1000 epochs, the loss resulted in 0.6142740845680237. 

Although the loss was in the range of 0.0 to 0.7, I still felt that it was somewhat high, and that the model could do better. But rather than changing the number of epochs during the training loop, I changed the learning rate to 0.001 instead, thus resulting in more impactful updates to the parameters/weights of the model. As I retrained the model, I found that the model began with a loss of 0.6992089152336121 and ended with a loss of 0.037341512739658356. This shows a major improvement in the model’s loss.

This brings to question: what learning rate would allow the model to produce the most optimal parameters, and therefore the lowest loss value?

To answer this question, I created a similar graph to the one for the data leakage model, where I plotted the number of epochs against the loss value for the learning rates of 0.01, 0.001, and 0.0001. In the code for this plot, I created individual models that each have a unique learning rate, and by using the training data given from the train dataloader to obtain predictions and to compute the loss value (over 100 epochs), I was able to obtain three different lines representing each unique model. The graph for this is attached below. The best performing learning rate was 0.01, which started with a loss of 0.6884 and by the end of the 100 epochs, it has a loss of about 0.0699. The next best learning rate was 0.001, which started with a loss of 0.6916 and ended with a loss of 0.7516, which signifies that the loss actually increased after training. The worst performing learning rate is 0.0001, which started with a loss of 0.676 and ended with a loss of 0.8324, also increasing its loss value after running 100 epochs. This is strange since it is supposed to be moving towards a lower loss value where the error between the prediction and the actual label is reduced. Some things to note include the fact that although the 0.01 learning rate had the lowest loss, this goes to show that it may be faster in terms of reaching the optimal parameters, but it is very unstable. 

However, it is highly possible that 100 epochs is not enough to allow the model to fully converge to a minimum loss value. To that end, I decided to try using 500 epochs and 1000 epochs to test if different outcomes may result. 

For the 500 epochs version, I found that the model with the 0.01 learning rate, it started with a loss of 0.6884 and ended with a loss of 0.0281. An observation that I had from its graph is that the line representing the 0.01 learning rate model was much more volatile and fluctuating compared to the rest, with a loss that spiked to 6.25 at some point in the training. For the model with the 0.001 learning rate, it started with a loss of 0.6916 and ended with a loss of 0.0236, which is a major improvement compared to the loss obtained from the training loop with only 100 epochs. From the graph, I noticed that the line representing the 0.001 learning rate model had a constantly decreasing trend, as it showed that as the number of epochs increased, the model would constantly trend towards a lower loss. Lastly, for the model with the 0.0001 learning rate, it started with a loss of 0.6767 and ended with a loss of 0.6389, making very little improvement over the span of the 500 epoch training loop. Looking at the graph, I noticed that the line representing the 0.0001 learning rate model remained relatively constant, with no major spikes but also no major shifts in loss.

For the 1000 epochs version, I found that the 0.01 learning rate model started with a loss of 0.6884, and after 1000 epochs, its loss ended up being 0.0006. However, when I analyzed the graph, I noticed several spikes in loss values throughout the span of the 1000 epochs, which could yield several possible explanations. For one, training with mini batches of 64 instances means that each batch is a slightly different sample of the data, which will cause wiggles/spikes due to the range of the batch. Another possible explanation is that if the learning rate is on the high side, then the optimizer might overshoot the minimum such that the loss will constantly fluctuate. Looking at the model with the 0.001 learning rate, it follows the same trend as shown in the 500 epoch training loop, where it has a constant but gradual downward trend towards a lower loss value. Its loss begins at 0.6916, and by the end of the 1000 epoch training loop, it has a loss of 0.005. Last of all, the model with the learning rate of 0.0001 starts with a loss of 0.6767 and ends with a loss of 0.4372. This model had the lowest change in loss, and just like in the 500 epoch training loop, the line representing this model remains stable with a very slight negative trend.

Additionally, I tried to implement another training loop along with a test loop that utilizes a learning rate scheduler, dataloaders to make use of the shuffle and batch features, and an accuracy metric to measure how accurate the model was. However, the results did not turn out as planned, as shown in the attached graph below. In the loss vs epoch graph, we see that the training loss and the test loss immediately diverge, thus showing a direct indication of overfitting since the training loss is decreasing while the test loss is increasing. This implies that the neural network is fitting to the training data too well, as it learns the training data patterns and consequently fails to generalize to new data. Looking at the accuracy vs epoch graph, both the test and training accuracy remain constant at 50%. 

___________________________________________________________________________________________

After training the model, I started to test the neural network in order to evaluate its performance on unseen data. I turned off backpropagation (since the model was previously trained with the most optimal parameters set) to prevent any further changes to the model weight, and I set the hyperparameters to 1000 for the number of epochs and 0.001 for the learning rate. To begin the testing phase, I computed the predictions on the test dataset (x features) using the forward() method and obtained the loss by using the BCE loss criterion on the actual test labels and the predictions from the model. Thereafter I checked the accuracy of the model by going through each instance in the test set, obtained the model’s prediction on the current instance, rounded the prediction based on whether it was over or under 0.5, and checked to see if the rounded prediction matched the actual label. It would do this for every instance in the test dataset, and when finished, it computes an accuracy score of correct predictions out of the total number of instances in the test set. In doing so, I obtained an accuracy score of 0.5384615384615384, which is just a little better than random guessing. This accuracy score is not as good compared to that obtained by the data leakage neural network, but we must take into account the fact that the data leakage model utilizes stats that are obtained only after the game is finished. This clearly explains why it would have a higher accuracy score, as the rolling averages model is a truly predictive model that uses stats based on previous games.

