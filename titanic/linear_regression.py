# Import Numpy
import numpy as np
# Import Pandas
import pandas as pd
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross-validation
from sklearn.cross_validation import KFold

from feature_preparation import prepare_features

# We can use the pandas library in Python to read in the CSV file
# This creates a pandas dataframe and assigns it to the titanic variable
titanic = pd.read_csv("train.csv")

# Print the first five rows of the dataframe
print(titanic.head(5))

print(titanic.describe())

# prepare all the necessary features
titanic = prepare_features(titanic)

'''
Here are the next steps:
    * Combine the first two parts, train a model, and make predictions on the third.    
    * Combine the first and third parts, train a model, and make predictions on the second.
    * Combine the second and third parts, train a model, and make predictions on the first.
This way, we generate predictions for the entire data set without ever evaluating accuracy on the 
same data we use to train our model.
'''

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()

# Generate cross-validation folds for the titanic data set
# It returns the row indices corresponding to train and test
# We set random_state to ensure we get the same splits every time we run this
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using to train the algorithm
    # Note how we only take the rows in the train folds
    train_predictors = (titanic[predictors].iloc[train, :])
    # The target we're using to train the algorithm
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)

# The predictions are in three separate NumPy arrays
# Concatenate them into a single array
# We concatenate them on axis 0, because they only have one axis
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (the only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)
