import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

from feature_preparation import prepare_features

# We can use the pandas library in Python to read in the CSV file
# This creates a pandas dataframe and assigns it to the titanic variable
titanic = pd.read_csv("train.csv")

# prepare all the necessary features
titanic = prepare_features(titanic)

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm
alg = LogisticRegression(random_state=1)

# Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

titanic_test = pd.read_csv("test.csv")

titanic_test = prepare_features(titanic_test)

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the data set
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })


