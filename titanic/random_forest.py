# Decision trees have a major flaw—they overfit to the training data.
# Because we build up a very "deep" decision tree in terms of splits,
# we end up with a lot of rules that are specific to the quirks of the training data,
# and not generalizable to new data sets.
#
# This is where the random forest algorithm can help.
# With random forests, we build hundreds of trees with slightly randomized input data,
# and slightly randomized split points. Each tree in a random forest gets a
# random subset of the overall training data. The algorithm performs each
# split point in each tree on a random subset of the potential columns
# to split on. By averaging the predictions of all of the trees,
# we get a stronger overall prediction and minimize overfitting.
# There's still more work you can do in feature engineering:
#
# Try using features related to the cabins.
# See if any family size features might help. Do the number of women in a family make the entire family more likely to
#  survive?
# Does the national origin of the passenger's name have anything to do with survival?
# There's also a lot more we can do on the algorithm side:
#
# Try the random forest classifier in the ensemble.
# A support vector machine might work well with this data.
# We could try neural networks.
# Boosting with a different base classifier might work better.
# And with ensembling methods:
#
# Could majority voting be a better ensembling method than averaging probabilities?

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import cross_validation
#from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

from feature_preparation import prepare_features

# We can use the pandas library in Python to read in the CSV file
# This creates a pandas dataframe and assigns it to the titanic variable
titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
# prepare all the necessary features
titanic = prepare_features(titanic)
titanic_test = prepare_features(titanic_test)

#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId",
# "NameLength"]

# Perform feature selection
#selector = SelectKBest(f_classif, k=5)
#selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform them from p-values into scores
#scores = -np.log10(selector.pvalues_)

# Plot the scores
# Do you see how "Pclass", "Sex", "Title", and "Fare" are the best features?
#plt.bar(range(len(predictors)), scores)
#plt.xticks(range(len(predictors)), predictors, rotation='vertical')
#plt.show()

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends
# (the bottom points of the tree)
#
# The first (and easiest) thing we can do to improve the accuracy of the random forest is to increase the number of
# trees we're using. Training more trees will take more time, but because we're averaging many predictions we made on
# different subsets of the data, having more trees will greatly increase accuracy (up to a point).
# We can also tweak the min_samples_split and min_samples_leaf variables to reduce overfitting.

# The algorithms we want to ensemble
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

predictions = predictions.astype(int)

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)
