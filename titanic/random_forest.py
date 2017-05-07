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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import KFold

from feature_preparation import prepare_features

# We can use the pandas library in Python to read in the CSV file
# This creates a pandas dataframe and assigns it to the titanic variable
titanic = pd.read_csv("train.csv")

# prepare all the necessary features
titanic = prepare_features(titanic)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
              "FamilySize", "Title", "FamilyId", "NameLength"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform them from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores
# Do you see how "Pclass", "Sex", "Title", and "Fare" are the best features?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

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

# Pick only the four best features
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

# Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

