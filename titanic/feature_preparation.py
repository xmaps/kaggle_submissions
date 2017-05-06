
def prepare_features(titanic_info):
    """
    In order for the algorithms to learn we need to prepare 
    the features (e.g. make them all integers)
    and come up with new features.
    
    :return: the titanic data set with the correct features. 
    """
    # The titanic variable is available here
    titanic_info["Age"] = titanic_info["Age"].fillna(titanic_info["Age"].median())

    # Replace all the occurrences of male with the number 0
    titanic_info.loc[titanic_info["Sex"] == "male", "Sex"] = 0

    # Replace all the occurrences of female with the number 1
    titanic_info.loc[titanic_info["Sex"] == "female", "Sex"] = 1

    # The most common embarkation port is S, so let's assume everyone who's missing an embarkation port got on there.
    titanic_info["Embarked"] = titanic_info["Embarked"].fillna("S")

    # We'll assign the code 0 to S, 1 to C, and 2 to Q.
    titanic_info.loc[titanic_info["Embarked"] == "S", "Embarked"] = 0
    titanic_info.loc[titanic_info["Embarked"] == "C", "Embarked"] = 1
    titanic_info.loc[titanic_info["Embarked"] == "Q", "Embarked"] = 2

    return titanic_info
