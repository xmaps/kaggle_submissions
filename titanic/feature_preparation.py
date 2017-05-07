import re
import operator

family_id_mapping = {}


def get_family_id(row):
    """A function to get the ID for a particular row"""
    # A dictionary mapping family name to ID
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family ID
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the ID in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum ID from the mapping, and add 1 to it if we don't have an ID
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]


def get_title(name):
    """A function to get the title from a name"""
    # Use a regular expression to search for a title
    # Titles always consist of capital and lowercase letters, and end with a period
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ""


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

    titanic_info["Fare"] = titanic_info["Fare"].fillna(titanic_info["Fare"].median())

    # We can also generate new features. Here are some ideas:
    # * The length of the name. This could pertain to how rich the person was, therefore their position on the Titanic.
    # * The total number of people in a family (SibSp + Parch).

    # Generating a familysize column
    titanic_info["FamilySize"] = titanic_info["SibSp"] + titanic_info["Parch"]

    # The .apply method generates a new series
    titanic_info["NameLength"] = titanic_info["Name"].apply(lambda x: len(x))

    # Get all of the titles, and print how often each one occurs
    titles = titanic_info["Name"].apply(get_title)

    # Map each title to an integer
    # Some titles are very rare, so they're compressed into the same codes as other titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                     "Mme": 8, "Dona": 9, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9,
                     "Capt": 7, "Ms": 2}
    for k, v in title_mapping.items():
        titles[titles == k] = v

    # Add in the title column
    titanic_info["Title"] = titles

    # We can also generate a feature that indicates which family passengers belong to.
    # Because survival was probably very dependent on your family and the people around you,
    # this has a good chance of being a helpful feature.

    # Get the family IDs with the apply method
    family_ids = titanic_info.apply(get_family_id, axis=1)

    # There are a lot of family IDs, so we'll compress all of the families with less than three members into one code
    family_ids[titanic_info["FamilySize"] < 3] = -1

    titanic_info["FamilyId"] = family_ids

    return titanic_info
