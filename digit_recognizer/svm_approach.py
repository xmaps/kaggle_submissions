

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

from digit_recognizer.utils import create_results_file, load_training_data

FLAGS = None


def train_svm_model(train_images, test_images, train_labels, test_labels):
    """
    Trains the selected model with the data.
    :param train_images: the training set with a list of images
    :param test_images: the test set to check the model accuracy
    :param train_labels: the labels of the columns in the training data
    :param test_labels: the labels of the columns in the test data
    :return: the trained model
    """
    # use the sklearn.svm module to create a vector classifier.
    clf = svm.SVC()
    # pass our training images and labels to the classifier's fit method, which trains our model.
    clf.fit(train_images, train_labels.values.ravel())
    # score method to see how well we trained our model. return a float between 0-1 indicating our accuracy.
    model_score = clf.score(test_images, test_labels)
    print(model_score)
    return clf


def main():
    images, labels = load_training_data()
    # break our data into two sets, one for training and one for testing
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels,
                                                                            train_size=0.8,
                                                                            random_state=0)

    # train the svm model (commented because it was for testing purposes only)
    #train_svm_model(train_images, test_images, train_labels)

    # simplify our images by making them true black and white.
    # any pixel with a value simply becomes 1 and everything else remains 0
    test_images[test_images > 0] = 1
    train_images[train_images > 0] = 1

    # train again with a simplified image
    clf = train_svm_model(train_images, test_images, train_labels, test_labels)

    # load test data to make predictions
    test_data = pd.read_csv('test.csv')
    # simplify the images on the test data
    test_data[test_data > 0] = 1

    # predict the test set values
    results = clf.predict(test_data)

    # print the results
    create_results_file(results)


if __name__ == '__main__':
    main()
