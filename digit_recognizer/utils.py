from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg


def load_training_data():
    """
    Reads the traning data csv and loads into a pandas dataframe.
    Then separates the training data into imagev info and the correct label.
    :return:
    images - list of images pixels
    labels - list with the correct values of each image
    """
    # load the training set to a pandas dataframe
    labeled_images = pd.read_csv('train.csv')
    # Then we separate our images and labels for supervised learning.
    images = labeled_images.iloc[:, 1:]
    labels = labeled_images.iloc[:, :1]
    return images, labels


def visualise_image(train_images, image_index, train_labels, cmap='gray'):
    """
    Draws the selected index image.
    :param train_images: the list of training images set
    :param image_index: The index of the image we want to see
    :param train_labels: the labels of the columns in the training data
    :param cmap: The scale we want to show the image (e.g. gray, black and white, etc.)
    """
    # he image is currently one-dimension, we load it into a numpy array
    img = train_images.iloc[image_index].as_matrix()
    # and reshape it so that it is two-dimensional (28x28 pixels)
    img = img.reshape((28, 28))
    # gray scale image
    plt.imshow(img, cmap=cmap)
    # puts the correct label as the title
    plt.title(train_labels.iloc[image_index, 0])
    plt.show()


def create_results_file(results):
    """
    Creates a submission csv file in the format ImageId, Label
    :param results: the list of predicted labels for the test data
    """
    df = pd.DataFrame(results)
    df.index.name = 'ImageId'
    df.index += 1
    df.columns = ['Label']
    df.to_csv('results.csv', header=True)
