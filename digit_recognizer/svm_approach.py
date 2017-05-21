import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm


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


def visualise_image(train_images, image_index, cmap='gray'):
    """
    Draws the selected index image.
    :param train_images: the list of training images set
    :param image_index: The index of the image we want to see
    :return: 
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


def train_svm_model(train_images, test_images):
    """
    Trains the selected model with the data.
    :param train_images: the training set with a list of images
    :param test_images: the test set to check the model accuracy
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

images, labels = load_training_data()
# break our data into two sets, one for training and one for testing
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# train the svm model (commented because it was for testing purposes only)
#train_svm_model(train_images, test_images)

# simplify our images by making them true black and white.
# any pixel with a value simply becomes 1 and everything else remains 0
test_images[test_images > 0] = 1
train_images[train_images > 0] = 1

# train again with a simplified image
clf = train_svm_model(train_images, test_images)

# load test data to make predictions
test_data = pd.read_csv('test.csv')
# simplify the images on the test data
test_data[test_data > 0] = 1

# predict the test set values
results = clf.predict(test_data)

#print the results
create_results_file(results)

