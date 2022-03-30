from base64 import encode
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize
from sklearn import tree
from keras.utils.np_utils import to_categorical 
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import ssl
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
ssl._create_default_https_context = ssl._create_unverified_context
#set initial values
#DATASET options are: "fashion_mnist, cifar100course, cifar100fine"

def load_data(data):
    if(data == "cifar100fine"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    if(data == "cifar100coarse"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")
    return(x_train, y_train, x_test, y_test)
    


def trainResNet(x_train, y_train, x_test, y_test, EPOCHS):

    model = keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3), classes=5, classifier_activation="softmax")

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(x_train, y_train, batch_size = 200, epochs = EPOCHS)
    model.evaluate(x_test, y_test)
    return model.get_weights(), history

def ResNetPreTrained(x_train, y_train, x_test, y_test, EPOCHS, WEIGHTS):
    print("Running ResNetPreTrained...")
    model = keras.applications.ResNet50(weights=None,
    input_shape=(32, 32, 3))
    if(WEIGHTS != None):
        model.set_weights(WEIGHTS)
    model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')
    history = model.fit(x_train, y_train, batch_size = 200, epochs = EPOCHS)
    print("test accuracy: {}".format(model.evaluate(x_test, y_test)))
    return history

def baselineClassifier(x_train, y_train, x_test, y_test):
    neigh = sklearn.neighbors.KNeighborsClassifier()
    neigh.fit(x_train, y_train)
    print("K-neighbours accuracy: {}".format(neigh.score(x_test, y_test)))
    return

def get_data(data, labels):
    x_train, y_train, x_test, y_test = load_data(data)
    y_train_test = []
    y_test_test = []
    x_train_test = []
    x_test_test = []
    for i in range(len(y_train)):
        if y_train[i] in labels:
            x_train_test.append(x_train[i].transpose().flatten().transpose())
            y_train_test.append(y_train[i].transpose())
    for i in range(len(y_test)):
        if y_test[i] in labels: 
            x_test_test.append(x_test[i].transpose().flatten().transpose())
            y_test_test.append(y_test[i].transpose())
    y_train_test =  np.array(y_train_test).ravel()
    y_test_test =  np.array(y_test_test).ravel()
    
    return x_train_test, y_train_test, x_test_test, y_test_test



def encodeoutput(y_train, y_test):
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)
    #print(y_test, len(y_test), y_test.shape, y_test.size)

    y_train, y_train_uniques = pd.factorize(y_train)
    trainOnehot = np.zeros((len(y_train), max(y_train)+1))
    trainOnehot[np.arange(len(y_train)),y_train] = 1
    y_test, y_test_uniques = pd.factorize(y_test)
    testOnehot = np.zeros((len(y_test), max(y_test)+1))
    testOnehot[np.arange(len(y_test)),y_test] = 1
    return trainOnehot, testOnehot

def saveweights(object):

 
    # Step 2
    with open('trainingweights.dictionary', 'wb') as config_dictionary_file:
 
    # Step 3
        pickle.dump(object, config_dictionary_file)
    return



def main():
    #define which labels we want to look at
    #aquatic animal labels (From aquatic animals and Fish)
    labels_fine = [36, 50, 65, 74, 80]
    #aquatic: [4, 30, 55, 72, 95]
    #small mammals: [36, 50, 65, 74, 80]
    #medium-sized mammals: [34, 63, 64, 66, 75]
    #chosen animal class labels
    #labels_coarse = [7, 8, 11, 12, 13, 15, 16]
    # all animal class labels [0, 1, 7, 8, 11, 12, 13, 15, 16]
    #labels_fine = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    #labels_coarse = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']
    #initialize data
    x_train, y_train, x_test, y_test = get_data("cifar100fine", labels_fine)
    baselineClassifier(x_train, y_train, x_test, y_test)

    x_train = np.array(x_train).reshape(len(x_train),32, 96)
    x_test = np.array(x_test).reshape((len(x_test)), 32, 96)

    #change y to be one-hot encoded
    y_test, y_train = encodeoutput(y_test, y_train)

    x_train = np.array(x_train).reshape(len(x_train),32, 32, 3)
    x_test = np.array(x_test).reshape((len(x_test)), 32, 32, 3)
    
    print("x_train shape: {} \n y_train shape: {} \n x_test shape: {} \n y_test shape: {} \n".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    #y_train, y_test = encodeoutput(y_train, y_test)
    print(y_train)

    train_weights, prehistory = trainResNet(x_train, y_train, x_test, y_test, 1)

    with open('prehistory.history', 'wb') as config_history_file:
        pickle.dump(object, config_history_file)
    
    #save the weights
    saveweights(train_weights)
    return
if __name__ == "__main__":
    main()