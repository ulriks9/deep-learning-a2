from base64 import encode
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt
import sklearn
import ssl
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
ssl._create_default_https_context = ssl._create_unverified_context

def plot_data(preaccuracy, ptaccuracy, accuracy, prevalaccuracy, ptvalaccuracy, valaccuracy):
    plt.plot(preaccuracy, 'b')
    plt.plot(prevalaccuracy, 'g')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../preaccuracy.jpg')
    plt.clf()

    plt.plot(ptaccuracy, 'b')
    plt.plot(ptvalaccuracy, 'g')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../ptaccuracy.jpg')
    plt.clf()

    plt.plot(accuracy, 'b')
    plt.plot(valaccuracy, 'g')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../accuracy.jpg')
    plt.clf()
    
    plt.plot(ptvalaccuracy, 'b')
    plt.plot(valaccuracy, 'g')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.savefig(f'../both_accuracy.jpg')
    plt.clf()

def load_data10pre(labels):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_test = []
    y_train_test = []
    x_test_test = []
    y_test_test = []
    for i in range(len(y_train)):
        if y_train[i] in labels:
            x_train_test.append(x_train[i])
            y_train_test.append(y_train[i])
    for i in range(len(y_test)):
        if y_test[i] in labels: 
            x_test_test.append(x_test[i])
            y_test_test.append(y_test[i])

    x_train = np.array(x_train_test)
    y_train = np.array(y_train_test)
    x_test = np.array(x_test_test)
    y_test = np.array(y_test_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # convert from integers to floats
    train_norm = x_train.astype('float32')
    test_norm = x_test.astype('float32')
    # normalize to range 0-1
    x_train = train_norm / 255.0
    x_test = test_norm / 255.0

    return (x_train, y_train, x_test, y_test)

def load_data10post(labels):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_test = []
    y_train_test = []
    x_test_test = []
    y_test_test = []
    for i in range(len(y_train)):
        if y_train[i] in labels:
            x_train_test.append(x_train[i])
            #this minus 5 is the reason why a seperate loading function was used
            y_train_test.append(y_train[i]-5)
    for i in range(len(y_test)):
        if y_test[i] in labels: 
            x_test_test.append(x_test[i])
            y_test_test.append(y_test[i]-5)

    x_train = np.array(x_train_test)
    y_train = np.array(y_train_test)
    x_test = np.array(x_test_test)
    y_test = np.array(y_test_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # convert from integers to floats
    train_norm = x_train.astype('float32')
    test_norm = x_test.astype('float32')
    # normalize to range 0-1
    x_train = train_norm / 255.0
    x_test = test_norm / 255.0

    return (x_train, y_train, x_test, y_test)


def trainResNet(x_train, y_train, x_test, y_test, EPOCHS):

    model = keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3), classes=5, classifier_activation="softmax")

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    print(model.summary())
    history = model.fit(x_train, y_train, batch_size = 200, epochs = EPOCHS, validation_data=(x_test, y_test))
    return model.get_weights(), history


def baselineClassifier(x_train, y_train, x_test, y_test):
    neigh = sklearn.neighbors.KNeighborsClassifier()
    neigh.fit(x_train, y_train)
    print("K-neighbours accuracy: {}".format(neigh.score(x_test, y_test)))
    return

def baseline_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    labels = [0,1,2,3,4]
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

def ResNetPreTrained(x_train, y_train, x_test, y_test, EPOCHS, WEIGHTS):
    
    model = keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3), classes = 5)
    if(WEIGHTS != None):
        print("Running ResNetPreTrained with pretrained weights...")
        model.set_weights(WEIGHTS)
    else:
        print("Running ResNetPreTrained with random weights...")
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(x_train, y_train, batch_size = 200, epochs = EPOCHS, validation_data=(x_test, y_test))
    #print("test accuracy: {}".format(model.evaluate(x_test, y_test)))
    return history



def main():
    ###Baseline###
    x_train, y_train, x_test, y_test = baseline_data()
    baselineClassifier(x_train, y_train, x_test, y_test)

    ###Pretraining###
    Epochs = 15
    x_train, y_train, x_test, y_test = load_data10pre([0,1,2,3,4])
    print("x_train shape: {} \n y_train shape: {} \n x_test shape: {} \n y_test shape: {} \n".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    train_weights, prehistory = trainResNet(x_train, y_train, x_test, y_test, Epochs)
    train_weights2, prehistory2 = trainResNet(x_train, y_train, x_test, y_test, Epochs)

    ###SecondDataset###
    x_trainc, y_trainc, x_testc, y_testc = load_data10post([5,6,7,8,9])

    pthistory = ResNetPreTrained(x_trainc, y_trainc, x_testc, y_testc, Epochs, train_weights)
    history = ResNetPreTrained(x_trainc, y_trainc, x_testc, y_testc, Epochs, None)

    pthistory2 = ResNetPreTrained(x_trainc, y_trainc, x_testc, y_testc, Epochs, train_weights2)
    history2 = ResNetPreTrained(x_trainc, y_trainc, x_testc, y_testc, Epochs, None)
    
    preaccuracy = np.add(prehistory.history["accuracy"], prehistory2.history["accuracy"]) / 2
    prevalaccuracy = np.add(prehistory.history["val_accuracy"], prehistory2.history["val_accuracy"]) / 2
    ptaccuracy= np.add(pthistory.history["accuracy"], pthistory2.history["accuracy"]) / 2
    ptvalaccuracy = np.add(pthistory.history["val_accuracy"], pthistory2.history["val_accuracy"]) / 2
    accuracy = np.add(history.history["accuracy"], history2.history["accuracy"]) / 2
    valaccuracy = np.add(history.history["val_accuracy"], history2.history["val_accuracy"]) / 2

    plot_data(preaccuracy, ptaccuracy, accuracy, prevalaccuracy, ptvalaccuracy, valaccuracy)
    
    
    return

if __name__ == "__main__":
    main()