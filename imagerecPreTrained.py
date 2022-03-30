import numpy as np
import pickle
import sklearn
import tensorflow as tf
import keras
import matplotlib as plt
import pandas as pd
from keras.utils.np_utils import to_categorical 
from skimage.transform import resize
from tensorflow import keras

def load_data(data):
    if(data == "cifar100fine"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    if(data == "cifar100coarse"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")
    return(x_train, y_train, x_test, y_test)

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


def ResNetPreTrained(x_train, y_train, x_test, y_test, EPOCHS, WEIGHTS):
    print("Running ResNetPreTrained...")
    model = keras.applications.ResNet50(weights=None,
    input_shape=(32, 32, 3), classes = 5)
    if(WEIGHTS != None):
        model.set_weights(WEIGHTS)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(x_train, y_train, batch_size = 200, epochs = EPOCHS)
    print("test accuracy: {}".format(model.evaluate(x_test, y_test)))
    return history


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


def loadweights():
    with open('trainingweights.dictionary', 'rb') as config_dictionary_file:
        weights = pickle.load(config_dictionary_file)
    return weights

def main():
    #labels_fine = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    labels_coarse = [34, 63, 64, 66, 75]

    ###TESTING###
    x_trainc, y_trainc, x_testc, y_testc = get_data("cifar100fine", labels_coarse)

    x_trainc = np.array(x_trainc).reshape(len(x_trainc),32, 32, 3)
    x_testc = np.array(x_testc).reshape((len(x_testc)), 32, 32, 3)

    y_trainc, y_testc = encodeoutput(y_trainc, y_testc)
    
    #train_weights = loadweights()
    print("x_trainc shape: {} \n y_trainc shape: {} \n x_testc shape: {} \n y_testc shape: {} \n".format(x_trainc.shape, y_trainc.shape, x_testc.shape, y_testc.shape))
    EPOCHS = 8
    #pthistory = ResNetPreTrained(x_trainc, y_trainc, x_testc, y_testc, EPOCHS, train_weights)
    history = ResNetPreTrained(x_trainc, y_trainc, x_testc, y_testc, EPOCHS, None)
    with open('history.history', 'wb') as config_history_file:
        pickle.dump(object, config_history_file)


    

    return
        
if __name__ == "__main__":
    main()