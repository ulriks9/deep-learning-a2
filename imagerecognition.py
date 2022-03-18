import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import matplotlib
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#set initial values
DATASET = "fashion_mnist"
TEACHER = "ResNet50"

def load_data(data):
    if(data == "fashion_mnist"):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        print("Fashion MNIST data loaded")
    
    if(data == "handwriting"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if(data == "cifar100"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    return(x_train, y_train, x_test, y_test)
    

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def trainResNetHandwriting():
    data = "handwriting"
    x_train, y_train, x_test, y_test = load_data(data)

    model = keras.applications.ResNet50(weights=None,
    input_shape=(28, 28, 1), classes=10)
    history = model.fit(x_train, y_train,
          batch_size=128, epochs=20,
          verbose=2,
          validation_data=(x_test, y_test))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')




    return

def RunResNet(x_train, y_train, x_test, y_test, epochs, pre_trained):
    if(pre_trained != None):
        model = pre_trained
    else:
        model = keras.applications.ResNet50(weights=None)

def baselineClassifier(x_train, y_train, x_test, y_test):
    



    print("works")
    
    return
    

def main():
    print("hello world")
    #initialize data
    #x_train, y_train, x_test, y_test = load_data(DATASET)
    x_train, y_train, x_test, y_test = load_data("cifar100")
    print(x_train[1])
    plt.imshow(x_train[1])
    #unpickle(r'C:/Users/wopke/Documents/Rug/CifarDataset/cifar-10-python.tar.gz')

    #RunResNet(x_train, y_train, x_test, y_test, 1, None)
    #trainResNetHandwriting()
    
if __name__ == "__main__":
    main()