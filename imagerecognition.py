import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#set initial values
#DATASET options are: "fashion_mnist, cifar100course, cifar100fine"
DATASET = "fashion_mnist"
TEACHER = "ResNet50"

def load_data(data):
    if(data == "fashion_mnist"):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        print("Fashion MNIST data loaded")
    
    if(data == "handwriting"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if(data == "cifar100fine"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    if(data == "cifar100coarse"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")
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
    labels_fine = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    labels_coarse = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']
    #initialize data
    #x_train, y_train, x_test, y_test = load_data(DATASET)
    x_train, y_train, x_test, y_test = load_data("cifar100coarse")
    i = 4538
    print(labels_coarse[int(y_train[i])])
    plt.imshow(x_train[i])
    plt.show()
    #unpickle(r'C:/Users/wopke/Documents/Rug/CifarDataset/cifar-10-python.tar.gz')

    #RunResNet(x_train, y_train, x_test, y_test, 1, None)
    #trainResNetHandwriting()
    
if __name__ == "__main__":
    main()