import os
import shutil
import numpy as np
from keras.utils import text_dataset_from_directory, get_file
from tensorflow.data import AUTOTUNE

class DataSets:
    def __init__(self):
        self.directory = "datasets/"

    # Downloads necessary datasets
    # 'IMDB' for the IMDB dataset, 'MNIST' for the MNIST dataset, 'all' for both
    def download(self, dataset='all'):
        if dataset == 'IMDB' or dataset == 'all':
            url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

            if os.path.exists('datasets'):
                print("INFO: Folder {} already exists, will not create folder.".format(self.directory))
            else:
                os.mkdir("datasets")

            dataset = get_file(
                'aclImdb_v1.tar.gz', url,
                untar=True, cache_dir=self.directory,
                cache_subdir='')

            dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
            train_dir = os.path.join(dataset_dir, 'train')
            remove_dir = os.path.join(train_dir, 'unsup')

            shutil.rmtree(remove_dir)

    # Returns train, val, and test sets
    # Test set cointains remaining split
    # Does not work well with low length and batch_size for some reason
    def IMDB(self, length=50000, train_split=0.60, val_split=0.30, batch_size=1):
        auto_tuner = AUTOTUNE
        DATA_PATH = "datasets/" + "aclImdb"

        # Load data from directory
        train = text_dataset_from_directory(
            DATA_PATH + '/train',
            batch_size=None)

        test = text_dataset_from_directory(
            DATA_PATH + '/test',
            batch_size=None)
            
        data = train.concatenate(test)

        # Split data into train, val, and test sets
        length = data.cardinality().numpy()
        train = data.take(int(length * train_split))
        val = data.take(int(length * train_split + length * val_split))
        val = val.skip(int(length * train_split))
        test = data.skip(int(length * train_split + length * val_split))

        # Batching data
        train = train.batch(batch_size)
        val = val.batch(batch_size)
        test = test.batch(batch_size)

        # Enable prefectching of samples during training
        train = train.cache().prefetch(buffer_size=auto_tuner)
        val = val.cache().prefetch(buffer_size=auto_tuner)
        test = test.cache().prefetch(buffer_size=auto_tuner)

        length = train.cardinality().numpy() + val.cardinality().numpy() + test.cardinality().numpy()

        print("\nINFO: Number of batches in full dataset: {}".format(length))
        print("INFO: Number of batches in training set: {}".format(train.cardinality().numpy()))
        print("INFO: Number of batches in validation set: {}".format(val.cardinality().numpy()))
        print("INFO: Number of batches in testing set: {}\n".format(test.cardinality().numpy()))

        return train, val, test

    # Load the IMDB dataset
    def std_IMDB(self, length=None, validation_split=0.25, batch_size=1):
        auto_tuner = AUTOTUNE
        DATA_PATH = "datasets/" + "aclImdb"
        SEED = 1

        train = text_dataset_from_directory(
            DATA_PATH + '/train',
            batch_size=batch_size,
            validation_split=validation_split,
            subset='training',
            seed=SEED)

        val = text_dataset_from_directory(
            DATA_PATH + '/train',
            batch_size=batch_size,
            validation_split=validation_split,
            subset='validation',
            seed=SEED)   

        test = text_dataset_from_directory(DATA_PATH + '/test', batch_size=batch_size)

        if length != None:
            train_length = int(length * (1 - validation_split))
            val_length = int(length * validation_split)

            train = train.take(train_length)
            val = val.take(val_length)

        train = train.cache().prefetch(buffer_size=auto_tuner)
        val = val.cache().prefetch(buffer_size=auto_tuner)
        test = test.cache().prefetch(buffer_size=auto_tuner)

        print("\nINFO: Number of batches in training set: {}".format(train.cardinality().numpy()))
        print("INFO: Number of batches in validation set: {}".format(val.cardinality().numpy()))
        print("INFO: Number of batches in testing set: {}".format(test.cardinality().numpy()))

        return train, val, test

    # Loads IMDB dataset
    # To download set download=True
    # TO-DO: Figure out relative paths
    # def IMDB(self, download=False, train_split=60, np=True):
    #     data = tfds.load(
    #         name="imdb_reviews", 
    #         split=('train[:{}%]'.format(train_split), 'train[{}%:]'.format(train_split), 'test'),
    #         as_supervised=True,
    #         download=download,
    #         batch_size=-1)

    #     train, val, test = tfds.as_numpy(data)
    #     print('INFO: Length of training set: {}'.format(len(train[0])))
    #     print('INFO: Length of validation set: {}'.format(len(val[0])))
    #     print('INFO: Length of testing set: {}'.format(len(test[0])))
        
    #     # train, val, test are tuples with format (sample, label)
    #     return train, val, test