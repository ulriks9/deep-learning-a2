
from asyncio.windows_events import NULL
from tensorflow.keras.utils import text_dataset_from_directory
from tensorflow.data import AUTOTUNE
import tensorflow as tf
import os
import nlpaug.augmenter.word as naw

DATA_PATH_READ = "aclImdb"
DATA_PATH_WRITE = "augmented_acllmdb"
LANGUAGES = ["de","fr","it","ru"]

def augmentData():
    train_ds = text_dataset_from_directory(
        DATA_PATH_READ  + '/train',
        batch_size = None
        ,shuffle=False)

    test_ds = text_dataset_from_directory(
        DATA_PATH_READ  + '/test',
        batch_size = None
        ,shuffle=False)


    try:
        os.makedirs(DATA_PATH_WRITE + '/train/pos') 
        os.makedirs(DATA_PATH_WRITE + '/test/pos') 
        os.makedirs(DATA_PATH_WRITE + '/train/neg') 
        os.makedirs(DATA_PATH_WRITE + '/test/neg') 
    except FileExistsError:
        pass


    for language in LANGUAGES:
        print("Starting to translate in "+ language)
        
        languageModel = getModel(language)

        positives = 0
        negatives = 0

        #training data
        for text, label in train_ds:
            review = str(text.numpy())
            if label.numpy() ==1:
                positives +=1
                label = 'pos'
                path = DATA_PATH_WRITE +'/train/' + label + '/' +str(positives)
            else:
                negatives +=1
                label = 'neg'
                path = DATA_PATH_WRITE +'/train/' + label + '/' +str(negatives)

            back_translated = languageModel.augment(review)

            with open(path + '_back-translated_' + language +".txt", 'w', encoding="utf-8") as f:
                f.write(back_translated)
        
        positives = 0
        negatives = 0
        #testing data
        for text, label in test_ds:
            review = str(text.numpy())
            if label.numpy() ==1:
                positives +=1
                label = 'pos'
                path = DATA_PATH_WRITE +'/test/' + label + '/' +str(positives)
            else:
                negatives +=1
                label = 'neg'
                path = DATA_PATH_WRITE +'/test/' + label + '/' +str(negatives)

            back_translated = languageModel.augment(review)

            with open(path + '_back-translated_' + language +".txt", 'w', encoding="utf-8") as f:
                f.write(back_translated)
                
def getModel(language):
    if language == "de":
        languageModel = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de',
        to_model_name='Helsinki-NLP/opus-mt-de-en')
    elif language == "fr":
        languageModel = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-fr',
        to_model_name='Helsinki-NLP/opus-mt-fr-en')
    elif language == "it":
        languageModel = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-it',
        to_model_name='Helsinki-NLP/opus-mt-it-en')
    elif language == "ru":
        languageModel = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-ru',
        to_model_name='Helsinki-NLP/opus-mt-ru-en')
    else:
        print("Error" + language + "-model not implemented")
        exit(0)
    return languageModel

if __name__ == '__main__':
    augmentData()