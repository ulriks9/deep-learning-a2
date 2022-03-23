import re
from turtle import update
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from load_data import *


#https://www.kaggle.com/valiantkevin/multinomialnb-spam-detection

stemmer= SnowballStemmer('english')   #to remove prefixes, suffixes etc. 
stop=set(stopwords.words('english'))   # “the”, “a”, “an”, “in”...

def preprocess_review(words):
    current_words=""
    words  = re.sub('[^0-9a-zA-Z]+', ' ', words)
    words = words.split(" ")
    for word in words:
        if word.lower() not in stop: 
            updated_word=stemmer.stem(word)
            current_words += updated_word.lower() + " "
    return current_words

def unpack_dataframe(ds):
    x = [preprocess_review(str(text_batch.numpy())) for text_batch, _ in ds]
    y = ["positive" if label_batch.numpy() == 1 else "negative" for _ , label_batch in ds ]
    return x, y

train_ds, test_ds = load_data()

x_train, y_train = unpack_dataframe(train_ds)
x_test, y_test = unpack_dataframe(test_ds)

    
cv=CountVectorizer()
x_train = cv.fit_transform(x_train) 
import copy
x_pre_transform = copy.deepcopy(x_test)
x_test = cv.transform(x_test) 

clf=MultinomialNB()
clf.fit(x_train,y_train)

prediction=clf.predict(x_train)
conf_mat=confusion_matrix(y_train, prediction)
print(conf_mat)
print("Training accuracy:"+str(accuracy_score(y_train,prediction)))

prediction=clf.predict(x_test)
conf_mat=confusion_matrix(y_test, prediction)
print(conf_mat)
print("Test Accuracy:"+str(accuracy_score(y_test,prediction)))
