import re
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


#https://www.kaggle.com/valiantkevin/multinomialnb-spam-detection

stemmer= SnowballStemmer('english')   #to remove prefixes, suffixes etc. 
stop=set(stopwords.words('english'))   # “the”, “a”, “an”, “in”...
print(stop)
def preprocess(words):
    current_words=""
    for word in words:
        if word.lower() not in stop: 
            updated_word=stemmer.stem(word)
            current_words += updated_word.lower() + " "
    return current_words

x_raw = [ ["This", "is", "the", "first","review"],
            ["This", "is", "an", "awesome", "movie"],
            ["Did", "not", "like", "the", "story"],
            ["Very", "boring", "movie", "film"],
            ["movie", "is", "was", "movies", "film.", "didn't"]]
y_train = ["positive","positive", "negative","negative","negative"]

x_train = [preprocess(review) for review in x_raw]

print(x_train)

cv=CountVectorizer() #transform texts into vectors
x_train_df=cv.fit_transform(x_train) 

clf=MultinomialNB()
clf.fit(x_train_df,y_train)
prediction=clf.predict(x_train_df)

conf_mat=confusion_matrix(y_train, prediction)
print(conf_mat)
print("Accuracy:"+str(accuracy_score(y_train,prediction)))