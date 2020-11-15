from tweet_score import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

review_train = pd.read_csv("amazon-review-scraper-labelled1.csv",encoding="ISO-8859-1")
review_df=pd.read_csv("review_with_product.csv")

print(review_train.head())

review_train=review_train.fillna('')
review_df=review_df.fillna('')

#review_train.describe(include=['O'])

#review_train.describe()

print("\n\n no of null values in each column \n",review_train.isnull().sum())

print("\n\n no of null values in each column \n",review_df.isnull().sum())

def pre_process(text):

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

textFeatures = review_train['content'].copy()
textFeatures = textFeatures.apply(pre_process)
print(textFeatures)

vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)

op_textFeatures = review_df['review'].copy()
op_textFeatures = op_textFeatures.apply(pre_process)
op_features = vectorizer.transform(op_textFeatures)

mnb = MultinomialNB(alpha=0.2)
mnb.fit(features,review_train['label'])
prediction = mnb.predict(op_features)
print("\nMNB Prediction:\n",prediction)

features_train, features_test, labels_train, labels_test = train_test_split(features, review_train['label'], test_size=0.3, random_state=111)

mnb1 = MultinomialNB(alpha=0.2)
mnb1.fit(features_train,labels_train)

prediction1 = mnb1.predict(features_test)
print("multinomial naive bayes",accuracy_score(labels_test,prediction1))
#print(confusion_matrix(labels_test, prediction1))
#print(classification_report(labels_test, prediction1))

review_df['label']=prediction
print(review_df)

review_df.label.describe(include=['O'])
review_result=review_df.label.max()
print("Most of the reviews are : ",review_result)

