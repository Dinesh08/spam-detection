from twitter_analysis import *
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

tweet_train = pd.read_csv("train.csv")
print(tweet_train.head())
tweet_train=tweet_train.drop(['location'],axis=1)
print(tweet_train.head())
print(tweet_train.describe(include=['O']))
print(tweet_train.describe())
print("\n\n no of null values in each column \n",tweet_train.isnull().sum())
tweet_train.dropna(subset=['actions','followers'],inplace=True)
values={'following':0}
tweet_train.fillna(value=values,inplace=True)
print("\n\n no of null values in each column \n",tweet_train.isnull().sum())

def pre_process(text):

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

textFeatures = tweet_train['Tweet'].copy()
textFeatures = textFeatures.apply(pre_process)
print(textFeatures)

vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)

op_textFeatures = tweet_df['Tweet'].copy()
op_textFeatures = op_textFeatures.apply(pre_process)
op_features = vectorizer.transform(op_textFeatures)

mnb = MultinomialNB(alpha=0.2)
mnb.fit(features,tweet_train['Type'])
prediction = mnb.predict(op_features)
print("\nMNB Prediction:\n",prediction)

svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(features,tweet_train['Type'])
prediction = svc.predict(op_features)
print("\nSVC Prediction\n",prediction)

features_train, features_test, labels_train, labels_test = train_test_split(features, tweet_train['Type'], test_size=0.3, random_state=111)

mnb1 = MultinomialNB(alpha=0.2)
mnb1.fit(features_train,labels_train)

prediction1 = mnb1.predict(features_test)
print("multinomial naive bayes",accuracy_score(labels_test,prediction1))
print(confusion_matrix(labels_test, prediction1))
print(classification_report(labels_test, prediction1))

svc1 = SVC(kernel='sigmoid', gamma=1.0)
svc1.fit(features_train,labels_train)

prediction2 = svc1.predict(features_test)
print("svc ",accuracy_score(labels_test,prediction2))
print(confusion_matrix(labels_test, prediction2))
print(classification_report(labels_test, prediction2))

objects = ("multinomialNB","svc")
y_pos = np.arange(len(objects))
performance = [accuracy_score(labels_test,prediction1),accuracy_score(labels_test,prediction2)]
#performance_test=[NBTest,RFTest,TreeTest,pred_test]
plt.figure(figsize=(3,3))
plt.bar(objects, performance,width=0.2,align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Performance of Models')
plt.show()

tweet_df['Type']=prediction
print(tweet_df)

print(tweet_df.Type.describe(include=['O']))

tweet_result=tweet_df.Type.max()
print("Most of the tweets are : ",tweet_result)
