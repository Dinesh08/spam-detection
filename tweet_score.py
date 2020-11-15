from tweet_spam import *
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

spam_words=["Ad","believe","Bargain","Amazed","Casino","Click","Collect","Cost","Coupons","Profits","Prize","Lucky","Congratulations","Credit","offers","Deal","Discount","Debt","Act","Additional","Billion","Bonus","Boss","Buy","free","hesitate","Extra","Double","Earn","Extra","Expire","Exciting","excited","Fantastic","gift","Get","Grab","Guarantee","Hurry","Increase","Junk","Limited","Link","Lowest","Luxury","Million","Make","miss","Money","marketing","Offer","opportunity","Order","Open","bucks","Please","Promise","Purchase","Rates","Refinance","Refund","Request","Risk","Spin","Sales","Satisfaction","Save","Score","Serious","Spam","Success","Supplies","Take","Terms","Traffic","Trial","Unlimited","Urgent","values","Weight","Win","Winner","waiting","lifetime","instant","delete","subscribe","unsubscribe"]

spam_sentence=["Double your","Earn extra cash","Earn per week","Expect to earn","Extra income","Home based","Home employment","Homebased business","Income from home","Make","Make money","click this link","Money making","Online biz opportunity","Online degree","While you sleep","Work at home","Work from home","One hundred percent free","Credit card offers","Lower interest rate","Month trial offer","you will not believe it","% off","percent off","no middleman","directly from company","limited period offer","you are a winner","you're a winner","free trial","apply","order","order now","new customers only","get it now","get started now","for just","great offer","give away","giving away","limited period offer","Incredible deal","Internet market","Internet marketing","It’s effective","Lower interest rate","Lowest interest rate","Lowest insurance rates","Lowest price","Luxury"]

spam_words=[x.lower() for x in spam_words]
spam_words = list(dict.fromkeys(spam_words))
print(spam_words)
print(len(spam_words))

spam_sentence=[x.lower() for x in spam_sentence]
spam_sentence = list(dict.fromkeys(spam_sentence))
print(spam_sentence)
print(len(spam_sentence))

spam_words = [word for word in spam_words if word.lower() not in stopwords.words('english')]

spam_sentence=[word for word in spam_sentence if word.lower() not in stopwords.words('english')]

print(spam_words)

print(spam_sentence)

stemmer = SnowballStemmer("english")
spam_words=[stemmer.stem(i) for i in spam_words]

print(spam_words)

#tweet="Hurry Hurry Hurry! Click this link to grab the offer! and be Amazed! and get a bonus of 1 lakh"

#tweet = tweet.translate(str.maketrans('', '', string.punctuation))
#tweet = [word for word in tweet.split() if word.lower() not in stopwords.words('english')]
#words = []
#for i in tweet:
#    stemmer = SnowballStemmer("english")
#    words.append(stemmer.stem(i))

#print(words)

#words = list(dict.fromkeys(words))

##print(words)

#print(len(words))

#count=0
#for i in words:
#    if i in spam_words:
#        count+=1
#print(count)

#score=(count/len(words))
#print(score)

#print(tweet_df)

tweet=tweet_df.copy()

print(tweet)

tweet['Score']=0.0

print("\n",tweet)

#tweet="Hurry Hurry Hurry! Click this link to grab the offer! and be Amazed! and get a bonus of 1 lakh"
index=0
for i in tweet['Tweet']:
    i = i.translate(str.maketrans('', '', string.punctuation))
    i = [word for word in i.split() if word.lower() not in stopwords.words('english')]
    #print(i)
    words = []
    for j in i:
        stemmer = SnowballStemmer("english")
        words.append(stemmer.stem(j))
    print(words)
    words = list(dict.fromkeys(words))
    #len(words)
    count=0
    for i in words:
        if i in spam_words:
            count+=1
    for sentence in spam_sentence:
        #x=tweet1.in(sentence)
        if sentence in i:
            #if(x):
            count+=1
    score=(count/len(words))
    tweet.Score[index]=score
    index+=1

print("\n",tweet)

print(tweet.info())

print(tweet.dtypes)

#tweet.Score=tweet.Score.astype(float)

#tweet.dtypes

print(tweet.Score.describe())
avg_tweet_score=tweet.Score.mean()
print("Average Tweet Spam Score = ",avg_tweet_score)
