from review_score import *
import tweepy
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys
import math
import re
import pandas as pd

consumer_key = "T5mrPqWj3JfkftRL86S2MFEze"
consumer_secret = "BLIjrePtYXecGFIt5zlQ9xXOkRmaAwHs0HiKxVXhgWqiaOWDap"
access_key = "754570151538757634-hOHyq1pwl7acWhZWxHLeLoSpR7dQFJd"
access_secret = "YmwyFupx8uNEMVeSzYb0s0pscQRmb6xRZiPfNSCE02Uup"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


df=pd.read_csv("review_with_product.csv")
print(df)

account_list = []
if (len(sys.argv) > 1):
  account_list = sys.argv[1:]

hashtags=[]
hash=0
product_name=[]
company_name=[]
promoted_products=0
producted_company=0
for name in df["product_name"]:
    product_name.append(name.replace(" ",""))      #if user_id==1
for name in df["company_name"]:
    company_name.append(name.replace(" ",""))

print("produt names",product_name)
print("company name",company_name)
if len(account_list) > 0:
  for target in account_list:
      for status in Cursor(api.user_timeline, id=target).items():
        hashtags=[]
        if hasattr(status, "entities"):
          entities = status.entities
      #    print(status.entities)
          if "hashtags" in entities:
            for ent in entities["hashtags"]:
              if ent is not None:
                if "text" in ent:
                  hashtag = ent["text"]
                  if hashtag is not None:
                    hashtags.append(hashtag)           #stores hashtags in a single tweet
            for names in product_name:
                y=-1
                for tags in hashtags:
                    x=tags.find(names)
                    if(x!=-1):
                        y=1
                if(y==1):
                    promoted_products+=1

print("no of promoted products",promoted_products)
