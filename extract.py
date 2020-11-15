import tweepy
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import numpy as np
import sys
import math
import re
import pandas as pd
import time

consumer_key = "T5mrPqWj3JfkftRL86S2MFEze"
consumer_secret = "BLIjrePtYXecGFIt5zlQ9xXOkRmaAwHs0HiKxVXhgWqiaOWDap"
access_key = "754570151538757634-hOHyq1pwl7acWhZWxHLeLoSpR7dQFJd"
access_secret = "YmwyFupx8uNEMVeSzYb0s0pscQRmb6xRZiPfNSCE02Uup"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

ids = []
statuses=None
def get_followers_id(person):
    followersid = []
    count=0
    influencer=api.get_user(screen_name=person)   #screen_name
    influencer_id=influencer.id
    number_of_followers=influencer.followers_count
    print("number of followers count : ",number_of_followers,'\n','user id : ',influencer_id)
    status = tweepy.Cursor(api.followers_ids, user_id=person, tweet_mode="extended").items() #screen_name
    for i in range(0,number_of_followers):
        try:
            user=next(status)
            followersid.append(user)
            count += 1
            #print("\n...testing...\n")
        except tweepy.TweepError:
            print('error limit of twiter sleep for 15 min')
            timestamp = time.strftime("%d.%m.%Y %H:%M:%S", time.localtime())
            print(timestamp)
            if len(followersid)>0 :
                print('the number get until this time :', count,'all followers count is : ',number_of_followers)
                followersid = np.array(str(followersid))
                save_followers_status(person, followersid)
                print("\n\n\n.....testing 1.....\n\n\n")
                followersid = []
            time.sleep(15*60)
            next(status)
        except :
            print('end of followers ', count, 'all followers count is : ', number_of_followers)
            followersid = np.array(str(followersid))
            save_followers_status(person, followersid)
            print("\n\n\n.....testing 2.....\n\n\n")
            followersid = []
    #save_followers_status(person, followersid)
    # followersid = np.array(map(str,followersid))
    return followersid

ids=get_followers_id(sys.argv[1])
print("follower ids :",ids)


account_list = []
if (len(sys.argv) > 1):
  account_list = sys.argv[1:]
else:
  print("Please provide a list of usernames at the command line.")
  sys.exit(0)

#description
if len(account_list) > 0:
  for target in account_list:
    print("Getting data for " + target)
    item = api.get_user(target)
    print("name: " + item.name)
    print("screen_name: " + item.screen_name)
    print("description: " + item.description)
    if(len(item.description)>0):
                description=1
    else:
                description=0
    print("statuses_count: " , item.statuses_count)
    print("friends_count: " , item.friends_count)
    print("followers_count: " , item.followers_count)
    print("favourites count:",item.favourites_count)
    print("verified: ",str(item.verified))
    print("lang: ",str(item.lang))
    if(item.statuses_count!=0):
     try:
      statuses=str(item.status)
     except:
      statuses=None
    print("status: ",statuses.encode('utf-8'))
    print("listed count: ",item.listed_count)
    print("default_profile: ",str(item.default_profile).encode('utf-8'))
    print("default_image: ",str(item.default_profile_image).encode('utf-8'))
    print("has extended profile: ",str(item.has_extended_profile).encode('utf-8'))
    print("id :",item.id)
    print("id_str :",str(item.id_str).encode('utf-8'))
    print("location :",str(item.location).encode('utf-8'))
    print("url : ",str(item.url).encode('utf-8'))
    #print("followers",item.followers)
    followers=item.followers_count
    following=item.friends_count
    tweets = item.statuses_count
    account_created_date = item.created_at
    print("creation date",account_created_date)
    print("created date = ",account_created_date)
    delta = datetime.utcnow() - account_created_date
    print(delta)
    account_age_days = delta.days
    print("Account age (in days): " + str(account_age_days))
    if account_age_days > 0:
      print("Average tweets per day: " + "%.2f"%(float(tweets)/float(account_age_days)))

    hashtags = []
    mentions = []
    tweet_count = 0
    sum=0
    hash=0
    end_date = datetime.utcnow() - timedelta(days=30)
    for status in Cursor(api.user_timeline, id=target).items():
      tweet_count += 1
      if hasattr(status, "entities"):
        entities = status.entities
    #    print(status.entities)
        if "hashtags" in entities:
          for ent in entities["hashtags"]:
            if ent is not None:
              if "text" in ent:
                hashtag = ent["text"]
                if hashtag is not None:
                  hashtags.append(hashtag)
                  hash+=1
        if "user_mentions" in entities:
          for ent in entities["user_mentions"]:
            if ent is not None:
              if "screen_name" in ent:
                name = ent["screen_name"]
                if name is not None:
                  mentions.append(name)
                  sum+=1
    #  if status.created_at < end_date:
    #    break
    print("tweet count",tweet_count)
    print()
    unique=0
    unique=len(set(mentions))
    print(unique)
    print(sum)
    if(tweet_count==0):
        mentions_ratio=0
    else:
        mentions_ratio="%.10f"%(sum/tweet_count)
    if(sum==0):
        unique_mentions=0
    else:
        unique_mentions="%.10f"%(unique/sum)

    print()
    print("Most used hashtags:")
    for item, count in Counter(hashtags).most_common(10):
      print(item + "\t" + str(count))
    if(tweets==0):
     hashtag_ratio=0.0
    else:
     hashtag_ratio="%.10f"%(hash/tweets)
    reputation="%.10f"%(followers/(following+followers))
    print("reputation = ",reputation,"\nmentions raitio = ",mentions_ratio,"\nhashtag_ratio =",hashtag_ratio)
    print("unique mentions =",unique_mentions)
    print("All done. Processed " + str(tweet_count) + " tweets.")

    #extract tweets and store it in list
    alltweets = []
    tweet_time=[]
    new_tweets = api.user_timeline(screen_name = target,count=200)
    #print(type(new_tweets))
    for tweet in new_tweets:
        alltweets.append(tweet.text)
        print(tweet.created_at)
        tweet_time.append(tweet.created_at)
        print(tweet.retweet_count)
    total=0
    for i in range(1,tweet_count):
        total+=(tweet_time[i-1]-tweet_time[i]).days
    print("total intertweet_delay",total)
    if(tweet_count==0):
     mean=0.0
    else:
     mean=float("%.10f"%(total/tweet_count))             #mean of inter tweet delay
    print("mean_of_intertweet_delay ", mean)

    total=0
    for i in range(tweet_count):
        total+=((tweet_time[i]).day-float(mean))**2
    if(tweet_count==0):
     variance=0.0
    else:
     variance=total/tweet_count
    sd=float("%.10f"%(math.sqrt(variance)))         # sd of intertweet delay
    print("sd of intertweet delay ",sd)
    avg_tweets_day=float("%.10f"%(tweet_count/account_age_days))
    avg_tweets_week=float("%.10f"%(tweet_count/(account_age_days/7)))
    print("avg_tweets day and week" , avg_tweets_day," ",avg_tweets_week)

    url_count=0
    for tweet in new_tweets:
        if(len(tweet.entities["urls"])>0):
                url_string=tweet.entities["urls"][0]['expanded_url']
                x=re.search("^https://twitter\.com.*",url_string)
                print("twitter urls :")
                if(x):
                    #url_count+=1
                    print(tweet.entities["urls"][0]['url'])
                    print(tweet.entities)
                    print("\n\n")

                else:
                    print("url posted by user: ")
                    url_count+=len(tweet.entities["urls"])
                    #print(tweet.entities)
                    print(tweet.entities["urls"][0]['url'])
    print("url count is ", url_count)
    if(tweet_count==0):
     url_ratio=0.0
    else:
     url_ratio=float("%.10f"%(url_count/tweet_count))
    print("url ratio is ",url_ratio)



#    print("list count", item.list_count)
    sum=0
    count=0
    for tweet in new_tweets:
    #    alltweets.append(tweet.text)
    #    print(tweet.created_at)
    #    tweet_time.append(tweet.created_at)
        count+=1
    #    print(tweet.retweet_count)
        retweet_count=tweet.retweet_count
        sum+=retweet_count/followers    #rt rate for each tweet
    if(count==0):
     retweet_rate=0.0
    else:
     retweet_rate  =float("%.10f"%(sum/count))
    print("retweet rate is ",retweet_rate)

    replies=0
    for tweet in new_tweets:
        print(tweet.in_reply_to_screen_name)
        #break
        if(tweet.in_reply_to_screen_name!=None):
            replies+=1
    if(count==0):
     reply_rate=0.0
    else:
     reply_rate  =float("%.10f"%(replies/count))
    print("retweet rate is ",reply_rate)

  for target in account_list:
    print("Getting data for " + target)
    item = api.get_user(target)
    if(item.verified==False):
            verified=0
    else:
        verified=1
    print(item.verified)
    print(item.listed_count)


tweet_df = pd.DataFrame({'Tweet': alltweets})
print("\n",tweet_df)
                                                                                                        #replyrate
extract_data=[[description,verified,account_age_days,followers,mentions_ratio,url_ratio,hashtag_ratio,retweet_rate,mean,sd,avg_tweets_day,avg_tweets_week]]
#print(extract_data)
extract_df=pd.DataFrame(extract_data,columns=["description","verified","age","followers","mentions","url_ratio","hashtag","retweet_rate","mean_of_intertweet_delay","SD","avg_tweets_day","avg_tweeets_week"])
print("\n",extract_df)


#id=ids[0]

followers_df = pd.DataFrame()
final=[]
if len(ids) > 0:
  for person in ids:
    print("\n\n")
    print("user_id is ",person)
    item=api.get_user(user_id=person)   #screen_name
    #item = influencer.screen_name
    print("name: " + item.name)
    print("screen_name: " + item.screen_name)
    print("description: " + item.description)
    if(len(item.description)>0):
                description=1
    else:
                description=0
    print("statuses_count: " , item.statuses_count)
    print("friends_count: " , item.friends_count)
    print("followers_count: " , item.followers_count)
    print("favourites count:",item.favourites_count)
    print("verified: ",item.verified)
    print("lang: ",item.lang)
    if(item.statuses_count!=0):
     try:
      statuses=str(item.status).encode('utf-8')
     except:
      statuses=None
    print("status: ",statuses)
    print("listed count: ",item.listed_count)
    print("default_profile: ",str(item.default_profile).encode('utf-8'))
    print("default_image: ",str(item.default_profile_image).encode('utf-8'))
    print("has extended profile: ",str(item.has_extended_profile).encode('utf-8'))
    print("id :",item.id)
    print("id_str :",str(item.id_str).encode('utf-8'))
    print("location :",str(item.location).encode('utf-8'))
    print("url : ",str(item.url).encode('utf-8'))
#  print("followers",item.followers)
    followers=item.followers_count
    following=item.friends_count
    tweets = item.statuses_count
    account_created_date = item.created_at
    print("created date = ",account_created_date)
    delta = datetime.utcnow() - account_created_date
    print(delta)
    account_age_days = delta.days
    print("Account age (in days): " + str(account_age_days).encode('utf-8'))

    if(item.verified==False):
            verified=0
    else:
        verified=1
    print(item.verified)
    followers_data=[]
    #followers_data=pd.Series([item.id,item.id_str,item.screen_name,item.location,item.description,item.url,item.followers_count,item.friends_count,item.listed_count,account_created_date,item.favourites_count,item.verified,item.statuses_count,item.lang,"none",str(item.default_profile),str(item.default_profile_image),str(item.has_extended_profile),item.name])
    followers_data=[item.id,item.id_str,item.screen_name,item.location,item.description,item.url,item.followers_count,item.friends_count,item.listed_count,account_created_date,item.favourites_count,item.verified,item.statuses_count,item.lang,statuses,str(item.default_profile).encode('utf-8'),str(item.default_profile_image).encode('utf-8'),str(item.has_extended_profile).encode('utf-8'),item.name]
    print("followers data is ",followers_data)
    final.append(followers_data)
    #followers_df=pd.concat([followers_data,followers_df],ignore_index=True)
#    followers_df.append(followers_data,ignore_index=True)
#print(final)
followers_df=pd.DataFrame(final,columns=['id','id_str','screen_name','location','description','url','followers_count','friends_count','listed_count','created_at','favourites_count','verified','statuses_count','lang','status','default_profile','default_profile_image','has_extended_profile','name'])
followers_df['bot']=''
print(followers_df)
