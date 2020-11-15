from review_hash import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("review_with_product.csv")
#print(df.product_name)
product=set()
company=set()
seller=set()
products=df.product_name
companies=df.company_name
sellers=df.seller
df1=df.groupby(["seller"])  #stores grouped mean of sellers
#print(df1.get_group("cloudtail"))
for i in products:
    product.add(i)
for i in companies:
    company.add(i)
for i in sellers:
    seller.add(i)
print("-------------------------------------------------------------")
i=0
total=0
#avg_rating=df.rating.mean(axis=0)
#print("avg rating is ",avg_rating)
avg_seller_rating={}
for seller_names in seller:
    #print(df1.get_group(seller_names))
    inter=df1.get_group(seller_names)
    avg_seller_rating[seller_names]=inter.rating.mean(axis=0)
print(avg_seller_rating)
print("--------------------------------------------------------------------")

#give the name of the person here
name="dinesh"
user_df=df.groupby("user_id")
user_df=user_df.get_group(7)
print(user_df)
user_avg_rating=user_df.rating.mean(axis=0)
no_reviews_user=user_df.shape[0]
print("user_avg_rating is ",user_avg_rating)
print("--------------------------------------------------------------")
user_df=user_df.groupby("seller")
user_avg_seller_rating={}
no_reviews={}
for seller_names in seller:
    print(df1.get_group(seller_names))
    inter=user_df.get_group(seller_names)
    no_reviews[seller_names]=inter.shape[0]
    user_avg_seller_rating[seller_names]=inter.rating.mean(axis=0)

print("avng seller rating by ",name,user_avg_seller_rating)

print("--------------------------------------------------------------------")
user_id_count=6;
inter=df.groupby("user_id")
user_reviews={}
for i in range(1,user_id_count+1):          #traversing each user
    each_user=df.loc[df['user_id'] ==i]
    print(each_user)
    user_name=each_user.iloc[0]["user_name"]
    print(user_name)
    user_reviews[i]={}          #creating a dict for each user, where productname:review stored
    reviews=[]
    rows=each_user.shape[0]
    for var in range(rows):
        user_reviews[i][each_user.iloc[var]["product_name"]]=each_user.iloc[var]["review"]
#        reviews.append(each_user.iloc[var]["review"])
    #user_reviews[i]=reviews
print(user_reviews)

print("-----------------------------------------------------------")

single_user_review=user_reviews[1]      #change userid for other users
sim_list=[]
fake_user=False
for id in user_reviews.keys():
  if(id!=1):                          #change
    print("comparing with user ",id)
    sim_list=[]
    count=0
    compare_user_review=user_reviews[id]
    for prod1 in single_user_review.keys():
        for prod2 in compare_user_review.keys():
            if(prod1==prod2):
                r1=single_user_review[prod1]
                r2=compare_user_review[prod2]
                tfidf_vectorizer = TfidfVectorizer()
                documents=(r1,r2)
                tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
                print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))
                similarity=cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
                sim_list.append(similarity[0][1])
                if(similarity[0][1]>0.4):
                    count+=1
    if(count/len(sim_list)>0.65):
                fake_user=True

print(no_reviews)
spam_user=False
if((user_avg_rating>=4.5 or user_avg_rating<=2) and no_reviews_user>4):
    print("potential spammer")
    spam_user=True
for key in avg_seller_rating.keys():
    if(avg_seller_rating[key]<2):
        if(no_reviews[key]>2):
            spam_user=True
            print("potential spammer")

for key in avg_seller_rating.keys():
    if((avg_seller_rating[key]>2 and avg_seller_rating[key]<4.3) or(avg_seller_rating[key]<2 and no_reviews[key]<2)):
        spam_user=False
print("is user a spammer ",spam_user)
print("does user have more accounts ",fake_user)
#print(similarity[0])
