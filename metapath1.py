from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("review_with_product1.csv")
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
avg_seller_rating={}
for seller_names in seller:
    #print(df1.get_group(seller_names))
    inter=df1.get_group(seller_names)
    avg_seller_rating[seller_names]=inter.rating.mean(axis=0)
print(avg_seller_rating)
print("--------------------------------------------------------------------")

#give the name of the person here
name="uday"
user_df=df.groupby("user_id")
user_df=user_df.get_group(1)
print(user_df)
print("--------------------------------------------------------------")
user_df=user_df.groupby("seller")
user_avg_seller_rating={}
for seller_names in seller:
    #print(df1.get_group(seller_names))
    inter=user_df.get_group(seller_names)
    user_avg_seller_rating[seller_names]=inter.rating.mean(axis=0)
print("avng seller rating by uday",user_avg_seller_rating)

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
for id in user_reviews.keys():
  if(id!=1):                          #change
    print("comparing with user ",id)
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
