from metapath import * 
print("USER REPORT\n")
print("Is the user a spammer based on...\n")
print("statistical data : ",twitter_result)
print("tweets (content based) : ",tweet_result)
print("spam score (tweets) : ",avg_tweet_score)
print("reviews (content based)  : ",review_result)
print("spam score (reviews) : ",avg_review_score)
print("promoted products : ",promoted_products)
print("has multiple accounts : ",fake_user)
print("Metapath : ",spam_user)
#print("\nAnalysing results...\nThe user is a ",spam_result)
if(twitter_result=='Spam'):
	twitter_result=True
else:
	twitter_result=False
if(spam_user and twitter_result ):
	print("The user is a product spammer")
if(avg_tweet_score>0.6 and avg_review_score>0.6):
	print("The user is a irrelevant spammer")
else:
	print("The user is not a spammer")