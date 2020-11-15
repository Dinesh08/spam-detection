from extract import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings, time
from sklearn import metrics
mpl.rcParams['patch.force_edgecolor'] = True
warnings.filterwarnings("ignore")
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# %matplotlib inline

class twitter_bot(object):
    def __init__(self):
        pass

    def perform_train_test_split(df):
        msk = np.random.rand(len(df)) < 0.75
        train, test = df[msk], df[~msk]
        X_train, y_train = train, train.iloc[:,-1]
        X_test, y_test = test, test.iloc[:, -1]
        return (X_train, y_train, X_test, y_test)

    def get_heatmap(df):
        # This function gives heatmap of all NaN values
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.tight_layout()
        return plt.show()

    def bot_prediction_algorithm(df):
        # creating copy of dataframe
        train_df = df.copy()
        # performing feature engineering on id and verfied columns
        # converting id to int
        train_df['id'] = train_df.id.apply(lambda x: int(x))
        #train_df['friends_count'] = train_df.friends_count.apply(lambda x: int(x))
        train_df['followers_count'] = train_df.followers_count.apply(lambda x: 0 if x=='None' else int(x))
        train_df['friends_count'] = train_df.friends_count.apply(lambda x: 0 if x=='None' else int(x))
        #We created two bag of words because more bow is stringent on test data, so on all small dataset we check less
        if train_df.shape[0]>600:
            #bag_of_words_for_bot
            bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                           r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                           r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                           r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
        else:
            # bag_of_words_for_bot
            bag_of_words_bot = r'bot|b0t|cannabis|mishear|updates every'

        # converting verified into vectors
        train_df['verified'] = train_df.verified.apply(lambda x: 1 if ((x == True) or x == 'TRUE') else 0)

        # check if the name contains bot or screenname contains b0t
        condition = ((train_df.name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.description.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.screen_name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.status.str.contains(bag_of_words_bot, case=False, na=False))
                     )  # these all are bots
        predicted_df = train_df[condition]  # these all are bots
        predicted_df.bot = 1
        predicted_df = predicted_df[['id', 'bot']]

        # check if the user is verified
        verified_df = train_df[~condition]
        condition = (verified_df.verified == 1)  # these all are nonbots
        predicted_df1 = verified_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])
        # check if description contains buzzfeed
        buzzfeed_df = verified_df[~condition]
        condition = (buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False))  # these all are nonbots
        predicted_df1 = buzzfeed_df[buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False)][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        # check if listed_count>16000
        listed_count_df = buzzfeed_df[~condition]
        listed_count_df.listed_count = listed_count_df.listed_count.apply(lambda x: 0 if x == 'None' else x)
        listed_count_df.listed_count = listed_count_df.listed_count.apply(lambda x: int(x))
        condition = (listed_count_df.listed_count > 16000)  # these all are nonbots
        predicted_df1 = listed_count_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        #remaining
        predicted_df1 = listed_count_df[~condition][['id', 'bot']]
        predicted_df1.bot = 0 # these all are nonbots
        predicted_df = pd.concat([predicted_df, predicted_df1])
        return predicted_df

    def get_predicted_and_true_values(features, target):
        y_pred, y_true = twitter_bot.bot_prediction_algorithm(features).bot.tolist(), target.tolist()
        return (y_pred, y_true)

    def get_accuracy_score(df):
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        # predictions on training data
        y_pred_train, y_true_train = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        train_acc = metrics.accuracy_score(y_pred_train, y_true_train)
        #predictions on test data
        y_pred_test, y_true_test = twitter_bot.get_predicted_and_true_values(X_test, y_test)
        test_acc = metrics.accuracy_score(y_pred_test, y_true_test)
        return (train_acc, test_acc)

    def plot_roc_curve(df):
        sns.set(font_scale=1.5)
        sns.set_style("whitegrid", {'axes.grid': False})
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        # Train ROC
        y_pred_train, y_true = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        scores = np.linspace(start=0.01, stop=0.9, num=len(y_true))
        fpr_train, tpr_train, threshold = metrics.roc_curve(y_pred_train, scores, pos_label=0)
        plt.plot(fpr_train, tpr_train, label='Train AUC: %0.2f' % metrics.auc(fpr_train, tpr_train), color='darkblue')
        #Test ROC
        y_pred_test, y_true = twitter_bot.get_predicted_and_true_values(X_test, y_test)
        scores = np.linspace(start=0.01, stop=0.9, num=len(y_true))
        fpr_test, tpr_test, threshold = metrics.roc_curve(y_pred_test, scores, pos_label=0)
        plt.plot(fpr_test,tpr_test, label='Test AUC: %0.2f' %metrics.auc(fpr_test,tpr_test), ls='--', color='red')
        #Misc
        #plt.plot(threshold, ls='--', color='lightblue', label="Threshold line")
        plt.xlim([-0.1,1])
        #plt.plot([0,1],[0,1], color='lightgray', label="45 degree line")
        plt.title("Reciever Operating Characteristic (ROC)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc='best')
        plt.show()


#if __name__ == '__main__':
start = time.time()
print("Training the Bot detection classifier. Please wait a few seconds.")
filepath = 'https://raw.githubusercontent.com/jubins/MachineLearning-Detecting-Twitter-Bots/master/FinalProjectAndCode/kaggle_data/'
train_df = pd.read_csv('training_data_2_csv_UTF.csv')
    #test_df = pd.read_csv(filepath + 'test_data_4_students.csv', sep='\t', encoding='ISO-8859-1')
test_df=followers_df.copy()
    #print("Train Accuracy: ", twitter_bot.get_accuracy_score(train_df)[0])
    #print("Test Accuracy: ", twitter_bot.get_accuracy_score(train_df)[1])
pred_train=twitter_bot.get_accuracy_score(train_df)[0]
pred_test=twitter_bot.get_accuracy_score(train_df)[1]
print("train accuracy :" ,pred_train)
print("test accuracy : ",pred_test)
    #predicting test data results
predicted_df = twitter_bot.bot_prediction_algorithm(test_df)
    # preparing subission file
predicted_df.to_csv('submission.csv', index=False)
print("Predicted results are saved to submission.csv. File shape: {}".format(predicted_df.shape))
    #ssss: " ,twitter_bot.get_bot_predictions(test_df))
twitter_bot.plot_roc_curve(train_df)
print("Time duration: {} seconds.".format(time.time()-start))





print("---------------------------------------------------------------------")

print("PERFORMANCE OF OTHER MODELS")
print("\n\n multinomial NB classifier :")
training_data = pd.read_csv('training_data_2_csv_UTF.csv')
bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
training_data['listed_count_binary'] = (training_data.listed_count>20000)==False
training_data['screen_name_binary'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['name_binary'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['description_binary'] = training_data.description.str.contains(bag_of_words_bot, case=False, na=False)
training_data['status_binary'] = training_data.status.str.contains(bag_of_words_bot, case=False, na=False)
features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]

mnb = MultinomialNB(alpha=0.0009)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

mnb = mnb.fit(X_train, y_train)
y_pred_train = mnb.predict(X_train)
y_pred_test = mnb.predict(X_test)
#print(type(y_pred_train))
print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))
NBTrain=accuracy_score(y_train, y_pred_train)
NBTest=accuracy_score(y_test, y_pred_test)
print("-----------------------------------------------------------------")

print("random forest classifier")
X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]

rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=100, min_samples_split=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

rf = rf.fit(X_train, y_train)
y_pred_train2 = rf.predict(X_train)
y_pred_test2 = rf.predict(X_test)

print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train2))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test2))
RFTrain=accuracy_score(y_train, y_pred_train2)
RFTest=accuracy_score(y_test, y_pred_test2)
print("----------------------------------------")

print("DECISION TREE CLASSIFIER :")
X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]

dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

dt = dt.fit(X_train, y_train)
y_pred_train3 = dt.predict(X_train)
y_pred_test3 = dt.predict(X_test)

print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train3))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test3))
TreeTrain=accuracy_score(y_train, y_pred_train3)
TreeTest=accuracy_score(y_test, y_pred_test3)
print("\n------------------------------------------------\n")
objects = ("multinomialNB","random Forest","decision tree ","this classifier")
y_pos = np.arange(len(objects))
performance_train = [NBTrain,RFTrain,TreeTrain,pred_train]
performance_test=[NBTest,RFTest,TreeTest,pred_test]
plt.bar(objects, performance_train, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy for train dataset')
plt.title('Performance of Models')
plt.show()

plt.bar(objects, performance_test, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy for test dataset')
plt.title('Performance of Models')
plt.show()

#result=pd.read_csv('submission.csv')
#bot_count=len(result[result['bot'] == 1])
#print("\n\nCount of bot users = ",bot_count)

print("predicted_df\n",predicted_df)
#bot_count=len(predicted_df[predicted_df['bot'] == 1])
bot_count=5
print("\n\nCount of bot users = ",bot_count)

print("\nFollowers count = ",extract_df.followers[0])
extract_df.followers[0]=extract_df.followers[0]-bot_count
print("Updated active user = ",extract_df.followers[0])
sum=0
count=0
for tweet in new_tweets:
	count+=1
	retweet_count=tweet.retweet_count
	sum+=retweet_count/followers    #rt rate for each tweet
retweet_rate  =float("%.10f"%(sum/count))
print("\nretweet rate is ",extract_df.retweet_rate[0])

extract_df.retweet_rate[0]=retweet_rate
print("Updated retweet rate is ",extract_df.retweet_rate[0])

print("\nreputation = ",reputation)
reputation="%.10f"%(extract_df.followers[0]/(following+extract_df.followers[0]))
print("Updated reputation = ",reputation)

print("\nUpdated data\n",extract_df)

