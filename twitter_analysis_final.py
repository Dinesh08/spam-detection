from bot_detection import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support,precision_score,recall_score,f1_score
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def scale(df):
    X = df.loc[:,:]
    scaler=StandardScaler()
    names=X.columns
    scaled_df = scaler.fit_transform(X)                    #our real time input
    scaled_df=pd.DataFrame(scaled_df,columns=names)
    return scaled_df


def model_eval(X,y,result):
    X_train, X_test, y_train, y_test = train_test_split(X,np.ravel(y), test_size=0.25,random_state=25)
    model=LogisticRegression(solver='liblinear')
    model.fit(X_train,y_train)
    print("----------------Logistic Regression-------------------")
    p=model.predict(X_test)
    ans=X_test.copy()
    ans["Actual"]=y_test
    ans["Predicted"]=p
    ans1=ans[ans['Actual']==0]
    y_true=ans1["Actual"]
    y_predict=ans1["Predicted"]
    prf=precision_recall_fscore_support(y_true, y_predict, average='micro')
    #print(X_test)
    print('Accuracy  : {:.2f}'.format(model.score(X_test, y_test)))
    print('Precision :',precision_score(y_test,p))
    print('Recall  :',recall_score(y_test,p))
    print('F1-score :',f1_score(y_test,p))
    print("----------------Logisticc Regression-------------------")



def svm(x,y,result):
    model=SVC()
    X_train, X_test, y_train, y_test = train_test_split(x,np.ravel(y), test_size=0.25,random_state=15)
    model.fit(X_train,y_train)
    print('\n\n\n')
    print("----------------SVM-------------------------")
    p=model.predict(X_test)
    ans=X_test.copy()
    ans["Actual"]=y_test
    ans["Predicted"]=p
    ans1=ans[ans['Actual']==0]
    y_true=ans1["Actual"]
    y_predict=ans1["Predicted"]
    prf=precision_recall_fscore_support(y_true, y_predict, average='micro')
    print('Accuracy  : {:.2f}'.format(model.score(X_test, y_test)))
    print('Precision :',precision_score(y_test,p))
    print('Recall  :',recall_score(y_test,p))
    print('F1-score :',f1_score(y_test,p))
    print("----------------SVM-------------------------")
    #predict_svm=model.predict(result)
    #print("predicted by svm ", predict_svm)


def random_forest(x,y,result):
    rf_model = RandomForestClassifier(n_estimators=100, # Number of trees  100 default
                                    #  max_features=2,    # Num features considered ,
                                      oob_score=True,
                                      min_samples_split=10,  #min no of samples required to split the tree
                                      max_depth=100,  #default is none or untill min_samples condition is met
                                      max_features='auto',
                                     # bootstrap=False  # if false, whole dataset is used for each tree
                                      )
    X_train, X_test, y_train, y_test = train_test_split(x,np.ravel(y), test_size=0.25,random_state=15)
    rf_model.fit(X_train,y_train)
    print('\n')
    print("----------------Random forest-------------------")
    p=rf_model.predict(X_test)
    ans=X_test.copy()
    ans["Actual"]=y_test
    ans["Predicted"]=p
#     print(ans)
    ans1=ans[ans['Actual']==0]
    y_true=ans1["Actual"]
    y_predict=ans1["Predicted"]
    prf=precision_recall_fscore_support(y_test, p, average='micro')
    #print(X_test)
    print('Accuracy  : {:.2f}'.format(rf_model.score(X_test, y_test)))
    print('Precision :',precision_score(y_test,p))
    print('Recall  :',recall_score(y_test,p))
    print('F1-score :',f1_score(y_test,p))
    print("----------------Random forest-------------------")
    prediction=rf_model.predict(result)
    print("\nPrediction : ",prediction[0])
    if(prediction[0]==1):
     user='Spam'
    elif(prediction[0]==0):
     user='Not Spam'
    print("Final User Result : ",user)
    #print('Accuracy : '.accuracy_score(y_test,predict1))
   #print("predicted by random forest ", predict1)

def rfe_model(df,y,result):
    X_train, X_test, y_train, y_test = train_test_split(df,np.ravel(y), test_size=0.25,random_state=25)
    model=LogisticRegression(solver='liblinear')
    rfe=RFE(model,18)
    rfe = rfe.fit(X_train,y_train)
    print(rfe.support_)
    print(rfe.ranking_)
    cols=["description","verified","age","followers","mentions","url_ratio","hashtag","retweet_rate","mean_of_intertweet_delay","SD","avg_tweets_day","avg_tweeets_week"]
    new_df=pd.DataFrame()
    for col in cols:
        new_df[col]=df[col]
    X_train, X_test, y_train, y_test = train_test_split(new_df,np.ravel(y), test_size=0.25,random_state=25)
    model=LogisticRegression(solver='liblinear')
    model.fit(X_train,y_train)
    model_eval(new_df,y,result)
    svm(new_df,y,result)
    random_forest(new_df,y,result)


#__main__
data=pd.read_csv('tweet_info.txt',sep=" ",header=None)
data.columns=["description","verified","age","following","followers","reputation","mentions","unique_mentions","url_ratio","hashtag","content_similarity","retweet_rate","reply_rate","no_of_tweets","mean_of_intertweet_delay","SD","avg_tweets_day","avg_tweeets_week","s1","s2","fofo","following_rate","0-3","3-6","6-9","9-12","12-15","15-18","18-21","21-24"]
label=pd.read_csv("label.txt",header=None)
data["label"]=label

print(data)

X = data.loc[:, data.columns != 'label']
y = data.loc[:, data.columns == 'label']


scaler=StandardScaler()
cols=["description","verified","age","followers","mentions","url_ratio","hashtag","retweet_rate","mean_of_intertweet_delay","SD","avg_tweets_day","avg_tweeets_week"]
new_df=pd.DataFrame()
for col in cols:
    new_df[col]=X[col]
print(new_df)

rfe_model(new_df,y,extract_df)

scaled_df = scaler.fit_transform(new_df)
scaled_df=pd.DataFrame(scaled_df,columns=cols)
result=scaler.transform(extract_df)
print("\n\nAfter Scaling\nresult is ", result)
rfe_model(scaled_df,y,result)

#def svm(x,y):
#    model=SVC()
#    X_train, X_test, y_train, y_test = train_test_split(x,np.ravel(y), test_size=0.25,random_state=15)
#    model.fit(X_train,y_train)
#    pred = model.predict(X_test)
#    print(confusion_matrix(y_test, pred))
#    print(classification_report(y_test, pred))

#svm(scaled_df,np.ravel(y))

#random_forest(scaled_df,np.ravel(y),result)

#svm(new_df,np.ravel(y))

#%matplotlib inline
matplotlib.style.use('ggplot')
np.random.seed(1)
df = pd.DataFrame({
    'logistic_reg': np.random.normal(0, 2, 10000),
    'svm': np.random.normal(5, 3, 10000),
    'random_forest': np.random.normal(-5, 5, 10000)
})

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=['logistic_reg', 'svm', 'random_forest'])
import seaborn as sns
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df['logistic_reg'], ax=ax1)
sns.kdeplot(df['svm'], ax=ax1)
sns.kdeplot(df['random_forest'], ax=ax1)
ax2.set_title('After Scaleing')
sns.kdeplot(scaled_df['logistic_reg'], ax=ax2)
sns.kdeplot(scaled_df['svm'], ax=ax2)
sns.kdeplot(scaled_df['random_forest'], ax=ax2)
plt.show()


