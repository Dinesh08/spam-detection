from bot_detection import *
import numpy as np
import pandas as pd
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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def scale(df):
    X = df.loc[:,:]
    scaler=StandardScaler()
    names=X.columns
    scaled_df = scaler.fit_transform(X)                    #our real time input
    scaled_df=pd.DataFrame(scaled_df,columns=names)
    return scaled_df

def model_eval1(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,np.ravel(y), test_size=0.25,random_state=25)
    model=LogisticRegression(solver='liblinear')
    model.fit(X_train,y_train)
    print("----------------Logistic Regression Before RFE-------------------")
    p=model.predict(X_test)
    ans=X_test.copy()
    ans["Actual"]=y_test
    ans["Predicted"]=p
    ans1=ans[ans['Actual']==0]
    y_true=ans1["Actual"]
    y_predict=ans1["Predicted"]
    prf=precision_recall_fscore_support(y_true, y_predict, average='micro')
    #print(X_test)
    global log_acc
    log_acc=model.score(X_test, y_test)
    print('Accuracy  : {:.2f}'.format(model.score(X_test, y_test)))
    print('Accuracy : ', accuracy_score(y_test,p))
    print('Precision spam : ',precision_score(y_test,p,pos_label=1))
    print('Recall spam : ',recall_score(y_test,p,pos_label=1))
    print('F1-score spam : ',f1_score(y_test,p,pos_label=1))
    print('Precision quality : ',precision_score(y_test,p,pos_label=0))
    print('Recall quality : ',recall_score(y_test,p,pos_label=0))
    print('F1-score quality : ',f1_score(y_test,p,pos_label=0))
    print('Confusion Matrix\n',confusion_matrix(y_test,p))
    print('Classification Report\n',classification_report(y_test,p))
    print("----------------Logistic Regression Before RFE-------------------")

def model_eval(X,y,result):
    X_train, X_test, y_train, y_test = train_test_split(X,np.ravel(y), test_size=0.25,random_state=25)
    model=LogisticRegression(solver='liblinear')
    model.fit(X_train,y_train)
    print("----------------Logistic Regression After RFE-------------------")
    p=model.predict(X_test)
    ans=X_test.copy()
    ans["Actual"]=y_test
    ans["Predicted"]=p
    ans1=ans[ans['Actual']==0]
    y_true=ans1["Actual"]
    y_predict=ans1["Predicted"]
    prf=precision_recall_fscore_support(y_true, y_predict, average='micro')
    #print(X_test)
    global log_acc
    log_acc=model.score(X_test, y_test)
    print('Accuracy  : {:.2f}'.format(model.score(X_test, y_test)))
    print('Accuracy : ', accuracy_score(y_test,p))
    print('Precision spam : ',precision_score(y_test,p,pos_label=1))
    print('Recall spam : ',recall_score(y_test,p,pos_label=1))
    print('F1-score spam : ',f1_score(y_test,p,pos_label=1))
    print('Precision quality : ',precision_score(y_test,p,pos_label=0))
    print('Recall quality : ',recall_score(y_test,p,pos_label=0))
    print('F1-score quality : ',f1_score(y_test,p,pos_label=0))
    print('Confusion Matrix\n',confusion_matrix(y_test,p))
    print('Classification Report\n',classification_report(y_test,p))
    print("----------------Logistic Regression After RFE-------------------")
    global prediction1, user_result1
    prediction1=model.predict(result)
    print("\nPrediction : ",prediction1[0])
    if(prediction1[0]==1):
     user_result1='Spam'
    elif(prediction1[0]==0):
     user_result1='Not Spam'
    print("Final User Result : ",user_result1)

def svm1(x,y):
    model=SVC()
    X_train, X_test, y_train, y_test = train_test_split(x,np.ravel(y), test_size=0.25,random_state=15)
    model.fit(X_train,y_train)
    print('\n\n\n')
    print("----------------SVM Before RFE-------------------------")
    p=model.predict(X_test)
    ans=X_test.copy()
    ans["Actual"]=y_test
    ans["Predicted"]=p
    ans1=ans[ans['Actual']==0]
    y_true=ans1["Actual"]
    y_predict=ans1["Predicted"]
    global svm_acc1
    svm_acc1=model.score(X_test, y_test)
    prf=precision_recall_fscore_support(y_true, y_predict, average='micro')
    print('Accuracy  : {:.2f}'.format(model.score(X_test, y_test)))
    print('Accuracy', accuracy_score(y_test,p))
    print('Precision spam : ',precision_score(y_test,p,pos_label=1))
    print('Recall spam : ',recall_score(y_test,p,pos_label=1))
    print('F1-score spam : ',f1_score(y_test,p,pos_label=1))
    print('Precision quality : ',precision_score(y_test,p,pos_label=0))
    print('Recall quality : ',recall_score(y_test,p,pos_label=0))
    print('F1-score quality : ',f1_score(y_test,p,pos_label=0))
    print('Confusion Matrix\n',confusion_matrix(y_test,p))
    print('Classification Report\n',classification_report(y_test,p))
    print("----------------SVM Before RFE-------------------------")

def svm(x,y,result):
    model=SVC()
    X_train, X_test, y_train, y_test = train_test_split(x,np.ravel(y), test_size=0.25,random_state=15)
    model.fit(X_train,y_train)
    print('\n\n\n')
    print("----------------SVM After RFE-------------------------")
    p=model.predict(X_test)
    ans=X_test.copy()
    ans["Actual"]=y_test
    ans["Predicted"]=p
    ans1=ans[ans['Actual']==0]
    y_true=ans1["Actual"]
    y_predict=ans1["Predicted"]
    prf=precision_recall_fscore_support(y_true, y_predict, average='micro')
    global svm_acc
    svm_acc=model.score(X_test, y_test)
    print('Accuracy  : {:.2f}'.format(model.score(X_test, y_test)))
    print('Accuracy', accuracy_score(y_test,p))
    print('Precision spam : ',precision_score(y_test,p,pos_label=1))
    print('Recall spam : ',recall_score(y_test,p,pos_label=1))
    print('F1-score spam : ',f1_score(y_test,p,pos_label=1))
    print('Precision quality : ',precision_score(y_test,p,pos_label=0))
    print('Recall quality : ',recall_score(y_test,p,pos_label=0))
    print('F1-score quality : ',f1_score(y_test,p,pos_label=0))
    print('Confusion Matrix\n',confusion_matrix(y_test,p))
    print('Classification Report\n',classification_report(y_test,p))
    print("----------------SVM After RFE-------------------------")
    global prediction2, user_result2
    prediction2=model.predict(result)
    print("\nPrediction : ",prediction2[0])
    if(prediction2[0]==1):
     user_result2='Spam'
    elif(prediction2[0]==0):
     user_result2='Not Spam'
    print("Final User Result : ",user_result2)
    #predict_svm=model.predict(result)
    #print("predicted by svm ", predict_svm)

def random_forest1(x,y):
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
    print("----------------Random Forest Before RFE-------------------")
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
    global rf_acc1
    rf_acc1 =rf_model.score(X_test, y_test)
    print('Accuracy  : {:.2f}'.format(rf_model.score(X_test, y_test)))
    print('Accuracy', accuracy_score(y_test,p))
    print('Precision spam : ',precision_score(y_test,p,pos_label=1))
    print('Recall spam : ',recall_score(y_test,p,pos_label=1))
    print('F1-score spam : ',f1_score(y_test,p,pos_label=1))
    print('Precision quality : ',precision_score(y_test,p,pos_label=0))
    print('Recall quality : ',recall_score(y_test,p,pos_label=0))
    print('F1-score quality : ',f1_score(y_test,p,pos_label=0))
    print('Confusion Matrix\n',confusion_matrix(y_test,p))
    print('Classification Report\n',classification_report(y_test,p))
    print("----------------Random Forest Before RFE-------------------")

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
    print("----------------Random Forest After RFE-------------------")
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
    global rf_acc
    rf_acc=rf_model.score(X_test, y_test)
    print('Accuracy  : {:.2f}'.format(rf_model.score(X_test, y_test)))
    print('Accuracy', accuracy_score(y_test,p))
    print('Precision : ',precision_score(y_test,p))
    print('Recall : ',recall_score(y_test,p))
    print('F1-score : ',f1_score(y_test,p))
    #print('Precision spam : ',precision_score(y_test,p,pos_label=1))
    #print('Recall spam : ',recall_score(y_test,p,pos_label=1))
    #print('F1-score spam : ',f1_score(y_test,p,pos_label=1))
    #print('Precision quality : ',precision_score(y_test,p,pos_label=0))
    #print('Recall quality : ',recall_score(y_test,p,pos_label=0))
    #print('F1-score quality : ',f1_score(y_test,p,pos_label=0))
    print('Confusion Matrix\n',confusion_matrix(y_test,p))
    print('Classification Report\n',classification_report(y_test,p))
    print("----------------Random Forest After RFE-------------------")
    global prediction3, user_result3
    prediction3=rf_model.predict(result)
    print("\nPrediction : ",prediction3[0])
    if(prediction3[0]==1):
     user_result3='Spam'
    elif(prediction3[0]==0):
     user_result3='Not Spam'
    print("Final User Result : ",user_result3)
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
data.columns=["description","verified","age","following","followers","reputation","mentions","unique_mentions","url_ratio","hashtag","content_similarity","retweet_rate","reply_rate","no_of_tweets","mean_of_intertweet_delay","SD","avg_tweets_day","avg_tweeets_week","s1","s2","followers_following_ratio","following_rate","0-3","3-6","6-9","9-12","12-15","15-18","18-21","21-24"]
label=pd.read_csv("label.txt",header=None)
data["label"]=label

print(data)

cols=["description","verified","age","following","followers","reputation","mentions","unique_mentions","url_ratio","hashtag","content_similarity","retweet_rate","reply_rate","no_of_tweets","mean_of_intertweet_delay","avg_tweets_day","avg_tweeets_week","followers_following_ratio","following_rate","label"]
data1=pd.DataFrame()
for col in cols:
    data1[col]=data[col]
print(data1)

fig = plt.figure(figsize = (15,20))
ax = fig.gca()
data1.hist(ax = ax)
#plt.show

data2=data[data['label']==0]
print(data2)
data4=pd.DataFrame()
for col in cols:
    data4[col]=data2[col]
print(data4)

data3=data[data['label']==1]
print(data3)
data5=pd.DataFrame()
for col in cols:
    data5[col]=data3[col]
print(data5)

for c in cols:
    if c=='description' or c=='verified' or c=='label':
        continue
    data6=data4[[str(c)]].copy()
    data6['cdf'] = data6.rank(method = 'average', pct = True)
    # Sort and plot
    #data6.sort_values(c).plot(x = str(c), y = 'cdf', grid = True)
    ax = data6.sort_values(c).plot(x = str(c), y ='cdf', grid = True)
    ax.set_xlabel(str(c))
    ax.set_ylabel("cdf")
    ax.legend(["non-spam"])

for c in cols:
    if c=='description' or c=='verified' or c=='label':
        continue
    data6=data5[[str(c)]].copy()
    data6['cdf'] = data6.rank(method = 'average', pct = True)
    # Sort and plot
    #data6.sort_values(c).plot(x = str(c), y = 'cdf', grid = True)
    ax = data6.sort_values(c).plot(x = str(c), y ='cdf', grid = True)
    ax.set_xlabel(str(c))
    ax.set_ylabel("cdf")
    ax.legend(["spam"])

'''for c in cols:
    if c=='description' or c=='verified' or c=='label':
       continue
    data6=data4[[str(c)]].copy()
    data7=data5[[str(c)]].copy()
    data6['cdf'] = data6.rank(method = 'average', pct = True)
    data7['cdf'] = data7.rank(method = 'average', pct = True)
    #Sort and plot
    #data6.sort_values(c).plot(x = str(c), y = 'cdf', grid = True)
    f, ax = plt.subplots(figsize=(8, 8))
    ax = data6.sort_values(c).plot(x = str(c), y ='cdf', grid = True, color = 'blue')
    ax = data7.sort_values(c).plot(x = str(c), y ='cdf', grid = True, color = 'red')
    ax.set_xlabel(str(c))
    ax.set_ylabel("cdf")
    ax.legend(["non-spam","spam"])'''

'''from empiricaldist import Cdf

def decorate_cdf(title, x, y):
    """Labels the axes.

    title: string
    """
    plt.xlabel(x)
    plt.ylabel(y)
    #plt.title(title)
    plt.legend(data1.groupby('label').groups.keys())
for c in cols:
    for name, group in data1.groupby('label'):
        Cdf.from_seq(group.).plot()
    x, y = str(c),'CDF'
    decorate_cdf(x,y)'''

#data.followers.max()

#data.followers=data.followers-5 if data.followers!=0 && data.followers-5!=0

#plt.plot(data.retweet_rate,data.label)

#plt.plot(data.reputation,data.label)

plt.figure(figsize=(20,25))
sns.heatmap(data.corr(method='spearman'), cmap='coolwarm', annot=True)
plt.tight_layout()
plt.show()

# # Without RFE and Standardization

X = data.loc[:, data.columns != 'label']
y = data.loc[:, data.columns == 'label']
#X_train, X_test, y_train, y_test = train_test_split(X,np.ravel(y), test_size=0.25,random_state=25)
model_eval1(X,y)
svm1(X,y)
random_forest1(X,y)
log1,svm11,rf1=log_acc,svm_acc1,rf_acc1
print(log1)
print(svm11)
print(rf1)

objects = ("logistic_reg","svm","random_forest")
y_pos = np.arange(len(objects))
performance = [log1,svm11,rf1]
#performance_test=[NBTest,RFTest,TreeTest,pred_test]
plt.figure(figsize=(7,7))
plt.bar(objects, performance,width=0.5,align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Performance of models without RFE and Standardization')
plt.show()

# # Scaling Before RFE

scaler=StandardScaler()
scaled_df = scaler.fit_transform(data)
X = data.loc[:, data.columns != 'label']
y = data.loc[:, data.columns == 'label']
#X_train, X_test, y_train, y_test = train_test_split(X,np.ravel(y), test_size=0.25,random_state=25)
model_eval1(X,y)
svm1(X,y)
random_forest1(X,y)
log2,svm2,rf2=log_acc,svm_acc1,rf_acc1
print(log2)
print(svm2)
print(rf2)

#print(log_acc)
#print(svm_acc1)
#print(rf_acc1)
#objects = ("logistic_reg","svm","random_forest")
#y_pos = np.arange(len(objects))
#performance = [log2,svm2,rf2]
#performance_test=[NBTest,RFTest,TreeTest,pred_test]
#plt.figure(figsize=(7,7))
#plt.bar(objects, performance,width=0.5,align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Accuracy')
#plt.title('Performance of Models After Scaling')
#plt.show()

cols=["description","verified","age","followers","mentions","url_ratio","hashtag","retweet_rate","mean_of_intertweet_delay","SD","avg_tweets_day","avg_tweeets_week"]
new_df=pd.DataFrame()
for col in cols:
    new_df[col]=X[col]
new_df

#for x in cols:
#    plt.plot(new_df[x],data.label)

# # After RFE

rfe_model(new_df,y,extract_df)
log3,svm3,rf3=log_acc,svm_acc,rf_acc
print(log3)
print(svm3)
print(rf3)

objects = ("logistic_reg","svm","random_forest")
y_pos = np.arange(len(objects))
performance = [log3,svm3,rf3]
#performance_test=[NBTest,RFTest,TreeTest,pred_test]
plt.figure(figsize=(7,7))
plt.bar(objects, performance,width=0.5,align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Performance of models after RFE')
plt.show()

# # After RFE and Standardization

scaled_df = scaler.fit_transform(new_df)
scaled_df=pd.DataFrame(scaled_df,columns=cols)
result=scaler.transform(extract_df)
print("result is ", result)
rfe_model(scaled_df,y,result)
log4,svm4,rf4=log_acc,svm_acc,rf_acc
print(log4)
print(svm4)
print(rf4)

objects = ("logistic_reg","svm","random_forest")
y_pos = np.arange(len(objects))
performance = [log_acc,svm_acc,rf_acc]
#performance_test=[NBTest,RFTest,TreeTest,pred_test]
plt.figure(figsize=(7,7))
plt.bar(objects, performance,width=0.5,align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Performance of models after RFE and Standardization')
plt.show()

#objects = ("logistic_reg","svm","random_forest")
#y_pos = np.arange(len(objects))
#performance = [log4,svm4,rf4]
##performance_test=[NBTest,RFTest,TreeTest,pred_test]
#plt.figure(figsize=(7,7))
#plt.bar(objects, performance,width=0.5,align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Accuracy')
#plt.title('Performance of Models After Bot Detection')
#plt.show()

#objects = ("logistic_reg","svm","random_forest")
#y_pos = np.arange(len(objects))
#performance = [log4,svm4,rf4]
##performance_test=[NBTest,RFTest,TreeTest,pred_test]
#plt.figure(figsize=(7,7))
#plt.bar(objects, performance,width=0.5,align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
##plt.ylabel('Accuracy')
#plt.title('Performance of RFE After Scaling')
#plt.show()

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

#new_df.plot.bar(figsize=(10,10))

#scaled_df.plot(figsize=(20,20))

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
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
ax1.set_title('Before Scaling')
sns.kdeplot(df['logistic_reg'], ax=ax1)
sns.kdeplot(df['svm'], ax=ax1)
sns.kdeplot(df['random_forest'], ax=ax1)
ax2.set_title('After Scaleing')
sns.kdeplot(scaled_df['logistic_reg'], ax=ax2)
sns.kdeplot(scaled_df['svm'], ax=ax2)
sns.kdeplot(scaled_df['random_forest'], ax=ax2)
plt.legend(loc='upper left')
plt.show()

print("Prediction for ",account_list[0])
print("Logistic Regression : ",user_result1)
print("SVM : ",user_result2)
print("Random Forest : ",user_result3)
user_result=[user_result1,user_result2,user_result3]
#print("\nPrediction : ",prediction[0])
twitter_result=max(user_result,key=user_result.count)
print("Final User Result : ",twitter_result)

