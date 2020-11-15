from extract import *
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
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def scale(df):
    X = df.loc[:,:]
    scaler=StandardScaler()
    names=X.columns
    scaled_df = scaler.fit_transform(X)                    #our real time input
    scaled_df=pd.DataFrame(scaled_df,columns=names)
    return scaled_df


# In[4]:


def model_eval(X,y):
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
    global log_acc
    log_acc=model.score(X_test, y_test)
    print('Accuracy  : {:.2f}'.format(model.score(X_test, y_test)))
    print('Precision :',precision_score(y_test,p))
    print('Recall  :',recall_score(y_test,p))
    print('F1-score :',f1_score(y_test,p))
    print("----------------Logistic Regression-------------------")
    


# In[5]:


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
    print('Precision :',precision_score(y_test,p))
    print('Recall  :',recall_score(y_test,p))
    print('F1-score :',f1_score(y_test,p))
    print("----------------SVM Before RFE-------------------------")


# In[6]:


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
    global svm_acc
    svm_acc=model.score(X_test, y_test)
    print('Accuracy  : {:.2f}'.format(model.score(X_test, y_test)))
    print('Precision :',precision_score(y_test,p))
    print('Recall  :',recall_score(y_test,p))
    print('F1-score :',f1_score(y_test,p))
    print("----------------SVM-------------------------")
    #predict_svm=model.predict(result)
    #print("predicted by svm ", predict_svm)


# In[7]:


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
    print("----------------Random forest Before RFE-------------------")
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
    print('Precision :',precision_score(y_test,p))
    print('Recall  :',recall_score(y_test,p))
    print('F1-score :',f1_score(y_test,p))
    print("----------------Random forest Before RFE-------------------")


# In[8]:


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
    global rf_acc
    rf_acc=rf_model.score(X_test, y_test)
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


# In[9]:


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
    model_eval(new_df,y)
    svm(new_df,y,result)
    random_forest(new_df,y,result)


# In[10]:


#__main__
data=pd.read_csv('tweet_info.txt',sep=" ",header=None)
data.columns=["description","verified","age","following","followers","reputation","mentions","unique_mentions","url_ratio","hashtag","content_similarity","retweet_rate","reply_rate","no_of_tweets","mean_of_intertweet_delay","SD","avg_tweets_day","avg_tweeets_week","s1","s2","fofo","following_rate","0-3","3-6","6-9","9-12","12-15","15-18","18-21","21-24"]
label=pd.read_csv("label.txt",header=None)
data["label"]=label


# In[11]:


print(data)


# In[12]:


#plt.plot(data.reputation,data.label)


# In[13]:


plt.figure(figsize=(20,25))
sns.heatmap(data.corr(method='spearman'), cmap='coolwarm', annot=True)
plt.tight_layout()
plt.show()


# In[14]:


X = data.loc[:, data.columns != 'label']
y = data.loc[:, data.columns == 'label']
#X_train, X_test, y_train, y_test = train_test_split(X,np.ravel(y), test_size=0.25,random_state=25)
model_eval(X,y)
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


# In[15]:


#type(log_acc)
#type(svm_acc1)
#type(rf_acc1)


# In[16]:


scaler=StandardScaler()
scaled_df = scaler.fit_transform(data)
X = data.loc[:, data.columns != 'label']
y = data.loc[:, data.columns == 'label']
#X_train, X_test, y_train, y_test = train_test_split(X,np.ravel(y), test_size=0.25,random_state=25)
model_eval(X,y)
svm1(X,y)
random_forest1(X,y)
log2,svm2,rf2=log_acc,svm_acc1,rf_acc1
print(log2)
print(svm2)
print(rf2)


# In[17]:


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



# In[18]:


cols=["description","verified","age","followers","mentions","url_ratio","hashtag","retweet_rate","mean_of_intertweet_delay","SD","avg_tweets_day","avg_tweeets_week"]
new_df=pd.DataFrame()
for col in cols:
    new_df[col]=X[col]
print(new_df)


# In[19]:


#for x in cols:
#    plt.plot(new_df[x],data.label)


# In[20]:


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

# In[21]:


scaled_df = scaler.fit_transform(new_df)
scaled_df=pd.DataFrame(scaled_df,columns=cols)
result=scaler.transform(extract_df)
print("result is ", result)
rfe_model(scaled_df,y,result)



# In[31]:

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
#plt.title('Performance of Models Before Bot Detection')
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


# In[ ]:





# In[23]:


#def svm(x,y):
#    model=SVC()
#    X_train, X_test, y_train, y_test = train_test_split(x,np.ravel(y), test_size=0.25,random_state=15)
#    model.fit(X_train,y_train)
#    pred = model.predict(X_test)
#    print(confusion_matrix(y_test, pred))
#    print(classification_report(y_test, pred))


# In[24]:


#svm(scaled_df,np.ravel(y))


# In[25]:


#random_forest(scaled_df,np.ravel(y),result)


# In[26]:


#svm(new_df,np.ravel(y))


# In[27]:


#new_df.plot.bar(figsize=(10,10))


# In[28]:


#scaled_df.plot(figsize=(20,20))


# In[29]:


matplotlib.style.use('ggplot')
np.random.seed(1)
df = pd.DataFrame({
    'logistic_reg': np.random.normal(0, 2, 10000),
    'svm': np.random.normal(5, 3, 10000),
    'random_forest': np.random.normal(-5, 5, 10000)
})


# In[30]:


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


# In[ ]:




