#!/usr/bin/env python
# coding: utf-8

# In[348]:


#Import the library
import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[349]:


train = pd.read_csv("C:\\Users\\USER\\Desktop\\DS\\titanic\\train.csv")
train.head()


# In[350]:


test = pd.read_csv("C:\\Users\\USER\\Desktop\\DS\\titanic\\test.csv")
test.head()


# In[351]:


test.info()


# In[352]:


train.info()


# In[353]:


train.describe()


# ### Dealing with Missing Values

# In[354]:


#function to print the total percentage of the missing values
def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(total/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])


# In[355]:


missing_percentage(train)


# In[356]:


missing_percentage(test)


# In[357]:


sns.heatmap(train.isnull(),cmap='viridis', yticklabels=False)
#more then 75% of missing values  found in both train and test is cabin. way is to drop the column or need advnced analysis


# In[358]:


sns.heatmap(test.isnull(),cmap='viridis',yticklabels=False)


# ### Working with missing values - Train Data

# In[359]:


#Drop cabin
train = train.drop(labels="Cabin", axis=1)


# In[360]:


def percent_value_counts(df, feature):
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    
    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)


# In[361]:


#Embarked
percent_value_counts(train, 'Embarked')


# In[362]:


train['Embarked'].mode()


# In[363]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Embarked', y='Fare', data=train, palette='prism_r')

#With respect to fare distribution C Embark shows mostly around 80.0 So lets fill it with C


# In[364]:


train["Embarked"] = train["Embarked"].fillna("C")


# In[365]:


percent_value_counts(train, 'Embarked')


# In[366]:


#Age
train['Age'].hist(bins=30, color='red', alpha=0.6)


# In[367]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[368]:


corr = numeric_features.corr()
corr


# In[369]:


print(corr['Survived'].sort_values(ascending=False))


# In[370]:


#correlation matrix
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);


# In[371]:


#Creating mean of age based on Pclass
class_1 = train[train['Pclass']==1]['Age'].mean()
class_2 = train[train['Pclass']==2]['Age'].mean()
class_3 = train[train['Pclass']==3]['Age'].mean()


# In[372]:


train.loc[(train['Age'].isnull()) & (train['Pclass']==1), 'Age']=class_1
train.loc[(train['Age'].isnull()) & (train['Pclass']==2), 'Age']=class_2
train.loc[(train['Age'].isnull()) & (train['Pclass']==3), 'Age']=class_3


# In[373]:


train.info()


# ### Working with missing values - Test Data

# In[374]:


test.info()


# In[375]:


missing_percentage(test)


# In[376]:


#Drop Cabin
test = test.drop(labels="Cabin", axis=1)


# In[377]:


#Fare
test[test['Fare'].isnull()]


# In[378]:


# We can go with average Fare with respect to class
meanFare_test = test.groupby('Pclass').mean()['Fare']
meanFare_test


# In[379]:


test['Fare'] = test['Fare'].fillna(meanFare_test[3])


# In[380]:


test.info()


# In[381]:


#Age
test['Age'].hist(bins=30,color='green',alpha=0.6)


# In[382]:


class_1_test = test[test['Pclass']==1]['Age'].mean()
class_2_test = test[test['Pclass']==2]['Age'].mean()
class_3_test = test[test['Pclass']==3]['Age'].mean()


# In[383]:


test.loc[(test['Age'].isnull()) & (test['Pclass']==1), 'Age']=class_1_test
test.loc[(test['Age'].isnull()) & (test['Pclass']==2), 'Age']=class_2_test
test.loc[(test['Age'].isnull()) & (test['Pclass']==3), 'Age']=class_3_test


# In[384]:


test.info()


# ### Encoding Categorical Values

# In[385]:


train.head()


# In[386]:


train.drop(['Name'],axis=1,inplace=True)


# In[402]:


df_Pclass = pd.get_dummies(train['Pclass'])
df_Embarked = pd.get_dummies(train['Embarked'])
df_Sex = pd.get_dummies(train['Sex'])

train_final = pd.concat([train, df_Pclass, df_Embarked, df_Sex], axis=1)


# In[403]:


train_final.head()


# In[410]:


train_final.drop(['PassengerId','Ticket'],axis=1,inplace=True)


# In[411]:


train_final.head()


# In[393]:


test.drop(['Name','Ticket'],axis=1,inplace=True)


# In[448]:


df_Pclass = pd.get_dummies(test['Pclass'])
df_Embarked = pd.get_dummies(test['Embarked'])
df_Sex = pd.get_dummies(test['Sex'])


test_final = pd.concat([test, df_Pclass, df_Embarked, df_Sex], axis=1)


# In[449]:


test_final.head()


# In[450]:


test_final.drop(['PassengerId','Pclass','Sex','Embarked'],axis=1,inplace=True)


# In[452]:


test_final.head()


# In[416]:


train_final.head()


# ### Modelling

# In[426]:


X = train_final.iloc[:,1:].values
y = train_final.iloc[:,0].values



# In[428]:


X


# In[427]:


y


# In[429]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[436]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[437]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[438]:


# Accuracy
from sklearn.metrics import accuracy_score
result = classifier.score(X_test, y_test)
result


# In[439]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:





# In[440]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)


# In[444]:


# Accuracy
from sklearn.metrics import accuracy_score
result1 = clf.score(X_test, y_test)
result1


# In[445]:


y_pred = clf.predict(X_test)


# In[446]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_clf = confusion_matrix(y_test, y_pred)
cm_clf


# In[447]:


(88+59) / (88+12+20+59)
#correct pred by total number of inputs


# In[453]:


X_submission = test_final.values


# In[455]:


y_submission = clf.predict(X_submission)
y_submission


# In[ ]:




