# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:15:23 2019

@author: AnasAhmed
"""

import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv(r'D:\Python\XYZCorp_LendingData.txt',  delimiter = "\t", index_col = 0, header = 0)
data.describe(include="all")
data.shape
data.isnull().sum()
data.isnull().sum().value_counts()
data.dtypes

data.sort_values(by=['member_id'])
#data.head(10)

#%%
half_count= len(data)/2
half_count

data_copy=data.dropna(thresh=half_count, axis=1)

data_copy.shape

data_copy.isnull().sum()
data_copy.dtypes
#data_copy.head(10)
#data['member_id'].unique()
#%%Treating missing value using linear reg

data_1 = data_copy.drop(['tot_coll_amt', 'tot_cur_bal', 'next_pymnt_d', 'emp_title', 'emp_length'],axis=1)
data_1.isnull().sum()
data_1.dtypes

#%%

data_dropcolumns1 =data_copy[['member_id', 'tot_coll_amt', 'tot_cur_bal', 'next_pymnt_d', 'emp_title', 'emp_length']]
data_dropcolumns1.shape
type(data_dropcolumns1)

#%%

data_1['collections_12_mths_ex_med'].fillna(data_1['collections_12_mths_ex_med'].mode()[0],inplace=True)
data_1['last_credit_pull_d'].fillna(data_1['last_credit_pull_d'].mode()[0],inplace=True)
data_1['last_pymnt_d'].fillna(data_1['last_pymnt_d'].mode()[0],inplace=True)
data_1['title'].fillna(data_1['title'].mode()[0],inplace=True)
data_1["revol_util"].fillna(int(data_1["revol_util"].mean()),inplace = True)
data_1.dtypes
data_1.isnull().sum()
#data_1.head(10)

#%%

colname = ['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'issue_d', 'pymnt_plan', 'purpose', 'title', 
           'zip_code', 'addr_state', 'earliest_cr_line', 'initial_list_status', 'last_pymnt_d', 'last_credit_pull_d', 
           'application_type']

from sklearn import preprocessing

le={}

for x in colname:
    le[x] = preprocessing.LabelEncoder()

for x in colname:
    data_1[x] = le[x].fit_transform(data_1[x]) 

data_1.dtypes
data_1.shape

data_1['home_ownership'].unique()

#%%

data_1 = data_1.drop(['total_rev_hi_lim'],axis=1)
data_1['total_rev_hi_lim'] = data_copy['total_rev_hi_lim']


#%%

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

from sklearn.model_selection import train_test_split
#%%

data_1_without_null = data_1.dropna()
data_1_without_null.shape
data_1_without_null.head()

data_1_without_null.sort_values(by=['member_id'])
#data_1_without_null['member_id'].unique()

data_0 = data_1

data_0['total_rev_hi_lim'] = data_0['total_rev_hi_lim'].replace({np.nan : 8241})
data_0.shape
data_0['total_rev_hi_lim'].unique()

data_1_withonly_null = data_0[data_0['total_rev_hi_lim']==8241]
data_1_withonly_null.isnull().sum()
data_1_withonly_null.shape
data_1_withonly_null.head()

data_1_withonly_null.sort_values(by=['member_id'])
#data_1_withonly_null['member_id'].unique()

data_1_withonly_null=data_1_withonly_null.drop('total_rev_hi_lim', axis=1)
data_1_withonly_null.shape

#%%
train_data_x = data_1_without_null.values[:,:-1]
train_data_y = data_1_without_null.values[:,-1]
train_data_x.shape
train_data_y.shape

test_data = data_1_withonly_null
test_data.shape
#%%Checking the accuracy b4 predicting the missing values

X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, test_size=0.3, random_state=10)

linreg.fit(X_train,Y_train)

Y_pred = linreg.predict(X_test)
print(Y_pred)

"""new_df = pd.DataFrame()
new_df = X_test
type(new_df)

new_df["Actual Sales"] = Y_test
new_df["Predicted Sales"] = Y_pred
print(new_df)
"""

from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

r2_score = r2_score(Y_test,Y_pred)
print(r2_score)

rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

Y_pred.mean()

#%%1. Recursive Feature Selection
colname = data_1_without_null.columns[:]

from sklearn.feature_selection import RFE
#rfe = RFE(linreg, 20)            #classifier algo(i.e Logistic algo) and retain 7 vars
#rfe = RFE(linreg, 10)
rfe = RFE(linreg, 15)
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ")
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ",model_rfe.ranking_)

Y_pred = model_rfe.predict(X_test)
#print(list(zip(Y_test, Y_pred)))
Y_test = Y_test.astype(int)
Y_pred = Y_pred.astype(int)

from sklearn.metrics import r2_score,mean_squared_error
r2_score = r2_score(Y_test,Y_pred)
print(r2_score)

rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

#%%
data_1_without_null.columns
data_1_without_null.shape

train_data_x = data_1_without_null.values[:,[4,5,8,11,13,22,23,30,32,33,34,35,40,43,44]]
train_data_y = data_1_without_null.values[:,-1]
train_data_x.shape
train_data_y.shape

test_data = data_1_withonly_null[['term', 'int_rate', 'sub_grade', 'verification_status',
                                  'pymnt_plan', 'open_acc', 'pub_rec', 'total_pymnt', 'total_rec_prncp', 
                                  'total_rec_int', 'total_rec_late_fee', 'recoveries', 
                                  'collections_12_mths_ex_med', 'acc_now_delinq', 'default_ind']]
test_data.shape
#%%

linreg.fit(train_data_x,train_data_y)

prediction = linreg.predict(test_data)
prediction

data_1_withonly_null['total_rev_hi_lim'] = prediction
data_1_withonly_null.shape
data_1_withonly_null.head()

#%%

data_1_withonly_null.columns
data_1_without_null.columns

#%%

data_temp = pd.concat([data_1_without_null, data_1_withonly_null])
data_temp.shape
data_temp.isnull().sum()
data_temp.sort_values(by=['member_id'])
data_temp.head(10)

data1 = data_temp.merge(data_dropcolumns1, on='member_id', how='left')
data1.shape
data1.isnull().sum()

#%%2nd missing value treatment

data1.dtypes

data_2 = data1.drop(['tot_coll_amt', 'next_pymnt_d', 'emp_title', 'emp_length'],axis=1)
data_2.isnull().sum()
data_2.dtypes
data_2.shape
data_2.head()

#%%

data_dropcolumns2 =data_copy[['member_id', 'tot_coll_amt', 'next_pymnt_d', 'emp_title', 'emp_length']]

linreg.fit(train_data_x,train_data_y)

prediction = linreg.predict(test_data)
prediction

data_1_withonly_null['total_rev_hi_lim'] = prediction
data_1_withonly_null.shape
data_1_withonly_null.head()
 
#%%

data_1_withonly_null.columns
data_1_without_null.columns

#%%

data_temp = pd.concat([data_1_without_null, data_1_withonly_null])
data_temp.shape
data_temp.isnull().sum()
data_temp.sort_values(by=['member_id'])
data_temp.head(10)

data1 = data_temp.merge(data_dropcolumns1, on='member_id', how='left')
data1.shape
data1.isnull().sum()
data_dropcolumns2.shape
type(data_dropcolumns2)

#%%

data_2_without_null = data_2.dropna()
data_2_without_null.shape
data_2_without_null.head()
 
data_2_without_null.sort_values(by=['member_id'])
#data_2_without_null['member_id'].unique()

data_0 = data_2
Y = data_0['tot_cur_bal']
print(max(Y))
print(min(Y))

data_0['tot_cur_bal'] = data_0['tot_cur_bal'].replace({np.nan : -1.5})
data_0.shape
data_0['tot_cur_bal'].unique()

data_2_withonly_null = data_0[data_0['tot_cur_bal']== -1.5]
data_2_withonly_null.isnull().sum()
data_2_withonly_null.shape
data_2_withonly_null.head()

data_2_withonly_null.sort_values(by=['member_id'])
#data_2_withonly_null['member_id'].unique()

data_2_withonly_null=data_2_withonly_null.drop('tot_cur_bal', axis=1)
data_2_withonly_null.shape

#%%

train_data_x = data_2_without_null.values[:,:-1]
train_data_y = data_2_without_null.values[:,-1]
train_data_x.shape
train_data_y.shape

test_data = data_2_withonly_null
test_data.shape

#%%Checking the accuracy b4 predicting the missing values

X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, test_size=0.3, random_state=10)

linreg.fit(X_train,Y_train)

Y_pred = linreg.predict(X_test)
print(Y_pred)

"""new_df = pd.DataFrame()
new_df = X_test
type(new_df)

new_df["Actual Sales"] = Y_test
new_df["Predicted Sales"] = Y_pred
print(new_df)
"""

from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

r2_score = r2_score(Y_test,Y_pred)
print(r2_score)

rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

Y_pred.mean()

#%%2nd Recursive Feature Selection
colname = data_2_without_null.columns[:]

from sklearn.feature_selection import RFE
rfe = RFE(linreg, 15)            #classifier algo(i.e Logistic algo) and retain 7 vars
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ")
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ",model_rfe.ranking_)

Y_pred = model_rfe.predict(X_test)
#print(list(zip(Y_test, Y_pred)))
Y_test = Y_test.astype(int)
Y_pred = Y_pred.astype(int)

from sklearn.metrics import r2_score,mean_squared_error
r2_score = r2_score(Y_test,Y_pred)
print(r2_score)

rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

#%%
data_2_without_null.columns
data_2_without_null.shape

train_data_x = data_2_without_null.values[:,[4,5,7,9,11,13,19,23,30,32,33,34,35,42,43]]
train_data_y = data_2_without_null.values[:,-1]
train_data_x.shape
train_data_y.shape

test_data = data_2_withonly_null[['term', 'int_rate', 'grade', 'home_ownership',
                                  'verification_status', 'pymnt_plan', 'delinq_2yrs', 'pub_rec', 'total_pymnt', 
                                  'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 
                                  'application_type', 'acc_now_delinq']]
test_data.shape

#%%

linreg.fit(train_data_x,train_data_y)

prediction = linreg.predict(test_data)
prediction

data_2_withonly_null['tot_cur_bal'] = prediction
data_2_withonly_null.shape
data_2_withonly_null.head()

#%%

data_2_withonly_null.columns
data_2_without_null.columns

#%%

data_temp = pd.concat([data_2_without_null, data_2_withonly_null])
data_temp.shape
data_temp.isnull().sum()
data_temp.sort_values(by=['member_id'])
data_temp.head(10)

data2 = data_temp.merge(data_dropcolumns2, on='member_id', how='left')
data2.shape
data2.isnull().sum()

#%%3rd missing value treatment

data2.dtypes

data_3 = data2.drop(['next_pymnt_d', 'emp_title', 'emp_length'],axis=1)
data_3.isnull().sum()
data_3.dtypes
data_3.shape
data_3.head()

#%%

data_dropcolumns3 =data_copy[['member_id','next_pymnt_d', 'emp_title', 'emp_length']]
data_dropcolumns3.shape
type(data_dropcolumns3)

#%%

data_3_without_null = data_3.dropna()
data_3_without_null.shape
data_3_without_null.head()
 
data_3_without_null.sort_values(by=['member_id'])
#data_3_without_null['member_id'].unique()

data_0 = data_3
Y = data_0['tot_coll_amt']
print(max(Y))
print(min(Y))

data_0['tot_coll_amt'] = data_0['tot_coll_amt'].replace({np.nan : -1.5})
data_0.shape
data_0['tot_coll_amt'].unique()

data_3_withonly_null = data_0[data_0['tot_coll_amt']== -1.5]
data_3_withonly_null.isnull().sum()
data_3_withonly_null.shape
data_3_withonly_null.head()

data_3_withonly_null.sort_values(by=['member_id'])
#data_3_withonly_null['member_id'].unique()

data_3_withonly_null=data_3_withonly_null.drop('tot_coll_amt', axis=1)
data_3_withonly_null.shape

#%%

train_data_x = data_3_without_null.values[:,:-1]
train_data_y = data_3_without_null.values[:,-1]
train_data_x.shape
train_data_y.shape

test_data = data_3_withonly_null
test_data.shape

#%%Checking the accuracy b4 predicting the missing values

X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, test_size=0.3, random_state=10)

linreg.fit(X_train,Y_train)

Y_pred = linreg.predict(X_test)
print(Y_pred)

"""new_df = pd.DataFrame()
new_df = X_test
type(new_df)

new_df["Actual Sales"] = Y_test
new_df["Predicted Sales"] = Y_pred
print(new_df)
"""

from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

r2_score = r2_score(Y_test,Y_pred)
print(r2_score)

rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

Y_pred.mean()

#%%3rd Recursive Feature Selection
colname = data_3_without_null.columns[:]

from sklearn.feature_selection import RFE
rfe = RFE(linreg, 10)            #classifier algo(i.e Logistic algo) and retain 7 vars
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ")
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ",model_rfe.ranking_)

Y_pred = model_rfe.predict(X_test)
#print(list(zip(Y_test, Y_pred)))
Y_test = Y_test.astype(int)
Y_pred = Y_pred.astype(int)

from sklearn.metrics import r2_score,mean_squared_error
r2_score = r2_score(Y_test,Y_pred)
print(r2_score)

rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)

#%%
data_3_without_null.columns
data_3_without_null.shape

train_data_x = data_3_without_null.values[:,[4,13,23,30,32,33,34,35,40,44]]
train_data_y = data_3_without_null.values[:,-1]
train_data_x.shape
train_data_y.shape

test_data = data_3_withonly_null[['term',
                                  'pymnt_plan', 'pub_rec', 'total_pymnt', 
                                  'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 
                                  'collections_12_mths_ex_med', 'default_ind']]
test_data.shape

#%%

linreg.fit(train_data_x,train_data_y)

prediction = linreg.predict(test_data)
prediction

data_3_withonly_null['tot_coll_amt'] = prediction
data_3_withonly_null.shape
data_3_withonly_null.head()

#%%

data_3_withonly_null.columns
data_3_without_null.columns

#%%

data_temp = pd.concat([data_3_without_null, data_3_withonly_null])
data_temp.shape
data_temp.isnull().sum()
data_temp.sort_values(by=['member_id'])
data_temp.head(10)

data3 = data_temp.merge(data_dropcolumns3, on='member_id', how='left')
data3.shape
data3.isnull().sum()

#%%4th missing value treatment

data3.dtypes

data_4 = data3.drop(['emp_title', 'emp_length'],axis=1)
data_4.isnull().sum()
data_4.dtypes
data_4.shape
data_4.head()

#%%

data_dropcolumns4 =data_copy[['member_id', 'emp_title', 'emp_length']]
data_dropcolumns4.shape
type(data_dropcolumns4)

#%%

data_4_without_null = data_4.dropna()
data_4_without_null.shape
data_4_without_null.head()
 
data_4_without_null.sort_values(by=['member_id'])
#data_4_without_null['member_id'].unique()

data_0 = data_4
Y = data_0['next_pymnt_d']
data_0['next_pymnt_d'].unique()

data_0['next_pymnt_d'] = data_0['next_pymnt_d'].replace({np.nan : -1.5})
data_0.shape
data_0['next_pymnt_d'].unique()

data_4_withonly_null = data_0[data_0['next_pymnt_d']== -1.5]
data_4_withonly_null.isnull().sum()
data_4_withonly_null.shape
data_4_withonly_null.head()

data_4_withonly_null.sort_values(by=['member_id'])
#data_4_withonly_null['member_id'].unique()

data_4_withonly_null=data_4_withonly_null.drop('next_pymnt_d', axis=1)
data_4_withonly_null.shape

#%%

train_data_x = data_4_without_null.values[:,:-1]
train_data_y = data_4_without_null.values[:,-1]
train_data_x.shape
train_data_y.shape

test_data = data_4_withonly_null
test_data.shape

#%%Checking the accuracy b4 predicting the missing values

X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, test_size=0.3, random_state=10)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(10,random_state=10)

#fit the model on the data and predict the values
RF.fit(X_train, Y_train)

Y_pred = RF.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(Y_test,Y_pred)) 
print(classification_report(Y_test,Y_pred))
print(accuracy_score(Y_test, Y_pred))

#%%4th Recursive Feature Selection
"""colname = data_4_without_null.columns[:]

from sklearn.feature_selection import RFE
rfe = RFE(RF, 10)            #classifier algo(i.e Logistic algo) and retain 7 vars
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ")
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ",model_rfe.ranking_)

Y_pred = model_rfe.predict(X_test)
#print(list(zip(Y_test, Y_pred)))

print(confusion_matrix(Y_test,Y_pred)) 
print(classification_report(Y_test,Y_pred))
print(accuracy_score(Y_test, Y_pred))

"""

#%%

RF.fit(train_data_x,train_data_y)

prediction = RF.predict(test_data)
prediction

data_4_withonly_null['next_pymnt_d'] = prediction
data_4_withonly_null.shape
data_4_withonly_null.head()
 
#%%

data_4_withonly_null.columns
data_4_without_null.columns

#%%

data_temp = pd.concat([data_4_without_null, data_4_withonly_null])
data_temp.shape
data_temp.isnull().sum()
data_temp.sort_values(by=['member_id'])
data_temp.head(10)

data4 = data_temp.merge(data_dropcolumns4, on='member_id', how='left')
data4.shape
data4.isnull().sum()

#%%
data_5 = data4
data_5.shape
data_5['emp_length'].unique()
data_5.dtypes

# Replacing the Values in "emp_length" from categorical to numerical.
# This is known as Manual Label Encoding

data_5['emp_length'] = data_5['emp_length'].replace({'2 years': 2, '1 year': 1, '4 years': 4, '8 years': 8,
                                                                 '10+ years': 10, '9 years': 9, '< 1 year': 0, '6 years': 6,
                                                                 '7 years': 7, '3 years': 3, '5 years': 5})
data_5['emp_length'].unique()
data_5.dtypes

data_5.isnull().sum()

# Finding the Mean Value of "Emp_Length"
    
print(round(data_5['emp_length'].mean(),0))

# Imputing Missing Values in "Emp_Length" by its Mean

data_5['emp_length'].fillna(round(data_5['emp_length'].mean(), 0), inplace = True)

data_5.dtypes
data_5.isnull().sum()

data_5['emp_title'] = data_5['emp_title'].replace({np.nan : 'others'})
data_5.shape
print(data_5['emp_title'].unique())

data_5.isnull().sum()
data_5['emp_title'].value_counts()

data_5 = data_5.drop(['issue_d'],axis=1)
data_5.shape

new_df = data_copy[['member_id', 'issue_d']]
new_df.shape

data_5 = data_5.merge(new_df, on='member_id', how='left')
data_5.shape

data5 = data_5
data5.shape
data5.dtypes
data5.isnull().sum()

data5 = data5.drop(['default_ind'],axis=1)
data5['default_ind'] = data_5['default_ind']
data5.dtypes
 
#%%

data5['default_ind'].value_counts()

data5['issue_d'] = pd.to_datetime(data5['issue_d'])
data5['issue_d'].head()

data5.dtypes
#%% SPLITTING THE DATA INTO TRAIN AND TEST

train_df = data5[data5['issue_d'] <= '2015-05-01']
train_df.shape

test_df = data5[data5['issue_d'] > '2015-05-01']
test_df

 
#%%

train_df['default_ind'].value_counts()
train_df.dtypes

"""#train_df.purpose.value_counts(ascending=False)
OUTPUT:-
debt_consolidation    353936
credit_card           139577
home_improvement       35177
other                  28461
major_purchase         11738
small_business          6989
car                     6133
medical                 5591
moving                  3627
vacation                3313
house                   2594
wedding                 1709
renewable_energy         361
educational              249"""

#%%
columns = ['next_pymnt_d', 'emp_title', 'issue_d']

from sklearn import preprocessing

le={}

for x in columns:
    le[x] = preprocessing.LabelEncoder()

for x in columns:
    train_df[x] = le[x].fit_transform(train_df[x])
    test_df[x] = le[x].fit_transform(test_df[x])

train_df.dtypes
test_df.dtypes

#%% DROPPING COLUMN : "Issue_D"

train_df1=train_df.drop('issue_d',axis=1)
test_df1=test_df.drop('issue_d',axis=1)

#%%CREATING X AND Y ARRAYS FOR TRAINING AND TESTING

X_train=train_df1.values[:, :-1]
Y_train=train_df1.values[:,-1] 

X_test=test_df1.values[:, : -1] 
Y_test=test_df1.values[:, -1]

#%% STANDARDIZATION OF THE DATA SET

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train) 
 
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)

#%%

#1.RUNNING A BASIC LOGISTIC REGRESSION MODEL

from sklearn.linear_model import LogisticRegression

# Create a model
classifier = LogisticRegression()

# Fitting Training data into the model
classifier.fit(X_train, Y_train) # fit is used to train the data  classifier.fit(dependent, independent)

Y_pred = classifier.predict(X_test)

#print(list(zip(Y_test, Y_pred)))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: ")
print()
print(cfm)

print("Classification Report:" )
print()
print(classification_report(Y_test, Y_pred))

acc = accuracy_score(Y_test, Y_pred)
print()
print("Accuracy of the Model: ", acc) 

#%%
y_pred_prob=classifier.predict_proba(X_test)
print(y_pred_prob)
print(list(zip(Y_test,Y_pred)))

for a in np.arange(0,1,0.05):                         #(0,1,0.01)-----for incrementing in steps of 0.01
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
        cfm[1,0]," , type 1 error:", cfm[0,1])

#%%Changing the threshold to 0.35
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.85:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

print()
print("Confusion Matrix:")
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm)
print()
print("Classification Report: ")
print(classification_report(Y_test,y_pred_class))
print()
acc = accuracy_score(Y_test,y_pred_class)
print("Accuracy of the model: ",acc)


#%%
from sklearn import metrics

#fpr, tpr, z = metrics.roc_curve(Y_test,y_pred_class)
fpr, tpr, z = metrics.roc_curve(Y_test,y_pred_prob[:,1])    #here we r passing the probablity array and allowing it choose the best threshold value
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)

import matplotlib.pyplot as plt
#%matplotlib inline
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#%%
from sklearn.metrics import precision_recall_curve 

precision, recall, thresholds = precision_recall_curve(Y_test, y_pred_prob[:,1])

# create plot
plt.plot(precision, recall, label='Precision-recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-recall curve')
plt.xlim([0.735, 1])
plt.ylim([0, 1.02])
plt.legend(loc="lower left")

# save figure
plt.savefig('precision_recall.png', dpi=200)

#%%

#2.Runninng ExtraTreesClassifier

#predicting using the ExtraTressClassifier
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(50,random_state=10)
#fit the model on the data and predict the values
model = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print()
print("Confusion Matrix:")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()
print("Classification Report: ")
print(classification_report(Y_test,Y_pred))
print()
acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)
#%%

#3.RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

model_RandomForest = RandomForestClassifier(50,random_state=10)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train, Y_train)

Y_pred = model_RandomForest.predict(X_test)

print()
print("Confusion Matrix:")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()
 print("Classification Report: ")
print(classification_report(Y_test,Y_pred))
print()
acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

#%%

#4.Predicting using the GradientBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting = GradientBoostingClassifier(random_state=10)
#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred = model_GradientBoosting.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print()
print("Confusion Matrix:")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()
print("Classification Report: ")
print(classification_report(Y_test,Y_pred))
print()
acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

#%%
y_pred_prob=model_GradientBoosting.predict_proba(X_test)
print(y_pred_prob)

print(list(zip(Y_test,Y_pred)))

for a in np.arange(0,1,0.05):                         #(0,1,0.01)-----for incrementing in steps of 0.01
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
        cfm[1,0]," , type 1 error:", cfm[0,1])

#%%Changing the threshold to 0.7
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.7:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

cfm=confusion_matrix(Y_test,y_pred_class)
print("Confusion Matrix:")
print(cfm)
print("Classification Report: ")
print()
print(classification_report(Y_test,y_pred_class))
print()
acc = accuracy_score(Y_test,y_pred_class)
print("Accuracy of the model: ",acc)

#%%Loss Calculation
test_df1.isnull().sum()

df1 = pd.DataFrame()
df1['member_id'] = test_df1['member_id']
df1['loan_amnt'] = test_df1['loan_amnt']
df1['Y_test'] = Y_test
df1['Y_pred'] = y_pred_class
df1.isnull().sum()
df1.shape

df1['Y_test'].value_counts()
df1['Y_pred'].value_counts()

loss1 = df1.loc[(df1['Y_test']==0.0) & (df1['Y_pred']==1)]
loss1.shape
loss1.isnull().sum()
type(loss1)
loss1.columns

loss2 = df1.loc[(df1['Y_test']==1.0) & (df1['Y_pred']==0)]
loss2.shape
loss2.isnull().sum()
type(loss2)
loss2.columns

loss = pd.concat([loss1, loss2])
loss.shape
loss.isnull().sum()

loss['loan_amnt']
loss['loan_amnt'].sum()

#%%
from sklearn import metrics

#fpr, tpr, z = metrics.roc_curve(Y_test,y_pred_class)
fpr, tpr, z = metrics.roc_curve(Y_test,y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)

import matplotlib.pyplot as plt
#%matplotlib inline
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#%%
from sklearn.metrics import precision_recall_curve 

precision, recall, thresholds = precision_recall_curve(Y_test, y_pred_prob[:,1])

# create plot
plt.plot(precision, recall, label='Precision-recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-recall curve')
plt.xlim([0.735, 1])
plt.ylim([0, 1.02])
plt.legend(loc="lower left")

# save figure
#plt.savefig('precision_recall.png', dpi=200)

#%%

#5.Predicting using the AdaBoostClassifier

#Running AdaBoostClassifier -> In AdaBoost we can specify the algorithm that we want it to run

from sklearn.ensemble import AdaBoostClassifier

#model_Adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
#                                    n_estimators=100, random_state=10)

model_Adaboost = AdaBoostClassifier(base_estimator=LogisticRegression(),
                                    n_estimators=10, random_state=10)

#fit the model on the data and predict the values
model_Adaboost.fit(X_train,Y_train)

Y_pred = model_Adaboost.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print()
print("Confusion Matrix:")
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print()
print("Classification Report: ")
print(classification_report(Y_test,Y_pred))
print()
acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)
#%%
y_pred_prob=model_Adaboost.predict_proba(X_test)
print(y_pred_prob)
print(list(zip(Y_test,Y_pred)))

for a in np.arange(0,1,0.05):                         #(0,1,0.01)-----for incrementing in steps of 0.01
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
        cfm[1,0]," , type 1 error:", cfm[0,1])


#%%
    
import statsmodels.formula.api as sm

col = data5.columns

#create a fitted model with all three features
lm_model = sm.ols(formula = 'default_ind ~ loan_amnt + funded_amnt + term + int_rate + installment + grade + member_id' , data = data5).fit()

#print the coefficients
print(lm_model.params)
print(lm_model.summary())
#%%
