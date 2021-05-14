#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import time
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

import pickle 
import math as m
import random as r
import numpy as np


# In[3]:


df = pd.read_excel("data/raw_data.xlsx",skiprows=1,usecols=["微博正文","点赞数","转发数","评论数"])
df.head()


# In[5]:


df = df.rename(columns = {'微博正文':'text', '点赞数':'like', '转发数':'comment','评论数':'forward'}, inplace=False)
df.head()


# In[29]:


f = open("data/tfidf.txt", 'r',encoding="utf8")
allsentences = f.readlines()

for i in range(len(allsentences)):
  allsentences[i] = allsentences[i].strip('\n')
y = df[['like','comment','forward']]

f.close()


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error




# In[31]:

train_errs=[]
test_errs=[]


for i in range(9,20):
    #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(max_features=100000)
    #该类会统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    #将文本转为词频矩阵并计算tf-idf
        
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(allsentences, y, train_size=.7, test_size=.3, random_state=0, shuffle=True, stratify=None)
    X_train = tf_idf_transformer.fit_transform(vectorizer.fit_transform(X_train))

    # normalize y_train, y_test
    y_train = np.log(y_train)
    y_train = y_train.replace(-np.Inf,0)
    y_test = np.log(y_test)
    
    y_test = y_test.replace(-np.Inf,0)
    
    timestamp = time.asctime( time.localtime(time.time()) )
    print(timestamp, ": start to train at depth= ",i+2)
    model = RandomForestRegressor(n_estimators=150,max_depth=i+2, random_state=0)
    
    model.fit(X_train,y_train)
    timestamp = time.asctime( time.localtime(time.time()) )
    
    print(timestamp, ": training done")
    
    #get error
    y_pred = model.predict(X_train)
    train_err = mean_absolute_error(np.exp(y_train.values), np.exp(y_pred),multioutput="raw_values")
    print("MAE ON TRAINING SET EXPONENTIAL: ",train_err)
    train_errs.append(train_err)
    
    
    idr = "Depth "+str(i+2)
    #write to file
    f = open("train_errs.txt","a",encoding='utf-8')
    for i in range(3):
        
        f.write(idr)
        f.write(" ")
        f.write(str(train_err[i]))
        f.write(" ")
    f.close()
    
    X_test = tf_idf_transformer.transform(vectorizer.transform(X_test))
    y_pred_test = model.predict(X_test)
    test_err = mean_absolute_error(np.exp(y_test.values), np.exp(y_pred_test),multioutput="raw_values")
    print("MAE ON TEST SET EXPONENTIAL: ",test_err)
    test_errs.append(test_err)
    
    f = open("test_errs.txt","w+",encoding='utf-8')
    for i in range(3):
        f.write(idr)
        f.write(" ")
        f.write(str(test_err[i]))
        f.write(" ")
    f.close()


# In[24]:



print(len(train_errs))
with open("test_errs.txt",'w',encoding = 'utf-8') as f:
    for i in range(len(test_errs)):
        for j in range(3):
            f.write(str(train_errs[i][j]))
            f.write(" ")
            


# In[10]:





print("MAE ON TRAINING SET BEFORE TAKING EXPONENTIAL: ",mean_absolute_error(y_train.values, y_pred,multioutput="raw_values"))
print("MAE ON TRAINING SET AFTER TAKING EXPONENTIAL: ",mean_absolute_error(np.exp(y_train.values), np.exp(y_pred),multioutput="raw_values"))



print("MAE ON TESTING SET BEFORE TAKING EXPONENTIAL: ",mean_absolute_error(y_test.values, y_pred_test,multioutput="raw_values"))
print("MAE ON TESTING SET AFTER TAKING EXPONENTIAL: ",mean_absolute_error(np.exp(y_test.values), np.exp(y_pred_test),multioutput="raw_values"))


# In[11]:


# y_pred.shape
# y_train.shape

# diff = np.absolute(np.subtract(np.exp(y_train),np.exp(y_pred)))
# diff

# find observations with likes less than 1000
import matplotlib.pyplot as plt

mask = np.exp(y_train).values[:,0]<=1000
# np.exp(y_train).values[mask,:]
# print(type(y_pred))
y_train_masked = np.exp(y_train.values)[mask,[0]] 
y_pred_masked = np.exp(y_pred)[mask,[0]]
y_pred_masked.shape[0]

# draw a plot
x_axis = np.arange(0,y_pred_masked.shape[0],1)
plt.plot(x_axis,y_pred_masked,label="prediciton")
plt.plot(x_axis,y_train_masked,label="truth")
plt.xlim([0, 100])
plt.legend()
plt.show()


# In[12]:


test_text = "你好 明天 岁末 年初 年终 盘点 一言难尽 新年 flag 仍然 计日可期 需要 抓住 联结 历史 未来 每一个 当下 今日 胜 昨日 奋斗 唯一 解 一时 千载千载 一时 面对 未来 值得 庆幸 努力 奔跑 都是 追梦人 再见 2018 你好 2019 原图"
x_test = [test_text]
x_test = tf_idf_transformer.transform(vectorizer.transform(x_test))
y_result = model.predict(x_test)
y_result = np.exp(y_result)
y_result


# In[15]:


# pickle.dump(model,open('randomForest.pkl','wb'))
get_ipython().system(' ')

