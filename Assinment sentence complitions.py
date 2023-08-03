#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests


# In[23]:


baseurl = "https://www.opindia.com/2023/02/ashwin-jadeja-india-australia-comprehensive-defeat-bgt/"


# In[24]:


requests.get(url)


# In[25]:


pageurls = []


# In[ ]:





# In[26]:


x = 1
for i in range(2,32,1):
    x = x + 1
    url = baseurl + str(x) + "/"
    pageurls.append(url)


# In[27]:


pageurls


# In[ ]:





# In[28]:


url = "https://www.opindia.com/2023/02/iqbal-khan-pfi-extremist-who-wanted-to-convert-india-into-an-islamic-state-by-2047/"
rawdata = requests.get(url)
soup = BeautifulSoup(rawdata.content)


# In[19]:


def headtimeart(url):
    data = requests.get(url)
    rawdata = data.content
    soup = BeautifulSoup(rawdata)
    heading = soup.find_all('h1')[0].text
    time =  soup.find_all("time")[0].text
    import re
    date = re.sub("[,]","",time).strip()
    
    art = ""
    for i in soup.find_all("div",class_="tdb-block-inner td-fix-index"):
        for j in i.find_all("p"):
            art = art + j.text
    return heading,date,art


# In[33]:


h,d,A = headtimeart("https://www.opindia.com/2023/02/ashwin-jadeja-india-australia-comprehensive-defeat-bgt/")


# In[34]:


data  = A


# In[35]:


data


# In[ ]:





# In[ ]:





# In[68]:


def cleaner(data):
    q0 = data.lower()
    import re
    q1 = re.sub("[^a-zA-Z0-9 ]","",q0)
    q2 = q1.split(" ")
    return q2


# In[69]:


cleaner(data)


# In[70]:


data1 = data.upper()


# In[71]:


data1


# In[72]:


tokenized = data1.split(" ")


# In[73]:


tokenized[5]


# In[74]:


len(tokenized)


# In[75]:


inp = []
op = []
for i in range(3,1076,1):
    w1 = tokenized[i-3]
    w2 = tokenized[i-2]
    w3 = tokenized[i-1]
    w4 = tokenized[i]
    inp.append(w1+" "+w2+" "+w3)
    op.append(w4)


# In[76]:


import pandas as pd
Q = pd.DataFrame([inp,op]).T
Q.columns=["Input","Output"]


# In[77]:


Q


# In[78]:


from numpy import unique


# In[79]:


uw = unique(tokenized)


# In[80]:


pd.set_option("display.max_columns",100)


# In[81]:


pd.DataFrame(columns=uw)


# # dimensions for cube

# In[82]:


n_lines = Q.shape[0]


# In[83]:


n_uw = len(uw)


# In[84]:


seq = 3


# In[85]:


print(n_lines,n_uw,seq)


# # create an empty cube filled with 0's

# In[86]:


from numpy import zeros
x_cube = zeros((n_lines,n_uw,seq),dtype=int)


# In[87]:


x_cube.shape


# # Enumerates

# In[89]:


list(enumerate(inp))


# In[90]:


uw = list(uw)


# In[94]:


for line_no,line in list(enumerate(inp)):
    for word_no,word in list(enumerate(line.split(" "))):
        x_cube[line_no,uw.index(word),word_no]=1


# In[95]:


x_cube


# In[96]:


for line_no, line in list(enumerate(inp)):
    for word_no, word in list(enumerate(line.split(" "))):
        print("Line_no",line_no,"Word_index_UW",uw.index(word),"seq",word_no,word)


# In[108]:


x_cube.shape


# # prepare output

# In[103]:


Y = zeros((n_lines,n_uw),dtype=int)


# In[104]:


for line_no,word in enumerate(op):
    Y[line_no,uw.index(word)]=1


# In[106]:


Y.shape


# # Deep learning model

# In[107]:


from keras.layers import LSTM,Dense
from keras.models import Sequential


# In[109]:


nn = Sequential()
nn.add(LSTM(238,input_shape=(486,3)))
nn.add(Dense(486,activation="softmax"))


# In[111]:


nn.compile(loss="categorical_crossentropy",metrics=["accuracy"])
model = nn.fit(x_cube,Y,validation_split=0.2,epochs=20)


# In[113]:


W = input("Enter words: ")
if(len(W.split(" "))!=3):
    print("3 words hone chahiye!")
    
W = W.upper()


# In[114]:


X_inp = zeros((1,n_uw,seq),dtype=int)
for word_no, word in list(enumerate(W.split(" "))):
    X_inp[0,uw.index(word),word_no]=1


# In[115]:


len(nn.predict(X_inp)[0])


# In[116]:


from numpy import argmax
argmax(nn.predict(X_inp)[0])


# In[117]:


pred = uw[429]
pred


# In[ ]:




