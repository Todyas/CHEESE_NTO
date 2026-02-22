#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from PIL import Image
from io import BytesIO
import base64

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyRegressor

import pickle


# In[2]:


test = pd.read_csv("test.csv")


# In[3]:


test["text"] = test["text"].fillna("")


# In[4]:


with open('text_vectorizer.pickle', 'rb') as f:
    text_vectorizer = pickle.load(f)


# In[5]:


def img_vectorizer(photo_base64):        
    img = np.array(Image.open(BytesIO(base64.b64decode(photo_base64))))    
    
    s = img.shape
    if (len(s) == 2):
        img = np.repeat(img, 3)
        
    h, w = s[0], s[1]
    img.resize((h * w, 3))    
    
    stats = []
    stats.append(np.array([h,w]))
    stats.append(img.min(axis=0))
    stats.append(img.max(axis=0))
    stats.append(img.mean(axis=0))
    stats.append(img.std(axis=0))
    stats.append(np.median(img, axis=0))
    cm = np.corrcoef(img.T)
    stats.append(cm[np.triu_indices(len(cm), k = 1)])
    return np.concatenate(stats) 


# In[6]:


X_img = np.vstack(test['photo'].map(img_vectorizer))


# In[7]:


X_text = text_vectorizer.fit_transform(test['text']).toarray()


# In[8]:


X = np.hstack([X_text, X_img])


# In[9]:


target = ["like", "comment", "hide", "expand", "open_photo", "open", "share_to_message"]


# In[10]:


prediction = pd.DataFrame()


# In[11]:


with open('model.pickle', 'rb') as f:
    model = pickle.load(f)


# In[12]:


for column in target:
    reg = model[column]
    y = reg.predict(X)
    prediction[column] = y * test['view'] 


# In[13]:


prediction.to_csv("submission.csv", index=False)


