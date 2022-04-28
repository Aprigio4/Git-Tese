#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.utils import shuffle


# In[4]:


def flip(X):
  return X[:,:,::-1,:,:]

def split_flip(X):
    
    #separate hemisfiers and flip
    X_HL = X[:,:,:64,:,:]
    X_HR = X[:,:,:64,:,:]
    X_HR = flip(X_HR)
    
    return [X_HL, X_HR]

def augmentate(X,Y):
    
    X = np.concatenate((flip(X), X), axis=0)
    Y = np.concatenate((Y,Y), axis=0)
    
    return shuffle(X,Y)


def merge_data(class1, class0, dir_read):
    
    c1_X_train = np.load(dir_read+class1+'_X_train.npy')
    c1_X_test = np.load(dir_read+class1+'_X_test.npy')
    
    c0_X_train = np.load(dir_read+class0+'_X_train.npy')
    c0_X_test = np.load(dir_read+class0+'_X_test.npy')
    
    c1_Y_test = np.ones(c1_X_test.shape[0])
    c1_Y_train = np.ones(c1_X_train.shape[0])
    
    c0_Y_test = np.zeros(c0_X_test.shape[0])
    c0_Y_train = np.zeros(c0_X_train.shape[0])
    
    X_test = np.concatenate((c1_X_test, c0_X_test), axis=0)
    Y_test = np.concatenate((c1_Y_test, c0_Y_test), axis=0)
    
    X_train = np.concatenate((c1_X_train, c0_X_train), axis=0)
    Y_train = np.concatenate((c1_Y_train, c0_Y_train), axis=0)
    
    X_train, Y_train = shuffle(X_train, Y_train)
    
    return X_train, Y_train, X_test, Y_test



# In[ ]:




