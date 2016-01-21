# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:21:13 2016

@author: Nataliya Portman 
"""

from sknn.mlp import Regressor, Layer
import numpy as np
import gzip
import cPickle

def rand_images(dim1):
    training_labels=np.zeros((dim1,1))
    training_data=np.random.uniform(low=0.0,high=255.0,size=(dim1,1,5,5)) / np.float32(256)
    # to match the format of the input data to MNIST dataset
    for i in range(dim1):
        image=training_data[i,0]
        training_labels[i]=np.mean(image)    

    # partitioning into the training, testing and validation datasets
    prop_train=0.8
    prop_val=0.1
    prop_test=0.1
    ind_train=np.around(len(training_data)*prop_train)
    ind_val=ind_train+np.around(len(training_data)*prop_val)
    train_X=training_data[:ind_train]
    train_y=training_labels[:ind_train]
    val_X=training_data[ind_train:ind_val]
    val_y=training_labels[ind_train:ind_val]
    test_X=training_data[ind_val:]
    test_y=training_labels[ind_val:]    
    train=train_X, train_y
    val=val_X,val_y
    test=test_X,test_y
    return (train, val, test)

def shape(data_xy):

    data_x, data_y = data_xy
    sz=np.size(data_x[0,0])
    dim1=len(data_x)
    dim2=sz
    data_x=np.reshape(data_x, (dim1,dim2))
    data_y=np.reshape(data_y, dim1)
    return data_x, data_y
err_mse=[]  
for dim in 6*10**np.arange(1,6):    
    [train_set, valid_set,test_set]=rand_images(dim)    
    test_set_x, test_set_y = shape(test_set)
    valid_set_x, valid_set_y = shape(valid_set)
    train_set_x, train_set_y = shape(train_set)
    sz=np.shape(train_set_x)
    nn = Regressor(
        layers=[
            Layer("Tanh", units=sz[1]),
            Layer("Linear")],
            learning_rate=0.02,
            n_iter=10,
            batch_size=20)
    nn.fit(train_set_x, train_set_y)
    y_pred=nn.predict(test_set_x)
    # compute mse error
    print '.... computing error of prediction for the dataset size', str(dim) 
    rel_error=[]
    I=0
    for x in test_set_y:
        if x!=0:
            rel_error.append(np.abs(y_pred[I]-x)/np.abs(x))
        else:
            rel_error.append(np.abs(y_pred[I]-x))
        I=I+1    
    err_mse.append(np.mean(rel_error))  
    #err_mse.append(np.mean((y_pred-test_set_y)**2))
    print err_mse
f = gzip.open('mlp_errors.pkl.gz','wb')
cPickle.dump(err_mse, f, protocol=2)
f.close()    
    
