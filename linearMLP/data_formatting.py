# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:55:27 2016

@author: Nataliya Portman
"""
import numpy as np
import cPickle
import gzip
import os
import sys

def shape(data_xy):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    sz=np.size(data_x[0,0])
    dim1=len(data_x)
    dim2=sz
    data_x=np.reshape(data_x, (dim1,dim2))
    data_y=np.reshape(data_y, dim1)

    return data_x, data_y

def get_data(filename):

    f =gzip.open(filename,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
  
    test_set_x, test_set_y = shape(test_set)
    valid_set_x, valid_set_y = shape(valid_set)
    train_set_x, train_set_y = shape(train_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
        (test_set_x, test_set_y)]
    return rval
