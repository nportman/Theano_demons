# Date: January 08, 2016
#Author: Nataliya Portman
# This script loads zipped data created by a user in the format appropriate for theano framework  
import numpy as np
import cPickle
import gzip
import os
import sys

import theano

def _shared_dataset(data_xy):
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
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y
    
def get_data(filename):

    f =gzip.open(filename,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an np.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #np.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

   
    test_set_x, test_set_y = _shared_dataset(test_set)
    valid_set_x, valid_set_y = _shared_dataset(valid_set)
    train_set_x, train_set_y = _shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
        (test_set_x, test_set_y)]
    return rval
