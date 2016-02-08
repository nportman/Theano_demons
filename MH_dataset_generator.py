# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:32:52 2016

@author: nportman
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 00:40:14 2016

@author: nportman

"""
# full (-1,1) dataset generation
#import sknn
#from sknn.mlp import Classifier, Layer
from itertools import product
import numpy as np
import random
import gzip
import cPickle
from sklearn.tree import DecisionTreeClassifier
import ising_model_princeton2 as I
from random import randint


def Hamiltonian(data):
    # define the energy of the ferromagnetic configuration
    sz=np.shape(data)
    E=0.0
    # compute horizontal interaction energies
    for i in range(sz[0]):
         for j in range(sz[0]):
            E-=data[i,j]*(data[i,(j+1)%sz[0]]+data[(i+1)%sz[0],j])
    return E        

def Magn(data):
    M=np.sum(data)
    return M
    
def complete_dataset(reps):
# generate cartesian product of 16 pairs of -ones and ones
    R=product([-1,1], repeat=reps)
    training_data=list(R)
    dim1=len(training_data)
    training_H=[]
    training_M=[]
    # compute Hamiltonian of each system
    # create lattice structures
    for i in range(dim1):
        config=training_data[i]
        config=np.reshape(config,(4,4))
        training_H.append(Hamiltonian(config))
        training_M.append(Magn(config))
    return (training_data,training_H,training_M)
    
def gen_train_labels(training_data, training_H, training_M):
    expr=input('Enter "H" for Hamiltonian energy estimation and "M" for magnetization -->')
    if expr=="H":
        #choose the seconf last column in the dataset
        train_labels=training_H # Hamiltonian
    else:
        train_labels=training_M
    
    # map Hamiltonian or magnetization data into classes    
    classes=np.unique(train_labels)
    num_classes=len(classes)
    labels=np.arange(0,num_classes)
    dict={}
    dict_back={}
    for i in range(num_classes):
        dict[classes[i]]=labels[i]
        dict_back[labels[i]]=classes[i]
    for i in range(len(train_labels)):
        val=train_labels[i]
        train_labels[i]=abs(dict[val])

    #training_100=training_data
    #labels_100=train_labels   
    print 'Computed training labels (Hamiltonians or Magnetizations)'
    return train_labels
#______________________________________________________________________________
def partition(training_data,train_labels):
    prop_train=0.9
    index=np.arange(0,len(training_data))
    random.shuffle(index)
    ind_train=int(np.around(len(training_data)*prop_train))
    training_data=[training_data[i] for i in index]
    train_labels=[train_labels[i] for i in index]
    train_X=training_data[:ind_train]
    train_y=train_labels[:ind_train]
    valid_X=training_data[ind_train:]
    valid_y=train_labels[ind_train:]  
    dim1=len(train_X)
    dim2=len(train_X[0])
    train_X=np.reshape(train_X, (dim1,dim2))
    train_y=np.reshape(train_y, dim1)
    dim1=len(valid_X)
    dim2=len(valid_X[0])
    valid_X=np.reshape(valid_X, (dim1,dim2))
    valid_y=np.reshape(valid_y, dim1)

    train=train_X, train_y
    valid=valid_X, valid_y
    return (train, valid) 
    
#____________________________________________________________________________

# having separated into training and testing datasets we run MH algo at 10 random seeds
# per temperature with 2000 configurations
def get_MH_samples(data):
    temp_val=[x for x in np.arange(0.001,20.0, 0.1)]
    iters=2000
    N_seeds=10    
    nr=4
    
    keys=['lattice','H','M']
    count=0
    all_M=[]
    all_H=[]
    repeats=np.zeros(len(temp_val))
    R=[]
    for T in temp_val:
        print T
        for n in range(N_seeds):
            a=I.Ising_lattice(nr)
            a.random_conf(data)
            #a.diagram()
            for k in range(iters): # num of MH steps
                i=randint(0,nr-1)
                j=randint(0,nr-1)
                En=a.cond_spin_flip(i,j,T)
                # record configuration and its properties
            
                #print "new configuration", new_config
                if En!=int(0.0):
                    repeats[count]=repeats[count]+1
                if k>=1000:
                    R.append(np.hstack((np.reshape(a._spins,(nr*nr)),a._E,a._M)))
                    all_M.append(a._M)
                    all_H.append(a._E)
                else:
                    continue

         #a.diagram()                
        count=count+1
    result=R
    result=np.array(result)
    result=result.reshape((len(result),nr*nr+2))
    return result
    

def histogr(output):    
    import pylab as P
    # the histogram of the data with histtype='step'
    P.figure()
    n, bins, patches = P.hist(output, 50, normed=1, histtype='stepfilled')
    P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    P.savefig('Hamil_trainnew.png')     
 

    #histogr(all_H)
 
  
    
