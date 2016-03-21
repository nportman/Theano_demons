# This script builds testing and training datasets and saves configurations in 
# png format and places them in the directories according to an energy class that 
# they belong 
"""
Created on Tue Mar 15 23:40:23 2016

@author: nportman
"""
from itertools import product
import numpy as np
import random
import gzip
import cPickle
import MH_dataset_generator as mh
import ising_model_princeton2 as I
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import json
from operator import itemgetter
from scipy import misc
import os

def Hamiltonian(data):
    # define the energy of the ferromagnetic configuration
    sz=np.shape(data)
    E=0.0
    # compute horizontal interaction energies
    for i in range(sz[0]):
         for j in range(sz[0]):
            E-=data[i,j]*(data[i,(j+1)%sz[0]]+data[(i+1)%sz[0],j])
    return E 
       
def partition(training_data,train_labels,prop_train):
    #prop_train=0.9
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
    
def image(data):
    from scipy import misc
    X = data
    misc.imsave('image1.png', X)

def create_subdir(classes,flag):
    r='/home/nportman/UOIT_research/data'
    if flag==1:
        dr='train'
    elif flag==2:
        dr='test'
        
    if dr=='train' or dr=='test':
        new_r=os.path.join(r,dr)
    else:
        new_r=r
    gh=classes
    for folder in gh:
        os.mkdir(os.path.join(new_r,str(folder)))

def upsample(data,nr):
    Hs=[]
    for k in range(len(data)):
        config=data[k] 
        conf=np.reshape(config,(nr,nr))
        rows0=[]        
        for i in range(4):
            blocks=[conf[i,j]*np.ones((7,7)) for j in range(4)]
            rows=np.concatenate(blocks,axis=1)
            if i==0:
                rows0=rows
            else:
                rows0=np.concatenate((rows0,rows))
        #misc.imsave('%05d.png' % k, rows0)    
        Hs.append(Hamiltonian(rows0))
    classes=np.unique(Hs)
    return classes
    
def calc(data,classes1,nr,flag):
    #nr=input('Enter the dimension N of the square lattice -->')
    Hs=[]
    matrices={}
    for k in range(len(data)):
        config=data[k] 
        conf=np.reshape(config,(nr,nr))
        rows0=[]        
        for i in range(4):
            blocks=[conf[i,j]*np.ones((7,7)) for j in range(4)]
            rows=np.concatenate(blocks,axis=1)
            if i==0:
                rows0=rows
            else:
                rows0=np.concatenate((rows0,rows))
        #misc.imsave('%05d.png' % k, rows0)    
        Hs.append(Hamiltonian(rows0))
        rows0=rows0>0.5        
        matrices[k]=1.0*rows0
    classes2=np.unique(Hs)
    classes=[val for val in classes1 if val in classes2]
    print classes
    classes_num=np.zeros(len(classes))
    classes_num=classes_num.astype(int)
    dict_c={}
    i=0
    for cl in classes1:
        dict_c[cl]=i
        i=i+1
    j=0    
    for cl in classes:
        ind=dict_c[cl]
        classes_num[j]=ind
        j=j+1
    print classes_num    
    cl_counts=[]
    create_subdir(classes_num,flag)
    root_dir='/home/nportman/UOIT_research/data'
    if flag==1:
        root_dir=os.path.join(root_dir,'train')
    elif flag==2:
        root_dir=os.path.join(root_dir,'test')
        
    dirlist=os.listdir(root_dir)
    for dirs in dirlist:
        if os.path.isdir(os.path.join(root_dir,dirs))==True:
            if int(dirs)>len(classes):
                label=classes[int(dirs)-1] 
            else:
                label=classes[int(dirs)]
            print int(dirs)
            ids=np.where(Hs==label)
            idd=ids[0]
            cl_counts.append(len(idd))# look at this level group
            os.chdir(os.path.join(root_dir,dirs))
            for k in idd:
                matr=matrices[k]
                misc.imsave('%05d.png' % k, matr)   
        else:
            pass
            
    print cl_counts                         
    
     
def main():
    nr=4
    reps=nr*nr # number of lattice sites
    (training_data,training_H,training_M)=mh.complete_dataset(reps)
    classes=upsample(training_data,nr) # unique energy levels of the complete dataset
    prop_train=0.8
    (train,test)=partition(training_data,training_H,prop_train)
    train_data=train[0]
    test_data=test[0]    
    calc(train_data,classes,nr,1) # flag=1 for training
    calc(test_data,classes,nr,2) # flag=2 for testing
    
if __name__=='__main__':
    main()    
              