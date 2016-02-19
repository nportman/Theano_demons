
# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 18th 00:40:14 2016

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
import matplotlib.pyplot as plt
from operator import itemgetter
import MH_dataset_generator as mh

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

(complete, Hs, Ms)=complete_dataset(16)
#const=8.6173324*10**(-5)
C=np.arange(0.1,40.0,0.1)
#T=C+273.15
Hs=np.array(Hs)
levels=np.unique(Hs)
weight=np.zeros(len(levels))
y_val1=np.zeros(len(C))
y_val2=np.zeros(len(C))
k=0
for labels in levels:
    ids=np.where(Hs==labels)
    weight[k]=len(ids[0])
    k=k+1
    
prob={}
i=0
for t in C:
    print t
    num=weight*np.exp(-levels/(t))
    denom=np.sum(num)
    prob[i]=num/denom
    y_val1[i]=prob[i][0]
    y_val2[i]=prob[i][1]
    i=i+1
    
col=[]
# display a graph of probability of the most likely state as a function of temperature

plt.plot(C, y_val1, 'r', label='en-32')
plt.plot(C, y_val2, 'b', label='en-24')
plt.xlabel('temperature')
plt.ylabel('Boltzmann probability')
#plt.axis([0, 600000.0, 0.0,15.0])
plt.axis([0.1, 40.0, 0.0, 1.0])
plt.legend(loc='upper center')
plt.savefig('temp1-2.png')    

#_____________________________________________________________________________________
# MH algorithm sampling IDs in the complete space
train_labels=mh.gen_train_labels(complete, Hs, Ms)
train=[]
for i in range(len(complete)):    
    row2=complete[i]
    train.append(np.hstack((row2,train_labels[i])))
np.reshape(train,(len(train),17))
train2=sorted(train,key=itemgetter(-1))

def get_MH_sampled_IDs(data,classes,temp): 
    temp_val=[temp]    
    iters=4000
    N_seeds=10    
    nr=4

    count=0
    all_M=[]
    all_H=[]
    #repeats=np.zeros(len(temp_val))
    R=[]
    Len=len(classes)
    dat=np.array(data)
    
    for T in temp_val:

        for n in range(N_seeds):           
            a=I.Ising_lattice(nr)
            a.random_conf(data)
            conf1=a._spins
            En_1=a._E
            M_1=a._M
            #a.diagram()
            for k in range(iters): # num of MH steps
                label=a.choose_level(data,classes) # next candidate from the label group
                ids=np.where(dat[:,-1]==label)
                idd=ids[0]
                group=dat[idd]# look at this level group
                #if len(group)>2:                 
                a.up_or_down2(group)
                conf2=a._spins 
                En_2=a._E
                M_2=a._M
                dE=En_1-En_2
                if (dE<0.0 or (T>0.0 and (np.random.random()<np.exp(-dE/T)))):
                    conf=conf1 # stay at the lattice value (i,j) with the probability
                    En=En_1
                    M=M_1
                    #repeats[count]=repeats[count]+1
                else:
                    conf=conf2 # update lattice value i,j and the corresponding energy                   
                    En=En_2
                    M=M_2
                conf1=conf
                En_1=En
                M_1=M
                if k>=1000:
                    R.append(np.hstack((np.reshape(conf2,(nr*nr)),En,M)))
                    all_M.append(M)
                    all_H.append(En)
               
        
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
    P.savefig('simulnt1.png')     

train_classes=np.arange(0,len(levels),1)
temp=5.0
train_res=get_MH_sampled_IDs(train2,train_classes,temp) 
H=[]
for i in range(len(train_res)):
    H.append(train_res[i][-2])
histogr(H)