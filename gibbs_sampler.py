# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:50:22 2016

@author: nportman

"""
# this script performs a random walk through the space of all possible configurations
# and records intermediate ones for each temperature in the range (0-15)

# also, it creates the dataset with translation-invariant energies 
import numpy as np
import ising_model_princeton as I
from random import randint
import cPickle
import gzip

def input_data():
    nr=input('Enter the dimension N of the square lattice -->')
    expr=input('Enter "testing"/"training" for testing/training dataset generation -->')
    if expr=="testing":
        temp_ran=input('Enter the temperature range [a,b] of the testing dataset -->')
        a=temp_ran[0]
        b=temp_ran[1]
        temp_val=[b*np.exp(-x) for x in np.arange(a, b, 0.1)]
        iters=10
        n_walks=1
    else:
        temp_ran=input('Enter the temperature range [a,b] of the training dataset -->')
        a=temp_ran[0]
        b=temp_ran[1]
        temp_val=[b*np.exp(-x) for x in np.arange(a, b, 0.1)]
        iters=1000
        n_walks=2
    a=I.Ising_lattice(nr)
    a.random_spins()
    a.diagram()
    return (a,nr,iters,temp_val,n_walks)
    
def translate(matr1, matr2, nr):
    new_a=np.ones((nr,nr))    
    for i in range(nr):
        for j in range(nr):
            new_a[i,j]=matr1[nr-i-1,j]
    matr_h=new_a
 
    for i in range(nr):
        for j in range(nr):
            new_a[j,i]=matr2[j,nr-i-1]
    matr_v=new_a    
    return matr_v, matr_h    

def generate_dataset(a,nr,iters,temp_val,n_walks):
    outer_set={}

    all_M=[]
    all_H=[]
    count=0
    
    for T in temp_val:
        print T
        rows1=[]
        rows2=[]
        rows3=[]
        inner_set={'lattice':[],'H':[],'M':[]}

        for N in range(n_walks):        
            a.random_spins()
        #a.diagram()
            for iter in range(iters):
                i=randint(0,nr-1)
                j=randint(0,nr-1)
                En=a.cond_spin_flip(i,j,T);
                # record configuration and its properties
                config=np.reshape(a._spins,(nr*nr))
                rows1.append(config)
                rows2.append(a._E)
                rows3.append(a._M)
                matr1=a._spins
                matr2=a._spins

                # check the condition if all spins are aligned
                if all(x==config[0] for x in config):
                    pass
                else: 
                    # generate diagonal translations of hte lattice

                    matr_d=np.transpose(matr1) #45 degree rotation
                    config2=np.reshape(matr_d,(nr*nr))
                    rows1.append(config2)
                    rows2.append(a._E)
                    rows3.append(a._M)
                    for k in range(nr-1):

                        # generate vertical and horizontal translations of the lattice
                        [matr_1, matr_2]=translate(matr1, matr2, nr)
                        matr1=matr_1
                        matr2=matr_2
                        rows1.append(np.reshape(matr_1,(nr*nr)))
                        rows2.append(a._E)
                        rows3.append(a._M)
                        rows1.append(np.reshape(matr_2,(nr*nr)))
                        rows2.append(a._E)
                        rows3.append(a._M)            
        #a.diagram()                
        inner_set['lattice']=rows1
        inner_set['H']=rows2 
        inner_set['M']=rows3           
        outer_set[count]=inner_set
        all_M=all_M+outer_set[count]['M']
        all_H=all_H+outer_set[count]['H']
        count=count+1
    #_____________________________________________________________________________

    # form and pickle the dataset

    L=len(outer_set)
    dataset=[]
    for i in range(L):
        for j in range(len(outer_set[i]['H'])):
            dataset.append(np.hstack((outer_set[i]['lattice'][j],outer_set[i]['H'][j],outer_set[i]['M'][j])))
    dataset=np.array(dataset)    
    dataset=dataset.reshape((len(dataset),nr*nr+2))
    return dataset