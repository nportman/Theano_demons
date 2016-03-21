
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
    
def image(data): # run this for testing imsave
    from scipy import misc
    X = data
    misc.imsave('image1.png', X)

def create_subdir(classes,nr):
    r='/home/nportman/UOIT_research/data'
    dr=str(nr)+'by'+str(nr)
    new_r=os.path.join(r,dr)    
    os.mkdir(new_r)
    for folder in classes:
        os.mkdir(os.path.join(new_r,str(folder)))

def calc(data,nr):
    #nr=input('Enter the dimension N of the square lattice -->')
    Hs=[]
    matrices={}
    for k in range(len(data)):
        config=data[k] 
        conf=np.reshape(config,(nr,nr))
        rows0=[]        
        for i in range(nr):
            blocks=[conf[i,j]*np.ones((7,7)) for j in range(nr)]
            rows=np.concatenate(blocks,axis=1)
            if i==0:
                rows0=rows
            else:
                rows0=np.concatenate((rows0,rows))
        #misc.imsave('%05d.png' % k, rows0)    
        Hs.append(Hamiltonian(rows0))
        rows0=rows0>0.5        
        matrices[k]=1.0*rows0
    classes=np.unique(Hs)
    print classes
    classes_num=np.arange(0,len(classes))
    cl_counts=[]
    create_subdir(classes_num,nr)
    
    root_dir='/home/nportman/UOIT_research/data'
    dr=str(nr)+'by'+str(nr)
    root_dir=os.path.join(root_dir,dr)    
    dirlist=os.listdir(root_dir)
    for dirs in dirlist:
        if os.path.isdir(os.path.join(root_dir,dirs))==True:
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
    calc(training_data,nr) 
    
if __name__=='__main__':
    main()    
              