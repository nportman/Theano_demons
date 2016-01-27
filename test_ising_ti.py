# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:08:00 2016

@author: nportman
"""
# this script performs a random walk through the space of all possible configurations
# and records intermediate ones for each temperature in the range (0-15)

# also, it creates the dataset with translation-invariant energies 
import numpy as np
import ising_model_princeton as I
from random import randint
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cPickle
import gzip

nr=input('Enter the dimension N of the square lattice -->')
expr=input('Enter "testing"/"training" for testing/training dataset generation -->')
if expr=="testing":
    temp_ran=input('Enter the temperature range [a,b] of the testing dataset -->')
    a=temp_ran[0]
    b=temp_ran[1]
    temp_val=[b*np.exp(-x) for x in np.arange(a, b, 0.1)]
    iters=100
else:
    temp_val=[20*np.exp(-x) for x in np.arange(0.0,20.0,0.1)]
    iters=1000    
a=I.Ising_lattice(nr)
a.random_spins()
a.diagram()
matr=a._spins

def translate(matr1, matr2):
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

keys=['lattice','H','M']
outer_set={}

rows1=[]
rows2=[]
rows3=[]

all_M=[]
all_H=[]
count=0
numelems=nr*nr
indexlist=np.arange(0,numelems) # default index ordering if "inorder" is chosen

for T in temp_val:
    print T
    a.random_spins()
    rows1=[]
    rows2=[]
    rows3=[]
    inner_set={'lattice':[],'H':[],'M':[]}
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
            config2=np.reshape(a._spins,(nr*nr))
            rows1.append(config2)
            rows2.append(a._E)
            rows3.append(a._M)
            for k in range(nr-1):

                # generate vertical and horizontal translations of the lattice
                [matr_1, matr_2]=translate(matr1, matr2)
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
# plotting data    
def show_results(outer_set,temp_val, all_M, all_H):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    L=len(outer_set)
    ax1.set_xlim(0, np.max(temp_val))
    ax1.set_ylim(np.min(all_M), np.max(all_M))
    ax2.set_xlim(0,len(outer_set[0]['H'])) # max temperature
    ax2.set_ylim(np.min(all_H), np.max(all_H))
    ax3.set_xlim(0,len(outer_set[L-1]['H'])) # min temperature
    ax3.set_ylim(np.min(all_H), np.max(all_H))
  
    curve1=Line2D([],[],color='blue')
    ax1.add_line(curve1)
    curve2=Line2D([],[],color='red')
    ax2.add_line(curve2)
    curve3=Line2D([],[],color='black')
    ax3.add_line(curve3)

    # find average magnetization per each temperature
    ave_M=[]
    for i in range(len(outer_set)):
        ave_M.append(np.mean(outer_set[i]['M']))
    curve1.set_data(temp_val,ave_M)
    #curve1.set_data(range(0,len(outer_set[0]['M'])),outer_set[0]['M'])
    curve2.set_data(range(0,len(outer_set[0]['H'])),outer_set[0]['H']) 
    curve3.set_data(range(0,len(outer_set[L-1]['H'])),outer_set[L-1]['H'])        
    plt.draw()

def histogr(output):    
    import pylab as P
    # the histogram of the data with histtype='step'
    P.figure()
    n, bins, patches = P.hist(output, 50, normed=1, histtype='stepfilled')
    P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    
train_labels=outer_set[0]['H'] # hot configuration
histogr(train_labels)    

# form and pickle the dataset

L=len(outer_set)
dataset=[]
for i in range(L):
    for j in range(len(outer_set[i]['H'])):
        dataset.append(np.hstack((outer_set[i]['lattice'][j],outer_set[i]['H'][j],outer_set[i]['M'][j])))
dataset=np.array(dataset)    
dataset=dataset.reshape((len(dataset),nr*nr+2))
    
def pickling(expr, dataset):
    if expr=="testing":
        f = gzip.open('testing_data_ising_ti.pkl.gz','wb') 
    else:    
        f = gzip.open('training_data_ising_ti.pkl.gz','wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()  
    
def load_data(data_file):
    with gzip.open(data_file,'rb') as f:
        data=f.read()    
    return data    
