# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:50:20 2016

@author: nportman
"""
# this script performs a random walk through the space of all possible configurations
# and records intermediate ones (if they are different form the previous ones) for each temperature in the range (0-15) 
import numpy as np
import ising_model_princeton as I
from random import randint
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
nr=input('Enter the dimension N of the square lattice -->')
a=I.Ising_lattice(nr)
a.random_spins()
#a.diagram()
temp_val=[20*np.exp(-x) for x in np.arange(0.0,20.0,0.1)]

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
    for iter in range(1000):
        i=randint(0,nr-1)
        j=randint(0,nr-1)
        En=a.cond_spin_flip(i,j,T);
        # record configuration and its properties
        rows1.append(np.reshape(a._spins,(nr*nr)))
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
    