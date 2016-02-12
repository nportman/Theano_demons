
"""
Created on Mon Feb  8 10:27:58 2016

@author: nportman
"""

#import sknn
#from sknn.mlp import Regressor, Layer
from itertools import product
import numpy as np
import random
import gzip
import cPickle
import MH_dataset_generator as mh
import ising_model_princeton2 as I
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from operator import itemgetter

nr=input('Enter the dimension N of the square lattice -->')
reps=nr*nr # number of lattice sites

# generate complete dataset
(training_data,training_H,training_M)=mh.complete_dataset(reps)
train_labels=mh.gen_train_labels(training_data, training_H, training_M)
#partition into training 90% and testing 10% datasets
[train,test]=mh.partition(training_data,train_labels)
train_set=train[0]
train_y=train[1]
test_set=test[0]
test_y=test[1]
test_classes=np.unique(test_y)
train_classes=np.unique(train_y)
training=[]
testing=[]
for i in range(len(train_set)):
    row=train_set[i]
    training.append(np.hstack((row,train_y[i])))
for i in range(len(test_set)):    
    row2=test_set[i]
    testing.append(np.hstack((row2,test_y[i])))
np.reshape(testing,(len(testing),17))
np.reshape(training,(len(training),17))
    
train2=sorted(training,key=itemgetter(-1))
test2=sorted(testing,key=itemgetter(-1))
# plotting data    
def show_results(outer_set,temp_val, all_M, all_H):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    L=len(outer_set)
    ax1.set_xlim(0, np.max(temp_val))
    ax1.set_ylim(np.min(all_M)-1, np.max(all_M))
    ax2.set_xlim(0,len(outer_set[0]['H'])) # max temperature
    ax2.set_ylim(np.min(all_H)-1, np.max(all_H))
    ax3.set_xlim(0,len(outer_set[L-1]['H'])) # min temperature
    ax3.set_ylim(np.min(all_H)-1, np.max(all_H))
  
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
    plt.savefig('new_dat.png') 
#show_results(outer_set,temp_val, all_M, all_H)

def get_MH_sampled_IDs(data,classes):
    #temp_val=[x for x in np.arange(0.001,20.0, 0.1)]
    temp_val=[0.0,0.1]
    iters=6000
    N_seeds=100    
    nr=4
 
    count=0
    all_M=[]
    all_H=[]
    #repeats=np.zeros(len(temp_val))
    R=[]
    Len=len(classes)
    dat=np.array(data)
    
    for T in temp_val:
        print T
        outer_set={}
        for n in range(N_seeds):
            Hs=[]
            a=I.Ising_lattice(nr)
            a.random_conf(data)
            conf1=a._spins
            En_1=a._E
            M_1=a._M
            #a.diagram()
            for k in range(iters): # num of MH steps
                label=a.choose_level(data,Len) # next candidate from the label group
                ids=np.where(dat[:,-1]==label)
                idd=ids[0]
                group=dat[idd]# look at this level group
                if len(group)>2:                 
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
                if k>4000:
                    R.append(np.hstack((np.reshape(conf,(nr*nr)),En,M)))
                    all_M.append(M)
                    all_H.append(En)
                    Hs.append(En)
            outer_set[n]=Hs
                # record configuration and its properties
            
                #if k>=3000:
                    #R.append(np.hstack((np.reshape(conf,(nr*nr)),En,M)))
                    #all_M.append(M)
                    #all_H.append(En)
                #else:
                    #continue
    
         #a.diagram()
        t1 = np.arange(0, len(outer_set[0]))
        t2 = np.arange(0, len(outer_set[99]))

        plt.figure(1)
        plt.subplot(211)
        plt.plot(t1, outer_set[0], 'b')
        print outer_set[0]
        plt.subplot(212)
        plt.plot(t2, outer_set[99], 'r')
        plt.show()                
        
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
    P.savefig('Hamil_test.png')     

test_res=get_MH_sampled_IDs(test2,test_classes) 
H=[]
for i in range(len(test_res)):
    H.append(test_res[i][-1])
histogr(H)    
 