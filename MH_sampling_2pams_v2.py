
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
import collections
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from operator import itemgetter
from scipy import stats
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
    
def rand_move(data,N):
    L=len(data)
    ID=np.random.randint(0,L)
    config=data[ID][:-1]
    conf=np.reshape(config,(N,N))
    En=Hamiltonian(conf)
    M=Magn(conf)
    return (config, En, M)
 
def choose_level(data,classes,deck):
    tot_len=np.max(classes)-np.min(classes)+1
    beta=np.random.random()
    if deck==np.min(classes):
        deck=deck+1
    elif deck==np.max(classes):
        deck=deck-1 
    else:           
        if beta<0.5:
            deck=deck-1
        else:    
            deck=deck+1
    return deck   # return  candidate energy level 
    
def choose_level2(classes, deck, freq):
    if deck==np.min(classes):
        deck=deck+1
    elif deck==np.max(classes):
        deck=deck-1
    else:
        pr=np.zeros(2)
        prob1=freq[deck-1]
        prob2=freq[deck+1]
        s=prob1+prob2
        pr[0]=prob1/s
        pr[1]=prob2/s
        xk=np.arange(2)
        custm=stats.rv_discrete(name='custm',values=(xk,pr))
        beta=custm.rvs(size=1)
        
        if beta[0]==0:
            deck=deck-1
        else:
            deck=deck+1
    return deck
              
def get_MH_sampled_IDs(data,classes,freq):
    #temp_val=[x for x in np.arange(0.001,20.0, 0.1)]
    #temp_val=[x for x in np.arange(0.001,20.0, 0.1)]
    temp_val=[40]    
    iters=4000
    N_seeds=10    
    nr=4
    count=0
    all_M=[]
    all_H=[]
    #repeats=np.zeros(len(temp_val))
    R=[]
    dat=np.array(data)
    
    for T in temp_val:
        print T
        outer_set={}
        for n in range(N_seeds):
            Hs=[]
            a1=I.Ising_lattice(nr)
            a1.random_conf(data)
            conf=a1._spins
            En=a1._E
            M=a1._M
            deck=a1._deck #https://www.google.ca/?gws_rd=ssl initialization
            for k in range(iters): # num of MH steps
                #print 'fix this'
                kB = 1.0
                # let us consider another configuration
                label=choose_level2(classes,deck,freq) # next candidate from the label group
                #label=randint(np.max([0,deck-1]),np.min([np.max(classes),deck+1]))                
                ids=np.where(dat[:,-1]==label)
                idd=ids[0]
                group=dat[idd]# look at this level group
                (conf2, En_2, M_2)=rand_move(group,nr)
                # calc whether it goes uphill
                dE=En_2-En
                if (dE<0.0):
                    #accept
                    conf=conf2 # go downhill
                    En=En_2
                    M=M_2
                    flag=1
                    deck=label
                else:  
                    #accept with prob
                    your_random_number = np.random.random()
                    if (your_random_number < np.exp(-dE/(kB*T))):
                        #go uphill
                        conf=conf2 # update lattice value i,j and the corresponding energy                   
                        En=En_2
                        M=M_2
                        flag=1
                        deck=label

#                we have now decided to either stay where we were, or change
#                provided we are past the warm up stage, let's store the result
                if k>=3000:
                    R.append(np.hstack((np.reshape(conf,(nr*nr)),En,M)))
                    all_M.append(M)
                    all_H.append(En)
                    Hs.append(En)
            outer_set[n]=Hs
        
        #t1 = np.arange(0, len(outer_set[0]))
        #t2 = np.arange(0, len(outer_set[99]))

        #plt.figure(1)
        #plt.subplot(211)
        #plt.plot(t1, outer_set[0], 'b')
        #print outer_set[0]
        #plt.subplot(212)
        #plt.plot(t2, outer_set[9], 'r')
        #plt.show()                
        
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
    P.savefig('Hamil_compl_40.png')     

def main():
    nr=input('Enter the dimension N of the square lattice -->')
    reps=nr*nr # number of lattice sites

    # generate complete dataset
    (training_data,training_H,training_M)=mh.complete_dataset(reps)
    train_labels=mh.gen_train_labels(training_data, training_H, training_M)
    # compute frequesncies
    counter=collections.Counter(train_labels)
    val=counter.values()
    freq=np.array(val)/np.float(len(train_labels))

    #partition into training 90% and testing 10% datasets
    #[train,test]=mh.partition(training_data,train_labels)
    #train_set=train[0]
    #train_y=train[1]
    #test_set=test[0]
    #test_y=test[1]
    #test_classes=np.unique(test_y)
    #train_classes=np.unique(train_y)
    #training=[]
    #testing=[]
    #for i in range(len(train_set)):
    #row=train_set[i]
    #training.append(np.hstack((row,train_y[i])))
    #for i in range(len(test_set)):    
    #row2=test_set[i]
    #testing.append(np.hstack((row2,test_y[i])))
    #np.reshape(testing,(len(testing),17))
    #np.reshape(training,(len(training),17))
    
    #train2=sorted(training,key=itemgetter(-1))
    #test2=sorted(testing,key=itemgetter(-1))
    train=[]
    for i in range(len(training_data)):    
        row2=training_data[i]
        train.append(np.hstack((row2,train_labels[i])))
    np.reshape(train,(len(train),17))
    
    train2=sorted(train,key=itemgetter(-1))
    train_classes=np.unique(train_labels)    
    train_res=get_MH_sampled_IDs(train2,train_classes,freq) 
    H=[]
    for i in range(len(train_res)):
        H.append(train_res[i][-2])
    histogr(H)    
 
if __name__=='__main__':
    main()