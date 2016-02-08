# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 23:29:41 2016

@author: nportman
"""

# full (-1,1) dataset generation
#import sknn
#from sknn.mlp import Regressor, Layer
from itertools import product
import numpy as np
import random
import gzip
import cPickle
import MH_dataset_generator as mh

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


nr=input('Enter the dimension N of the square lattice -->')
reps=nr*nr # number of lattice sites

# generate complete dataset
(training_data,training_H,training_M)=mh.complete_dataset(reps)
train_labels=mh.gen_train_labels(training_data, training_H, training_M)
[train,test]=mh.partition(training_data,train_labels)
train_set=train[0]
train_y=train[1]
test_set=test[0]
test_y=test[1] 
train_data=mh.get_MH_samples(train_set)
print "... Computed training dataset"
test_data=mh.get_MH_samples(test_set)
print "... Computed testing dataset"

expr=input('Enter "H" for Hamiltonian energy estimation and "M" for magnetization -->')
if expr=="H":
    train_labels=train_data[:,-2] # Hamiltonian
    test_labels=test_data[:,-2]

else:
    train_labels=train_data[:,-1]
    test_labels=test_data[:,-1]
    
# map Hamiltonian data into classes    
train_dat=train_data[:,:-2]
train_dat=scale(train_dat,axis=0)

#pca.fit(train_dat)
#new_data=pca.fit_transform(train_dat)
#__________________________________________________________________________________    
classes=np.unique(np.hstack((train_labels,test_labels)))
num_classes=len(classes)
labels=np.arange(0,num_classes)
dict={}
dict_back={}
for i in range(num_classes):
    dict[classes[i]]=labels[i]
    dict_back[labels[i]]=classes[i]
# mapping Hamiltonian energies into their ID's    
for i in range(len(train_labels)):
    val=train_labels[i]
    train_labels[i]=np.int(dict[val])
 
for i in range(len(test_labels)):
    val=test_labels[i]
    test_labels[i]=np.int(dict[val])
  
print 'Computed training labels (Hamiltonians)'
#________________________________________________
# plot PC-transformed data
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

# identify different energy subsets
def visualize(classes,new_data):
    subsets={}
    for i in range(len(classes)):
        subsets[i]=new_data[train_labels==float(i),:]
    new_d=subsets[1]    
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(new_data[:,0], new_data[:,1], new_data[:,2])
    pyplot.savefig('en_all.png')
    pyplot.show()
#_________________________________________________
# partitioning into training and testing
def partition(training_data,training_labels, percentage):
    prop_train=percentage
    index=np.arange(0,len(training_data))
    random.shuffle(index)
    ind_train=int(np.around(len(training_data)*prop_train))
    training_data=[training_data[i] for i in index]
    training_labels=[training_labels[i] for i in index]
    train_X=training_data[:ind_train]
    train_y=training_labels[:ind_train] 
    dim1=len(train_X)
    dim2=len(train_X[0])
    train_X=np.reshape(train_X, (dim1,dim2))
    train_y=np.reshape(train_y, dim1)

    train=train_X, train_y
    #test=test_X, test_y
    return train    

valid_set_x=test_data[:,:-2]
valid_set_x=scale(valid_set_x,axis=0)
#new_valid_set_x=pca.transform(valid_set_x)
valid_set_y=test_labels
# random sampling
# average error over 1000 random drawings 
 

def write_to_csv(data):# specify a csv file where you want to write you results
    with open('/home/nportman/RV_IsingModel/results_class.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', dialect='excel')#, delimiter='\n ',  counts)
        for x in data:        
            #csvwriter.writerow([x,])
            csvwriter.writerow(x)
        csvfile.close()
 
print 'Building decision tree'
for k in range(1): 
    #print 'epoch', str(k)     
    train_set_x = train_dat
    train_set_y = train_labels

    sz=np.shape(train_set_x)
    clf=DecisionTreeClassifier(criterion='entropy')
    clf.fit(train_set_x,train_set_y)
    #clf=DecisionTreeRegressor()
    y_pred=clf.predict(valid_set_x)
    # compute mse error
    diff=np.array(y_pred)-np.array(valid_set_y)
    res=np.where(diff!=0)
    #failures=len(res[0])
    #err[l,k]=float(failures)/len(y_pred)        
    rel_error=[]
    I=0
    row=[]
    for x in valid_set_y:
        En_true=dict_back[np.int(x)]
        En_pred=dict_back[np.int(y_pred[I])]
        row.append([x, En_true, y_pred[I],En_pred])             
        if En_true!=0:
            rel_error.append(np.abs(En_true-En_pred)/np.abs(En_true))
        else:
            rel_error.append(np.abs(En_true-En_pred))
        I=I+1    
    err=np.mean(rel_error)           
    print '.... computing error of prediction'
    print err

