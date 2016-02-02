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
import sympy
from sympy.functions.special.tensor_functions import KroneckerDelta
from sklearn.tree import DecisionTreeClassifier
# generate cartesian product of 25 pairs of -ones and ones
R=product([-1,1], repeat=16)
training_data=list(R)
dim1=len(training_data)
training_labels=[]
def Hamiltonian(data):
    # define unnormalized Gibbs measure of the ferromagnetic configuration
    sz=np.shape(data)
    E=0.0
    # compute horizontal interaction energies
    for i in range(sz[0]):
         for j in range(sz[0]):
            E-=data[i,j]*(data[i,(j+1)%sz[0]]+data[(i+1)%sz[0],j])
    return E        
# error for full dataset [0.92006853321127036, 0.84903594496999291, 0.78891385667785552, 0.733462007934086, 0.679338775302614, 0.63999872851185036]    
def Hamiltonian2(data):
    sz=np.shape(data)
    sm=0.0
    # compute horizontal interaction energies
    for i in range(sz[0]):
        for j in range(sz[1]-1):
            val1=data[i,j]
            val2=data[i,j+1]
            sm=sm-KroneckerDelta(val1,val2)
    # compute vertical interaction energies        
    for j in range(sz[1]):
        for i in range(sz[0]-1):
            val1=data[i,j]
            val2=data[i+1,j]
            sm=sm-KroneckerDelta(val1,val2)
    return int(sm)    
     
    
# compute Hamiltonian of each system
# create lattice structures
for i in range(dim1):
    config=training_data[i]
    config=np.reshape(config,(4,4))
    training_labels.append(Hamiltonian(config))
# map Hamiltonian data into classes    

classes=np.unique(training_labels)
num_classes=len(classes)
labels=np.arange(0,num_classes)
dict={}
dict_back={}
for i in range(num_classes):
    dict[classes[i]]=labels[i]
    dict_back[labels[i]]=classes[i]
for i in range(len(training_labels)):
    val=training_labels[i]
    training_labels[i]=abs(dict[val])

training_100=training_data
labels_100=training_labels   
print 'Computed training labels (Hamiltonians)'

def get_valid_set(training_data,training_labels):
    prop_train=0.9
    index=np.arange(0,len(training_data))
    random.shuffle(index)
    ind_train=int(np.around(len(training_data)*prop_train))
    training_data=[training_data[i] for i in index]
    training_labels=[training_labels[i] for i in index]
    train_X=training_data[:ind_train]
    train_y=training_labels[:ind_train]
    valid_X=training_data[ind_train:]
    valid_y=training_labels[ind_train:]  
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
    #test_X=training_data[ind_train:]
    #test_y=training_labels[ind_train:]  
    dim1=len(train_X)
    dim2=len(train_X[0])
    train_X=np.reshape(train_X, (dim1,dim2))
    train_y=np.reshape(train_y, dim1)
    #dim1=len(test_X)
    #dim2=len(test_X[0])
    #test_X=np.reshape(test_X, (dim1,dim2))
    #test_y=np.reshape(test_y, dim1)

    train=train_X, train_y
    #test=test_X, test_y
    return train    

[train,valid]=get_valid_set(training_data,training_labels)
train_data=train[0]
train_labels=train[1]
valid_set_x=valid[0]
valid_set_y=valid[1]
#keep real energy values in validation dataset y
valid_set_y_en=np.zeros(len(valid_set_y))
for j in range(len(valid_set_y)):
    En=dict_back[int(valid_set_y[j])]
    valid_set_y_en[j]=En

# random sampling
err=np.zeros((7,100))
err_ave=[] # average error over 1000 random drawings 
l=0 

def write_to_csv(data):# specify a csv file where you want to write you results
    with open('/home/nportman/RV_IsingModel/results_class.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', dialect='excel')#, delimiter='\n ',  counts)
        for x in data:        
            #csvwriter.writerow([x,])
            csvwriter.writerow(x)
        csvfile.close()

for percentage in np.arange(0.5,1.1,0.1): 
    print 'Building decision tree for training data percentage', str(percentage)
    for k in range(100): 
        if percentage==1.1:
            train_set_x=training_100
            train_set_y=labels_100
        else:    
            #print 'epoch', str(k)
            train_set=partition(train_data,train_labels,percentage) #get training subset     
            train_set_x = train_set[0]
            train_set_y = train_set[1]

        sz=np.shape(train_set_x)
        #_______________________________________________________________________
        #nn = Classifier(
        #    layers=[
        #        Layer("Tanh", units=sz[1]),
        #        Layer("Linear")],
        #        learning_rate=0.02,
        #        n_iter=10,
        #        batch_size=2)
        #nn.fit(train_set_x, train_set_y)
        #y_pred=nn.predict(test_set_x)
        #_______________________________________________________________________
        clf=DecisionTreeClassifier(criterion='entropy')
        clf=clf.fit(train_set_x,train_set_y)
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
            En_true=dict_back[int(x)]
            En_pred=dict_back[int(y_pred[I])]
            row.append([x, En_true, y_pred[I],En_pred])             
            if En_true!=0:
                rel_error.append(np.abs(En_true-En_pred)/np.abs(En_true))
            else:
                rel_error.append(np.abs(En_true-En_pred))
            I=I+1    
        err[l,k]=np.mean(rel_error)          
        #err_mse[l,k]=np.mean((y_pred-test_set_y)**2)  
    print '.... computing error of prediction for the training dataset percentage', str(percentage) 
    err_ave.append(np.mean(err[l,:]))
    print err_ave
    l=l+1
f = gzip.open('dt4c_errors.pkl.gz','wb')
cPickle.dump(err, f, protocol=2)
f.close()
#write_to_csv(row)  
#[0.91661545112399567, 0.85129663818533219, 0.79044708066320835, 0.73672408707150849, 0.68756385413487953, 0.62642482962058776]  