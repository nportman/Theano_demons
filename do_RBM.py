# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:47:34 2016

@author: nportman
"""

import csv
    
from sklearn.metrics import classification_report
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


filename1='/home/nportman/UOIT_research/Theano_demons/training.csv'
filename2='/home/nportman/UOIT_research/Theano_demons/testing.csv'
def read_file(filename):
    data=[]
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            floats=[float(x) for x in row]
            data.append(floats)
    return data        
            
train=read_file(filename1)
test=read_file(filename2)

def train_test(data):
    X=np.asarray(data)
    X1=X[:,:-2]
    X2=X1>0.5 # transform data into binary format
    En=X[:,-2]
    M=X[:,-1]
    return (X1,X2,En,M)

(X_train_orig, X_train,Y_train_en,Y_train_m)=train_test(train)
(X_test_orig, X_test,Y_test_en,Y_test_m)=train_test(test)

def get_train_labels(En,start_in,end_in):
    train_labels=En
    classes=np.unique(En)
    labels=np.arange(start_in,end_in+1)
    num_classes=len(labels)
    dict={}
    dict_back={}
    for i in range(num_classes):
        dict[classes[i]]=labels[i]
        dict_back[labels[i]]=classes[i]
    for i in range(len(En)):
        val=En[i]
        train_labels[i]=abs(dict[val])
    return train_labels  

train_cl=np.unique(Y_train_en)
test_cl=np.unique(Y_test_en)
if len(test_cl)<len(train_cl):
    if test_cl[0]!=train_cl[0] and test_cl[0]==train_cl[1]:
        start_ind=1 # for testing
        end_ind=len(train_cl)-1
    elif test_cl[0]==train_cl[0] and test_cl[-1]!=train_cl[-1]:
        end_ind=len(train_cl)-2
        start_ind=0
    else:
        start_ind=1
        end_ind=len(train_cl)-2
else:
    start_ind=0
    end_ind=len(train_cl)-1        
        
Y_train=get_train_labels(Y_train_en,0,len(train_cl)-1)
Y_test=get_train_labels(Y_test_en,start_ind,end_ind)

regressor=DecisionTreeRegressor()
classifier=DecisionTreeClassifier()
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, batch_size=10, n_iter=50, verbose=True, random_state=None)
print 'Ready to run the Pipeline'
clf = Pipeline(steps=[('rbm', rbm), ('clf', classifier)])
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
#print 'Score:',(metrics.classification_report(Y_test,Y_pred))
rbm_res=rbm.components_
# Plotting
def plotting_rbm(rbm_res):
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm_res):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((4, 4)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()
    
# error analysisn(errors))
def error_analysis(Y_test,Y_pred):    
    res_rbm=np.abs(Y_test-Y_pred)
    total_n=float(len(res_rbm))
    errors=np.unique(res_rbm)
    rbm_r=np.zeros(len(errors))
    for x in errors:
        ind=np.where(res_rbm==x)
        rbm_r[x]=len(ind[0])/total_n*100.0
        print [int(x), rbm_r[x]]
print 'Prediction of energies, RBM+decision tree classification'  
error_analysis(Y_test,Y_pred)
  
classifier.fit(X_train,Y_train)
Y_pred2=classifier.predict(X_test)
print 'Prediction of energies, raw pixels, decision treew classification'
error_analysis(Y_test,Y_pred2)
    
regressor.fit(X_train_orig,Y_train_m)
Y_pred_m=regressor.predict(X_test_orig) 
res_regr=np.abs(Y_test_m-Y_pred_m)
total_n=float(len(res_regr))
errors=np.unique(res_regr)
regr_r=np.zeros(len(errors))
print 'Prediction of magnetization, decision tree regression, raw pixels'
i=0
for x in errors:
    ind=np.where(res_regr==x)
    regr_r[i]=len(ind[0])/total_n*100.0
    print [int(x), regr_r[i]]
    i=i+1       
    
plotting_rbm(rbm_res)    