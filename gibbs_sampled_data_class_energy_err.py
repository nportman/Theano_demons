
"""
Created on Wed Jan 27 00:41:43 2016

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
import gibbs_sampler as gs
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# generate testing dataset
(a,nr,iters,temp_val,n_walks)=gs.input_data()
test_data=gs.generate_dataset(a,nr,iters,temp_val,n_walks)
# generate training dataset
(a,nr,iters,temp_val,n_walks)=gs.input_data()
dataset=gs.generate_dataset(a,nr,iters,temp_val,n_walks)
pca=PCA(n_components=3)
# create training labels

expr=input('Enter "H" for Hamiltonian energy estimation and "M" for magnetization -->')
if expr=="H":
    #choose the seconf last column in the dataset
    train_labels=dataset[:,-2] # Hamiltonian
    test_labels=test_data[:,-2]
    method="classification"
else:
    train_labels=dataset[:,-1]
    test_labels=test_data[:,-1]
    method="regression"
# map Hamiltonian data into classes    
train_data=dataset[:,:-2]
train_data=scale(train_data,axis=0)
pca.fit(train_data)
new_data=pca.fit_transform(train_data)
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

#[train,valid]=get_valid_set(training_data,training_labels)

valid_set_x=test_data[:,:-2]
valid_set_x=scale(valid_set_x,axis=0)
new_valid_set_x=pca.transform(valid_set_x)
valid_set_y=test_labels
# random sampling
err=np.zeros((7,10))
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
    for k in range(10): 
        #print 'epoch', str(k)
        train_set=partition(train_data,train_labels,percentage) #get training subset     
        train_set_x = train_set[0]
        train_set_y = train_set[1]

        sz=np.shape(train_set_x)
        #_______________________________________________________________________
        #nn = Regressor(
        #    layers=[
        #        Layer("Tanh", units=sz[1]),
        #        Layer("Tanh")],
        #        learning_rate=0.02,
        #        n_iter=10,
        #        batch_size=2)
        #nn.fit(train_set_x, train_set_y)
        #y_pred=nn.predict(valid_set_x)
        #_______________________________________________________________________
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
        err[l,k]=np.mean(rel_error)          
        #err_mse[l,k]=np.mean((y_pred-test_set_y)**2)  
    print '.... computing error of prediction for the training dataset percentage', str(percentage) 
    err_ave.append(np.mean(err[l,:]))
    print err_ave
    l=l+1
# for PC features with prediction of M: [0.58600039520632519, 0.58663713402306406,
# 0.58663564484557473, 0.58139713979506991, 0.5806999247878547, 0.57981761598554604]    
f = gzip.open('dt5_errors.pkl.gz','wb')
cPickle.dump(err, f, protocol=2)
# for no PC features, Magnetization :[1.23090821164319, 1.2341450621356629, 1.2265436507785299, 1.23060747126214, 1.2304768219589151, 1.2363765443211761]
f.close()
