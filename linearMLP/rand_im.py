#

# MNIST data reader
import numpy as np
import theano
import cPickle
import gzip
data_file1='digits-2.0/mnist/train-images-idx3-ubyte.gz'
data_file2='digits-2.0/mnist/train-labels-idx1-ubyte.gz'
data_file11='digits-2.0/mnist/t10k-images-idx3-ubyte.gz'
data_file12='digits-2.0/mnist/t10k-labels-idx1-ubyte.gz'
def load_mnist_images(data_file):
    with gzip.open(data_file,'rb') as f:
        data=np.frombuffer(f.read(),np.uint8,offset=16)
    data=data.reshape(-1,1,28,28)    
    return data / np.float32(256)

def load_mnist_labels(data_file):
    with gzip.open(data_file,'rb') as f:
        data=np.frombuffer(f.read(),np.uint8,offset=8) 
    return data
X_train=load_mnist_images(data_file1)
y_train=load_mnist_labels(data_file2)
X_test=load_mnist_images(data_file11)
y_test=load_mnist_labels(data_file12)
    
training_labels=np.zeros(60000, dtype=np.float32)
training_data=np.random.uniform(0.0,255.0,size=(60000,1,5,5)) / np.float32(256)
training_data.dtype=np.float32
# to match the format of the input data to MNIST dataset
for i in range(60000):
    image=training_data[i,0]
    training_labels[i]=np.mean(image)    

# partitioning into the training, testing and validation datasets
prop_train=0.8
prop_val=0.1
prop_test=0.1
ind_train=np.around(len(training_data)*prop_train)
ind_val=ind_train+np.around(len(training_data)*prop_val)
train_X=training_data[:ind_train]
train_y=training_labels[:ind_train]
val_X=training_data[ind_train:ind_val]
val_y=training_labels[ind_train:ind_val]
test_X=training_data[ind_val:]
test_y=training_labels[ind_val:]    
train_set=train_X, train_y
val_set=val_X,val_y
test_set=test_X,test_y
dataset=[train_set, val_set, test_set]

f = gzip.open('random_images.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()

