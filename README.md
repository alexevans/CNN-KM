# CNN-KM
Convolutional Neural Network for feature extraction with unsupervised feature clustering.
Using the <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">Cifar-10 dataset</a> on <a href="https://github.com/BVLC/caffe/" target="_blank">Caffe</a>.

## How to Use

###Cifar10

#### Training
First, following the <a href="https://caffe.berkeleyvision.org/gathered/examples/cifar10.html" target="_blank">Caffe Cifar-10 Tutorial</a>, download and convert the data into leveldb format for training the network using the scripts provided in Caffe: /data/cifar10/get\_cifar10.sh and /examples/cifar10/create\_cifar10.sh. Run these scripts from the root directory of Caffe. This should give a mean.binaryproto file along with the leveldb data, it is convenient to copy these files to the net directory. Now, train the network using train.sh.

#### Running
After training, you should have a .caffemodel file that holds the weights of the network. You will also need the file batches.meta.txt, which contains the label names, this should have come with the data downloaded earlier, and should be able to be found in $CAFFE_ROOT/data/cifar10. You can use the cifar10 train lmdb you generated earlier to supply test images and their labels. Now set all the variables to point to the correct files in main.py. Finally, run main.py. After a few minutes, the results should be saved along with clustering metrics.   

