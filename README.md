# CNN-KM
Convolutional Neural Network for feature extraction and K-means for clustering.
Using the <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">Cifar-10 dataset</a> on <a href="https://github.com/BVLC/caffe/" target="_blank">Caffe</a>.

## How to Use

### Training
First, following the <a href="https://caffe.berkeleyvision.org/gathered/examples/cifar10.html" target="_blank">Caffe Cifar-10 Tutorial</a>, download and convert the data into leveldb format for training the network using the scripts provided in Caffe: /data/cifar10/get\_cifar10.sh and /examples/cifar10/create\_cifar10.sh. Run these scripts from the root directory of Caffe. This should give a mean.binaryproto file along with the leveldb data, remember the path of this file or copy it somewhere handy since it will be used later. Now, train the network using one of the training scripts in the Caffe examples\/cifar10 directory.

### Running
After training, you should have a .caffemodel file that holds the weights of the network. It is convenient to copy this and the deploy .prototxt corresponding to whichever training .prototxt you used to this repository's directory along with the mean.binaryproto file from earlier. You will also need the file batches.meta.txt, which contains the label names, this should have come with the data downloaded earlier, and should be able to be found in $CAFFE_ROOT/data/cifar10. You will also need images to test with, for this the Cifar-10 test images at <a href="https://www.kaggle.com/c/cifar-10/data">Kaggle</a> work well. Now set all the paths to point to the correct files in cifar10\_feat\_KM.py and the parameters. Finally, run main.py. After a few minutes, the results should be saved.   

