import numpy
from PIL import Image
import caffe
import os
from sklearn.cluster import KMeans
import time
from copy import deepcopy

num_samples = 10000
miter = 800
num_clusters = 10
images_dir = '/home/alex/git/caffe/examples/cifar10/cifar10_images'
net_definition = 'cifar10_full.prototxt'
trained_weights = 'trained_full/cifar10_full_iter_70000.caffemodel'
mean_file = 'mean.binaryproto'
names_file = 'batches.meta.txt'
feat_layer = 'ip1'

#get image samples
def get_image_arr():
  images = []
  count=0
  for file in os.listdir(path=images_dir):
    images.append(caffe.io.load_image(images_dir+'/'+file))
    count+=1
    if(count>=num_samples):
      break
  return numpy.array(images)

#convert image samples to displayable uint8 format
def get_disp_images(img_arr):
  imgs1 = deepcopy(img_arr)
  imgs1*=255
  return imgs1.astype(numpy.uint8)

#Convert the mean binaryproto file to numpy format
#https://github.com/BVLC/caffe/issues/290
def mean_as_numpy(meanfile):
  bl = caffe.proto.caffe_pb2.BlobProto()
  orig_mean = open(meanfile, 'rb').read()
  bl.ParseFromString(orig_mean)
  np_mean = numpy.array(caffe.io.blobproto_to_array(bl))
  return np_mean[0]

#gets predictions and features for all samples
def get_predictions(input_images):
  cnn = caffe.Classifier(net_definition, trained_weights, image_dims=(32,32), mean=mean_as_numpy(mean_file), raw_scale=255, channel_swap=(2,1,0))
  predictions=[]
  features=[]
  for x in range(0, num_samples):
    predictions.append(numpy.argmax(cnn.predict([input_images[x]])))
    out = cnn.forward()
    features.append((cnn.blobs[feat_layer].data).flatten())
  return numpy.array(predictions), numpy.array(features)

#get clusters of feature array
def get_clusters(feat_arr):
  kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feat_arr)
  return kmeans.labels_

#returns label names of the label numbers from batches.meta file
def get_label_names():
  names=[]
  fil = open(names_file, 'r')
  for lin in fil.readlines():
    names.append(lin)
  return names

#Save images in folders corresponding to their cluster
#Images are saved in format <CNN prediction>_<sample number>.png
def save_clustered(cluster_labels, images, predictions, label_names):
  timestmp=time.strftime('%c')
  os.mkdir('Cluster_Results_'+timestmp)
  os.chdir('Cluster_Results_'+timestmp)
  for x in range(0,num_clusters):
    os.mkdir(str(x))
  count=0
  for x in range(0,num_samples):
    os.chdir(str(cluster_labels[x]))
    im = Image.fromarray(images[x])
    im.save(str(label_names[predictions[x]])+'_'+str(count)+'.png')
    count+=1
    os.chdir('..')



