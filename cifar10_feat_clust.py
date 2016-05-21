import numpy
from PIL import Image
import caffe
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn import metrics
import time
from copy import deepcopy
import lmdb


#get image samples
def get_image_arr():
  images = []
  count=0
  for file in os.listdir(path=images_dir):
    images.append(caffe.io.load_image(images_dir+'/'+file))
    count+=1
    if(count>=num_samples):
      break
  return numpy.array(images), 

#get image samples from lmdb 
#also returns labels for each image, useful for metrics
def get_image_arr_lmdb(images_lmdb, num_samples):
  ims=[]
  labels=[]
  crs = lmdb.open(images_lmdb).begin().cursor()
  caffe_dat = caffe.proto.caffe_pb2.Datum()
  count=0
  for k, v in crs:
    caffe_dat.ParseFromString(v)
    labels.append(caffe_dat.label)
    #need to change the image to 32,32,3 shape for display
    roll_img = numpy.rollaxis(caffe.io.datum_to_array(caffe_dat), 0, 3)
    ims.append(roll_img)
    count+=1
    if(count>=num_samples):
      break
  ims1=numpy.array(ims, numpy.float32)
  ims1/=255
  return ims1, numpy.array(labels)


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
def get_predictions(input_images, net_definition, trained_weights, mean_file, feat_layer, num_samples):
  cnn = caffe.Classifier(net_definition, trained_weights, image_dims=(32,32), mean=mean_as_numpy(mean_file), raw_scale=255, channel_swap=(2,1,0))
  predictions=[]
  features=[]
  for x in range(0, num_samples):
    predictions.append(numpy.argmax(cnn.predict([input_images[x]])))
    out = cnn.forward()
    features.append((cnn.blobs[feat_layer].data).flatten())
  return numpy.array(predictions), numpy.array(features)

#get clusters of feature array
def get_clusters(feat_arr, num_clusters, mode):
  if (mode=='kmeans'):
    kmeans = KMeans(n_clusters=num_clusters).fit(feat_arr)
    return kmeans.labels_
  elif (mode=='agglomerative'):
    agg = AgglomerativeClustering(n_clusters=num_clusters).fit(feat_arr)
    return agg.labels_
  elif (mode=='birch'):
    birch = Birch(n_clusters=num_clusters).fit(feat_arr)
    return birch.labels_
  else:
    pass

#returns label names of the label numbers from batches.meta file
def get_label_names(names_file):
  names=[]
  fil = open(names_file, 'r')
  for lin in fil.readlines():
    names.append(lin)
  return names

#Save images in folders corresponding to their cluster
#Images are saved in format <CNN prediction>_<sample number>.png
def save_clustered(cluster_labels, ground_truth, images, predictions, label_names, num_clusters, num_samples, mode):
  timestmp=time.strftime('%c')
  os.mkdir(mode+'_Results_'+timestmp)
  os.chdir(mode+'_Results_'+timestmp)
  for x in range(0,num_clusters):
    os.mkdir(str(x))
  count=0
  if (mode=='dbscan'):
    os.mkdir('noisy')
  for x in range(0,num_samples):
    os.chdir(str(cluster_labels[x]))
    im = Image.fromarray(images[x])
    im.save(str(label_names[predictions[x]])+'_'+str(count)+'.png')
    count+=1
    os.chdir('..')
  ARI_clust = metrics.adjusted_rand_score(ground_truth, cluster_labels)
  ARI_cnn = metrics.adjusted_rand_score(ground_truth, predictions)
  homogeneity = metrics.homogeneity_score(ground_truth, cluster_labels)
  completeness = metrics.completeness_score(ground_truth, cluster_labels)
  fil = open('metrics.txt', 'w')
  fil.write('Adjusted Rand Index Score of ' + mode + ': ' + str(ARI_clust))
  fil.write('\nAdjusted Rand Index Score of CNN: ' + str(ARI_cnn))
  fil.write('\nHomogeneity Score of ' + mode + ': ' + str(homogeneity))
  fil.write('\nCompleteness Score of ' + mode + ': ' + str(completeness))



