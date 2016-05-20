import cifar10_feat_KM as cnnkm

num_samples = 10000
num_clusters = 10
images_dir = ''
images_lmdb = 'cifar10_test_lmdb/'
net_definition = 'cifar10_full.prototxt'
trained_weights = 'trained_full/cifar10_full_iter_70000.caffemodel'
mean_file = 'mean.binaryproto'
names_file = 'batches.meta.txt'
feat_layer = 'ip1'
mode='kmeans'

if __name__=='__main__':
  label_names = cnnkm.get_label_names(names_file)
  in_ims, gt = cnnkm.get_image_arr_lmdb(images_lmdb, num_samples)
  disp_ims = cnnkm.get_disp_images(in_ims)
  preds, feats = cnnkm.get_predictions(in_ims, net_definition, trained_weights, mean_file, feat_layer, num_samples)
  clustered = cnnkm.get_clusters(feats, num_clusters, mode)
  cnnkm.save_clustered(clustered, gt, disp_ims, preds, label_names, num_clusters, num_samples, mode)
