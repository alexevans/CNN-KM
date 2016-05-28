import cifar10_feat_clust as cnncl

num_samples = 10000
num_clusters = 10
images_dir = ''
images_lmdb = 'cifar10_test_lmdb/'
net_definition = 'cifar10_full.prototxt'
trained_weights = 'trained_full/cifar10_full_iter_70000.caffemodel'
mean_file = 'mean.binaryproto'
names_file = 'batches.meta.txt'
feat_layer = 'ip1'
mode='birch'

if __name__=='__main__':
  label_names = cnncl.get_label_names(names_file)
  in_ims, gt = cnncl.get_image_arr_lmdb(images_lmdb, num_samples)
  disp_ims = cnncl.get_disp_images(in_ims)
  preds, feats = cnncl.get_predictions(in_ims, net_definition, trained_weights, mean_file, feat_layer, num_samples)
  
  print('Starting K-Means with 10 clusters')
  clustered_km10 = cnncl.get_clusters(feats, 10, 'kmeans')
  cnncl.save_clustered(clustered_km10, gt, disp_ims, feats, preds, label_names, 10, num_samples, 'kmeans')

  print('Starting Agglomerative with 10 clusters')
  clustered_agg10 = cnncl.get_clusters(feats, 10, 'agglomerative')
  cnncl.save_clustered(clustered_agg10, gt, disp_ims, feats, preds, label_names, 10, num_samples, 'agglomerative')

  print('Starting Birch with 10 clusters')
  clustered_birch10 = cnncl.get_clusters(feats, 10, 'birch')
  cnncl.save_clustered(clustered_birch10, gt, disp_ims, feats, preds, label_names, 10, num_samples, 'birch')

  print('Starting K-Means with 5 clusters')
  clustered_km5 = cnncl.get_clusters(feats, 5, 'kmeans')
  cnncl.save_clustered(clustered_km5, gt, disp_ims, feats, preds, label_names, 5, num_samples, 'kmeans')

  print('Starting Agglomerative with 5 clusters')
  clustered_agg5 = cnncl.get_clusters(feats, 5, 'agglomerative')
  cnncl.save_clustered(clustered_agg5, gt, disp_ims, feats, preds, label_names, 5, num_samples, 'agglomerative')

  print('Starting Birch with 5 clusters')
  clustered_birch5 = cnncl.get_clusters(feats, 5, 'birch')
  cnncl.save_clustered(clustered_birch5, gt, disp_ims, feats, preds, label_names, 5, num_samples, 'birch')

  print('Starting K-Means with 2 clusters')
  clustered_km2 = cnncl.get_clusters(feats, 2, 'kmeans')
  cnncl.save_clustered(clustered_km2, gt, disp_ims, feats, preds, label_names, 2, num_samples, 'kmeans')

  print('Starting Agglomerative with 2 clusters')
  clustered_agg2 = cnncl.get_clusters(feats, 2, 'agglomerative')
  cnncl.save_clustered(clustered_agg2, gt, disp_ims, feats, preds, label_names, 2, num_samples, 'agglomerative')

  print('Starting Birch with 2 clusters')
  clustered_birch2 = cnncl.get_clusters(feats, 2, 'birch')
  cnncl.save_clustered(clustered_birch2, gt, disp_ims, feats, preds, label_names, 2, num_samples, 'birch')
