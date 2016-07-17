import cifar10_feat_clust as cnncl

num_samples = 10000
num_clusters = 10
images_dir = ''
images_lmdb = 'asirra_val_lmdb/'
net_definition = 'net/deploy.prototxt'
trained_weights = ''
mean_file = 'asirra_mean.binaryproto'
names_file = 'net/batches.meta.txt'
feat_layer = 'fc7'

if __name__=='__main__':
  label_names = cnncl.get_label_names(names_file)
  in_ims, gt = cnncl.get_image_arr_lmdb(images_lmdb, num_samples)
  disp_ims = cnncl.get_disp_images(in_ims)
  preds, feats = cnncl.get_predictions(in_ims, net_definition, trained_weights, mean_file, feat_layer, num_samples)

  print('Starting K-Means with 2 clusters')
  clustered_km2 = cnncl.get_clusters(feats, 2, 'kmeans')
  cnncl.save_clustered(clustered_km2, gt, disp_ims, feats, preds, label_names, 2, num_samples, 'kmeans')

  print('Starting Agglomerative with 2 clusters')
  clustered_agg2 = cnncl.get_clusters(feats, 2, 'agglomerative')
  cnncl.save_clustered(clustered_agg2, gt, disp_ims, feats, preds, label_names, 2, num_samples, 'agglomerative')

  print('Starting Birch with 2 clusters')
  clustered_birch2 = cnncl.get_clusters(feats, 2, 'birch')
  cnncl.save_clustered(clustered_birch2, gt, disp_ims, feats, preds, label_names, 2, num_samples, 'birch')
