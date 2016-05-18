import cifar10_feat_KM as cnnkm


if __name__=='__main__':
  label_names = cnnkm.get_label_names()
  in_ims = cnnkm.get_image_arr()
  disp_ims = cnnkm.get_disp_images(in_ims)
  preds, feats = cnnkm.get_predictions(in_ims)
  clustered = cnnkm.get_clusters(feats)
  cnnkm.save_clustered(clustered, disp_ims, preds, label_names)
