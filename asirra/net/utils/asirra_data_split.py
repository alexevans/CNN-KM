import os
import numpy
import shutil

asirra_dir = '/path/to/untarred/asirra/PetImages/'
train_val_dir = '/path/to/output/dir/'
val_size = 5000

os.mkdir(train_val_dir+'train')
os.mkdir(train_val_dir+'val')

#get cat and dog train and val splt
cats_files = os.listdir(asirra_dir+'Cat')
cats_val = numpy.random.choice(cats_files, int(val_size/2), False)
cats_train = list(set(cats_files)-set(cats_val))

dogs_files = os.listdir(asirra_dir+'Dog')
dogs_val = numpy.random.choice(dogs_files, int(val_size/2), False)
dogs_train = list(set(dogs_files)-set(dogs_val))

#write cat and dog train ans val files and copy files to dirs
val_file = open(train_val_dir+'val.txt', 'a+')
train_file = open(train_val_dir+'train.txt', 'a+')

print('Saving val samples for Cat')
for file in (cats_val):
  shutil.copy(asirra_dir+'Cat/'+file, train_val_dir+'val/cat_'+file)
  val_file.write('cat_'+file+' 0\n')

print('Saving val samples for Dog')
for file in (dogs_val):
  shutil.copy(asirra_dir+'Dog/'+file, train_val_dir+'val/dog_'+file)
  val_file.write('dog_'+file+' 1\n')

print('Saving train samples for Cat')
for file in (cats_train):
  shutil.copy(asirra_dir+'Cat/'+file, train_val_dir+'train/cat_'+file)
  train_file.write('cat_'+file+' 0\n')

print('Saving train samples for Dog')
for file in (dogs_train):
  shutil.copy(asirra_dir+'Dog/'+file, train_val_dir+'train/dog_'+file)
  train_file.write('dog_'+file+' 1\n')


