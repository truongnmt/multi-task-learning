# Script to reformat the IMDB data from matlab to numpy arrays.
# Flattens all images to gray scale and resizes to (128,128)
# Reference: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
# usage:
# ./imdb_preprocess --partial 1000
# (Will return <1000 samples from the training set as a result of filtering.
#
# Note: If --partial is not specified it will attempt to process
#   the entire data set.

import os
import argparse
import pickle
import numpy as np
import scipy.io as sio
import scipy.misc as spm
import datetime
import matplotlib.image as plt

IMG_DIR = r'/home/sl/coding/cnn/gr2/datasets/imdb_crop'
MAT_FILE = r'./data/imdb.mat'


def reformat_date(mat_date):
  """ Extract only the year.

    Necessary for calculating the age of the individual in the image.
  Args:
    mat_date - raw date format.
  """
  # Take account for difference in convention between matlab and python.
  dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
  return dt


def create_path(path):
  """ Creates path to full path to image.

  Args:
    path - incomplete path
  Returns:
    fullpath
  """
  return os.path.join(IMG_DIR, path[0])


def reformat_imdb():
  """ Opens .mat file and reformats.

    Matlab struct format to dictionary of numpy arrays.
  Returns:
    imdb_dict - dict of numpy arrays.
  """
  mat_struct = sio.loadmat(MAT_FILE)
  data_set = [data[0] for data in mat_struct['imdb'][0, 0]]

  keys = ['dob',
    'photo_taken',
    'full_path',
    'gender',
    'name',
    'face_location',
    'face_score',
    'second_face_score',
    'celeb_names',
    'celeb_id'
    ]

  imdb_dict = dict(zip(keys, np.asarray(data_set)))
  imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
  imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

  # Add 'age' key to the dictionary
  imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

  return imdb_dict


def create_and_dump(imdb_dict, partial):
  """ Creates dictionary of inputs and labels and pickles.
  Args:
    img_paths - full path to image.
  """

  raw_path = imdb_dict['full_path']
  raw_age = imdb_dict['age']
  raw_sface = imdb_dict['second_face_score']

  if partial != 0:
        raw_path = imdb_dict['full_path'][:partial]
        raw_age = imdb_dict['age'][:partial]
        raw_sface = imdb_dict['second_face_score'][:partial]

  age = []
  imgs = []
  for i, sface in enumerate(raw_sface):
    if not np.isnan(sface) and raw_age[i] >= 0:
      age.append(raw_age[i])
      imgs.append(raw_path[i])

  # Convert images path to images.
  imgs = [np.asarray(spm.imresize(spm.imread(img_path, flatten=1), (128, 128)), dtype=np.float32)
    for img_path in imgs
    ]

  data = {'image_inputs': np.array(imgs),
      'age_labels': np.array(age)
      }

  print("Number of samples reduced to: {}".format(len(data['image_inputs'])))
  with open(os.path.join(IMG_DIR,"pkl_folder/imdb_data_{}.pkl".format(partial)),'wb') as f:
    pickle.dump(data, f)


def main():
  parser=argparse.ArgumentParser(description='IMDB data reformat script.')
  parser.add_argument('--partial', '-p', type=int, default=0,
                        help='The number of samples to use.')
  parser.add_argument('--imgpath', '-ip', default=r'/home/sl/coding/cnn/gr2/datasets/imdb_crop',
                        help='Image folder path.')
  parser.add_argument('--labelpath', '-lp', default=r'./data/imdb.mat',
                        help='Label file path.')
  args=parser.parse_args()

  IMG_DIR = args.imgpath
  MAT_FILE = args.labelpath

  imdb_dict=reformat_imdb()
  print("Dictionary created...")

  print("Converting {} samples. (0=all samples)".format(args.partial))
  create_and_dump(imdb_dict, args.partial)

  print("File dumped to imdb_data_{}.pkl.".format(args.partial))


if __name__ == "__main__":
  main()
