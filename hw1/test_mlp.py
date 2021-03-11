#!/bin/python

import argparse
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import sys
import numpy as np

# Apply the MLP model to the testing videos;
# Output prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat1_dir")
parser.add_argument("feat2_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load mlp model
  mlp = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    feat1_filepath = os.path.join(args.feat1_dir, video_id + args.feat_appendix)
    feat2_filepath = os.path.join(args.feat2_dir, video_id + args.feat_appendix)
    if not (os.path.exists(feat1_filepath) and os.path.exists(feat2_filepath)):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      # print(feat_filepath)
      original = np.genfromtxt(feat1_filepath, delimiter=";", dtype="float")
      # print(original)
      # print(original.shape)
      compressed = np.genfromtxt(feat2_filepath, delimiter=",", dtype="float")
      # print(compressed)
      # print(compressed)
      compressed_sum = np.sum(compressed, axis=0)
      # print(compressed_sum.shape)
      complete_feat = np.concatenate((original ,compressed_sum))
      # print(complete_feat.shape)

      feat_list.append(compressed_sum)

  X = np.array(feat_list)

  # 3. Get predictions
  # (num_samples) with integer
  pred_classes = mlp.predict(X)

  # 4. save for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
