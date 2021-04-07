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
parser.add_argument("video_model_file")
parser.add_argument("sound_model_file")
parser.add_argument("comb_model_file")
parser.add_argument("visual_feat_dir")
parser.add_argument("sound_feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load mlp model
  sound_mlp = pickle.load(open(args.sound_model_file, "rb"))
  video_mlp = pickle.load(open(args.video_model_file, "rb"))
  comb_mlp = pickle.load(open(args.comb_model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    video_feat_filepath = os.path.join(args.visual_feat_dir, video_id + args.feat_appendix)
    sound_feat_filepath = os.path.join(args.sound_feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(video_feat_filepath) or not os.path.exists(sound_feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      sound = np.genfromtxt(sound_feat_filepath, delimiter=",", dtype="float")
      compressed_sound = np.average(sound, axis=0)
      sound_prob = sound_mlp.predict_proba(compressed_sound.reshape(1,-1))

      video = np.genfromtxt(video_feat_filepath, delimiter=";", dtype="float")
      video_prob = video_mlp.predict_proba(video.reshape(1,-1))

      complete_feat = np.concatenate((sound_prob, video_prob), axis=1)
      print(complete_feat[0].shape)

      feat_list.append(complete_feat[0])

  X = np.array(feat_list)

  # 3. Get predictions
  # (num_samples) with integer
  pred_classes = comb_mlp.predict(X)

  # 4. save for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))