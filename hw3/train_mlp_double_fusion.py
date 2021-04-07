#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pickle
import argparse
import sys

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("visual_feat_dir_1")
parser.add_argument("visual_feat_dir_2")
parser.add_argument("sound_feat_dir_1")
parser.add_argument("sound_feat_dir_2")
parser.add_argument("list_videos")
parser.add_argument("video_model_file")
parser.add_argument("sound_model_file")
parser.add_argument("prob_output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--max_iter", type=int, default=300)

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  video_feat_list = []
  sound_feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    video_feat_filepath_1 = os.path.join(args.visual_feat_dir_1, video_id + args.feat_appendix)
    sound_feat_filepath_1 = os.path.join(args.sound_feat_dir_1, video_id + args.feat_appendix)
    video_feat_filepath_2 = os.path.join(args.visual_feat_dir_2, video_id + args.feat_appendix)
    sound_feat_filepath_2 = os.path.join(args.sound_feat_dir_2, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(video_feat_filepath_1) and os.path.exists(sound_feat_filepath_1)
    and os.path.exists(video_feat_filepath_2) and os.path.exists(sound_feat_filepath_2):
      sound_1 = np.genfromtxt(sound_feat_filepath_1, delimiter=",", dtype="float")
      compressed_sound_1 = np.average(sound_1, axis=0)
      sound_2 = np.genfromtxt(sound_feat_filepath_2, delimiter=",", dtype="float")
      compressed_sound_1 = np.average(sound_2, axis=0)
      sound = np.concatenate((sound_1, sound_2), axis=1)

      video_1 = np.genfromtxt(video_feat_filepath_1, delimiter=";", dtype="float")
      video_2 = np.genfromtxt(video_feat_filepath_2, delimiter=";", dtype="float")
      video = np.concatenate((video_1, video_2), axis=1)

      video_feat_list.append(video)
      sound_feat_list.append(sound)

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(label_list))
  y = np.array(label_list)
  X_sound = np.array(sound_feat_list)
  X_video = np.array(video_feat_list)

  sound_mlp = pickle.load(open(args.sound_model_file, "rb"))
  video_mlp = pickle.load(open(args.video_model_file, "rb"))

  sound_prob = sound_mlp.predict_proba(X_sound)
  video_prob = video_mlp.predict_proba(X_video)

  complete_proba = np.concatenate((sound_prob, video_prob), axis=1)

  X_train, X_test, y_train, y_test = train_test_split(complete_proba, y, test_size=0.2, random_state=0)
  combined_clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=1000)
  combined_clf.fit(X_train, y_train)

  print(combined_clf.score(X_test, y_test))
  # scores = cross_val_score(clf, X, y, cv=2)
  plot_confusion_matrix(combined_clf, X_test, y_test)
  plt.savefig('confusion_matrix_late.png')
  combined_clf.fit(complete_proba, y)
  print(combined_clf.score(X_test, y_test))

  # save trained MLP in output_file
  pickle.dump(combined_clf, open(args.prob_output_file, 'wb'))
  print('MLP classifier trained successfully')