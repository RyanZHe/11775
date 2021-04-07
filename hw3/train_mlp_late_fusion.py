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
parser.add_argument("visual_feat_dir")
parser.add_argument("sound_feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("video_output_file")
parser.add_argument("sound_output_file")
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
    video_feat_filepath = os.path.join(args.visual_feat_dir, video_id + args.feat_appendix)
    sound_feat_filepath = os.path.join(args.sound_feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(video_feat_filepath) and os.path.exists(sound_feat_filepath):
      sound = np.genfromtxt(sound_feat_filepath, delimiter=",", dtype="float")
      compressed_sound = np.average(sound, axis=0)
      video = np.genfromtxt(video_feat_filepath, delimiter=";", dtype="float")

      video_feat_list.append(video)
      sound_feat_list.append(compressed_sound)

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(label_list))
  y = np.array(label_list)
  X_sound = np.array(sound_feat_list)
  X_video = np.array(video_feat_list)

  sound_clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=1000)
  sound_clf.fit(X_sound, y)

  video_clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=1000)
  video_clf.fit(X_video, y)

  sound_prob = sound_clf.predict_proba(X_sound)
  video_prob = video_clf.predict_proba(X_video)

  complete_proba = np.concatenate((sound_prob, video_prob), axis=1)

  X_train, X_test, y_train, y_test = train_test_split(complete_proba, y, test_size=0.2, random_state=45)
  combined_clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=1000)
  combined_clf.fit(X_train, y_train)

  print(combined_clf.score(X_test, y_test))
  # scores = cross_val_score(clf, X, y, cv=2)
  plot_confusion_matrix(combined_clf, X_test, y_test)
  plt.savefig('confusion_matrix_late.png')
  combined_clf.fit(complete_proba, y)
  print(combined_clf.score(X_test, y_test))

  # save trained MLP in output_file
  pickle.dump(video_clf, open(args.video_output_file, 'wb'))
  print('MLP classifier trained successfully')

  pickle.dump(sound_clf, open(args.sound_output_file, 'wb'))
  print('MLP classifier trained successfully')

  pickle.dump(combined_clf, open(args.prob_output_file, 'wb'))
  print('MLP classifier trained successfully')