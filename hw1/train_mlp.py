#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pickle
import argparse
import sys

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat1_dir")
parser.add_argument("feat2_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat1_filepath = os.path.join(args.feat1_dir, video_id + args.feat_appendix)
    feat2_filepath = os.path.join(args.feat2_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(feat2_filepath) and os.path.exists(feat1_filepath):
      # print(feat_filepath)
      original = np.genfromtxt(feat1_filepath, delimiter=";", dtype="float")
      # print(original.shape)
      compressed = np.genfromtxt(feat2_filepath, delimiter=",", dtype="float")
      # print(compressed)
      compressed_sum = np.average(compressed, axis=0)
      print(compressed_sum.shape)
      complete_feat = np.concatenate((original ,compressed_sum))
      print(complete_feat.shape)

      feat_list.append(compressed_sum)

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", max_iter=5000)
  # clf = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf")
  # clf = RandomForestClassifier(max_depth=3, random_state=0)
  clf.fit(X_train, y_train)
  print(clf.score(X_test, y_test))
  # scores = cross_val_score(clf, X, y, cv=2)
  plot_confusion_matrix(clf, X_test, y_test)
  plt.savefig('confusion_matrix.png')
  clf.fit(X, y)
  print(clf.score(X_test, y_test))
  # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
