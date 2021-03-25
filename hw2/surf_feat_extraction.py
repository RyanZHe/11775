#!/usr/bin/env python3

import argparse
import os
import sys
import cv2
import pickle
from tqdm import tqdm
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("video_path")
parser.add_argument("output_path")
parser.add_argument("--feat_fps", type=float, default=1.5,
                    help="frame rate for getting features")
parser.add_argument("--hessian_threshold", type=float, default=400,
                    help="hyperparam for surf feature")

def get_surf_features_from_video(surf_detector,
                                 video_filepath,
                                 surf_feat_filepath,
                                 keyframe_interval):
  "Receives filename of downsampled video and of output path for features."
  " Extracts features in the given keyframe_interval. "
  " Saves features in pickled file."

  surf_feat = []

  for keyframe in get_keyframes(video_filepath, keyframe_interval):
    # key_points: the location of the detected key points of current frame
    # feat: the corresponding surf features around the key points
    key_points, feat = surf_detector.detectAndCompute(keyframe, None)
    # (num_keypoints, 64)
    surf_feat.append(feat)

  pickle.dump(surf_feat, open(surf_feat_filepath, 'wb'))
  # to open this file use:
  # pickle.load(open(surf_feat_video_filename, 'rb'), encoding='latin1')


def get_keyframes(video_filepath, keyframe_interval):
  """Generator function which returns the next keyframe.
  Args:
      video_filepath (string): path to the video
      keyframe_interval (int): return a frame every k frame
  Returns:
      frame (np.array): opencv loaded RGB frame object
  """

  # video_cap = cv2.VideoCapture(video_filepath)
  # TODO: implement your code here
  # it should be a generator that yields a frame when it should
  frames = []

  frames_path = 'rgb/'
  video_id = video_filepath.split("/")[-1].split(".")[0]
  # print(video_id)
  frames_dir = sorted(os.listdir(frames_path+video_id))
  for idx in range(0, len(frames_dir)):
    if idx % keyframe_interval == 0:
      # print(frames_path + "/" + video_id + '/' + frames_dir[idx])
      img = cv2.imread(frames_path + "/" + video_id + '/' + frames_dir[idx])
      frames.append(img)
  return frames
  # raise Exception("Please implement get_keyframes")
  # video_cap.release()

if __name__ == '__main__':
  args = parser.parse_args()

  # Create SURF object
  # 1. need to reinstall opencv 4.5 from the following url
  # https://github.com/opencv/opencv-python/issues/126#issuecomment-621583923
  # 2. or downgrade:
  # $ sudo pip install opencv-contrib-python==3.4.2.16
  surf = cv2.xfeatures2d.SURF_create(hessianThreshold=args.hessian_threshold)

  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  video_fps = 30.0  # assuming the video frame rate
  # extract feature from every k frame
  frame_interval = int(video_fps / args.feat_fps)

  # Loop over all videos (training, val, testing)
  video_files = glob(os.path.join(args.video_path, "*.mp4"))

  for video_file in tqdm(video_files):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    surf_feat_outfile = os.path.join(args.output_path, video_name + ".p")

    # Get SURF features for one video
    get_surf_features_from_video(surf, video_file,
                                 surf_feat_outfile, frame_interval)
