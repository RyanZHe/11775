# Instructions for hw2

In this homework we will perform a video classification task using visual features.

## Video Data and Labels

Please download the videos from [this link](https://drive.google.com/file/d/1Oyzv7eC0QDrg0vX3AdSXYzdsFpIsdzT-/view?usp=sharing) or [this link](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/hw_11775_videos.zip). Then download the labels from [here](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/labels_v2.zip). Extract these zip files to get videos and their labels by:
```
$ unzip hw_11775_videos.zip
$ unzip labels_v2.zip
```

## Overirew
A video-based MED system is composed of mainly three parts: 1) Video pre-processing, 2) video feature extraction, and 3) classification. You will mainly focus on learning how to use two-types of visual features in this homework. You may simply reuse the classifcation codebase you finished in HW1.

## Video Pre-Processing
There are many types of video features. For simplicity, we ask you to extract and aggregate features of RGB frames from videos.
In our example code, we directly read frames from the videos to extract features.
Alternatively, you can speed up the feature extraction process by extracting the frames first onto the disk. You can do this by:

```
for video in videos/*.mp4
do
    base=$(basename -- "$video")
    base="${base%.*}"
    mkdir -p rgb/$base
    ffmpeg -i "$video"  rgb/"$base"/frame_%05d.jpg -hide_banner
done
```

## Extract SURF features
For the hand-crafted visual feature, we ask you to Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ..., etc.). We use OpenCV to extract SURF feature. To do so, you need to downgrade OpenCV:
```
$ sudo pip uninstall opencv-python -y
$ sudo pip uninstall opencv-contrib-python -y
$ sudo pip install opencv-contrib-python==3.4.2.16
```
Or you can follow this [link](https://github.com/opencv/opencv-python/issues/126#issuecomment-621583923) to build the latest OpenCV from source.

+ Step 1. Extract SURF keypoint features at a frame rate of 1.5, meaning that we will extract one frame of features every 20 frames. *Please implement the get_keyframes function.*
```
$ python surf_feat_extraction.py videos surf_feat/ --feat_fps 1.5
```
This will take 2-4 hours and 25 GB disk space.

+ Step 2. K-Means clustering

First randomly select a subset of the features to speed things up. This may affect the quality of the clusters.
```
$ python select_frames.py labels/trainval.csv surf_feat/ 0.01 selected.surf.csv
```

K-Means clustering with 256 centers (1-3 hrs):
```
$ python train_kmeans.py selected.surf.csv 256 surf.kmeans.256.model
```

+ Step 3. Get Bag-of-Words features
```
$ ls videos/|while read line;do filename=$(basename $line .mp4);echo $filename;done > videos.name.lst
$ python old_code/get_bof.py surf.kmeans.256.model surf_feat/ 256  videos.name.lst surf_bof
```

### [bonus] VLAD Encoding
Please follow instructions in the writeup to implement this part.


## Extract CNN features
In HW2 we ask you to extract CNN-based feature for each frame.
We provide an example code template using ResNet, a type of CNN model that has been widely used in many computer vision tasks.
After extracting CNN-based feature for each video frame, you can choose to apply and build bag-of-words feature as in HW1 or simply do a global average or maximum pooling to aggregate the frame-level features (For example, for a 300-frame video you have a 300x512 feature matrix, you average over its 300 frames and get the final 512-dim video representaion.).
You are also encouraged to try some more advanced model such as DenseNet and RexNeXt.

In the code template, we use PyTorch (torch 1.6.0, torchvision 0.7.0). [Here](https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c) is a good tutorial of how to extract feature from an image. *Please implement Get_CNN object in cnn_feat_extraction.py*

```
python cnn_feat_extraction.py videos cnn_feat/ --feat_fps 1.5 --model resnet101 --feat_size 2048
```

## Training and Testing Classifiers
As you already have the code to train and test SVM and MLP.
**You can reuse your code from HW1 for bag-of-words and SVM/MLP training.**

```
python3 train_mlp.py surf_bof/ 256 labels/trainval.csv models/surfbof.mlp.model
python3 test_mlp.py models/surfbof.mlp.model surf_bof/ 256 labels/test_for_student.label surf.csv
```

## Potential Improvement
Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:
+ Use more advanced visual models/features such as DenseNet and ResNeXt.
+ End-to-end training of these models instead of just extracting features.
+ Use video models/features such as I3D that takes video clips as its input insteadt of frames.
