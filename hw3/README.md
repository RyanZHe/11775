Train and test early fusion
```
$ python3 train_mlp_early_fusion.py hw2/resnet101_feat/ hw1/SoundNet-tensorflow/converted_18 2304 hw2/labels/trainval.csv early_fusion.mlp.model
$ python3 test_mlp_early_fusion.py early_fusion.mlp.model hw2/resnet101_feat/ hw1/SoundNet-tensorflow/converted_18 2304 hw2/labels/test_for_student.label best_early_fusion.csv
```

Train and test late fusion
```
python3 train_mlp_late_fusion.py hw2/resnet101_feat/ hw1/SoundNet-tensorflow/converted_18 2304 hw2/labels/trainval.csv late_fusion.visual_mlp.model late_fusion.sound_mlp.model late_fusion.prob_mlp.model
python3 test_mlp_late_fusion.py late_fusion.visual_mlp.model late_fusion.sound_mlp.model late_fusion.prob_mlp.model hw2/resnet101_feat/ hw1/SoundNet-tensorflow/converted_18 20 hw2/labels/test_for_student.label best_late_fusion.csv
```

Train and evaluate double fusion
```
$ python3 train_mlp_early_fusion.py hw1/SoundNet-tensorflow/converted hw1/SoundNet-tensorflow/converted_18 hw2/labels/trainval.csv sound_early.mlp.model
$ python3 train_mlp_early_fusion.py hw2/resnet18-feat/ hw2/resnet101_feat/ hw2/labels/trainval.csv video_early.mlp.model
$ python3 train_mlp_double_fusion.py hw1/SoundNet-tensorflow/converted hw1/SoundNet-tensorflow/converted_18 hw2/resnet18-feat/ hw2/resnet101_feat/ 20 hw2/labels/trainval.csv  video_early.mlp.model sound_early.mlp.model double_fusion.prob_mlp.model
```