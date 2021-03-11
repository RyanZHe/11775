First extract the audio using ffmpeg
Then extract the MFCCs using openSMILE
The script to extract SoundNet is in extract_feat.py
Use convert_npy_to_csv to convert .npy to .csv format
Then use train_mlp.py to train the MLP model using the MFCC features and SoundNet features
Use test_mlp.py to produce the labels for the test set with MFCC and SoundNet features
