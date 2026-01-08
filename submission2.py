"""
Video Anomaly Detection using Convolutional Autoencoder

this code detects anomalies in surveillance videos by training an autoencoder
on normal data and measuring reconstruction errors on test data

approach:
1. create temporal frame stacks (K consecutive frames)
2. train a convolutional autoencoder to reconstruct normal patterns
3. measure reconstruction error on test frames where higher error = more anomalous
4. rank errors within each video to generate anomaly probabilities

"""

#include neccessary libraries
import os
import glob
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from scipy.stats import rankdata
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

#dataset paths
TRAIN_MAT_DIR = "/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset/training_vol/"
TEST_MAT_DIR = "/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset/testing_vol/"
TEST_JPG_DIR = "/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset/testing_videos/"

# hyperparameters
K = 5 #depth size for each stack
BATCH_SIZE = 8 #shifted from 32 to 8 
EPOCHS = 50 
LR = 1e-4  #learning rate
SEED = 42 
np.random.seed(SEED) # randomness reproducibility
tf.random.set_seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# data loading functions
def load_mat_volume(mat_path):
    """
    load and normalize a video volume from .mat file
    """
    mat_data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # mat_path is the path of .mat files and squeeze_me makes sure that there is no extra dimension with size=1
    #struct_as_record makes it easy to read complex structs for beginners
    volume = mat_data['vol']
    volume = volume.astype(np.float32) / 255.0
    #converting into float32 and make it to range [0,1]
    return volume

def create_frame_stacks(volume, K=5):
    """
    create temporal stacks of K consecutive frames
    temporal information helps the model learn normal motion patterns
    """
    H, W, num_frames = volume.shape
    stacks = []

    # create stacks for frames that have K previous frames
    for i in range(num_frames - K + 1):
        #create stacks 
        stack = volume[:, :, i:i+K]
        stack = stack.transpose(2, 0, 1)  # change to (K, H, W) format
        stacks.append(stack)

    # handle the first (K-1) frames by padding with zeros
    # add padding so that the first (k-1) frames arent left
    for i in range(K - 1):
        pad_stack = np.zeros((K, H, W), dtype=np.float32)
        available = i + 1
        # fill available frames, rest remain zeros
        pad_stack[:available] = volume[:, :, :available].transpose(2, 0, 1) 
        # transpose to make it of form (k,h,w) from (h,w,k)
        stacks.insert(i, pad_stack)
    
    return np.array(stacks)

# prepare training data
# get MATLAB files
mat_files = sorted([f for f in os.listdir(TRAIN_MAT_DIR) if f.endswith('.mat')]) #sort the .mat files
print(f"\nFound {len(mat_files)} training videos")

# determine frame dimensions from first video
# load video volumes
sample_volume = load_mat_volume(os.path.join(TRAIN_MAT_DIR, mat_files[0]))
H, W, _ = sample_volume.shape
del sample_volume
gc.collect()

print(f"Frame dimensions: H={H}, W={W}\n")


# autoencoder architecture
def func_autoencoder(input_shape=(120, 160, 5)):
    """
    architecture:
    - encoder: 4 conv layers with stride 2 (downsampling)
    - decoder: 4 transposed conv layers with stride 2 (upsampling)
    - bottleneck captures compressed representation of normal patterns
    """
    
    inputs = keras.Input(shape=input_shape)
    
    # encoder: progressively compress spatial dimensions and increase channels
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)   # 32 filters
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)        # 64 filters
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)       # 128 filters
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)       # 256 filters (bottleneck)

    # decoder: reconstruct original dimensions
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(5, 3, strides=2, padding='same', activation='sigmoid')(x)  # 5 channels for K=5 frames
    #activation at last layer is kept sigmoid to get [0,1]

    # cropping 4 pixels from top and bottom to match 120*160 size of the frames (128-4-4=120)
    outputs = layers.Cropping2D(cropping=((4, 4), (0, 0)))(x)
    
    model = Model(inputs, outputs, name='autoencoder')
    return model


# build and compile model
autoencoder = func_autoencoder(input_shape=(H, W, K))
autoencoder.summary()

optimizer = keras.optimizers.Adam(learning_rate=LR) #loss function
# mean squared error penalizes reconstruction differences
autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# training loop
# train video-by-video to handle memory constraints
print("\nTraining autoencoder video-by-video...\n")

# iterate through epochs
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    epoch_losses = []
    
    # process each training video
    for mat_file in tqdm(mat_files, desc=f"Training videos"):
        mat_path = os.path.join(TRAIN_MAT_DIR, mat_file)
        
        # load video and create frame stacks
        volume = load_mat_volume(mat_path)
        stacks = create_frame_stacks(volume, K=K)
        stacks = stacks.transpose(0, 2, 3, 1)  # change to (num_stacks,H,W,K) for keras
        
        # train on this video (autoencoder learns to reconstruct its input)
        history = autoencoder.fit(
            stacks, stacks,  # input = output for autoencoder
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=0
        )
        
        epoch_losses.append(history.history['loss'][0])
        
        # free memory after each video
        del stacks, volume
        gc.collect()
    
    # print average loss across all videos for this epoch
    avg_loss = np.mean(epoch_losses)
    print(f"Average loss: {avg_loss:.6f}")

print("\nTraining completed\n")

# test data processing
# correction code for the mismatch in the .mat files and .jpg files 
# total .jpg files = number of rows required in csv file = 11706
# total frames of all videos combined ~ 15k , not all have to be there in the csv file
# example : for video01, frame numbering starts from 939 and not 0 so this code fixess the mismatch
def get_jpg_frame_numbers(jpg_dir, video_num):
    """
    extract frame numbers from jpg filenames for a given video
    this ensures we only predict for frames that exist in the test set
    """
    jpg_folder = os.path.join(jpg_dir, f"{video_num:02d}")
    jpg_files = sorted(glob.glob(os.path.join(jpg_folder, "*.jpg")))
    frame_numbers = []
    
    # get frame number from each jpg filename
    for jpg_file in jpg_files:
        frame_filename = os.path.basename(jpg_file).replace('.jpg', '')
        if '_' in frame_filename:
            frame_num = int(frame_filename.split('_')[-1])
        else:
            frame_num = int(frame_filename)
        frame_numbers.append(frame_num)
    
    return frame_numbers

# process all test videos
mat_files = sorted([f for f in os.listdir(TEST_MAT_DIR) if f.endswith('.mat')])
print(f"Processing {len(mat_files)} test videos\n")

all_test_errors = []  # reconstruction errors for all frames
all_test_ids = []     # frame identifiers
video_boundaries = []  # number of frames per video (for per-video ranking)

for mat_file in tqdm(mat_files, desc="Processing test videos"):
    mat_path = os.path.join(TEST_MAT_DIR, mat_file)
    video_num = int(mat_file.replace('vol', '').replace('.mat', ''))
    
    # load video and create stacks
    volume = load_mat_volume(mat_path)
    stacks = create_frame_stacks(volume, K=K)
    test_data = stacks.transpose(0, 2, 3, 1)
    
    # get reconstructions from trained autoencoder
    reconstructed = autoencoder.predict(test_data, batch_size=BATCH_SIZE, verbose=0)
    
    # calculate reconstruction error (MSE)
    # higher error means the frame is more different from normal patterns
    errors = np.mean((test_data - reconstructed) ** 2, axis=(1, 2, 3))  # average over H, W, K dimensions
    
    # get frame numbers that actually exist in jpg files
    jpg_frame_numbers = get_jpg_frame_numbers(TEST_JPG_DIR, video_num)
    
    # collect errors and ids only for frames that exist in test set
    video_errors = []
    video_ids = []
    for frame_num in jpg_frame_numbers:
        if frame_num < len(errors):
            video_errors.append(errors[frame_num])
            video_ids.append(f"{video_num}_{frame_num}")
    
    all_test_errors.extend(video_errors)
    all_test_ids.extend(video_ids)
    video_boundaries.append(len(video_ids))  # track video length for per-video ranking
    
    # free memory after each video
    del stacks, test_data, volume, reconstructed
    gc.collect()

print(f"\nTotal test frames: {len(all_test_ids)}")
print(f"Expected: 11706, Match: {len(all_test_ids) == 11706}")

# convert reconstruction errors to probabilities
# ranking is done per video to normalize for different video characteristics
all_test_errors = np.array(all_test_errors)
probabilities = np.zeros(len(all_test_errors))

start_idx = 0
# rank errors within each video separately
for video_len in tqdm(video_boundaries, desc="Ranking probabilities"):  #tqdm to track progress by progressbar
    end_idx = start_idx + video_len
    video_errors = all_test_errors[start_idx:end_idx]
    
    # convert to rank-based probabilities
    video_probs = rankdata(video_errors) / len(video_errors)
    probabilities[start_idx:end_idx] = video_probs  
    start_idx = end_idx

# create submission file
submission = pd.DataFrame({
    'Id': all_test_ids,
    'Predicted': probabilities
})

#final submission to CSV 
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission created")
print(f"Total predictions: {len(submission)}")
print(f"Probability range: [{submission['Predicted'].min():.4f}, {submission['Predicted'].max():.4f}]")
print(f"Mean probability: {submission['Predicted'].mean():.4f}")
print(f"No NaN: {not submission.isnull().any().any()}")
print(f"\nFirst 10:\n{submission.head(10)}")
print(f"\nLast 10:\n{submission.tail(10)}")