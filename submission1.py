"""
Video Anomaly Detection using Optical Flow and Isolation Forest

This code detects the anomalies in surviellance video by analyzing optical flow patterns.
It uses the .mat files in dataset provided in the kaggle competition "PixelPlay'26" 

Approach:
1. Extract optical flow features from consecutive video frames
2. Train an Isolation Forest model on normal training data
3. Score test frames and apply temporal smoothing
4. Generate probability predictions for anomaly detection

"""
# IMPORTS
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import cv2
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from scipy.ndimage import uniform_filter1d
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # suppress warnings for clearer outputs


# DATASET PATHS
TRAIN_MAT_DIR = "/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset/training_vol/"
TEST_MAT_DIR = "/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset/testing_vol/"
TEST_JPG_DIR = "/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset/testing_videos/"

# RANDOM SEED FOR REPRODUCIBILITY
SEED = 42
np.random.seed(SEED)

# PRINTING INFORMATION ABOUT DATASET 
print(f"Training .mat files: {len([f for f in os.listdir(TRAIN_MAT_DIR) if f.endswith('.mat')])}")
print(f"Testing .mat files: {len([f for f in os.listdir(TEST_MAT_DIR) if f.endswith('.mat')])}")

# COMPUTE OPTICALFLOW
"""
    Compute dense optical flow between two consecutive frames using Farneback method.
    Optical flow captures motion patterns between frames
"""
def compute_optical_flow(frame1, frame2):
    """
    Extract statistical features from optical flow using spatial grid division.
    
    The frame is divided into a grid, and statistical features are computed
    for each grid cell.
    It captures local and spatial motion patterns.
    
    Features per grid cell (11 total):
    - Mean, std, max, median of flow magnitude
    - 75th and 90th percentile of magnitude
    - Mean and std of horizontal flow (x-direction)
    - Mean and std of vertical flow (y-direction)
    - Ratio of pixels with magnitude > 95th percentile 
    """
    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

# FEATURE EXTRACTION
def extract_flow_features(flow, grid=(2, 2)):
    H, W, _ = flow.shape
    gh, gw = grid
    fh, fw = H // gh, W // gw
    all_grid_feats = []
    
    # for each grid cell
    for i in range(gh):
        for j in range(gw):
            # Extract flow for current grid cell
            block = flow[i*fh:(i+1)*fh, j*fw:(j+1)*fw]
            flow_x = block[:, :, 0]  # horizontal component
            flow_y = block[:, :, 1]  # vertical component
            
            # compute flow magnitude 
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            
            if magnitude.size > 0:
                m_95 = np.percentile(magnitude, 95)
                
                # compute comprehensive statistics for this grid cell
                block_feats = [
                    np.mean(magnitude),                          # average motion
                    np.std(magnitude),                           # Motion variability
                    np.max(magnitude),                           # Max motion
                    np.median(magnitude),                        # Robust center
                    np.percentile(magnitude, 75),                # Upper quartile
                    np.percentile(magnitude, 90),                # Strong motion  threshold
                    np.mean(flow_x),                             # Average horizontal motion
                    np.std(flow_x),                              # Horizontal variability
                    np.mean(flow_y),                             # Average vertical motion
                    np.std(flow_y),                              # Vertical variability
                    np.sum(magnitude > m_95) / magnitude.size    # High motion ratio.
                ]
            else:
                # zero features for exmpty blocks
                block_feats = [0.0] * 11
            
            all_grid_feats.extend(block_feats)
    
    return np.array(all_grid_feats)


def process_mat_file(mat_path, normalize=True):
    """
    Process one .mat video file and extract flow features for all frames.
    """
    # load the MATLAB file containing video
    mat_data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    volume = mat_data['vol']
    num_frames = volume.shape[2]
    
    # initialize with zero features 
    features = [np.zeros(44)]
    
    # Process consecutive frame pairs
    for i in range(1, num_frames):
        frame_prev = volume[:, :, i-1].astype(np.uint8)
        frame_curr = volume[:, :, i].astype(np.uint8)
        
        # Compute optical flow and extract features
        flow = compute_optical_flow(frame_prev, frame_curr)
        features.append(extract_flow_features(flow))
    
    features = np.array(features)
    
    # Apply z-score normalization per video. 
    #so, mean=0 and std=1
    if normalize:
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    
    return features

#TRAINING DATA PROCESSING

# Get the MATLAB files
mat_files = sorted([f for f in os.listdir(TRAIN_MAT_DIR) if f.endswith('.mat')])
print(f"Processing {len(mat_files)} training videos with per-video normalization...")

# extract features from training videos
all_features = []
for mat_file in tqdm(mat_files):
    mat_path = os.path.join(TRAIN_MAT_DIR, mat_file)
    features = process_mat_file(mat_path, normalize=True)
    all_features.append(features)

# put all the features in one array
train_features = np.vstack(all_features)
print(f"\nTraining features shape: {train_features.shape}")
print(f"Normalized feature stats: Mean={train_features.mean():.4f}, Std={train_features.std():.4f}")

# MODEL TRAINING BY ISOLATION FOREST
"""
IsolationForest is an unsupervised anomaly detection algorithm.
It works this way: 
1. Randomly selecting a feature and split value
2. Recursively partitioning the data
3. Anomalies are isolated faster (fewer splits needed)
4. Shorter paths in trees indicate anomalies

i have chosen this because 
- We have normal training data (no anomalies labeled)
- Anomalies should have unusual motion patterns (easier to isolate)
"""
iso_forest = IsolationForest(
    contamination='auto',    # auto-determine contamination level
    random_state=SEED,       # Reproducibility
    n_estimators=300,        # number of trees 
    max_samples='auto',      # Samples per tree (auto = min(256, n_samples))
    max_features=1.0,        # using all the features
    n_jobs=-1,               # all CPU cores being used 
    verbose=1                # progress is shown
)

#train the model on normal data
iso_forest.fit(train_features)

# Compute anomaly scores for training data 
train_scores = iso_forest.decision_function(train_features)
print(f"Training score stats: Mean={train_scores.mean():.4f}, Std={train_scores.std():.4f}")

# TEST DATA PROCESSING

def extract_test_features_with_jpg_ids(mat_dir, jpg_dir, normalize=True):
    """
    Extract features from test videos and create frame IDs matching jpg filenames.
    
    This function processes test videos and creates identifiers for each frame
    that match the jpg files in the testing_videos directory.

    It is the correction code for the mismatch in the .mat files and .jpg files 
    total .jpg files = number of rows required in csv file = 11706
    total frames of all videos combined ~ 15k , not all have to be there in the csv file
    example : for video01, frame numbering starts from 939 and not 0 so this code fixess the mismatch
    """
    all_features = []
    all_ids = []
    mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat')])
    print(f"Processing {len(mat_files)} test videos...")
    
    for mat_file in tqdm(mat_files):
        mat_path = os.path.join(mat_dir, mat_file)
        # extract video number from filename
        video_num = int(mat_file.replace('vol', '').replace('.mat', ''))
        
        # load the video volume
        mat_data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        volume = mat_data['vol']
        
        # get corresponding jpg files for the video
        jpg_folder = os.path.join(jpg_dir, f"{video_num:02d}")
        jpg_files = sorted(glob.glob(os.path.join(jpg_folder, "*.jpg")))
        
        # extract the flow features for all frames
        features = [np.zeros(44)]  # first frame has no flow
        for i in range(1, volume.shape[2]):
            frame_prev = volume[:, :, i-1].astype(np.uint8)
            frame_curr = volume[:, :, i].astype(np.uint8)
            flow = compute_optical_flow(frame_prev, frame_curr)
            features.append(extract_flow_features(flow))
        
        features = np.array(features)
        
        # normalize per video
        if normalize:
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
        
        # create IDs matching jpg filenames
        for jpg_file in jpg_files:
            frame_filename = os.path.basename(jpg_file).replace('.jpg', '')
            frame_num = int(frame_filename.split('_')[-1]) if '_' in frame_filename else int(frame_filename)
            
            # ensure frame number is within the bounds
            if frame_num < len(features):
                all_features.append(features[frame_num])
                all_ids.append(f"{video_num}_{frame_num}")
    
    return np.array(all_features), all_ids

# extract test features and IDs
test_features, test_ids = extract_test_features_with_jpg_ids(TEST_MAT_DIR, TEST_JPG_DIR, normalize=True)
print(f"\nTest features shape: {test_features.shape}")
print(f"Total test IDs: {len(test_ids)}")
print(f"Expected: 11706, Match: {len(test_ids) == 11706}")
print(f"Sample IDs: {test_ids[:5]}")


#ANOMALY SCORING AND PROBABILITY CALCULATION

# compute anomaly scores using the trained Isolation Forest
# lesser the score, more anomaly it has.
test_scores = iso_forest.decision_function(test_features)
print(f"Test score stats: Mean={test_scores.mean():.4f}, Std={test_scores.std():.4f}")

# convert scores -> probabilities using rank-based approach
# higher probability means it is more likely to be anomalous
# rankdata(-test_scores) ranks from most to least anomalous
probabilities = rankdata(-test_scores) / len(test_scores)
print(f"Probability stats: Mean={probabilities.mean():.4f}, Std={probabilities.std():.4f}")

# VIDEO LENGTH CALCULATION
def get_video_lengths_from_ids(test_ids):
    """
    Calculate the number of frames in each test video from frame IDs.
    This is needed for temporal smoothing, which should be applied
    within each video separately and not across video boundaries.
    """
    lengths = []
    current_video, current_count = None, 0
    for frame_id in test_ids:
        video_num = int(frame_id.split('_')[0])
        
        if current_video is None:
            # first video
            current_video, current_count = video_num, 1
        elif video_num == current_video:
            # increment count
            current_count += 1
        else:
            # new video saving previous count and reset
            lengths.append(current_count)
            current_video, current_count = video_num, 1
    
    # last video
    lengths.append(current_count)
    return lengths

video_lengths = get_video_lengths_from_ids(test_ids)
print(f"Number of videos: {len(video_lengths)}")
print(f"Video lengths: {video_lengths}")

#TEMPORAL SMOOTHING
def apply_temporal_smoothing(probabilities, video_lengths, window_size=14):
    """
    Apply temporal smoothing to anomaly probabilities within each video.
    
    It is applied because
    Single-frame predictions can be noisy
    Anomalies in videos often persist for multiple consecutive frames
    So, 
    Temporal smoothing reduces false positives
    Smoothing is applied per video to avoid smoothing across video boundaries
    
    """
    smoothed = np.copy(probabilities)
    start_idx = 0
    
    # apply smoothing to each video separately
    for video_len in video_lengths:

        end_idx = start_idx + video_len
        video_probs = probabilities[start_idx:end_idx]
        
        # moving average-uniform filter for temporal smoothing
        smoothed[start_idx:end_idx] = uniform_filter1d(
            video_probs, 
            size=window_size, 
            mode='nearest'  # handling the boundaries by repeating the edge values
        )
        start_idx = end_idx
    
    return smoothed

# apply temporal smoothing with window size of 14 frames
# 14 is chosen as it gives best score in range 5 to 18.
probabilities_final = apply_temporal_smoothing(probabilities, video_lengths, window_size=14)
print(f"probability stats: Mean={probabilities_final.mean():.4f}, Std={probabilities_final.std():.4f}")

# SUBMISSION FILE GENERATION

# Creating submission dataframe
submission = pd.DataFrame({
    'Id': test_ids,
    'Predicted': probabilities_final
})

# Save to CSV.
submission.to_csv('submission.csv', index=False)

# print(display) submission stats
print(f"Total predictions: {len(submission)}")
print(f"Probability range: [{submission['Predicted'].min():.4f}, {submission['Predicted'].max():.4f}]")
print(f"Mean probability: {submission['Predicted'].mean():.4f}")
print(f"Median probability: {submission['Predicted'].median():.4f}")
