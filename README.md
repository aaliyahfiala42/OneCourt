# OneCourt

The purpose of this repository is to store the object and event detection models for various sports.

# Overview

## Ball Tracking
### Demo
* Description: Tracks a tennis ball using the HSV color live using a laptop camera. For demo purposes only.
* Filepath: tennis\demo\live_ball_tracking.py
* Input: Live camera feed
* Output: 
    * in terminal, prints X and Y coordinates of the pixel location on the camera
    * Opens local camera and displays visual output of ball tracking

### TrackNet
* TrackNet is a CNN model trained to track tennis ball location, both using a single frame (TrackNet I) and multiple frames (TrackNet II) as input. 
    * Original Paper: TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications 
    * Source code & Dataset: https://nol.cs.nctu.edu.tw/ndo3je6av9/ 
* Filepath: tennis\models\TrackNet
* Input: tennis video path
* Output: heatmap frames of predicted tennis ball location

### UNET
* Documentation: tennis\models\UNET\documentation
* Filepath: tennis\models\UNET
* Input:  tennis video path
* Output: frames of predicted tennis ball location

## Action Tracking

## Apply Homogenous Projection

## Tracking with Music 
 
## Player Tracking
  * UNet Model 
  * LSTM Model

### Tennis Key milestones
 - [X] Train tennis model to track ball location
 - [X] Train tennis model to track player locations
 - [ ] Train tennis model to track key actions (bounce, hit, air)
 - [X] Homography estimations for tennis court positions
 - [ ] Train tennis model to track game events (score, out, etc.)
 - [ ] Explore designing a single model architecture to track all of the details that we want to improve inference and scalability
 - [ ] Upload model to server and connect it to the device
 - [ ] Test the complete-end-to-end process with our own generated tennis footage*
 - [ ] Research how to improve homographic projections
 - [ ] Explore ways to improve overall performance and inference
 - [ ] Test process end-to-end in a live tennis match on UW tennis court*


### Sources
Tennis: 
* Tennis Tracking: https://github.com/ArtLabss/tennis-tracking 
* TrackNet: https://arxiv.org/abs/1907.03698 
* YOLOv3: https://pjreddie.com/darknet/yolo/ 
