# Parking Monitoring System
This repository contains the custom weights file and code to run the website.
## Description
The aim of this project is to use a live feed of a parking lot and apply object detection and object tracking to provide users with different statistics about the parking lot. For object tracking, we use StrongSORT which is an upgraded version of DeepSORT, with YOLOv5 detections, which is fine tuned on custom dataset, to track cars in a parking lot. It has an additional security feature implemented which activates an alert as soon as a car moves from its position. The video below shows the result of tracking on our parking lot. The website has multiple features such as interactive dashboard displaying maximum, average counts of cars in tabular and graphic form, hourly detections of car, security section which idsplays id number and detected car picture if it moves and a live stream feed page, visualized with detected bounding boxes.

https://user-images.githubusercontent.com/58269425/188257175-6d130d9e-d25e-4e55-9e16-a8d3f8290799.mp4

## Installation

### Initialize repository 
%cd Object_Tracking-with-StrongSORT

!pip install -r requirements.txt  # install dependencies

### Initialize torchreid
!git clone https://github.com/KaiyangZhou/deep-person-reid.git

%cd deep-person-reid

!pip install -r requirements.txt

!python setup.py develop

%cd ../

### Initialize yolov5 repositrory
!git clone https://github.com/ultralytics/yolov5


### Start website and begin tracking
!python track.py 

## Acknowledgments
The project is supported by abd belongs to Neurog.ai (https://neurog.ai/).
