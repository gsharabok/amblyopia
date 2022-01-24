import numpy as np
import cv2
from cv2.data import haarcascades
import random
from typing import Optional
from skimage import measure
import argparse
import datetime
import time

from detection.mask import create_mask
from detection.face import detect_face, get_largest_frame
from detection.eye import detect_eyes,_cut_eyebrows
from detection.blob import blob_track
from detection.pupil import pupil_detect
from ball.ball_tracking import *
from distance_measure.distance import get_distance_face, get_distance_ball
import models


# Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--filepath", help="path to video file")
parser.add_argument("--pupilthresh", help="pupil threshold", type=int)
parser.add_argument("--camera", help="use live video feed", action="store_true")

args = parser.parse_args()

filepath = 'data/video/eye_training_stable.MOV'
if args.filepath:
    filepath = args.filepath
if args.camera:
    filepath = 0


# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# Maybe lower for high quality video and higher for live video
pupil_threshold = 60 #25
# if args.camera:
#     pupil_threshold = bw_threshold
if args.pupilthresh:
    pupil_threshold = args.pupilthresh


# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = ""
not_weared_mask = ""


detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blob_detector = cv2.SimpleBlobDetector_create(detector_params)
previous_right_blob_area = 1

previous_left_blob_area = 1
previous_right_blob_area = 1
previous_left_keypoints = None
previous_right_keypoints = None


# Read video
cap = cv2.VideoCapture(filepath)

# Create video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

currentTime = time.time() # seconds from UTC
currentTime = time.ctime(currentTime)
currentTime = str(currentTime).replace(" ", "_").replace(":", "_").strip()

capWriter_filename = "data/results/" + currentTime + ".avi"
capWriter = cv2.VideoWriter(capWriter_filename, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# init_get_distance()

while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img,1)

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image in black and white
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow("B&W", black_and_white)

    # detect face
    faces = models.face_cascade.detectMultiScale(gray, 1.1, 4)
    faces = get_largest_frame(faces)

    # face = detect_face(faces)

    # Face prediction for black and white
    faces_bw = models.face_cascade.detectMultiScale(black_and_white, 1.1, 4)
    # face_bw = detect_face(faces_bw)

    ball = track_ball(img)

    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw rectangle on face
        for (x, y, w, h) in faces:
            get_distance_face(img, faces)
            get_distance_ball(img, ball)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            fr = gray[y : y + h, x : x + w]
            left_eye, right_eye, left_coord, right_coord = detect_eyes(fr)

            if left_eye is not None:
                x0 = x+left_coord[0]
                x1 = x+left_coord[0] + left_coord[2]
                y0 = y+left_coord[1]
                y1 = y+left_coord[1] + left_coord[3]
                
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
                crop_left = img[y0:y1, x0:x1]
                
                pupil_detect(crop_left, pupil_threshold)

            if right_eye is not None:
                x0 = x+right_coord[0]
                x1 = x+right_coord[0] + right_coord[2]
                y0 = y+right_coord[1]
                y1 = y+right_coord[1] + right_coord[3]

                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
                crop_right = img[y0:y1, x0:x1]
                
                pupil_detect(crop_right, pupil_threshold)

            # Detect lips counters
            mouth_rects = models.mouth_cascade.detectMultiScale(gray, 1.5, 5)

        # Face detected but Lips not detected which means person is wearing mask
        if(len(mouth_rects) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if(y < my < y + h):
                    # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                    # person is not waring mask
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)

                    #cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                    break

    # Show frame with results
    #dst = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
    #dst = cv2.inpaint( img, create_mask(img), 0.8, cv2.INPAINT_NS)

    #cv2.imshow('Gray', dst)
    capWriter.write(img)
    cv2.imshow('Amblyopia Treatment', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Press Esc to kill the program
        break

# Release video
cap.release()
capWriter.release()
cv2.destroyAllWindows()
