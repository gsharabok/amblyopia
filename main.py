import numpy as np
import cv2
from cv2.data import haarcascades
import random
from typing import Optional
from skimage import measure
import argparse
import datetime
import time
import matplotlib.pyplot as plt

from detection.mask import create_mask
from detection.face import detect_face, get_largest_frame
from detection.eye import detect_eyes,_cut_eyebrows
from detection.blob import blob_track
from detection.pupil import pupil_detect
from ball.ball_tracking import *
from face_landmarks.eye_corners import get_eye_corners
from distance_measure.distance import get_distance_face, get_distance_ball
import models

def test():
    print("test")


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

previous_left_pupil_coords = None
previous_left_pupil_direction = None
previous_right_pupil_coords = None
previous_right_pupil_direction = None
previous_ball_to_face_distance = None

direction_change_count = 0
frame_count = 0
direction_change_frames = 10
direction_change_thresh = 6

ball_distance_array = []
left_eye_distance_center_array = []
right_eye_distance_center_array = []

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

def find_pupil_direction(pupil_coords, previous_pupil_coords):
    if previous_pupil_coords is None:
        return None
    if pupil_coords is None:
        return None

    x_diff = pupil_coords[0] - previous_pupil_coords[0]
    # y_diff = pupil_coords[1] - previous_pupil_coords[1]

    if x_diff > 0:
        return "right"
    elif x_diff < 0:
        return "left"
    else:
        return None

def is_direction_change(direction, previous_direction):
    if previous_direction is None:
        return False
    if direction is None:
        return False
    if direction != previous_direction:
        return True
    return False


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
            face_distance = get_distance_face(img, faces)
            ball_distance = get_distance_ball(img, ball)
            ball_to_face_distance = face_distance-ball_distance

            ball_distance_array.append(ball_to_face_distance)

            left_eye_corner, right_eye_corner = get_eye_corners(img, gray, (x, y, w, h))

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            fr = gray[y : y + h, x : x + w]
            left_eye, right_eye, left_coord, right_coord = detect_eyes(fr)

            frame_count += 1
            if frame_count >= direction_change_frames:
                print(direction_change_count)
                if direction_change_count >= direction_change_thresh:
                    print("WIGGLE")
                direction_change_count = 0
                frame_count = 0
            right_direction_change = left_direction_change = False

            if left_eye is not None:
                # Eye rectangle coordinates x0, y0 = top left corner, x1, y1 = bottom right corner 
                x0 = x+left_coord[0]
                x1 = x+left_coord[0] + left_coord[2]
                y0 = y+left_coord[1]
                y1 = y+left_coord[1] + left_coord[3]

                # Left eye rectangle width and height
                w = left_coord[2]
                h = left_coord[3]

                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
                crop_left = img[y0:y1, x0:x1]
                
                pupil_coords = pupil_detect(crop_left, pupil_threshold)

                direction = find_pupil_direction(pupil_coords, previous_left_pupil_coords)
                left_direction_change = is_direction_change(direction, previous_left_pupil_direction)

                previous_left_pupil_coords = pupil_coords
                previous_left_pupil_direction = direction

                left_eye_distance_center_array.append(w - (pupil_coords[0] + pupil_coords[2]/2))

            if right_eye is not None:
                # Eye rectangle coordinates x0, y0 = top left corner, x1, y1 = bottom right corner
                x0 = x+right_coord[0]
                x1 = x+right_coord[0] + right_coord[2]
                y0 = y+right_coord[1]
                y1 = y+right_coord[1] + right_coord[3]

                # Right eye rectangle width and height
                w = right_coord[2]
                h = right_coord[3]

                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
                crop_right = img[y0:y1, x0:x1]
                
                pupil_coords = pupil_detect(crop_right, pupil_threshold)

                direction = find_pupil_direction(pupil_coords, previous_right_pupil_coords)
                right_direction_change = is_direction_change(direction, previous_right_pupil_direction)

                previous_right_pupil_coords = pupil_coords
                previous_right_pupil_direction = direction

                right_eye_distance_center_array.append(pupil_coords[0]+pupil_coords[2]/2)

            if previous_ball_to_face_distance is None or ball_to_face_distance is None:
                ball_closer = False
            else:
                ball_closer = ball_to_face_distance < previous_ball_to_face_distance
            
            if left_direction_change and right_direction_change and ball_closer:
                direction_change_count += 1

            previous_ball_to_face_distance = ball_to_face_distance

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

def plot_data():
    indx = [x for x in range(0, len(ball_distance_array))]
    plt.scatter(indx, ball_distance_array, color='red', label='Ball Distance')
    plt.savefig('data/results/plots/ball_distance.png', bbox_inches='tight')
    plt.close()

    indx = [x for x in range(0, len(left_eye_distance_center_array))]
    plt.scatter(indx, left_eye_distance_center_array, color='blue', label='Left Eye Distance')
    plt.savefig('data/results/plots/left_eye_distance.png', bbox_inches='tight')
    plt.close()

    indx = [x for x in range(0, len(right_eye_distance_center_array))]
    plt.scatter(indx, right_eye_distance_center_array, color='green', label='Right Eye Distance')
    plt.savefig('data/results/plots/right_eye_distance.png', bbox_inches='tight')
    plt.close()

plot_data()
