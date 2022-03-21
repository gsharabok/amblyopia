import numpy as np
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from cv2.data import haarcascades
import random
from typing import Optional
from skimage import measure
import argparse
import datetime
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib
import requests
import imutils
from PIL import Image as im

from detection.mask import create_mask
from detection.face import detect_face, get_largest_frame
from detection.eye import detect_eyes,_cut_eyebrows
from detection.blob import blob_track
from detection.pupil import pupil_detect
from ball.ball_tracking import *
from face_landmarks.eye_corners import get_eye_corners
# from distance_measure.distance import get_distance_face, get_distance_ball
from distance_measure.distance_class import Distance
import models

class Runner:
    def __init__(self): 
        # Adjust threshold value in range 80 to 105 based on your light.
        self.bw_threshold = 80

        # Maybe lower for high quality video and higher for live video
        self.pupil_threshold = 95 #25

        # User message
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.org = (30, 30)
        self.weared_mask_font_color = (255, 255, 255)
        self.not_weared_mask_font_color = (0, 0, 255)
        self.thickness = 2
        self.font_scale = 1
        self.weared_mask = ""
        self.not_weared_mask = ""

        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1500
        self.blob_detector = cv2.SimpleBlobDetector_create(detector_params)
        self.previous_right_blob_area = 1
        self.previous_left_blob_area = 1
        self.previous_right_blob_area = 1
        self.previous_left_keypoints = None
        self.previous_right_keypoints = None

        self.ball1_lower = (0, 0, 0)
        self.ball1_upper = (255, 255, 255)
        self.ball2_lower = (0, 0, 0)
        self.ball2_upper = (255, 255, 255)

        self.previous_left_pupil_coords = None
        self.previous_left_pupil_direction = None
        self.previous_right_pupil_coords = None
        self.previous_right_pupil_direction = None
        self.previous_ball_distance = 0
        self.previous_ball_to_face_distance = None

        self.right_pupil_to_center_distance = None
        self.left_pupil_to_center_distance = None
        self.previous_right_pupil_to_center_distance = None
        self.previous_left_pupil_to_center_distance = None

        self.ball_closer_count = 0
        self.eyes_moving_out = 0
        self.eyes_not_moving_in = 0
        self.direction_change_count = 0
        self.frame_count = 0
        self.direction_change_frames = 10
        self.direction_change_thresh = 6

        self.ball_distance_array = []
        self.ball_to_face_distance_array = []
        self.left_eye_distance_center_array = []
        self.right_eye_distance_center_array = []

        self.user_positioned = False
        self.user_positioned_audio = False
        self.user_wiggling = False
        self.user_wiggling_audio = False

    def init_writer(self, img):
        # Create video writer
        frame_width, frame_height, _ = img.shape

        # size = (frame_width, frame_height)
        size = (frame_height, frame_width)

        currentTime = time.time() # seconds from UTC
        currentTime = time.ctime(currentTime)
        currentTime = str(currentTime).replace(" ", "_").replace(":", "_").strip()

        capWriter_filename = "data/results/" + currentTime + ".avi"
        self.capWriter = cv2.VideoWriter(capWriter_filename, 
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 20, size)

        self.Distance = Distance(self.ball1_lower, self.ball1_upper)

    def find_pupil_direction(self, pupil_coords, previous_pupil_coords):
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

    def is_direction_change(self, direction, previous_direction):
        if previous_direction is None:
            return False
        if direction is None:
            return False
        if direction != previous_direction:
            return True
        return False

    def check_for_wiggle(self):
        print("Direction changes: ", self.direction_change_count)
        print("Eyes not moving in: ", self.eyes_not_moving_in)
        print("Eyes moving out: ", self.eyes_moving_out)
        print("Ball closer count: ", self.ball_closer_count, "\n")

        if self.direction_change_count >= self.direction_change_thresh:
            self.user_wiggling = True
            print("WIGGLE")

        if self.eyes_moving_out > 8 and self.ball_closer_count >= 3:
            print("WIGGLE (eyes cannot focus anymore)")

        self.direction_change_count = 0
        self.frame_count = 0
        self.eyes_not_moving_in = 0
        self.eyes_moving_out = 0
        self.ball_closer_count = 0


    def run_detection_calibration(self, img):
        # ret, img = cap.read()
        # img = cv2.flip(img,1)

        # Convert Image into gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert image in black and white
        # (thresh, black_and_white) = cv2.threshold(gray, self.bw_threshold, 255, cv2.THRESH_BINARY)

        # detect face
        faces = models.face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = get_largest_frame(faces)

        # Face prediction for black and white
        # faces_bw = models.face_cascade.detectMultiScale(black_and_white, 1.1, 4)
        # face_bw = detect_face(faces_bw)

        ball = track_ball(img, self.ball1_lower, self.ball1_upper)

        if(len(faces) == 0):
            cv2.putText(img, "No face found...", self.org, self.font, self.font_scale, self.weared_mask_font_color, self.thickness, cv2.LINE_AA)
        else:
            # Draw rectangle on face
            for (x, y, w, h) in faces:
                face_distance = self.Distance.get_distance_face(img, faces)

                if len(ball) != 0:
                    ball_distance = self.Distance.get_distance_ball(img, ball)
                else:
                    ball_distance = self.previous_ball_distance

                self.ball_distance_array.append(ball_distance)

                if ball_distance > face_distance and self.previous_ball_distance < face_distance:
                    ball_distance = self.previous_ball_distance
                elif ball_distance > face_distance:
                    ball_distance = np.average(self.ball_distance_array)
                
                self.previous_ball_distance = ball_distance

                ball_to_face_distance = None
                ball_to_face_distance = face_distance-ball_distance

                # print(self.user_positioned, " ", ball_to_face_distance, " ", ball_distance)
                if not self.user_positioned and ball_to_face_distance > 0 and ball_distance != 0:
                    print("User positioned")
                    self.user_positioned = True

                self.ball_to_face_distance_array.append(ball_to_face_distance)

                left_eye_corner, right_eye_corner = get_eye_corners(img, gray, (x, y, w, h))

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                # roi_gray = gray[y:y + h, x:x + w]
                # roi_color = img[y:y + h, x:x + w]

                fr = gray[y : y + h, x : x + w]
                left_eye, right_eye, left_coord, right_coord = detect_eyes(fr)

                # Checks if the eyes are wiggling
                self.frame_count += 1
                if self.frame_count >= self.direction_change_frames:
                    self.check_for_wiggle()

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

                    pupil_coords = pupil_detect(crop_left, self.pupil_threshold)

                    left_corner_x, left_corner_y = left_eye_corner
                    left_corner_x = left_corner_x - x0
                    left_corner_y = left_corner_y - y0

                    direction = self.find_pupil_direction(pupil_coords, self.previous_left_pupil_coords)
                    left_direction_change = self.is_direction_change(direction, self.previous_left_pupil_direction)

                    self.previous_left_pupil_coords = pupil_coords
                    self.previous_left_pupil_direction = direction

                    self.previous_left_pupil_to_center_distance = self.left_pupil_to_center_distance
                    self.left_pupil_to_center_distance = left_corner_x - (pupil_coords[0] + pupil_coords[2]/2)

                    self.left_eye_distance_center_array.append(self.left_pupil_to_center_distance)

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

                    pupil_coords = pupil_detect(crop_right, self.pupil_threshold)

                    right_corner_x, right_corner_y = right_eye_corner
                    right_corner_x = right_corner_x - x0
                    right_corner_y = right_corner_y - y0

                    direction = self.find_pupil_direction(pupil_coords, self.previous_right_pupil_coords)
                    right_direction_change = self.is_direction_change(direction, self.previous_right_pupil_direction)

                    self.previous_right_pupil_coords = pupil_coords
                    self.previous_right_pupil_direction = direction

                    self.previous_right_pupil_to_center_distance = self.right_pupil_to_center_distance
                    self.right_pupil_to_center_distance = (pupil_coords[0]+pupil_coords[2]/2)-right_corner_x

                    self.right_eye_distance_center_array.append(self.right_pupil_to_center_distance)

                if self.previous_ball_to_face_distance is None or ball_to_face_distance is None:
                    ball_closer = False
                else:
                    ball_closer = ball_to_face_distance < self.previous_ball_to_face_distance

                if left_direction_change and right_direction_change and ball_closer:
                    self.direction_change_count += 1

                self.ball_closer_count += 1 if ball_closer else 0

                if (ball_closer and (self.right_pupil_to_center_distance is not None and self.left_pupil_to_center_distance is not None and self.previous_left_pupil_to_center_distance is not None and self.previous_right_pupil_to_center_distance is not None) and
                    (not self.right_pupil_to_center_distance < self.previous_right_pupil_to_center_distance and not self.left_pupil_to_center_distance < self.previous_left_pupil_to_center_distance)): 
                    self.eyes_not_moving_in += 1

                if (self.previous_left_pupil_to_center_distance is not None and self.previous_right_pupil_to_center_distance is not None and
                    self.left_pupil_to_center_distance is not None and self.right_pupil_to_center_distance is not None and
                    (self.right_pupil_to_center_distance > self.previous_right_pupil_to_center_distance or self.left_pupil_to_center_distance > self.previous_left_pupil_to_center_distance)):
                    self.eyes_moving_out += 1

                # print("Ball to face: (cur, prev) ", ball_to_face_distance, self.previous_ball_to_face_distance)
                # print("Left/Right direction change: ", left_direction_change, right_direction_change)

                self.previous_ball_to_face_distance = ball_to_face_distance

        #cv2.imshow('Gray', dst)
        
        self.capWriter.write(img)
        # cv2.imshow('Visual Exercises', img)


    def run_detection_training(self, img):
        # ret, img = cap.read()
        # img = cv2.flip(img,1)

        # Convert Image into gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert image in black and white
        # (thresh, black_and_white) = cv2.threshold(gray, self.bw_threshold, 255, cv2.THRESH_BINARY)

        # detect face
        faces = models.face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = get_largest_frame(faces)

        # Face prediction for black and white
        # faces_bw = models.face_cascade.detectMultiScale(black_and_white, 1.1, 4)
        # face_bw = detect_face(faces_bw)

        ball1 = track_ball(img, self.ball1_lower, self.ball1_upper)
        ball2 = track_ball(img, self.ball2_lower, self.ball2_upper)

        if(len(faces) == 0):
            cv2.putText(img, "No face found...", self.org, self.font, self.font_scale, self.weared_mask_font_color, self.thickness, cv2.LINE_AA)
        else:
            # Draw rectangle on face
            for (x, y, w, h) in faces:
                face_distance = self.Distance.get_distance_face(img, faces)

                if len(ball1) != 0:
                    ball1_distance = self.Distance.get_distance_ball(img, ball1)
                else:
                    ball1_distance = self.previous_ball_distance

                if len(ball2) != 0:
                    ball2_distance = self.Distance.get_distance_ball(img, ball2, True)
                else:
                    ball2_distance = self.previous_ball_distance


                self.ball_distance_array.append(ball1_distance)

                if ball1_distance > face_distance and self.previous_ball_distance < face_distance:
                    ball1_distance = self.previous_ball_distance
                elif ball1_distance > face_distance:
                    ball1_distance = np.average(self.ball_distance_array)
                
                self.previous_ball_distance = ball1_distance

                ball_to_face_distance = None
                ball_to_face_distance = face_distance-ball1_distance

                if not self.user_positioned and ball_to_face_distance > 0 and ball1_distance != 0:
                    self.user_positioned = True

                self.ball_to_face_distance_array.append(ball_to_face_distance)

                left_eye_corner, right_eye_corner = get_eye_corners(img, gray, (x, y, w, h))

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                # roi_gray = gray[y:y + h, x:x + w]
                # roi_color = img[y:y + h, x:x + w]

                fr = gray[y : y + h, x : x + w]
                left_eye, right_eye, left_coord, right_coord = detect_eyes(fr)

                # Checks if the eyes are wiggling
                self.frame_count += 1
                if self.frame_count >= self.direction_change_frames:
                    self.check_for_wiggle()

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

                    pupil_coords = pupil_detect(crop_left, self.pupil_threshold)

                    left_corner_x, left_corner_y = left_eye_corner
                    left_corner_x = left_corner_x - x0
                    left_corner_y = left_corner_y - y0

                    direction = self.find_pupil_direction(pupil_coords, self.previous_left_pupil_coords)
                    left_direction_change = self.is_direction_change(direction, self.previous_left_pupil_direction)

                    self.previous_left_pupil_coords = pupil_coords
                    self.previous_left_pupil_direction = direction

                    self.previous_left_pupil_to_center_distance = self.left_pupil_to_center_distance
                    self.left_pupil_to_center_distance = left_corner_x - (pupil_coords[0] + pupil_coords[2]/2)

                    self.left_eye_distance_center_array.append(self.left_pupil_to_center_distance)

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

                    pupil_coords = pupil_detect(crop_right, self.pupil_threshold)

                    right_corner_x, right_corner_y = right_eye_corner
                    right_corner_x = right_corner_x - x0
                    right_corner_y = right_corner_y - y0

                    direction = self.find_pupil_direction(pupil_coords, self.previous_right_pupil_coords)
                    right_direction_change = self.is_direction_change(direction, self.previous_right_pupil_direction)

                    self.previous_right_pupil_coords = pupil_coords
                    self.previous_right_pupil_direction = direction

                    self.previous_right_pupil_to_center_distance = self.right_pupil_to_center_distance
                    self.right_pupil_to_center_distance = (pupil_coords[0]+pupil_coords[2]/2)-right_corner_x

                    self.right_eye_distance_center_array.append(self.right_pupil_to_center_distance)

                if self.previous_ball_to_face_distance is None or ball_to_face_distance is None:
                    ball_closer = False
                else:
                    ball_closer = ball_to_face_distance < self.previous_ball_to_face_distance

                if left_direction_change and right_direction_change and ball_closer:
                    self.direction_change_count += 1

                self.ball_closer_count += 1 if ball_closer else 0

                if (ball_closer and (self.right_pupil_to_center_distance is not None and self.left_pupil_to_center_distance is not None and self.previous_left_pupil_to_center_distance is not None and self.previous_right_pupil_to_center_distance is not None) and
                    (not self.right_pupil_to_center_distance < self.previous_right_pupil_to_center_distance and not self.left_pupil_to_center_distance < self.previous_left_pupil_to_center_distance)): 
                    self.eyes_not_moving_in += 1

                if (self.previous_left_pupil_to_center_distance is not None and self.previous_right_pupil_to_center_distance is not None and
                    self.left_pupil_to_center_distance is not None and self.right_pupil_to_center_distance is not None and
                    (self.right_pupil_to_center_distance > self.previous_right_pupil_to_center_distance or self.left_pupil_to_center_distance > self.previous_left_pupil_to_center_distance)):
                    self.eyes_moving_out += 1

                # print("Ball to face: (cur, prev) ", ball_to_face_distance, self.previous_ball_to_face_distance)
                # print("Left/Right direction change: ", left_direction_change, right_direction_change)

                self.previous_ball_to_face_distance = ball_to_face_distance

        #cv2.imshow('Gray', dst)
        
        self.capWriter.write(img)
        # cv2.imshow('Visual Exercises', img)

    def plot_data(self):
        indx = [x for x in range(0, len(self.ball_to_face_distance_array))]
        plt.scatter(indx, self.ball_to_face_distance_array, color='red', label='Ball Distance')
        plt.savefig('data/results/plots/ball_distance.png', bbox_inches='tight')
        plt.close()

        indx = [x for x in range(0, len(self.left_eye_distance_center_array))]
        plt.scatter(indx, self.left_eye_distance_center_array, color='blue', label='Left Eye Distance')
        plt.savefig('data/results/plots/left_eye_distance.png', bbox_inches='tight')
        plt.close()

        indx = [x for x in range(0, len(self.right_eye_distance_center_array))]
        plt.scatter(indx, self.right_eye_distance_center_array, color='green', label='Right Eye Distance')
        plt.savefig('data/results/plots/right_eye_distance.png', bbox_inches='tight')
        plt.close()

    def finish(self):
        print("Finishing")
        self.capWriter.release()
        cv2.destroyAllWindows()

        self.plot_data()
