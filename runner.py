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
import urllib
import requests
import imutils

from detection.mask import create_mask
from detection.face import detect_face, get_largest_frame
from detection.eye import detect_eyes,_cut_eyebrows
from detection.blob import blob_track
from detection.pupil import pupil_detect
from ball.ball_tracking import *
from face_landmarks.eye_corners import get_eye_corners
from distance_measure.distance import get_distance_face, get_distance_ball
import models

class Runner:
    def __init__(self): 
        # Adjust threshold value in range 80 to 105 based on your light.
        self.bw_threshold = 80

        # Maybe lower for high quality video and higher for live video
        self.pupil_threshold = 60 #25

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

        self.previous_left_pupil_coords = None
        self.previous_left_pupil_direction = None
        self.previous_right_pupil_coords = None
        self.previous_right_pupil_direction = None
        self.previous_ball_to_face_distance = None

        self.direction_change_count = 0
        self.frame_count = 0
        self.direction_change_frames = 10
        self.direction_change_thresh = 6

        self.ball_distance_array = []
        self.left_eye_distance_center_array = []
        self.right_eye_distance_center_array = []

        self.user_positioned = False
        self.user_positioned_audio = False
        self.user_wiggling = False
        self.user_wiggling_audio = False

    def init_writer(self, img):
        # Create video writer
        frame_width, frame_height, _ = img.shape

        size = (frame_width, frame_height)

        currentTime = time.time() # seconds from UTC
        currentTime = time.ctime(currentTime)
        currentTime = str(currentTime).replace(" ", "_").replace(":", "_").strip()

        capWriter_filename = "data/results/" + currentTime + ".avi"
        self.capWriter = cv2.VideoWriter(capWriter_filename, 
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)

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


    def run_detection(self, img):
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

        ball = track_ball(img)

        if(len(faces) == 0):
            cv2.putText(img, "No face found...", self.org, self.font, self.font_scale, self.weared_mask_font_color, self.thickness, cv2.LINE_AA)
        else:
            # Draw rectangle on face
            for (x, y, w, h) in faces:
                face_distance = get_distance_face(img, faces)
                ball_distance = get_distance_ball(img, ball)
                ball_to_face_distance = face_distance-ball_distance

                if not self.user_positioned and ball_to_face_distance > 0 and ball_distance != 0:
                    self.user_positioned = True

                self.ball_distance_array.append(ball_to_face_distance)

                left_eye_corner, right_eye_corner = get_eye_corners(img, gray, (x, y, w, h))

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                # roi_gray = gray[y:y + h, x:x + w]
                # roi_color = img[y:y + h, x:x + w]

                fr = gray[y : y + h, x : x + w]
                left_eye, right_eye, left_coord, right_coord = detect_eyes(fr)

                self.frame_count += 1
                if self.frame_count >= self.direction_change_frames:
                    print(self.direction_change_count)
                    if self.direction_change_count >= self.direction_change_thresh:
                        self.user_wiggling = True
                        print("WIGGLE")
                    self.direction_change_count = 0
                    self.frame_count = 0
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

                    self.left_eye_distance_center_array.append(left_corner_x - (pupil_coords[0] + pupil_coords[2]/2))

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

                    self.right_eye_distance_center_array.append((pupil_coords[0]+pupil_coords[2]/2)-right_corner_x)

                if self.previous_ball_to_face_distance is None or ball_to_face_distance is None:
                    ball_closer = False
                else:
                    ball_closer = ball_to_face_distance < self.previous_ball_to_face_distance

                if left_direction_change and right_direction_change and ball_closer:
                    self.direction_change_count += 1

                self.previous_ball_to_face_distance = ball_to_face_distance


        #cv2.imshow('Gray', dst)
        self.capWriter.write(img)
        # cv2.imshow('Amblyopia Treatment', img)


    def plot_data(self):
        indx = [x for x in range(0, len(self.ball_distance_array))]
        plt.scatter(indx, self.ball_distance_array, color='red', label='Ball Distance')
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
