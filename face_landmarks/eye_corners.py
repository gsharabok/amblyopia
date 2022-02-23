# Landmark 39: left eye corner; 42: right eye corner
# 0 indexed

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face_landmarks/shape_predictor_68_face_landmarks.dat')

# Args: full gray image, face bounding box
# Returns: left eye corner, right eye corner
def get_eye_corners(image, gray, face):
	rect = bb_to_rect(face)

	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# (x, y, w, h) = face

	# for (x, y) in shape:
	# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	# left
	x_left, y_left = shape[39]
	cv2.circle(image, (x_left, y_left), 1, (0, 255, 0), -1)

	# right
	x_right, y_right = shape[42]
	cv2.circle(image, (x_right, y_right), 1, (0, 255, 0), -1)

	return (x_left, y_left), (x_right, y_right)

def bb_to_rect(face):
	(x, y, w, h) = face
	return dlib.rectangle(x,y,x+w,y+h)

