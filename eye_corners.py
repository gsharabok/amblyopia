# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import sys
import os.path

if getattr(sys, 'frozen', False):
	EXE_LOCATION = os.path.dirname( sys.executable ) # cx_Freeze frozen
else:
	EXE_LOCATION = os.path.dirname( os.path.realpath( __file__ ) ) # Other packers
	
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join( EXE_LOCATION, 'shape_predictor_68_face_landmarks.dat'))

# Landmark 39: left eye corner; 42: right eye corner
# 0 indexed

# Args: full gray image, face bounding box
# Returns: left eye corner, right eye corner
def get_eye_corners(image, gray, face):
	rect = bb_to_rect(face)

	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

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

