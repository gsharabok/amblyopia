import cv2
from cv2.data import haarcascades
import sys
import os.path

if getattr(sys, 'frozen', False):
	EXE_LOCATION = os.path.dirname( sys.executable ) # cx_Freeze frozen
	is_frozen = True
else:
	EXE_LOCATION = os.path.dirname( os.path.realpath( __file__ ) ) # Other packers
	is_frozen = False

if is_frozen:
	face_cascade = cv2.CascadeClassifier(os.path.join( EXE_LOCATION,'cv2','data','haarcascade_frontalface_default.xml'))
	eye_cascade = cv2.CascadeClassifier(os.path.join( EXE_LOCATION, 'cv2','data','haarcascade_eye_tree_eyeglasses.xml'))
else:
	face_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
	eye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# eye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_eye.xml")
# lefteye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_lefteye_2splits.xml")
# righteye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_righteye_2splits.xml")
# mouth_cascade = cv2.CascadeClassifier("data/xml/haarcascade_mcs_mouth.xml")
# upper_body = cv2.CascadeClassifier(haarcascades + "haarcascade_upperbody.xml")

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blob_detector = cv2.SimpleBlobDetector_create(detector_params)