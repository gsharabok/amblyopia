import cv2
from cv2.data import haarcascades

face_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
# eye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_eye.xml")
# lefteye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_lefteye_2splits.xml")
# righteye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_righteye_2splits.xml")
mouth_cascade = cv2.CascadeClassifier("data/xml/haarcascade_mcs_mouth.xml")
# upper_body = cv2.CascadeClassifier(haarcascades + "haarcascade_upperbody.xml")

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blob_detector = cv2.SimpleBlobDetector_create(detector_params)