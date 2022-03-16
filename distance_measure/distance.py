import cv2
from ball.ball_tracking import track_ball
from detection.face import get_largest_frame
import models

# Small string 34 cm
# Face width 14.3 cm
# Ball width 1.6 cm
# KNOWN_DISTANCE_FACE = 34  # centimeter
# KNOWN_WIDTH_FACE = 14.3  # centimeter

# KNOWN_DISTANCE_BALL = 17  # centimeter
# KNOWN_WIDTH_BALL = 1.6  # centimeter


# Testing distances
KNOWN_DISTANCE_FACE = 34  # centimeter
KNOWN_WIDTH_FACE = 14.3  # centimeter

KNOWN_DISTANCE_BALL = 12  # centimeter
KNOWN_WIDTH_BALL = 1.6  # centimeter


# variables
# distance from camera to object(face) measured
# KNOWN_DISTANCE = 76.2  # centimeter
# width of face in the real world or Object Plane
# KNOWN_WIDTH = 14.3  # centimeter
# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(0)

# face detector object
face_detector = models.face_cascade


def get_distance_face(frame, faces):
    if len(faces) == 0: return 0
    
    faces = faces[0]
    _,_,w,_ = faces
    face_width_in_frame = w
    # finding the distance by calling function Distance
    Distance = 0
    if face_width_in_frame != 0:
        Distance = distance_finder(focal_length_found, KNOWN_WIDTH_FACE, face_width_in_frame)
        # Drwaing Text on the screen
        cv2.putText(
            frame, f"Face Distance = {round(Distance,2)} CM", (30, 30), fonts, 0.6, (WHITE), 1
        )
    # cv2.imshow("frame", frame)    
    return Distance

def get_distance_ball(frame, ball):
    if len(ball) == 0: return 0

    ((x, y), radius) = ball
    ball_width_in_frame = radius*2

    Distance = 0
    if ball_width_in_frame != 0:
        Distance = distance_finder(focal_ball_length_found, KNOWN_WIDTH_BALL, ball_width_in_frame, 0.8)

        cv2.putText(
            frame, f"Ball Distance = {round(Distance,2)} CM", (30, 50), fonts, 0.6, (WHITE), 1
        )

    return Distance

# focal length finder function
def focal_length(measured_distance, real_width, width_in_rf_image):
    """
    This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using
    MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
    :param1 Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image

    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 14.3 centimeters)
    :param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
    :retrun focal_length(Float):"""
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


# distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame, angle_adjustment=1):
    """
    This Function simply Estimates the distance between object and camera using arguments(focal_length, Actual_object_width, Object_width_in_the_image)
    :param1 focal_length(float): return by the focal_length_Finder function

    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
    :param3 object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
    :return Distance(float) : distance Estimated
    """
    distance = (real_face_width * focal_length * angle_adjustment) / face_width_in_frame
    return distance


# face detector function
def face_data(image):
    """
    This function Detect the face
    :param Takes image as argument.
    :returns face_width in the pixels
    """

    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = models.face_cascade.detectMultiScale(gray_image, 1.1, 4)
    faces = get_largest_frame(faces)
    # faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        face_width = w

    return face_width


# TODO: expects a face and a ball to be present for calibration

ref_image = cv2.imread("data/video/reference_color.jpg")
ref_image = cv2.flip(ref_image,1)
ref_image_face_width = face_data(ref_image)
focal_length_found = focal_length(KNOWN_DISTANCE_FACE, KNOWN_WIDTH_FACE, ref_image_face_width)
# print(focal_length_found)

ball = track_ball(ref_image)
((x, y), radius) = ball
ref_image_ball_width = radius*2
focal_ball_length_found = focal_length(KNOWN_DISTANCE_BALL, KNOWN_WIDTH_BALL, ref_image_ball_width)
