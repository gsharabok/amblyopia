import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import argparse
import matplotlib.pyplot as plt
import imutils

from imutils.video import WebcamVideoStream

from detection.face import get_largest_frame
from detection.eye import detect_eyes
from detection.pupil import pupil_detect
from ball.ball_tracking import *
from eye_corners import get_eye_corners
from distance_measure.distance import get_distance_face, get_distance_ball
import models

# Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--filepath", help="path to video file")
parser.add_argument("--pupilthresh", help="pupil threshold", type=int)
parser.add_argument("--camera", help="use live video feed", action="store_true")
parser.add_argument("--remote", help="stream camera")

args = parser.parse_args()

filepath = 'data/video/eye_training_stable.MOV'
if args.filepath:
    filepath = args.filepath
if args.camera:
    filepath = 1
if args.remote:
    filepath = args.remote


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
previous_ball_distance = 0
previous_ball_to_face_distance = None

direction_change_count = 0
frame_count = 0
direction_change_frames = 10
direction_change_thresh = 6

ball_distance_array = []
left_eye_distance_center_array = []
right_eye_distance_center_array = []

# Threaded video reading
cap = WebcamVideoStream(src=1).start()

# Normal video reading
# cap = cv2.VideoCapture(filepath)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# Create video writer
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
   
# size = (frame_width, frame_height)

# currentTime = time.time() # seconds from UTC
# currentTime = time.ctime(currentTime)
# currentTime = str(currentTime).replace(" ", "_").replace(":", "_").strip()

# capWriter_filename = "data/results/" + currentTime + ".avi"
# capWriter = cv2.VideoWriter(capWriter_filename, 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)

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
    # Normal Read
    # ret, img = cap.read()
    # img = cv2.flip(img,1)

    # Threaded Read
    img = cap.read()
    img = imutils.resize(img, width=400)

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
            if len(ball) != 0:
                ball_distance = get_distance_ball(img, ball)
            else:
                ball_distance = previous_ball_distance
            previous_ball_distance = ball_distance
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
                print("Direction changes: ", direction_change_count)
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

                left_corner_x, left_corner_y = left_eye_corner
                left_corner_x = left_corner_x - x0
                left_corner_y = left_corner_y - y0

                direction = find_pupil_direction(pupil_coords, previous_left_pupil_coords)
                left_direction_change = is_direction_change(direction, previous_left_pupil_direction)

                previous_left_pupil_coords = pupil_coords
                previous_left_pupil_direction = direction

                left_eye_distance_center_array.append(left_corner_x - (pupil_coords[0] + pupil_coords[2]/2))

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

                right_corner_x, right_corner_y = right_eye_corner
                right_corner_x = right_corner_x - x0
                right_corner_y = right_corner_y - y0

                direction = find_pupil_direction(pupil_coords, previous_right_pupil_coords)
                right_direction_change = is_direction_change(direction, previous_right_pupil_direction)

                previous_right_pupil_coords = pupil_coords
                previous_right_pupil_direction = direction

                right_eye_distance_center_array.append((pupil_coords[0]+pupil_coords[2]/2)-right_corner_x)

            if previous_ball_to_face_distance is None or ball_to_face_distance is None:
                ball_closer = False
            else:
                ball_closer = ball_to_face_distance < previous_ball_to_face_distance
            
            if left_direction_change and right_direction_change and ball_closer:
                direction_change_count += 1

            previous_ball_to_face_distance = ball_to_face_distance


    #cv2.imshow('Gray', dst)
    # capWriter.write(img)
    cv2.imshow('Visual Exercises', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Press Esc to kill the program
        break

# Release video
cap.release()
# capWriter.release()
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
