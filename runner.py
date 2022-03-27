import numpy as np
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
# from moviepy.editor import concatenate_audioclips, AudioFileClip
from moviepy.audio.io import AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips

from detection.mask import create_mask
from detection.face import detect_face, get_largest_frame
from detection.eye import detect_eyes,_cut_eyebrows
from detection.blob import blob_track
from detection.pupil import pupil_detect
from ball.ball_tracking import *
from eye_corners import get_eye_corners
# from distance_measure.distance import get_distance_face, get_distance_ball
from distance_measure.distance_class import Distance
# from local_client import send_feedback_audio
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
        self.blur_margin = 0

        self.ball1_lower = (0, 0, 0)
        self.ball1_upper = (255, 255, 255)
        self.ball2_lower = (0, 0, 0)
        self.ball2_upper = (255, 255, 255)

        self.previous_left_pupil_coords = None
        self.previous_left_pupil_direction = None
        self.previous_right_pupil_coords = None
        self.previous_right_pupil_direction = None
        self.previous_ball_distance = 0
        self.ball_to_face_distance = None
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
        self.direction_change_thresh = 5

        self.ball_distance_array = []
        self.ball_to_face_distance_array = []
        self.left_eye_distance_center_array = []
        self.right_eye_distance_center_array = []

        self.user_positioned = False
        self.user_positioned_audio = False
        self.user_wiggling = False
        self.user_wiggling_audio = False
        self.user_wiggling2 = False
        self.user_wiggling2_audio = False

        self.is_training = False
        self.training_reps = 0
        self.training_session_time = 0
        self.training_start_time = None
        self.training_ball1_distances = []
        self.training_ball2_distances = []
        self.average_ball1_distance = 0
        self.average_ball2_distance = 0
        self.training_finished = False
        self.training_feedback_ready = False


    def init_writer(self, img):
        # Create video writer
        frame_width, frame_height, _ = img.shape

        # size = (frame_width, frame_height)
        size = (frame_height, frame_width)

        currentTime = time.time() # seconds from UTC
        currentTime = time.ctime(currentTime)
        currentTime = str(currentTime).replace(" ", "_").replace(":", "_").strip()

        path = 'data/results'
        if not os.path.exists(path):
            os.makedirs(path)

        path = 'data/results/plots'
        if not os.path.exists(path):
            os.makedirs(path)

        if models.is_frozen:
            capWriter_filename = os.path.join(models.EXE_LOCATION,'data','results',currentTime + ".avi")
        else:
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
        # print("Direction changes: ", self.direction_change_count)
        # print("Eyes not moving in: ", self.eyes_not_moving_in)
        # print("Eyes moving out: ", self.eyes_moving_out)
        # print("Ball closer count: ", self.ball_closer_count, "\n")

        if self.direction_change_count >= self.direction_change_thresh:
            self.user_wiggling = True
            print("WIGGLE")

        if self.eyes_moving_out >= 6 and self.ball_closer_count >= 5:
            print("WIGGLE (eyes cannot focus anymore)")
            if self.ball_to_face_distance < 10 or self.previous_ball_to_face_distance < 10:
                print("WIGGLE (ball is too close to face)")
                self.user_wiggling2 = True

        self.direction_change_count = 0
        self.frame_count = 0
        self.eyes_not_moving_in = 0
        self.eyes_moving_out = 0
        self.ball_closer_count = 0

    def blur_img(self, img, factor = 20):
        kW = int(img.shape[1] / factor)
        kH = int(img.shape[0] / factor)
    
        #ensure the shape of the kernel is odd
        if kW % 2 == 0: kW = kW - 1
        if kH % 2 == 0: kH = kH - 1
    
        blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
        return blurred_img

    def run_detection_calibration(self, img):
        # ret, img = cap.read()
        # img = cv2.flip(img,1)

        # Convert Image into gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = self.blur_img(img, factor = 2)

        # Convert image in black and white
        # (thresh, black_and_white) = cv2.threshold(gray, self.bw_threshold, 255, cv2.THRESH_BINARY)

        # detect face
        faces = models.face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = get_largest_frame(faces)

        # Face prediction for black and white
        # faces_bw = models.face_cascade.detectMultiScale(black_and_white, 1.1, 4)
        # face_bw = detect_face(faces_bw)


        if(len(faces) == 0):
            cv2.putText(img, "No face found...", self.org, self.font, self.font_scale, self.weared_mask_font_color, self.thickness, cv2.LINE_AA)
        else:
            # Draw rectangle on face
            for (x, y, w, h) in faces:
                detected_face = img[int(y)-self.blur_margin:, int(x)-self.blur_margin:int(x+w)+self.blur_margin]
                blurred_img[y-self.blur_margin:, x-self.blur_margin:x+w+self.blur_margin] = detected_face
                img = blurred_img

                ball = track_ball(img, self.ball1_lower, self.ball1_upper)

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

                self.ball_to_face_distance = None
                self.ball_to_face_distance = face_distance-ball_distance

                # print(self.user_positioned, " ", ball_to_face_distance, " ", ball_distance)
                if not self.user_positioned and self.ball_to_face_distance > 0 and ball_distance != 0:
                    print("User positioned")
                    self.user_positioned = True

                self.ball_to_face_distance_array.append(self.ball_to_face_distance)

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

                if self.previous_ball_to_face_distance is None or self.ball_to_face_distance is None:
                    ball_closer = False
                else:
                    ball_closer = self.ball_to_face_distance < self.previous_ball_to_face_distance

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

                self.previous_ball_to_face_distance = self.ball_to_face_distance

        #cv2.imshow('Gray', dst)
        
        self.capWriter.write(img)
        return img
        # cv2.imshow('Visual Exercises', img)


    def run_detection_training(self, img):
        # ret, img = cap.read()
        # img = cv2.flip(img,1)

        # Convert Image into gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = self.blur_img(img, factor = 2)

        # Convert image in black and white
        # (thresh, black_and_white) = cv2.threshold(gray, self.bw_threshold, 255, cv2.THRESH_BINARY)

        # detect face
        faces = models.face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = get_largest_frame(faces)

        # Face prediction for black and white
        # faces_bw = models.face_cascade.detectMultiScale(black_and_white, 1.1, 4)
        # face_bw = detect_face(faces_bw)

        if(len(faces) == 0):
            cv2.putText(img, "No face found...", self.org, self.font, self.font_scale, self.weared_mask_font_color, self.thickness, cv2.LINE_AA)
        else:
            # Draw rectangle on face
            for (x, y, w, h) in faces:
                detected_face = img[int(y)-self.blur_margin:, int(x)-self.blur_margin:int(x+w)+self.blur_margin]
                blurred_img[y-self.blur_margin:, x-self.blur_margin:x+w+self.blur_margin] = detected_face
                img = blurred_img

                ball1 = track_ball(img, self.ball1_lower, self.ball1_upper)
                ball2 = track_ball(img, self.ball2_lower, self.ball2_upper)

                face_distance = self.Distance.get_distance_face(img, faces)

                if len(ball1) != 0:
                    ball1_distance = self.Distance.get_distance_ball(img, ball1)
                else:
                    ball1_distance = self.previous_ball_distance

                if len(ball2) != 0:
                    ball2_distance = self.Distance.get_distance_ball(img, ball2, True)
                else:
                    ball2_distance = self.previous_ball_distance

                self.training_ball1_distances.append(face_distance - ball1_distance)
                self.training_ball2_distances.append(face_distance - ball2_distance)

                self.ball_distance_array.append(ball1_distance)

                if ball1_distance > face_distance and self.previous_ball_distance < face_distance:
                    ball1_distance = self.previous_ball_distance
                elif ball1_distance > face_distance:
                    ball1_distance = np.average(self.ball_distance_array)
                
                self.previous_ball_distance = ball1_distance

                self.ball_to_face_distance = None
                self.ball_to_face_distance = face_distance-ball1_distance

                if not self.user_positioned and self.ball_to_face_distance > 0 and ball1_distance != 0:
                    self.user_positioned = True

                self.ball_to_face_distance_array.append(self.ball_to_face_distance)

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

                if self.previous_ball_to_face_distance is None or self.ball_to_face_distance is None:
                    ball_closer = False
                else:
                    ball_closer = self.ball_to_face_distance < self.previous_ball_to_face_distance

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

                self.previous_ball_to_face_distance = self.ball_to_face_distance

        #cv2.imshow('Gray', dst)
        
        self.capWriter.write(img)
        return img

    def write_training_log(self):
        path = 'data/results/log'
        if not os.path.exists(path):
            os.makedirs(path)

        if models.is_frozen:
            filename = os.path.join(models.EXE_LOCATION,'data','results','log',datetime.today().strftime('%Y-%m-%d') + '.txt')
        else:
            filename = 'data/results/log/' + datetime.today().strftime('%Y-%m-%d') + '.txt'

        session_count = 0

        if os.path.exists(filename):
            with open(filename, "r+") as file:
                for line in file:
                    if "Session" in line.rstrip():
                        session_count += 1

                file.close()

        self.average_ball1_distance = sum(self.training_ball1_distances)/len(self.training_ball1_distances)
        self.average_ball2_distance = sum(self.training_ball2_distances)/len(self.training_ball2_distances)

        more_lines = []
        more_lines.append("Session " + str(session_count + 1) + ": ")
        more_lines.append("\tTraining finished: " + str(self.training_finished))
        more_lines.append("\tTraining time: " + str(self.training_session_time))
        more_lines.append("\tTraining reps: " + str(self.training_reps))
        more_lines.append("\tAverage first ball distance: " + str(self.average_ball1_distance))
        more_lines.append("\tAverage second ball distance: " + str(self.average_ball2_distance))
        more_lines.append("\n")
        
        file = open(filename, 'a+')
        file.writelines('\n'.join(more_lines))
        file.close()

        if session_count > 0:
            self.send_training_feedback(session_count+1)

    def concatenate_audio_moviepy(self, audio_clip_paths, output_path = "static/audio/combined/combined.mp3"):
        """Concatenates several audio files into one audio file using MoviePy
        and save it to `output_path`. Note that extension (mp3, etc.) must be added to `output_path`"""
        clips = [AudioFileClip("static/audio/base_files/" + c + ".mp3") for c in audio_clip_paths]
        final_clip = concatenate_audioclips(clips)
        final_clip.write_audiofile(output_path)

    def get_pos_nums(self, num):
        pos_nums = []
        while num != 0:
            pos_nums.append(num % 10)
            num = num // 10

        pos_nums = pos_nums[::-1]
        return pos_nums

    def get_times(self, time):
        print(time)
        times = []
        if time > 100 and time < 1000:
            nums = self.get_pos_nums(time)
            times.append(str(nums[0] * 100))
            times.append(str(nums[1] * 10))
            times.append(str(nums[2]))
        elif time > 10 and time < 100:
            nums = self.get_pos_nums(time)
            times.append(str(nums[0] * 10))
            times.append(str(nums[1]))
        elif time < 10:
            times.append(str(time))

        return times

    def send_training_feedback(self, current_session):
        audios = ["from_your_last_session"]
        send_audio = False

        if models.is_frozen:
            filename = os.path.join(models.EXE_LOCATION,'data','results','log',datetime.today().strftime('%Y-%m-%d') + '.txt')
        else:
            filename = 'data/results/log/' + datetime.today().strftime('%Y-%m-%d') + '.txt'

        session_found = False
        past_training_time = 0
        past_training_reps = 0
        past_average_ball1_distance = 0
        past_average_ball2_distance = 0

        with open(filename) as file:
            for line in file:
                if "Session " + str(current_session-1) in line:
                    session_found = True

                if "Session " + str(current_session) in line:
                    break

                line = line.rstrip()
                if session_found:
                    if "Training time: " in line:
                        past_training_time = int(float(line.split(": ")[1]))
                    if "Training reps: " in line:
                        past_training_reps = int(float(line.split(": ")[1]))
                    if "Average first ball distance: " in line:
                        past_average_ball1_distance = float(line.split(": ")[1])
                    if "Average second ball distance: " in line:
                        past_average_ball2_distance = float(line.split(": ")[1])

            file.close()

        print("Calculating...")
        print(self.training_session_time, past_training_time)
        if int(self.training_session_time) > int(past_training_time):
            send_audio = True
            audios.append("you_have_trained_for")
            
            time = int(self.training_session_time - past_training_time)
            times = self.get_times(time)

            audios.extend(times)
            audios.append("seconds_longer")

        print(self.training_reps, past_training_reps)
        if self.training_reps > past_training_reps:
            send_audio = True
            audios.append("you_have_done")

            time = int(self.training_reps - past_training_reps)
            times = self.get_times(time)

            audios.extend(times)
            audios.append("more_reps")

        print(self.average_ball1_distance, past_average_ball1_distance)
        if self.average_ball1_distance < past_average_ball1_distance:
            send_audio = True
            audios.append("your_first_ball_is")

            time = int(past_average_ball1_distance - self.average_ball1_distance)
            times = self.get_times(time)

            audios.extend(times)
            audios.append("cm_closer")

        if send_audio:
            audios.append("keep_it_up")

            self.concatenate_audio_moviepy(audios)

            print("Sending training feedback...")
            self.training_feedback_ready = True


    def plot_data(self):
        indx = [x for x in range(0, len(self.ball_to_face_distance_array))]
        plt.scatter(indx, self.ball_to_face_distance_array, color='red', label='Ball Distance')
        if models.is_frozen:
            plt.savefig(os.path.join(models.EXE_LOCATION,'data','results','plots','ball_distance.png'), bbox_inches='tight')
        else:
            plt.savefig('data/results/plots/ball_distance.png', bbox_inches='tight')
        
        plt.close()

        indx = [x for x in range(0, len(self.left_eye_distance_center_array))]
        plt.scatter(indx, self.left_eye_distance_center_array, color='blue', label='Left Eye Distance')
        if models.is_frozen:
            plt.savefig(os.path.join(models.EXE_LOCATION,'data','results','plots','left_eye_distance.png'), bbox_inches='tight')
        else:
            plt.savefig('data/results/plots/left_eye_distance.png', bbox_inches='tight')
        plt.close()

        indx = [x for x in range(0, len(self.right_eye_distance_center_array))]
        plt.scatter(indx, self.right_eye_distance_center_array, color='green', label='Right Eye Distance')
        if models.is_frozen:
            plt.savefig(os.path.join(models.EXE_LOCATION,'data','results','plots','right_eye_distance.png'), bbox_inches='tight')
        else:
            plt.savefig('data/results/plots/right_eye_distance.png', bbox_inches='tight')
        plt.close()

    def finish(self):
        print("Finishing")
        self.capWriter.release()
        cv2.destroyAllWindows()

        if self.is_training:
            self.write_training_log()
        self.plot_data()
