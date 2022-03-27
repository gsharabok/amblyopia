import cv2
import os.path
from ball.ball_tracking import track_ball
from detection.face import get_largest_frame
import models

class Distance:
    def __init__(self, lower_ball_color, higher_ball_color):
        # Small string 34 cm
        # Face width 14.3 cm
        # Ball width 1.6 cm
        # KNOWN_DISTANCE_FACE = 34  # centimeter
        # KNOWN_WIDTH_FACE = 14.3  # centimeter

        # KNOWN_DISTANCE_BALL = 17  # centimeter
        # KNOWN_WIDTH_BALL = 1.6  # centimeter


        # Testing distances
        self.KNOWN_DISTANCE_FACE = 34  # centimeter
        self.KNOWN_WIDTH_FACE = 14.3  # centimeter

        self.KNOWN_DISTANCE_BALL = 20  # centimeter
        self.KNOWN_WIDTH_BALL = 1.6  # centimeter


        # variables
        # distance from camera to object(face) measured
        # KNOWN_DISTANCE = 76.2  # centimeter
        # width of face in the real world or Object Plane
        # KNOWN_WIDTH = 14.3  # centimeter
        # Colors
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        self.fonts = cv2.FONT_HERSHEY_COMPLEX

        # face detector object
        self.face_detector = models.face_cascade

        # TODO: expects a face and a ball to be present for calibration

        if models.is_frozen:
            self.ref_image = cv2.imread(os.path.join(models.EXE_LOCATION,'data','reference_color.jpg'))
        else:
            self.ref_image = cv2.imread(os.path.join(models.EXE_LOCATION,'data','reference_color.jpg'))


        self.ref_image = cv2.flip(self.ref_image,1)
        self.ref_image_face_width = self.face_data(self.ref_image)
        self.focal_length_found = self.focal_length(self.KNOWN_DISTANCE_FACE, self.KNOWN_WIDTH_FACE, self.ref_image_face_width)
        # print(focal_length_found)

        ball = track_ball(self.ref_image, lower_ball_color, higher_ball_color)
        ((x, y), radius) = ball
        self.ref_image_ball_width = radius*2
        self.focal_ball_length_found = self.focal_length(self.KNOWN_DISTANCE_BALL, self.KNOWN_WIDTH_BALL, self.ref_image_ball_width)

        cv2.imwrite("data/video/reference_color_detection.jpg", self.ref_image)


    def get_distance_face(self, frame, faces):
        if len(faces) == 0: return 0
        
        faces = faces[0]
        _,_,w,_ = faces
        face_width_in_frame = w
        # finding the distance by calling function Distance
        Distance = 0
        if face_width_in_frame != 0:
            Distance = self.distance_finder(self.focal_length_found, self.KNOWN_WIDTH_FACE, face_width_in_frame)
            # Drwaing Text on the screen
            cv2.putText(
                frame, f"Face Distance = {round(Distance,2)} CM", (30, 30), self.fonts, 0.6, (self.WHITE), 1
            )
        # cv2.imshow("frame", frame)    
        return Distance
    
    def get_distance_ball(self, frame, ball, second_ball=False):
        if len(ball) == 0: return 0
    
        ((x, y), radius) = ball
        ball_width_in_frame = radius*2
    
        Distance = 0
        if ball_width_in_frame != 0:
            Distance = self.distance_finder(self.focal_ball_length_found, self.KNOWN_WIDTH_BALL, ball_width_in_frame, 0.6)
            # print(Distance)
    
            if not second_ball:
                cv2.putText(
                    frame, f"Ball1 Distance = {round(Distance,2)} CM", (30, 50), self.fonts, 0.6, (self.WHITE), 1
                )
            else:
                cv2.putText(
                    frame, f"Ball2 Distance = {round(Distance,2)} CM", (30, 70), self.fonts, 0.6, (self.WHITE), 1
                )
    
        return Distance
    
    # focal length finder function
    def focal_length(self, measured_distance, real_width, width_in_rf_image):
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
    def distance_finder(self, focal_length, real_face_width, face_width_in_frame, angle_adjustment=1):
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
    def face_data(self, image):
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
            cv2.rectangle(image, (x, y), (x + w, y + h), self.WHITE, 1)
            face_width = w
    
        return face_width
