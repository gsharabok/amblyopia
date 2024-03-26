# Amblyopia Gymnastics - Visual Training
Amblyopia (“Lazy eye”) is a serious issue described as the reduction in the vision of one of the eyes. Amblyopia treatment to this day has been primarily focused on refractive correction, occlusion, penalization, and medical involvement. The early-stage diagnosis and treatment heavily rely on the doctor's subjective measurement, which is highly prone to human errors. As take-home vision strengthening exercises are increasing in popularity, the measurement of progress lacks standardization. Using previous research on the effectiveness of such exercises and common issues faced by kids, this project presents a working system for conducting visual training exercises for the use of eye care professionals. The program provides a graphical user interface designed specifically for the needs of young users, while also standardizing the tracking and measurement process. 

## Quick Start
To run:
python main.py
python ball\color_picker.py -i 1 --video

Parameters:
Path to a video file: --filepath data/video/video.mp4
Define threshold for pupil tracking: --pupilthresh 100
Use live camera: --camera

## Files overview
runner.py -> backend for tracking and recognition
server.py -> for remote mobile use (limited by network)
client.js -> code for remote streaming

main.py -> local runner
local_client.py -> local flask stuff
camera_opencv, base_camera -> local opencv stuff

## Details
For ball tracking:
Program relies on the color of the ball
To find the proper settings, run color_picker.py on a screenshot from the video
Then put the measure min and max into ball_tracking.py

For distance measurement:
Take a reference photo: ref_ball.jpg
And put the measured distance from camera to object into distance.py

## Other
Tests:
For g_train2.mp4 use ref_ball.JPG for reference and run:
python main.py --filepath data\video\g_train2.mp4 --pupilthresh 40

Serve over the Internet:
python frontend_main.py
ngrok http 5000
(when on mobile make sure to add https://)

## TODO
in fit_circle only feed the eye part to make sure the circle fits into the eye -> then just use the circle as the pupil
When video feed ends fix so it doesnt crush:
    Traceback (most recent call last):
    File "main.py", line 137, in <module>
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.error: OpenCV(4.5.4) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'