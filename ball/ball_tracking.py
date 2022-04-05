import cv2

# Define color range for the ball used
# Can find the value using color_picker
# greenLower = (163, 165, 61) 
greenLower = (164, 48, 83)
# greenUpper = (185, 255, 224) 
greenUpper = (182, 214, 255)

def track_ball(frame, lower_color=(0,0,0), upper_color=(255,255,255)):
    # frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    ball = []
    ball_width = 0

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ball = cv2.minEnclosingCircle(c)
        ((x, y), radius) = ball
        ball_width = radius*2

        if radius > 3:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 1)

    return ball

    # cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)





# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help="path to the (optional) video file")
# args = vars(ap.parse_args())

# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)

# if not args.get("video", False):
#     camera = cv2.VideoCapture(0)
# else:
#     camera = cv2.VideoCapture(args["video"])

# while True:
#     (grabbed, frame) = camera.read()

#     if args.get("video") and not grabbed:
#         break

#     frame = imutils.resize(frame, width=600)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     mask = cv2.inRange(hsv, greenLower, greenUpper)
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)

#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#     center = None

#     if len(cnts) > 0:
#         c = max(cnts, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(c)

#         if radius > 10:
#             cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

#     cv2.imshow("Frame", frame)
#     cv2.imshow("Mask", mask)

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break

# camera.release()
# cv2.destroyAllWindows()

