import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
# import imutils
import numpy as np
import models
from detection.face import get_largest_frame

colors = []
lower = np.array([0, 0, 0])
upper = np.array([255, 255, 255])

blur_margin = 0

def on_mouse_click(event, x, y, flags, image):
    if event == cv2.EVENT_LBUTTONUP:
        print("Clicked at: ", x, y)
        colors.append(image[y,x].tolist())

def on_trackbar_change(position):
    return

def reset_colors():
    global colors, lower, upper
    colors = []
    lower = np.array([0, 0, 0])
    upper = np.array([255, 255, 255])

def blur_img(img, factor = 20):
    kW = int(img.shape[1] / factor)
    kH = int(img.shape[0] / factor)

    #ensure the shape of the kernel is odd
    if kW % 2 == 0: kW = kW - 1
    if kH % 2 == 0: kH = kH - 1

    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    return blurred_img

def extract_color():
    vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    _,image = vidcap.read()

    # Load image, resize to 600 width, and convert color to HSV

    # image = imutils.resize(image, width=600)

    blurred_img = blur_img(image, factor = 2)
    faces = models.face_cascade.detectMultiScale(image, 1.1, 4)
    faces = get_largest_frame(faces)
    for (x, y, w, h) in faces:
        detected_face = image[int(y)-blur_margin:, int(x)-blur_margin:int(x+w)+blur_margin]
        blurred_img[y-blur_margin:, x-blur_margin:x+w+blur_margin] = detected_face
        image = blurred_img

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create window
    cv2.namedWindow('mask')
    cv2.setWindowProperty('mask', cv2.WND_PROP_TOPMOST, 1)
    cv2.namedWindow('image')
    cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)

    # Set mouse callback to capture HSV value on click
    cv2.setMouseCallback("image", on_mouse_click, hsv)
    cv2.createTrackbar("Min H", "image", int(lower[0]), 255, on_trackbar_change)
    cv2.createTrackbar("Min S", "image", int(lower[1]), 255, on_trackbar_change)
    cv2.createTrackbar("Min V", "image", int(lower[2]), 255, on_trackbar_change)
    cv2.createTrackbar("Max H", "image", int(upper[0]), 255, on_trackbar_change)
    cv2.createTrackbar("Max S", "image", int(upper[1]), 255, on_trackbar_change)
    cv2.createTrackbar("Max V", "image", int(upper[2]), 255, on_trackbar_change)

    # Show HSV image
    cv2.imwrite("data/video/reference_color.jpg", image)
    cv2.imshow("image", hsv)
    # cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)

    while True:
        # Get trackbar positions and set lower/upper bounds
        lower[0] = cv2.getTrackbarPos("Min H", "image")
        lower[1] = cv2.getTrackbarPos("Min S", "image")
        lower[2] = cv2.getTrackbarPos("Min V", "image")
        upper[0] = cv2.getTrackbarPos("Max H", "image")
        upper[1] = cv2.getTrackbarPos("Max S", "image")
        upper[2] = cv2.getTrackbarPos("Max V", "image")

        # Create a range mask then erode and dilate to reduce noise
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Show range mask
        cv2.imshow("mask", mask)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    # if not colors:
    #   exit

    if len(colors) > 0:
      minh = min(c[0] for c in colors)
      mins = min(c[1] for c in colors)
      minv = min(c[2] for c in colors)

      maxh = max(c[0] for c in colors)
      maxs = max(c[1] for c in colors)
      maxv = max(c[2] for c in colors)

      print("Mouse Click Selection: ")
      print([minh, mins, minv])
      print([maxh, maxs, maxv])

    print("Manual Values: ")
    print("({}, {}, {})".format(lower[0], lower[1], lower[2]))
    print("({}, {}, {})".format(upper[0], upper[1], upper[2]))

    return [lower, upper]

