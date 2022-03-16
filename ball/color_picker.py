import argparse
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
# import imutils
import numpy as np

colors = []
lower = np.array([0, 0, 0])
upper = np.array([255, 255, 255])

def on_mouse_click(event, x, y, flags, image):
    if event == cv2.EVENT_LBUTTONUP:
        print("Clicked at: ", x, y)
        colors.append(image[y,x].tolist())

def on_trackbar_change(position):
    return

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="path to the image file")
parser.add_argument("--video", help="use video", action="store_true")
parser.add_argument("-l", "--lower", help="HSV lower bounds")
parser.add_argument("-u", "--upper", help="HSV upper bounds")
args = vars(parser.parse_args())

# Parse lower and upper bounds arguments (--lower "1, 2, 3" --upper "4, 5, 6")
if args["lower"] and args["upper"]:
  lower = np.fromstring(args["lower"], sep=",")
  upper = np.fromstring(args["upper"], sep=",")

if args["video"]:
  src = args["image"]
  if len(args["image"]) == 1:
    src = int(args["image"])
  vidcap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
  _,image = vidcap.read()
else:
  image = cv2.imread(args["image"])

# Load image, resize to 600 width, and convert color to HSV

# image = imutils.resize(image, width=600)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create window
cv2.namedWindow('image')

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

