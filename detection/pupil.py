from unittest.util import three_way_cmp
from cv2 import threshold
import numpy as np
import cv2
from skimage import io, color, measure, draw, img_as_bool, feature
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from scipy import optimize
import matplotlib.pyplot as plt

def pupil_thresh(self):
    kernel = np.ones((1, 1), np.uint8)
    self.source[:] = cv2.threshold(cv2.GaussianBlur(cv2.erode(self.source, kernel, iterations = 1), self.blur, 0), self.binarythreshold, 255, cv2.THRESH_BINARY_INV)[1]

def fit_circle(threshInv):
    # regions = measure.regionprops(threshInv)
    # bubble = regions[0]

    # y0, x0 = bubble.centroid
    # r = bubble.major_axis_length / 2.

    # def cost(params):
    #     x0, y0, r = params
    #     coords = draw.circle_perimeter(int(y0), int(x0), int(r), shape=threshInv.shape)
    #     template = np.zeros_like(threshInv)
    #     template[coords] = 1
    #     return -np.sum(template == threshInv)

    # x0, y0, r = optimize.fmin(cost, (x0, y0, r))

    # f, ax = plt.subplots()
    # circle = plt.Circle((x0, y0), r)
    # ax.imshow(threshInv, cmap='gray', interpolation='nearest')
    # ax.add_artist(circle)
    # plt.show()

    img = feature.canny(threshInv).astype(np.uint8)
    img[img > 0] = 255

    coords = np.column_stack(np.nonzero(img))

    model, inliers = measure.ransac(coords, measure.CircleModel,
                                    min_samples=3, residual_threshold=1,
                                    max_trials=500)

    print(model.params)

    ##
    # Detect two radii
    # hough_radii = np.arange(250, 300, 10)
    # hough_res = hough_circle(coords, hough_radii)

    # # Select the most prominent 5 circles
    # accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
    #                                            total_num_peaks=3)

    # # Draw them
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    # image = color.gray2rgb(img)
    # for center_y, center_x, radius in zip(cy, cx, radii):
    #     circy, circx = circle_perimeter(center_y, center_x, radius)
    #     image[circy, circx] = (220, 20, 20)

    # ax.imshow(image, cmap=plt.cm.gray)
    # plt.show()
    ##

    rr, cc = draw.circle_perimeter(int(model.params[0]), int(model.params[1]), int(model.params[2]),
                         shape=img.shape)

    img[rr, cc] = 128

    cv2.imshow("circle", img)

def circlee(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    cv2.imshow("fg", sure_fg)
    cv2.imshow("bg", sure_bg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    return img
    

    # cv2.imshow("Threshold", img)


def pupil_detect(eyeArr, bw_threshold):
    rows, cols, _ = eyeArr.shape

    # eyeArr = eyeArr[4:rows - 4, 4:cols - 4]

    gray_eye = cv2.cvtColor(eyeArr, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.GaussianBlur(gray_eye, (9, 9), 0)
    gray_eye = cv2.medianBlur(gray_eye, 3)

    # cv2.imshow("Eye1", gray_eye)

    threshold = cv2.threshold(gray_eye, bw_threshold, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow("Threshold1", threshold)

    ### Thresh (working part)
    # gray = cv2.cvtColor(eyeArr[4:rows - 4, 4:cols - 4], cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # blurred = cv2.medianBlur(blurred, 3)
    
    # (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.imshow("Threshold", threshInv)
    # print("[INFO] otsu's thresholding value: {}".format(T))


    # fit_circle(threshInv)
    # img = circlee(eyeArr[4:rows - 4, 4:cols - 4])
    ##### (ends here)

    # th2 = cv2.threshold(gray_eye,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # g1 = cv2.cvtColor(eyeArr, cv2.COLOR_BGR2GRAY)
    # g1 = cv2.GaussianBlur(g1, (7, 7), 0)
    # th3 = cv2.adaptiveThreshold(gray_eye[4:rows - 4, 4:cols - 4],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,3) # 15,2.5

    # cv2.imshow("TH", threshold)
    # cv2.imshow("TH3", th3)

    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    shape = (0, 0, 0, 0)
    for cnt in contours:
        shape = cv2.boundingRect(cnt)
        (x, y, w, h) = shape
        cv2.circle(eyeArr, (x + int(w/2), y + int(h/2)), int((h)/3), (0, 0, 255), 1)
        cv2.line(eyeArr, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
        cv2.line(eyeArr, (0, y + int(h/2)), (cols , y + int(h/2)), (0, 255, 0), 1)
        break
    
    # cv2.imwrite(out_dir + 'res_' + f, eyeArr)
    return shape