from cv2 import threshold
import numpy as np
import cv2

def pupil_thresh(self):
    kernel = np.ones((1, 1), np.uint8)
    self.source[:] = cv2.threshold(cv2.GaussianBlur(cv2.erode(self.source, kernel, iterations = 1), self.blur, 0), self.binarythreshold, 255, cv2.THRESH_BINARY_INV)[1]


def pupil_detect(eyeArr, bw_threshold):
    rows, cols, _ = eyeArr.shape

    gray_eye = cv2.cvtColor(eyeArr, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.GaussianBlur(gray_eye, (9, 9), 0)
    gray_eye = cv2.medianBlur(gray_eye, 3)

    threshold = cv2.threshold(gray_eye, bw_threshold, 255, cv2.THRESH_BINARY_INV)[1]
    # threshold = cv2.threshold(gray_eye,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # threshold = cv2.adaptiveThreshold(gray_eye,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,100)[1]

    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.circle(eyeArr, (x + int(w/2), y + int(h/2)), int((h)/3), (0, 0, 255), 1)
        cv2.line(eyeArr, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 1)
        cv2.line(eyeArr, (0, y + int(h/2)), (cols , y + int(h/2)), (0, 255, 0), 1)
        break
    
    # cv2.imwrite(out_dir + 'res_' + f, eyeArr)
    return eyeArr