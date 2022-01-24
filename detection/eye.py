import numpy as np
from typing import Optional
import models

def _cut_eyebrows(img, coord):
        """
        Primitively cut eyebrows out of an eye frame by simply cutting the top ~30% of the frame
        """
        if img is None:
            return img, coord
        height, width = img.shape[:2]
        img = img[15:height, 0:width]  # cut eyebrows out (15 px)

        as_list = list(coord)
        as_list[1] += 15
        as_list[3] -= 15
        coord = tuple(as_list)

        return img, coord

def detect_eyes(
        face_img: np.ndarray, cut_brows=False
    ) -> (Optional[np.ndarray], Optional[np.ndarray]):
        """
        Detect eyes, optionally cut the eyebrows out
        """
        coords = models.eye_cascade.detectMultiScale(face_img, 1.3, 5)

        left_eye = right_eye = None
        left_coord = right_coord = None

        if coords is None or len(coords) == 0:
            return left_eye, right_eye, left_coord, right_coord
        for coord in coords:
            x, y, w, h = coord

            eye_center = int(float(x) + (float(w) / float(2)))
            if int(face_img.shape[0] * 0.1) < eye_center < int(face_img.shape[1] * 0.4):
                left_eye = face_img[y : y + h, x : x + w]
                left_coord = (x, y, w, h)
            elif int(face_img.shape[0] * 0.5) < eye_center < int(face_img.shape[1] * 0.9):
                right_eye = face_img[y : y + h, x : x + w]
                right_coord = (x, y, w, h)
            else:
                pass  # false positive - nostrill

            if cut_brows:
                left_eye, left_coord = _cut_eyebrows(left_eye, left_coord)
                right_eye, right_coord = _cut_eyebrows(right_eye, right_coord)
                #return _cut_eyebrows(left_eye, left_coord), _cut_eyebrows(right_eye, right_coord)
        return left_eye, right_eye, left_coord, right_coord

# def detect_lefteye(
#         face_img: numpy.ndarray, cut_brows=False
#     ) -> (Optional[numpy.ndarray], Optional[numpy.ndarray]):

#         #(thresh, im_bw) = cv2.threshold(face_img, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         coords = lefteye_cascade.detectMultiScale(im_bw, 1.3, 5)

#         left_eye = left_coord = None

#         if coords is None or len(coords) == 0:
#             return left_eye, left_coord
#         for (x, y, w, h) in coords:
#             eye_center = int(float(x) + (float(w) / float(2)))
#             if int(face_img.shape[0] * 0.1) < eye_center < int(face_img.shape[1] * 0.4):
#                 left_eye = face_img[y : y + h, x : x + w]
#                 left_coord = (x, y, w, h)
#             else:
#                 pass

#             if cut_brows:
#                 return _cut_eyebrows(left_eye), _cut_eyebrows(right_eye)
#             return left_eye, left_coord,