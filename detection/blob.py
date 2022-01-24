import cv2
import models

# Not in use
def blob_track(img, threshold, prev_area, eyeX, eyeY):
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=4)
        img = cv2.medianBlur(img, 5)
        keypoints = models.blob_detector.detect(img)
        # print([i.pt for i in keypoints])
        # (x,y) = point.pt
        for point in keypoints:
            (x,y) = point.pt
            point.pt = (x+eyeX,y+eyeY)
        # cv2.KeyPoint(
        #         keypoint.pt[0],
        #         keypoint.pt[1],
        #         keypoint.size,
        #         keypoint.angle,
        #         keypoint.response,
        #         keypoint.octave,
        #         keypoint.class_id,
        #     )

        if keypoints and len(keypoints) > 1:
            tmp = 1000
            for keypoint in keypoints:  # filter out odd blobs
                if abs(keypoint.size - prev_area) < tmp:
                    ans = keypoint
                    tmp = abs(keypoint.size - prev_area)

            keypoints = (ans,)
        return keypoints