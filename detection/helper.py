import cv2

# Not in use
class CV2Error(Exception):
    pass

def draw(source, keypoints, dest=None):
        try:
            if dest is None:
                dest = source
            return cv2.drawKeypoints(
                source,
                keypoints,
                dest,
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
        except cv2.error as e:
            raise CV2Error(str(e))