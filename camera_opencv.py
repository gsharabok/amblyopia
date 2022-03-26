import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0
    dim1, dim2, dim3 = (0,0,0)

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # encode as a jpeg image and return it
            Camera.dim1, Camera.dim2, Camera.dim3 = img.shape
            yield cv2.imencode('.jpg', img)[1].tobytes()

    def get_dimensions(self):
        return self.dim1, self.dim2, self.dim3
