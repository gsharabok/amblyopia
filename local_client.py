#!/usr/bin/env python
from importlib import import_module
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
from flask import Flask, render_template, Response

from runner import Runner
runner = Runner()
is_writer_init = False
continue_running = True

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera_opencv import Camera

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    """Visual Training home page."""
    return render_template('local_implementation/index.html')

@app.route('/tutorial')
def tutorial():
    """Tutorial page."""
    return render_template('local_implementation/tutorial.html')

@app.route('/color_setup')
def color_setup():
    """Tutorial page."""
    return render_template('local_implementation/color_setup.html')

@app.route('/calibration')
def calibration():
    """Ball Distance Setup Page."""
    return render_template('local_implementation/calibration.html')

@app.route('/training')
def training():
    """Training page."""
    return render_template('local_implementation/training.html')

def gen(camera):
    """Video streaming generator function."""
    
    global is_writer_init, runner, continue_running

    yield b'--frame\r\n'
    while continue_running:
        frame = camera.get_frame()

        nparr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if not is_writer_init:
            is_writer_init = True
            runner.init_writer(frame)

        runner.run_detection(frame)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

    runner.finish()


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finish_recording')
def finish_recording():
    global continue_running
    continue_running = False
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
