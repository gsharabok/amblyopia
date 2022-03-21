#!/usr/bin/env python
from cProfile import run
from importlib import import_module
import os

from ball.color_picker_func import extract_color
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import sched, time
import subprocess
import threading
import webbrowser
import time
from flask import Flask, render_template, Response, send_from_directory
from flask_socketio import SocketIO, emit

from runner import Runner
runner_calibration = Runner()
runner_training = Runner()
is_writer_init = False
continue_running = True

second_eye = True
send_eye_switch = False
# loop = sched.scheduler(time.time, time.sleep)

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera_opencv import Camera

app = Flask(__name__)
socketio = SocketIO(app)

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
    return render_template('local_implementation/calibration.html', sound="well_done.mp3")

@app.route('/training')
def training():
    """Training page."""
    return render_template('local_implementation/training.html')

def gen_calibration(camera):
    """Video streaming generator function."""
    
    global is_writer_init, runner_calibration, continue_running
    # runner_calibration.ball1_lower = (107, 43, 135)
    # runner_calibration.ball1_upper = (216, 215, 255)

    yield b'--frame\r\n'
    while continue_running:
        frame = camera.get_frame()

        nparr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if not is_writer_init:
            is_writer_init = True
            runner_calibration.init_writer(frame)

        runner_calibration.run_detection_calibration(frame)
        # emit('update value', "hello", broadcast=True)

        if runner_calibration.user_positioned and not runner_calibration.user_positioned_audio:
            print("Sent User Positioned Audio")
            socketio.emit('play', 'correct_position.mp3')
            runner_calibration.user_positioned_audio = True

        if runner_calibration.user_wiggling and not runner_calibration.user_wiggling_audio:
                print("Sent User Wiggling Audio")
                socketio.emit('play', 'stop_the_ball.mp3')
                runner_calibration.user_wiggling_audio = True

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

    runner_calibration.finish()


def switch_eye():
    global second_eye, send_eye_switch
    starttime = time.time()

    i = 0
    while i<20:
        # print("tick")
        time.sleep(6.0 - ((time.time() - starttime) % 6.0))
        send_eye_switch = True
        i+=1

def gen_training(camera):
    """Video streaming generator function."""
    
    global is_writer_init, runner_training, continue_running
    global send_eye_switch, second_eye
    # runner_training.ball1_lower = (164, 48, 83)
    # runner_training.ball1_upper = (182, 214, 255)
    # runner_training.ball2_lower = (40, 10, 207)
    # runner_training.ball2_upper = (81, 77, 255)

    yield b'--frame\r\n'
    while continue_running:
        frame = camera.get_frame()

        nparr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if not is_writer_init:
            is_writer_init = True
            runner_training.init_writer(frame)

        runner_training.run_detection_training(frame)

        if runner_training.user_positioned and not runner_training.user_positioned_audio:
            print("Sent User Second Ball Audio")
            socketio.emit('play', 'second_ball0.mp3')
            second_eye = False

            runner_training.user_positioned_audio = True

            th = threading.Thread(target=switch_eye)
            th.start()
            # loop.enter(60, 1, switch_eye, (loop,))
            # loop.run()

        if send_eye_switch:
            print("Sent Eye Switch Audio")
            if second_eye:
                socketio.emit('play', 'second_ball1.mp3')
                second_eye = False
            else:
                socketio.emit('play', 'first_ball1.mp3')
                second_eye = True
            send_eye_switch = False

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

    th.join()

    runner_training.finish()


@app.route('/video_feed_calibration')
def video_feed_calibration():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_calibration(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_training')
def video_feed_training():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_training(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finish_recording')
def finish_recording():
    global continue_running
    continue_running = False
    return '', 204

@app.route('/music/<path:filename>')
def download_file(filename):
    return send_from_directory('static/audio/', filename)

@app.route('/extract_color1')
def extract_color1():
    global runner_calibration, runner_training
    lower, upper = extract_color()
    runner_calibration.ball1_lower = lower
    runner_calibration.ball1_upper = upper

    runner_training.ball1_lower = lower
    runner_training.ball1_upper = upper

    return '', 204

@app.route('/extract_color2')
def extract_color2():
    global runner_training
    lower, upper = extract_color()

    runner_training.ball2_lower = lower
    runner_training.ball2_upper = upper

    return '', 204

@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)


def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    socketio.run(app, host='0.0.0.0')

    # app.run(host='0.0.0.0', threaded=True)
