from importlib import import_module
import os
import sys

# from gevent import monkey
# monkey.patch_all()

from ball.color_picker_func import extract_color, reset_colors
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
import threading
import webbrowser
import time
from flask import Flask, render_template, Response, send_from_directory, request
from flask_socketio import SocketIO, emit, send

from threading import Lock
import eventlet
from engineio.async_drivers import eventlet as evt
eventlet.monkey_patch()
# async_mode = "eventlet"


from runner import Runner
runner_calibration = Runner()
runner_training = Runner()
runner_training.is_training = True

is_writer_init = False
continue_running = True

second_eye = True
send_eye_switch = False
diverse_ball_sound = False

to_send = None
# loop = sched.scheduler(time.time, time.sleep)

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera_opencv import Camera

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
else:
    app = Flask(__name__)
# app = Flask(__name__)

socketio = SocketIO(app) # , async_mode=async_mode

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
    return render_template('local_implementation/calibration_test.html', sound="well_done.mp3") #, async_mode=socketio.async_mode

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

        frame = runner_calibration.run_detection_calibration(frame)
        # emit('update value', "hello", broadcast=True)

        if runner_calibration.user_positioned and not runner_calibration.user_positioned_audio:
            print("Sent User Positioned Audio")
            runner_calibration.user_positioned_audio = True
            global to_send
            to_send = 'correct_position.mp3'
            # socketio.send('play', 'correct_position.mp3') #, namespace="/audio"
            # emit('play', 'correct_position.mp3')
            # emit('play', 'correct_position.mp3',broadcast=True)

        if runner_calibration.user_wiggling and not runner_calibration.user_wiggling_audio:
                print("Sent User Wiggling Audio")
                runner_calibration.user_wiggling_audio = True
                socketio.emit('play', 'stop_the_ball.mp3')

        if runner_calibration.user_wiggling2 and not runner_calibration.user_wiggling2_audio:
                print("Check if user wiggled")
                runner_calibration.user_wiggling2_audio = True
                socketio.emit('play', 'check_wiggle.mp3')

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

    runner_calibration.finish()


def switch_eye():
    global second_eye, send_eye_switch, runner_training, diverse_ball_sound
    starttime = time.time()

    i = 0
    while i<6:
        # print("tick")
        time.sleep(6.0 - ((time.time() - starttime) % 6.0))

        if i != 0 and i % 5 == 0:
            diverse_ball_sound = True
        send_eye_switch = True

        i+=1
        runner_training.training_reps += 1

    runner_training.training_finished = True
    runner_training.training_session_time = time.time() - runner_training.training_start_time

def gen_training(camera):
    """Video streaming generator function."""
    
    global is_writer_init, runner_training, continue_running
    global send_eye_switch, second_eye, diverse_ball_sound
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

        frame = runner_training.run_detection_training(frame)

        if runner_training.user_positioned and not runner_training.user_positioned_audio:
            print("Sent User Second Ball Audio")
            socketio.emit('play', 'second_ball0.mp3')
            second_eye = False

            runner_training.user_positioned_audio = True
            runner_training.training_start_time = time.time()

            th = threading.Thread(target=switch_eye)
            th.start()
            # loop.enter(60, 1, switch_eye, (loop,))
            # loop.run()

        if send_eye_switch:
            print("Sent Eye Switch Audio")
            if second_eye:
                if diverse_ball_sound:
                    socketio.emit('play', 'second_ball2.mp3')
                    diverse_ball_sound = False
                else:
                    socketio.emit('play', 'second_ball1.mp3')
                second_eye = False
            else:
                if diverse_ball_sound:
                    socketio.emit('play', 'first_ball2.mp3')
                    diverse_ball_sound = False
                else:
                    socketio.emit('play', 'first_ball1.mp3')
                second_eye = True
            send_eye_switch = False

        if runner_training.training_finished:
            print("Training Finished")
            continue_running = False
            socketio.emit('play', 'session_complete.mp3')

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

    runner_training.finish()

    while not runner_training.training_feedback_ready:
        time.sleep(1)
    print("Sent")
    socketio.emit('play', 'combined/combined.mp3')

    th.join()


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
    global continue_running, runner_training
    continue_running = False

    if not runner_training.training_finished and runner_training.training_start_time != None:
        runner_training.training_session_time = time.time() - runner_training.training_start_time
    
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

    reset_colors()

    return '', 204


@app.route('/extract_color2')
def extract_color2():
    global runner_training
    lower, upper = extract_color()

    runner_training.ball2_lower = lower
    runner_training.ball2_upper = upper

    return '', 204

@app.route('/check_wiggle_result', methods=['POST'])
def check_wiggle_result():
    global runner_calibration

    jsdata = request.form['result']

    if jsdata == 'true':
        runner_calibration.user_wiggling = True

    socketio.emit('play', 'stop_the_ball.mp3')
    return '', 204


def background_thread():
    """Example of how to send server generated events to clients."""
    # count = 0
    while True:
        socketio.sleep(1)
        # count += 1
        global to_send
        if to_send is not None:
            socketio.emit('my_response', to_send)
        # else:
        #     socketio.emit('my_response',
        #               {'data': 'Server generated event', 'count': count})

# @socketio.event
# def my_ping():
#     print("pong")
#     global to_send
#     if to_send is not None:
#         print('sending: ', to_send)
#         emit(to_send)
#         to_send = None
#     else:
#         emit('my_pong')

# @socketio.event
# def connect():
#     emit('my_response', {'data': 'Connected', 'count': 0})

# @socketio.on('connect')
# def test_connect(auth):
#     emit('my response', {'data': 'Connected'})

# @socketio.on('my event')
# def handle_my_custom_event(json):
#     print('received json: ' + str(json))

# @socketio.on('message')
# def handle_message(data):
#     print('received message: ' + data)

# @socketio.on('message', namespace='/video')
# def on_connect1():
#     print("I'm connected to the /video namespace!")

# @socketio.on('message', namespace='/audio')
# def on_connect2():
#     print("I'm connected to the /audio namespace!")

thread = None
thread_lock = Lock()

@socketio.event
def connect():
    print("Connected")
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__.endswith('__main__'):
    threading.Timer(1, open_browser).start()
    socketio.run(app, host='0.0.0.0')

    # app.run(host='0.0.0.0', threaded=True)
