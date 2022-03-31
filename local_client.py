from importlib import import_module
import os
import sys

# from gevent import monkey
# monkey.patch_all()


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

from ball.color_picker_func import extract_color, reset_colors
# from ball.color_setup_new import ColorSetup
from ball.color_setup_new_func import run_color, reset_color
import models

from runner import Runner
runner_calibration = Runner()
runner_training = Runner()
runner_training.is_training = True

is_writer_init = False
continue_running = True
# server_training_count = 0
# server_calibration_count = 0

second_eye = True
send_eye_switch = False
diverse_ball_sound = False

to_send = None
experimental_color = True
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


communication_thread = None
communication_thread_lock = Lock()
extract_color1_thread = None
extract_color1_thread_lock = Lock()
extract_color2_thread = None
extract_color2_thread_lock = Lock()

lower_range = np.array([0,0,0])
higher_range = np.array([255,255,255])

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
    global to_send

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

        print(runner_calibration.ball1_lower, runner_calibration.ball1_upper)

        frame = runner_calibration.run_detection_calibration(frame)
        # emit('update value', "hello", broadcast=True)

        if runner_calibration.user_positioned and not runner_calibration.user_positioned_audio:
            print("Sent User Positioned Audio")
            runner_calibration.user_positioned_audio = True
            to_send = 'correct_position.mp3'
            # socketio.send('play', 'correct_position.mp3') #, namespace="/audio"
            # emit('play', 'correct_position.mp3')
            # emit('play', 'correct_position.mp3',broadcast=True)

        if runner_calibration.user_wiggling and not runner_calibration.user_wiggling_audio:
                print("Sent User Wiggling Audio")
                runner_calibration.user_wiggling_audio = True
                # socketio.emit('play', 'stop_the_ball.mp3')
                to_send = 'stop_the_ball.mp3'

        if runner_calibration.user_wiggling2 and not runner_calibration.user_wiggling2_audio:
                print("Check if user wiggled")
                runner_calibration.user_wiggling2_audio = True
                # socketio.emit('play', 'check_wiggle.mp3')
                to_send = 'check_wiggle.mp3'

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

    continue_running = True
    runner_calibration.finish()


def switch_eye():
    global second_eye, send_eye_switch, runner_training, diverse_ball_sound
    starttime = time.time()

    i = 0
    while i<6:
        # print("tick")
        # eventlet.sleep(5)
        eventlet.sleep(6.0 - ((time.time() - starttime) % 6.0))

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
    global to_send
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
            # socketio.emit('play', 'second_ball0.mp3')
            to_send = 'second_ball0.mp3'
            second_eye = False

            runner_training.user_positioned_audio = True
            runner_training.training_start_time = time.time()

            th = threading.Thread(target=switch_eye)
            th.start()
            # loop.enter(60, 1, switch_eye, (loop,))
            # loop.run()

        if send_eye_switch:
            print("Sent Eye Switch Audio")
            send_eye_switch = False
            if second_eye:
                if diverse_ball_sound:
                    # socketio.emit('play', 'second_ball2.mp3')
                    to_send = 'second_ball2.mp3'
                    diverse_ball_sound = False
                else:
                    # socketio.emit('play', 'second_ball1.mp3')
                    to_send = 'second_ball1.mp3'
                second_eye = False
            else:
                if diverse_ball_sound:
                    # socketio.emit('play', 'first_ball2.mp3')
                    to_send = 'first_ball2.mp3'
                    diverse_ball_sound = False
                else:
                    # socketio.emit('play', 'first_ball1.mp3')
                    to_send = 'first_ball1.mp3'
                second_eye = True

        if runner_training.training_finished:
            print("Training Finished")
            continue_running = False
            # socketio.emit('play', 'session_complete.mp3')
            to_send = 'session_complete.mp3'

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

    continue_running = True
    runner_training.finish()

    while not runner_training.training_feedback_ready:
        eventlet.sleep(1)
    print("Sent")
    # socketio.emit('play', 'combined/combined.mp3')
    to_send = 'combined/combined.mp3'

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

@app.route('/prepare_color_data', methods=['POST'])
def prepare_color_data():
    global runner_calibration, experimental_color
    print("Preparing color. Experimental: ", experimental_color)

    if not experimental_color:
        return '', 204        

    filename = os.path.join(models.EXE_LOCATION,'data','results','color','red.txt')
    if os.path.exists(filename):
        f = open(filename, "r")
        content = f.read()
        content = content.splitlines( )
        f.close()
        l = content[0].strip("[]").split()
        u = content[1].strip("[]").split()
        low = np.array([int(l[0]),int(l[1]),int(l[2])])
        up = np.array([int(u[0]),int(u[1]),int(u[2])])

        runner_calibration.ball1_lower = low
        runner_calibration.ball1_upper = up
        runner_training.ball1_lower = low
        runner_training.ball1_upper = up

    filename = os.path.join(models.EXE_LOCATION,'data','results','color','green.txt')
    if os.path.exists(filename):
        f = open(filename, "r")
        content = f.read()
        content = content.splitlines( )
        f.close()
        l = content[0].strip("[]").split()
        u = content[1].strip("[]").split()
        low = np.array([int(l[0]),int(l[1]),int(l[2])])
        up = np.array([int(u[0]),int(u[1]),int(u[2])])

        runner_training.ball2_lower = low
        runner_training.ball2_upper = up

    return '', 204


@app.route('/music/<path:filename>')
def download_file(filename):
    return send_from_directory('static/audio/', filename)

# def func():
#     lower, upper, done = run_color()
#     return [lower,upper]

# def callback(gt, *args, **kwargs): 
#     """ this function is called when results are available """ 
#     result = gt.wait() 
#     print("[cb] %s" % result)

#     lower, upper = result

#     global runner_calibration, runner_training
#     runner_calibration.ball1_lower = lower
#     runner_calibration.ball1_upper = upper

#     runner_training.ball1_lower = lower
#     runner_training.ball1_upper = upper

@app.route('/extract_color1')
def extract_color1():
    global experimental_color
    experimental_color = False

    global runner_calibration, runner_training
    lower, upper = extract_color()

    runner_calibration.ball1_lower = lower
    runner_calibration.ball1_upper = upper

    runner_training.ball1_lower = lower
    runner_training.ball1_upper = upper

    reset_colors()

    return '', 204

@app.route('/extract_color1_ex')
def extract_color1_ex():
    # global runner_calibration, runner_training
    # lower, upper = extract_color()

    # greenth = eventlet.spawn(func) 
    # greenth.link(callback)

    # lower, upper, done = run_color()

    # print(lower, upper)
    # global lower_range, higher_range
    # lower_range = lower
    # higher_range = upper


    # exec(open('ball/countdown.py').read())

    global extract_color1_thread
    with extract_color1_thread_lock:
        if extract_color1_thread is None:
            # thread = socketio.start_background_task(background_thread)
            extract_color1_thread = socketio.start_background_task(back_temp1)
    # # lower, upper = color_setup.run()

    # runner_calibration.ball1_lower = lower
    # runner_calibration.ball1_upper = upper

    # runner_training.ball1_lower = lower
    # runner_training.ball1_upper = upper

    # reset_color()
    # reset_colors()

    return '', 204


@app.route('/extract_color2')
def extract_color2():
    global experimental_color
    experimental_color = False

    global runner_training
    lower, upper = extract_color()

    runner_training.ball2_lower = lower
    runner_training.ball2_upper = upper

    return '', 204

@app.route('/extract_color2_ex')
def extract_color2_ex():
    global extract_color2_thread
    with extract_color2_thread_lock:
        if extract_color2_thread is None:
            # thread = socketio.start_background_task(background_thread)
            extract_color2_thread = socketio.start_background_task(back_temp2)

    return '', 204

@app.route('/check_wiggle_result', methods=['POST'])
def check_wiggle_result():
    global runner_calibration

    jsdata = request.form['result']

    if jsdata == 'true':
        runner_calibration.user_wiggling = True

    # socketio.emit('play', 'stop_the_ball.mp3')
    global to_send
    to_send = 'stop_the_ball.mp3'
    return '', 204


def background_thread():
    """Example of how to send server generated events to clients."""
    # count = 0
    while True:
        socketio.sleep(0.5)
        # count += 1
        global to_send
        if to_send is not None:
            socketio.emit('my_response', to_send)
            to_send = None
        # else:
        #     socketio.emit('my_response',
        #               {'data': 'Server generated event', 'count': count})


def back_temp1():
    """Background thread for extracting first color"""
    # count = 0
    # while True:
        # socketio.sleep(0.5) 
    # lower, upper = run_color()
    lower_green, higher_green, lower_red, higher_red = run_color()

    path = 'data/results/color'
    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.join(models.EXE_LOCATION,'data','results','color','red.txt')
    f = open(filename, "w+")
    f.write(str(lower_red))
    f.write('\n')
    f.write(str(higher_red))
    f.close()

    filename = os.path.join(models.EXE_LOCATION,'data','results','color','green.txt')
    f = open(filename, "w+")
    f.write(str(lower_green))
    f.write('\n')
    f.write(str(higher_green))
    f.close()

    reset_color()

    global extract_color1_thread
    extract_color1_thread = None
        # np.savetxt(filename, np.array([lower, upper]), delimiter =", ")
        # for num in lower:
        #     f.write("%s\n" % num)

        # global runner_calibration, runner_training
        # runner_calibration.ball1_lower = lower
        # runner_calibration.ball1_upper = upper

        # runner_training.ball1_lower = lower
        # runner_training.ball1_upper = upper

def back_temp2():
    """Background thread for extracting second color"""
    # count = 0
    # while True:
        # socketio.sleep(0.5) 
    lower, upper = run_color()
    path = 'data/results/color'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(models.EXE_LOCATION,'data','results','color','color2.txt')
    f = open(filename, "w+")
    f.write(str(lower))
    f.write('\n')
    f.write(str(upper))
    f.close()

    reset_color()


# def run_color_setup1():
#     color_setup = ColorSetup()

#     lower, upper = color_setup.run()

#     print(lower, upper)

#     global runner_calibration, runner_training

#     runner_calibration.ball1_lower = lower
#     runner_calibration.ball1_upper = upper

#     runner_training.ball1_lower = lower
#     runner_training.ball1_upper = upper

# def run_color_setup2():
#     color_setup = ColorSetup()

#     lower, upper = color_setup.run()

#     # global runner_training

#     # runner_training.ball2_lower = lower
#     # runner_training.ball2_upper = upper


@socketio.event
def connect():
    print("Connected")
    global communication_thread
    with communication_thread_lock:
        if communication_thread is None:
            communication_thread = socketio.start_background_task(background_thread)
    # emit('my_response', {'data': 'Connected', 'count': 0})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__.endswith('__main__'):
    threading.Timer(1, open_browser).start()
    socketio.run(app, host='0.0.0.0')

    # app.run(host='0.0.0.0', threaded=True)
