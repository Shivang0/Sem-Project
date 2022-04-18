from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from detector import Detective
import os
import cv2

app = Flask(__name__, template_folder="templates", static_url_path='/static')


# for ip camera use -
# rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
# for local webcam use 
# cv2.VideoCapture(0)


def gen_frames():
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = webcam.read()  # read the camera frame
        if not success:
            break
        else:
            # Detective.detect(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def gen_ip_frames(ipadd):
    # rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
    # RTSP_URL = 'rtsp://192.168.0.101:8080/h264_ulaw.sdp'
    RTSP_URL = 'rtsp://'+ipadd+'/h264_ulaw.sdp'

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

    webcam = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    while True:
        success, frame = webcam.read()  # read the camera frame
        if not success:
            break
        else:
            # Detective.detect(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/localcam')
def localcam():
   # return render_template('localcam.html',gen_frames = gen_frames,Response = Response )
   return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/ipcam', methods=['POST'])
def ipcam():
    ipadd = request.form['ip']
    print('got ip')
    print(ipadd)
    return Response(gen_ip_frames(ipadd), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


@app.route('/test', methods=['GET'])
def test():
    return "hello world!"


@app.route('/submit', methods=['POST'])
def submit():
    image = request.args.get('image')

    print(type(image))
    return ""
