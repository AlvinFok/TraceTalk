# -*- coding: utf-8 -*-
from glob import glob
from itertools import filterfalse
from charset_normalizer import detect
from cv2 import VideoCapture
from flask import Flask, render_template, Response, request
import cv2
import random
import numpy as np

from libs.YOLO import YoloDevice
import libs.DAN as DAN
import LineNotify

app = Flask(__name__)
#demo video
demo = cv2.VideoCapture("./demo.mp4")


ServerURL = 'https://edgecore.iottalk.tw'    
Reg_addr = '555642434' #if None, Reg_addr = MAC address

DAN.profile['dm_name']='Yolo_Device'
DAN.profile['df_list']=['yPerson-I',]
DAN.profile['d_name']= 'YOLO_Alvin'

# DAN.device_registration_with_retry(ServerURL, Reg_addr)
# DAN.deregister()  #if you want to deregister this device, uncomment this line
# exit()            #if you want to deregister this device, uncomment this line   
    

# results:[(class, confidence, (center_x, center_y, width, height), id, [pose_centerX, pose_centerY]), (...)]


def on_data(image, group, alias, results): 
    
    '''
    if len(results) > 10:
        msg = f"{len(results)}人進入廣場。"
        LineNotify.line_notify(msg)
        print(msg)
    '''
    
    
    
    
def getReturnFrame(image, format):
    if image is None:
        
        image = np.zeros((1080,1920,4))
        
    # image = cv2.resize(image, (1280, 720))
    
    ret, buffer = cv2.imencode(format, image)
    frame = buffer.tobytes()
    return frame
    
    
frame = None
bboxImage = None
distanceImage = None

@app.route('/yoloImages', methods=['POST'])
def yoloImages():
    if request.method == 'POST':
        
        global frame, bboxImage, distanceImage
        frame = request.files['frame'].read()
        bboxImage = request.files['bboxImage'].read()
        distanceImage = request.files['distanceImage'].read()

        return {'status':200}
    

#get demo frame
        
@app.route('/demoVideo')
def demoVideo():
    def gen_demoFrames(): 
        global demo
        while True: 
            ret, frame = demo.read()
            if not ret:#demo video ended
                demo = VideoCapture("./demo.mp4")#read video again
                ret, frame = demo.read()
                
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.imencode('.jpg', frame)[1]
            frame = frame.tobytes()
            yield  (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_demoFrames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

'''
#get demo bbox image
@app.route('/demoBbox')
def demoBbox():
    def gen_demoBbox():
        while True: 
            global bboxImage
            
            yield  (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + bboxImage + b'\r\n')
    
    return Response(gen_demoBbox(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
#get demo social distance image
@app.route('/demoSocialDistance')
def demoSocialDistance():
    def gen_demoSocialDistance():
        while True: 
            global distanceImage
            yield  (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + distanceImage + b'\r\n')
    
    return Response(gen_demoSocialDistance(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    



'''
@app.route('/liveVideo')
def liveVideo():
    def gen_liveFrames(): 
        while True: 
            global frame
            yield  (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen_liveFrames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


#get live bbox image
@app.route('/liveBbox')
def liveBbox():
    def gen_liveBbox():
        while True: 
            
            global bboxImage
            yield  (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + bboxImage + b'\r\n')
    
    return Response(gen_liveBbox(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
#get live social distance image
@app.route('/liveSocialDistance')
def liveSocialDistance():
    def gen_liveSocialDistance():
        while True: 
            
            global distanceImage
            yield  (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + distanceImage + b'\r\n')
    
    return Response(gen_liveSocialDistance(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    


@app.route("/videoControl", methods=['POST'])
def liveVideoControl():
    if request.method == 'POST':
        name = request.form.get("name")
        
        if name == 'live_bbox':
            yoloLive.draw_bbox = not yoloLive.draw_bbox
        elif name == 'demo_bbox':
            yoloDemo.draw_bbox = not yoloDemo.draw_bbox
            
        elif name == 'live_distance':
            yoloLive.draw_socialDistanceInfo = not yoloLive.draw_socialDistanceInfo
        elif name == 'demo_distance':
            yoloDemo.draw_socialDistanceInfo = not yoloDemo.draw_socialDistanceInfo
            
            
    return "success"



@app.route("/", methods=['GET', 'POST'])
def index():
    
    return render_template('index.html')


    

if __name__ == '__main__':
    
    
    # yoloLive.start()
    
    app.run('0.0.0.0', threaded=True, port=11030, debug=True)