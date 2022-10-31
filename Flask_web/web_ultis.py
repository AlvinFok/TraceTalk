import os
import shutil
import sys
import json
import cv2
import numpy as np


# import below is jim's YOLOtalk code
import sys
sys.path.append("..") 
from darknet import darknet
from libs.utils import *


def read_all_fences():
    oldFences = os.listdir(r'static/alias_pict')
    print(f"oldFences : {oldFences}")
    all_fences_names = []
    for name in oldFences :
        if "ipynb" in name:
            continue
        name = name[:-4]
        all_fences_names.append(name)
    
    return all_fences_names


def on_data(img_path, group, alias, results): 

    for det in results:
        class_name = det[0]
        confidence = det[1]
        center_x, center_y, width, height = det[2]
        left, top, right, bottom = darknet.bbox2points(det[2])
#             print(class_name, confidence, center_x, center_y)
#         if len(results) > 0:            
#             LineNotify.line_notify(class_name)   # LINENOTIFY.py  token
#             DAN.push('yPerson-I', str(class_name), center_x, center_y, img_path)


def transform_vertex(old_vertex):
    # transform vertex from [x1,y1,x2,y2,x3,....]  to [(x1,y1),(x2,y2),...]
    new_vertex = []
    a = old_vertex.split(",") 

    for i in range(0,len(a),2):
        x = (int(a[i]), int(a[i+1]))
        new_vertex.append(x)    

    return new_vertex


def gen_frames(yolo):

    filepath = "static/Json_Info/camera_info_" + str(yolo.alias) + ".json"
    with open(filepath, 'r', encoding='utf-8') as f:                    
        Jdata = json.load(f)

    key_list = list(Jdata["fence"].keys())
    vertex = {}
    detect_target = 0
    while True:
        frame = yolo.get_current_frame()
        
        for key in key_list :

            old_vertex = Jdata["fence"][key]["vertex"][1:-1]
            new_vertex = transform_vertex(old_vertex)
            vertex[key] = new_vertex

        frame = draw_polylines(frame, vertex)  # draw the polygon

        if  len(yolo.detect_target) != 0 :
            
            detect_target = 0
            mask = np.zeros((frame.shape), dtype = np.uint8)
            pts = []

            for singal_vertex in vertex.values():
                temp =[]
                for point in singal_vertex :
                    temp.append(point)
                pts.append(np.array(temp, dtype=np.int32))

            mask = cv2.fillPoly(mask, pts, (180,0,255))  # Filling the mask of polygon 
            frame =  0.5 * mask + frame  

        else:
            if detect_target < 5 :
                mask = np.zeros((frame.shape), dtype = np.uint8)
                pts = []

                for singal_vertex in vertex.values():
                    temp =[]
                    for point in singal_vertex :
                        temp.append(point)
                    pts.append(np.array(temp, dtype=np.int32))
                mask = cv2.fillPoly(mask, pts, (180,0,255))  # Filling the mask of polygon 
                frame =  0.5 * mask + frame    
                detect_target +=1
                
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def replot(alias, URL, Addtime):
    data  = {"alias":"",
        "viedo_url":"",
        "add_time":"",
        "fence": {}
        }
    print("REPLOT")
    IMGpath = "static/alias_pict/"+str(alias)+".jpg"
    data["alias"]=alias
    data["viedo_url"]=URL
    data["add_time"]=Addtime

    filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
    with open(filepath, 'r', encoding='utf-8') as f:                    
        Jdata = json.load(f)
    old_fence_data = Jdata['fence']

    fences = list(old_fence_data.keys())

    vertexs = []

    fig = cv2.imread(IMGpath)
    shape = fig.shape

    for fence in fences:
        old_vertex = old_fence_data[fence]['vertex']
        vertexs.append(old_vertex)

    return IMGpath, shape