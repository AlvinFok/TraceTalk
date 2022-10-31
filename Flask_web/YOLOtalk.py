from flask import Flask, render_template,Response ,request ,jsonify,redirect
import time
import numpy as np
import threading
import cv2
import requests
import json
import os
import jinja2

from web_ultis import *
# import below is jim's YOLOtalk code
import sys
sys.path.append("..") 
from libs.YOLO import YoloDevice
from darknet import darknet
from libs.utils import *


app = Flask(__name__)



# Restart YoloDevice


def Restart_YoloDevice():
    oldJson = os.listdir(r"static/Json_Info")
    actived_yolo = {}
    for j in oldJson :
        if "ipynb" in j:
            continue

        path = os.path.join("static/Json_Info", j)
        with open(path, 'r', encoding='utf-8') as f:             
            Jdata = json.load(f)

        alias = Jdata["alias"]
        URL   = Jdata["viedo_url"]

        if Jdata["fence"]  == {}:
            vertex = None

        else:
            key_list = list(Jdata["fence"].keys())
            vertex = {}

            for key in key_list :
                old_vertex = Jdata["fence"][key]["vertex"][1:-1]
                # vertex
                new_vertex = transform_vertex(old_vertex)
                vertex[key] = new_vertex 
                # sensitivity
                old_sensitivity = float(Jdata["fence"][key]["Sensitivity"])

                yolo1 = YoloDevice(
                                config_file = '../darknet/cfg/yolov4-tiny.cfg',
                                data_file = '../darknet/cfg/coco.data',
                                weights_file = '../weights/yolov4-tiny.weights',
                                thresh = old_sensitivity,                 
                                output_dir = './static/record/',              
                                video_url = URL,              
                                is_threading = True,          
                                vertex = new_vertex,                 
                                alias = alias,                
                                display_message = True,
                                obj_trace = True,        
                                save_img = False,
                                save_video = False,           
                                target_classes = ["person"],
                                auto_restart = False,
                                )    
                yolo1.set_listener(on_data)
                yolo1.start()
                actived_yolo[alias] = yolo1
        # ======== FOR YOLO ========
    return actived_yolo
actived_yolo = Restart_YoloDevice()

@app.route('/',methods=[ 'GET','POST'])
def home():
    
    all_fences_names = read_all_fences()

    if request.method == 'POST':

        alias  = request.form.get('area')
        URL        = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]

        
        if URL =="REPLOT":
            IMGpath, shape =  replot(alias, URL, Addtime)
            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape)

        else:    

            fig = cv2.VideoCapture(str(URL))
            stat, I = fig.read()

            # Temporary information
            data  = {"alias":"",
                    "viedo_url":"",
                    "add_time":"",
                    "fence": {}
                    }

            data["alias"]     = alias
            data["viedo_url"] = URL
            data["add_time"]  = Addtime
            

            filepath = "static/Json_Info/camera_info_" + data["alias"] + ".json"
            with open(filepath, 'w', encoding='utf-8') as file:             
                json.dump(data, file,separators=(',\n', ':'),indent = 4)
        
        
            IMGpath = "static/alias_pict/"+str(alias)+".jpg"
            all_fences_names.append(str(alias))
            cv2.imwrite(IMGpath,I)

    # ======== FOR YOLO ========
            yolo1 = YoloDevice(
                    config_file = '../darknet/cfg/yolov4-tiny.cfg',
                    data_file = '../darknet/cfg/coco.data',
                    weights_file = '../weights/yolov4-tiny.weights',
                    thresh = 0.3,                 # need modify (ok)
                    output_dir = './static/record/',              
                    video_url = URL,              # need modify (ok) 
                    is_threading = True,          # rtsp  ->true  video->false (ok)
                    vertex = None,                # need modify (ok)    
                    alias = alias,                  # need modify (ok)
                    display_message = True,
                    obj_trace = True,        
                    save_img = False,
                    save_video = False,           # modify to False
                    target_classes = ["person"],
                    auto_restart = False,
                    )    
            yolo1.set_listener(on_data)
            yolo1.start()
            actived_yolo[alias] = yolo1
    # ======== FOR YOLO ========
        
            return render_template('plotarea.html', data = IMGpath, name=str(alias), shape = I.shape)
     
    return render_template('home.html', navs = all_fences_names)
    


@app.route('/plotarea',methods=[ 'GET','POST'])
def plotarea():

    all_fences_names = read_all_fences()

    if request.method == 'POST':
        
        print("plotarea enter & POST")
        
        alias = request.form['alias']
        FenceName = request.form['FenceName']   # plot name
        vertex = request.form['vertex']         # plot point

        IMGpath, shape =  replot(alias, URL, Addtime)

        Fence_info={'vertex':vertex,
                    'Group':'-',                
                    'Alarm_Level':'General',    # General, High
                    'Note':' - ',           
                    'Sensitivity':'0.5',
                    'Schedule':{
                                    '1':{'Start_time':'--:--','End_time':'--:--'},
                               }
                   }
     
        filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
        print("Json File Path : " , filepath)
        if os.path.isfile(filepath):                                        # If file is exist
            with open(filepath, 'r', encoding='utf-8') as f:                
                Jdata = json.load(f)

            if vertex == "DELETE" :
                Jdata["fence"].pop(FenceName)                              
            else:
                Jdata["fence"][FenceName]=Fence_info                        
                # ======== FOR YOLO ========
                old_vertex = vertex[1:-1]
                new_vertex = transform_vertex(old_vertex)
                data = { FenceName : new_vertex }
                actived_yolo[alias].vertex = data
                print(f"alias:{alias}")
                print(actived_yolo[alias].vertex)
                # ======== FOR YOLO ======== 
            with open(filepath, 'w', encoding='utf-8') as file:             
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)
            
        else:                                                               
            print("\nFile doesn't exist !! \n")
                                  # 
            data["fence"][FenceName]=Fence_info                             
            with open(filepath, 'w', encoding='utf-8') as f:                
                json.dump(data, f,separators=(',\n', ':'),indent = 4)
            data['fence']={}                                                


    return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape)



@app.route('/management',methods=[ 'GET','POST'])
def management():

    all_fences_names = read_all_fences()
    # nav Replot 
    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
        
        IMGpath, shape =  replot(alias, URL, Addtime)
        
        if URL =="REPLOT":
            
            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape)

        if (URL == "Edit"):
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            Group       = request.form.get('Group')
            Alarm_Level = request.form.get('Alarm_Level')
            Note        = request.form.get('Note')
            Sensitivity = request.form.get('Sensitivity') 

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    
                Jdata = json.load(f)

            Jdata['fence'][FenceName]['Group']       = Group
            Jdata['fence'][FenceName]['Alarm_Level'] = Alarm_Level
            Jdata['fence'][FenceName]['Note']        = Note

            old_Sensitivity = Jdata['fence'][FenceName]['Sensitivity']
            Jdata['fence'][FenceName]['Sensitivity'] = Sensitivity

            if old_Sensitivity != Sensitivity :
                print(f"actived_yolo[alias].thresh : {actived_yolo[alias].thresh}")
                actived_yolo[alias].thresh = float(Sensitivity)

            with open(filepath, 'w', encoding='utf-8') as file:                 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)

            return render_template('management.html', navs=all_fences_names)

        if (URL == "Delete"):    
            print("Enter Delete")
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            print("alias : ", alias)

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    
                Jdata = json.load(f)

            Jdata['fence'].pop(FenceName)
            print("\n Jdata :", Jdata, "\n")

            with open(filepath, 'w', encoding='utf-8') as file:                 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)

        
    # management items 
    items=[]
    filepath_list = os.listdir("./static/Json_Info/")
    
    for path in filepath_list:
        if ".ipynb" in path :
            continue
        else:
            path = "./static/Json_Info/" + path
            with open(path, 'r', encoding='utf-8') as f:
                file = json.load(f)
                items.append(file)
    return render_template('management.html', navs=all_fences_names, items=items)


@app.route('/streaming',methods=[ 'GET','POST'])
def streaming():

    all_fences_names = read_all_fences()

    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
        
        if URL =="REPLOT":

            IMGpath, shape =  replot(alias, URL, Addtime)
            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape)

    alias_list = os.listdir(r'static/alias_pict')
    alias_list.remove('.ipynb_checkpoints')

    return render_template('streaming.html', navs=all_fences_names, alias_list=alias_list, length=len(alias_list))


@app.route('/schedule',methods=[ 'GET','POST'])
def schedule():
    
    all_fences_names = read_all_fences()

    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
            
        if URL =="REPLOT":

            IMGpath, shape =  replot(alias, URL, Addtime)
            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape)

        
        if (URL == "Edit_time"):
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            # Group       = request.form.get('Group')
            Order       = str(request.form.get('Order'))
            Start_time  = request.form.get('start_time')
            End_time    = request.form.get('end_time') 

            new_schedule = {'Start_time':Start_time,'End_time':End_time}

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                     
                Jdata = json.load(f)

            Schedule_keys = list(Jdata['fence'][FenceName]['Schedule'].keys())

            if Order in Schedule_keys:
                # Jdata['fence'][FenceName]['Group'] = Group
                Jdata['fence'][FenceName]['Schedule'][Order]['Start_time']  = Start_time
                Jdata['fence'][FenceName]['Schedule'][Order]['End_time']    = End_time

            else:
                # data['fence'][FenceName]['Group'] = Group
                Jdata['fence'][FenceName]['Schedule'][Order] = new_schedule

            with open(filepath, 'w', encoding='utf-8') as file:                  
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)


        if (URL == "Delete_Schedule"):
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            Order       = request.form.get('Order')
            
            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    
                Jdata = json.load(f)

            Jdata['fence'][FenceName]['Schedule'][Order]['Start_time']  = "--:--"
            Jdata['fence'][FenceName]['Schedule'][Order]['End_time']    = "--:--"

            with open(filepath, 'w', encoding='utf-8') as file:                 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)
    items=[]
    filepath_list = os.listdir("./static/Json_Info/")
    
    for path in filepath_list:
        if ".ipynb" in path :
            continue
        else:
            path = "./static/Json_Info/" + path
            with open(path, 'r', encoding='utf-8') as f:
                file = json.load(f)
                items.append(file)
  
    return render_template('schedule.html', navs=all_fences_names, items=items)


@app.route('/video/<order>',methods=[ 'GET','POST'])
def video_feed(order):
    alias_list = os.listdir(r'static/alias_pict')
    alias_list.remove('.ipynb_checkpoints')

    if len(actived_yolo) > int(order) : 
        print(f"actived_yolo len = {len(actived_yolo)} , order = {order}")
        alias = alias_list[int(order)].replace('.jpg','')
        return Response(gen_frames(actived_yolo[alias]),mimetype='multipart/x-mixed-replace; boundary=frame' )
    else:
        return 'Error'


@app.route('/test',methods=[ 'GET','POST'])
def test():
    
    return render_template('base.html', )

@app.route('/base2',methods=[ 'GET','POST'])
def base2():
    
    return render_template('test_base.html', )  
if __name__ == '__main__':
    app.run(debug = True, host="0.0.0.0",port="10328")
