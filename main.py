from libs.YOLO import YoloDevice
from darknet import darknet
import libs.DAN as DAN
import LineNotify

import numpy as np
from shapely.geometry import Point, Polygon

if __name__ == '__main__': 
    
    ServerURL = 'https://edgecore.iottalk.tw'    
    Reg_addr = '555642434' #if None, Reg_addr = MAC address
    DAN.profile['dm_name']='Yolo_Device'
    DAN.profile['df_list']=['yPerson-I',]
    DAN.profile['d_name']= 'YOLO_Alvin'

    # DAN.device_registration_with_retry(ServerURL, Reg_addr)
    # DAN.deregister()  #if you want to deregister this device, uncomment this line
    # exit()            #if you want to deregister this device, uncomment this line   

    totalIn = 0
    lastCentroids = dict()
    squareArea = np.array([[7, 384], [0, 273], [185, 86], [320, 82], [401, 136], [381, 209], [509, 345], [541, 230], [640, 282], [640, 384] ], np.int32)#The polygon of the area you want to count people inout
    squareAreaPolygon = Polygon(squareArea)#crate a polygon of square for counting people
    
    # results:[(class, confidence, (center_x, center_y, width, height), id, [pose_centerX, pose_centerY]), (...)]
    def on_data(img_path, group, alias, results): 
        pass
            # left, top, right, bottom = darknet.bbox2points(det[2])
            # print(class_name, confidence, center_x, center_y)
        if len(results) > 10:
            msg = f"{len(results)}人進入廣場。"
            LineNotify.line_notify(msg)
            print(msg)
        #     # DAN.push('yPerson-I', str(class_name), center_x, center_y, img_path)
        #     print("hi")

    #廣場: http://125.228.228.122:8080/video.mjpg
    yolo1 = YoloDevice(
        config_file = './cfg_person/yolov4-tiny-person.cfg',
        data_file = './cfg_person/person.data',
        weights_file = './weights/yolov4-tiny-person_best.weights',
        thresh = 0.6,
        output_dir = '',
        video_url = './1.mp4',
        is_threading = False,
        vertex = [[0, 762],[537, 244],[1152, 286],[1920, 599],[1920, 1080],[0, 1080]],
        alias="1",
        display_message = False,
        obj_trace = True,        
        save_img = False,
        save_video = True,        
        target_classes=["person"],
        auto_restart = True,
        
        count_people=True,
        draw_peopleCounting=True,
        draw_pose=True,
        social_distance=True,
        draw_socialDistance=True,
        
     )    

    yolo1.set_listener(on_data)
    yolo1.start()  
