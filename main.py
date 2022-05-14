# coding:utf-8

from libs.YOLO import YoloDevice
from darknet import darknet
import libs.DAN as DAN
import LineNotify

if __name__ == '__main__': 
    
    ServerURL = 'https://edgecore.iottalk.tw'    
    Reg_addr = '555642434' #if None, Reg_addr = MAC address
    
    DAN.profile['dm_name']='Yolo_Device'
    DAN.profile['df_list']=['yPerson-I',]
    DAN.profile['d_name']= 'YOLO_Alvin'

    # DAN.device_registration_with_retry(ServerURL, Reg_addr)
    # DAN.deregister()  #if you want to deregister this device, uncomment this line
    # exit()            #if you want to deregister this device, uncomment this line   
    
    # results:[(class, confidence, (center_x, center_y, width, height), id, [pose_centerX, pose_centerY]), (...)]
    def on_data(img_path, group, alias, results): 
        pass
            # left, top, right, bottom = darknet.bbox2points(det[2])
            # print(class_name, confidence, center_x, center_y)
        '''
        if len(results) > 10:
            msg = f"{len(results)}人進入廣場。"
            LineNotify.line_notify(msg)
            print(msg)
        '''
        #     # DAN.push('yPerson-I', str(class_name), center_x, center_y, img_path)
        #     print("hi")

    #廣場: http://125.228.228.122:8080/video.mjpg
    yolo1 = YoloDevice(
        config_file = './cfg_person/yolov4-tiny-person.cfg',
        weights_file = './weights/yolov4-tiny-person_70000.weights',
        # config_file = './cfg_person/yolov4.cfg',
        # weights_file = './weights/yolov4.weights',
        
        data_file = './cfg_person/person.data',
        thresh = 0.3,
        output_dir = '',
        video_url = './0325.mp4',
        is_threading = False,
        vertex = [[0, 1080],[0, 764],[544, 225],[1014, 229],[1920, 809],[1920, 1080]],
        draw_polygon=False,
        alias="0325",
        display_message = True,
        obj_trace = True,        
        save_img = False,
        save_video = True,        
        target_classes=["person"],
        auto_restart = True,
        skip_frame=None,
        count_people=True,
        draw_peopleCounting=True,
        draw_pose=True,
        social_distance=True,
        draw_socialDistanceInfo=True,
        
     )    

    yolo1.set_listener(on_data)
    yolo1.start()  
