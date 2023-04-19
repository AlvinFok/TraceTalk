from libs.YOLO_SSIM import YoloDevice
from libs.utils import *
from darknet import darknet
import libs.DAN as DAN
import LineNotify
import json


if __name__ == '__main__': 
    ServerURL = 'https://6.iottalk.tw'    
    Reg_addr = '567890' #if None, Reg_addr = MAC address
    DAN.profile['dm_name']='Yolo'
    DAN.profile['df_list']=['yperson(json)-I', 'yperson(json)-O']
    DAN.profile['d_name']= 'Yolo(56789)'

    DAN.device_registration_with_retry(ServerURL, Reg_addr)
    
    # DAN.deregister()  #if you want to deregister this device, uncomment this line
    # exit()            #if you want to deregister this device, uncomment this line   


    def on_data(img_path, group, alias, results, frameID): 
        """
        When target objects are detected, this function will be called.
        
        Args:
            img_path: 
                The path of the stored frame. (string)
            group:
                The group of this YoloDevice. (string)
            alias:
                The alias of this YoloDevice. (string)
        
            results:
                The detection results. (list)  
                results element in each list: (class, confidence, (center_x, center_y, width, height), object_ID)
        """
        detections = []
        
        for det in results:
            class_name = det[0]
            confidence = det[1]
            center_x, center_y, width, height = det[2]
            left, top, right, bottom = darknet.bbox2points(det[2])
            # print(type(frameID), type(class_name), type(confidence), type(center_x), type(center_y), type(img_path))
            detections.append([class_name, int(confidence), [float(center_x), float(center_y), float(width), float(height)]])
        
        data = {
            "detections":detections,
            "frameID":frameID,
            "imgPath":img_path,
        }
        data = json.dumps(data)
        # print(DAN.push('yperson(json)-I', data))
            
        # if len(results) > 0:
        #     LineNotify.line_notify(class_name)
            # DAN.push('SFDB-I', str(class_name), float(center_x), float(center_y), str(img_path))
      
    
    args = parse_args() # Parse args from command linec
    
    # Create the YoloDevice object
    yolo1 = YoloDevice(
        config_file = './cfg_person/yolov4-tiny.cfg',
        data_file = './cfg_person/coco.data',
        weights_file = './weights/yolov4-tiny.weights',
        thresh = args.thresh,
        output_dir = '',
        video_url = args.video_url,
        is_threading = args.is_threading,
        vertex = None,
        alias = args.alias,       
        save_img = True,
        save_video = True,     
        using_SSIM = args.using_SSIM,
        target_classes=["person"],
        obj_trace=True,#enable tracking
        tracker_mode=5,
        track_buffer=30,#track buffer
        enable_people_counting=True,#enable people counting
        count_in_vertex=[[0, 1100],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1100, 590],[1425, 1007],[1525, 985],[1574, 814],[1930, 1100] ],#count in vertex
        count_out_vertex=[[0, 1080],[0, 0],[877, 0],[1019, 257],[1007, 360],[1177, 501],[1165, 595],[1512, 962],[1609, 578], [1980, 728], [1980, 1080]],#count out vertex
     )    

    # yolo1.set_listener(on_data) # Set callback function
    yolo1.start() # Start the program

    
