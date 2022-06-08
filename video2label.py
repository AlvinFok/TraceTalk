from libs.YOLO import YoloDevice
from darknet import darknet

#yolov4
yolo1 = YoloDevice(
          #config_file = './cfg_person/yolov4-tiny-person.cfg',
          #weights_file = './weights/yolov4-tiny-person_best.weights',
         config_file = './cfg_person/yolov4.cfg',
         weights_file = './weights/yolov4.weights',
        
         data_file = './cfg_person/person.data',
         thresh = 0.5,
         output_dir = 'videoTest',
         video_url = './0325.mp4',
         is_threading = False,
         vertex = [[0, 1080],[0, 764],[544, 225],[1014, 229],[1920, 809],[1920, 1080]],
         draw_polygon=False,
         alias="Test",
         display_message = True,
         obj_trace = True,        
         save_img = False,
         save_video = True,        
         target_classes=["person"],
         auto_restart = True,
         skip_frame=3,
         count_people=True,
         draw_peopleCounting=True,
         draw_pose=True,
         social_distance=True,
         draw_socialDistanceInfo=True,
         testMode=True,
         repeat=False,
        
     )    

    
# yolo1.video2Label("./usedVideos/", "./labeledData/")
yolo1.generateTrainTxt("./labeledData/")
