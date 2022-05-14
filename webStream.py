from libs.YOLO import YoloDevice
import requests
import cv2

yolo = YoloDevice(
        config_file = './cfg_person/yolov4-tiny-person.cfg',
        weights_file = './weights/yolov4-tiny-person_70000.weights',
        data_file = './cfg_person/person.data',
        thresh = 0.3,
        output_dir = '',
        video_url = 'http://125.228.228.122:8080/video.mjpg',#廣場: http://125.228.228.122:8080/video.mjpg
        is_threading = False,
        vertex = [[0, 1080],[0, 764],[544, 225],[1014, 229],[1920, 809],[1920, 1080]],
        draw_polygon=False,
        alias="live",
        display_message = False,
        obj_trace = True,        
        save_img = False,
        save_video = False,        
        target_classes=["person"],
        auto_restart = True,
        skip_frame=2,
        count_people=True,
        draw_peopleCounting=True,
        draw_pose=True,
        social_distance=True,
        draw_socialDistanceArea=True,
        draw_square=True,
        draw_socialDistanceInfo=True,
        
        )

def webStream(frame, bboxImage, distanceImage):
    url = 'http://panettone.iottalk.tw:11030/yoloImages'
    
    #resize image for faster post requests
    frame = cv2.resize(frame, (1280, 720))
    bboxImage = cv2.resize(bboxImage, (1280, 720))
    distanceImage = cv2.resize(distanceImage, (1280, 720))
                       
    
    frame = cv2.imencode(".jpg", frame)[1]
    bboxImage = cv2.imencode(".png", bboxImage)[1]
    distanceImage = cv2.imencode(".png", distanceImage)[1]
    
    files = {'frame': ('frame.jpg', frame.tostring(), 'image/jpeg', {'Expires': '0'}),
             'bboxImage': ('bboxImage.png', bboxImage.tostring(), 'image/png', {'Expires': '0'}),
             'distanceImage': ('distanceImage.png', distanceImage.tostring(), 'image/png', {'Expires': '0'})}
    
    requests.post(url, files=files)
    
    
yolo.start()
yolo.set_web_listener(webStream)