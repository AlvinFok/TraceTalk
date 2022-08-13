from libs.YOLO_centroid import YoloDevice
import argparse


#yolov4 tiny
yolo1 = YoloDevice(
        config_file = './cfg_person/yolov4-tiny-person.cfg',
        weights_file = './weights/yolov4-tiny-person_final.weights',
        # config_file = './cfg_person/yolov4.cfg',
        # weights_file = './weights/yolov4.weights',
        
        data_file = './cfg_person/person.data',
        thresh = 0.5,
        output_dir = 'videoTest_centroid',
        video_url = './0325.mp4',
        is_threading = False,
        vertex = [[0, 1080],[0, 764],[544, 225],[1014, 229],[1920, 809],[1920, 1080]],
        draw_polygon=False,
        alias="Test",
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
        draw_socialDistanceInfo=True,
        testMode=True,
        repeat=False,
        gpu=1,
    )    

    
    
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True,  help='video file or folder')
args = parser.parse_args()

# yolo1.test("usedVideos")
# yolo1.test("videoClips")
yolo1.test(args.video)
