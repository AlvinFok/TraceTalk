import cv2
from matplotlib.pyplot import draw
import numpy as np
import datetime
import time
import os
import shutil
import threading
import  random
import sys
sys.path.append("/home/alvin/TraceTalk/ByteTrack")

from tqdm import tqdm
import pathlib
import json
import torch
from loguru import logger
#people counting
from shapely.geometry import Point, Polygon

#BYTE tracker
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess
from ByteTrack.yolox.tracking_utils.timer import Timer
from ByteTrack.yolox.exp import get_exp


# SmartFence  lib
from libs.utils import *

from darknet import darknet




class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def loadModel(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # output_dir = os.path.join(exp.output_dir, args.experiment_name)
    # os.makedirs(output_dir, exist_ok=True)

    # if args.save_result:
    #     vis_folder = os.path.join(output_dir, "track_vis")
    #     os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            # ckpt_file = os.path.join(output_dir, "best_ckpt.pth.tar")
            pass
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(output_dir, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    # current_time = time.localtime()
    # if args.demo == "image":
    #     image_demo(predictor, vis_folder, current_time, args)
    # elif args.demo == "video" or args.demo == "webcam":
    #     imageflow_demo(predictor, vis_folder, current_time, args)
    return predictor

def loadPredictor(args):
    exp = get_exp(args.exp_file, args.name)
    predictor = loadModel(exp, args)
    return predictor

class YoloDevice:
    def __init__(self, video_url="", output_dir="", run=True, auto_restart=False, repeat=False, obj_trace = False,
                 display_message=True, data_file="", config_file="", weights_file="", 
                 names_file="", thresh=0.5, vertex=None, target_classes=None, draw_bbox=True, draw_polygon=True, draw_square=True,
                 draw_socialDistanceArea=False, draw_socialDistanceInfo=False,  social_distance=False, draw_pose=False, count_people=False, draw_peopleCounting=False,
                 alias="", group="", place="", cam_info="", warning_level=None, is_threading=True, skip_frame=None,
                 schedule=[], save_img=True, save_original_img=False, save_video=False, save_video_original=False, testMode=False, gpu=0, args=None,
                 ):
        

        
        
        self.video_url = video_url
        self.output_dir = output_dir
        self.run = run
        self.auto_restart = auto_restart
        self.repeat = repeat#auto repeat prediction
        self.display_message = display_message
        self.data_file = data_file
        self.config_file = config_file
        self.weights_file = weights_file
        self.names_file = names_file
        self.thresh = thresh
        self.skip_frame = skip_frame
        self.vertex = vertex # set None to detect the full image
        self.target_classes = target_classes # set None to detect all target
        self.draw_bbox = draw_bbox
        self.draw_polygon = draw_polygon
        self.alias = alias
        self.group = group
        self.place = place
        self.obj_trace = obj_trace
        self.cam_info = cam_info
        self.warning_level = warning_level
        self.is_threading = is_threading # set False if the input is video file
        self.schedule = schedule
        self.save_img = save_img
        self.save_img_original = save_original_img # set True to restore the original image
        self.save_video = save_video # set True to save the result to video
        self.save_video_original = save_video_original # set True to save the video stream
        self.testMode = testMode#set True for test mode
        self.IDInfo = dict()
        self.testWithTxt = False
        #load yolox model and predictor at start()
        self.predictor = None
        self.args = args
        
        self.social_distance = social_distance
        self.draw_socialDistanceArea=draw_socialDistanceArea
        self.draw_pose=draw_pose
        self.count_people=count_people
        
        
        # callback function
        self.detection_listener = None
        self.web_listener = None
        
        # path initilize
        self.output_dir_img = os.path.join(output_dir, alias, "img")
        self.output_dir_video = os.path.join(output_dir, alias, "video")
        self.output_dir_img_draw = os.path.join(output_dir, alias, "img_draw")
        self.output_dir_video_draw = os.path.join(output_dir, alias, "video_draw")
        self.output_dir_json = os.path.join(output_dir, self.alias+ "_IDInfo.json")
        
        # Object Tracking
        self.id_storage = [] # save the trace id
        
        self.tracker = BYTETracker(args)

        self.bbox_colors = {}
        
        # Video initilize
        self.frame = np.zeros((1080,1920,4))
        self.drawImage = None
        self.cap = cv2.VideoCapture(self.video_url)        
        self.ret = False
        self.H = int(self.cap.get(4))
        self.W = int(self.cap.get(3))      
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')              
        self.frame_id = 0
        self.obj_id = None
        self.retry = 0 # reconntecting counts of the video capture
        
        
        #web streaming image
        self.bboxImage = np.zeros((self.H,self.W,4))
        self.socialDistanceImage = np.zeros((self.H,self.W,4))
        self.infoImage = np.zeros((self.H,self.W,4))
        
        #people counting
        self.totalIn = 0
        self.currentIn = 0
        self.totalOut = 0
        self.draw_square = draw_square
        if "square" in video_url:
            self.countInArea_cal = np.array([[0, 1100],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1100, 590],[1425, 1007],[1525, 985],[1574, 814],[1930, 1100] ])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.countOutArea = np.array([[0, 1080],[0, 0],[877, 0],[1019, 257],[1007, 360],[1177, 501],[1165, 595],[1512, 962],[1609, 578], [1980, 728], [1980, 1080]])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.suspiciousArea = np.array([[1070, 589],[846, 590],[890, 684],[1024, 732],[1129, 905],[1350, 927]])#This area use to handle occlusion when people get in square
            self.mergeIDArea = np.array([[144, 1074],[511, 465],[1099, 485],[1643, 1080]])#only in this area id can merge
            self.vertex = [[180, 873],[483, 266],[1124, 289],[1769, 870]]
        elif "FirstRestaurant_2" in video_url:
            self.countInArea_cal = np.array([[441, 600],[620, 280],[820, 280],[946, 597],])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.countOutArea = np.array([[300, 712],[470, 320],[569, 320],[588, 250],[834, 250],[861, 320],[988, 320],[1078, 707],])
            self.suspiciousArea = None#This area use to handle occlusion when people get in square
            
            
        elif "FirstRestaurant_1" in video_url:
            self.countInArea_cal = np.array([[16, 750],[16, 445],[79, 326],[380, 200],[463, 0],[1280, 280],[1170, 790], [880, 850]])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.countOutArea = np.array([[-18, 990], [-18, 283],[318, 130],[369, 144],[463, 0],[1280, 278],[1280, 750],[1080, 1050]])
            self.suspiciousArea = None#This area use to handle occlusion when people get in square
            
        
        elif "0325__12__12"  in video_url:
            self.countInArea_cal = np.array([[0, 1100],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1100, 590],[1425, 1007],[1525, 985],[1574, 814],[1930, 1100] ])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.countOutArea = np.array([[0, 1080],[0, 0],[877, 0],[1019, 257],[1007, 360],[1177, 501],[1165, 595],[1512, 962],[1609, 578], [1980, 728], [1980, 1080]])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.suspiciousArea = np.array([[1070, 589],[846, 590],[890, 684],[1024, 732],[1129, 905],[1350, 927]])#This area use to handle occlusion when people get in square
            
            self.vertex = [[180, 873],[483, 266],[1124, 289],[1769, 870]]
        elif "入口人流Oct25" in video_url:
            self.countInArea_cal = np.array([[1719, 1513], [1749, 910], [2551, 913], [2606, 1327]])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.countOutArea = np.array([[1656, 1634], [1714, 841], [2583, 828], [2655, 1418]])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.suspiciousArea = None
            
        elif "場內分區熱點監測" in video_url:
            self.countInArea_cal = np.array([[1217, 1480], [1102, 1297], [1416, 1143], [1707, 1210], [1953, 1222], [2303, 1177], [2639, 1457], [2195, 1694], [1864, 1702]])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.countOutArea = np.array([[1093, 1552], [1044, 1253], [1420, 1103], [1727, 1154], [1953, 1165], [2292, 1119], [2852, 1535], [2139, 1908], [1512, 1901]])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
            self.suspiciousArea = None
            
            
        
        self.lastCentroids = dict()
        self.IDsInLastSuspiciousArea = set()
        self.suspiciousAreaIDTracker = dict()
        self.IDSwitch = {
                        "frame":1000,
                        "amount":0
                        }
        self.lastDetections = list()#for merge method
        self.mergedIDs = dict()
        self.AllIDtracker = dict()
        self.mergeDistanceThreshold = 70
        self.splitDistanceThreshold = 100
        #social distance
        self.socialDistanceArea = np.array([ [378, 1080],[585, 345],[939, 339],[1590, 1080] ], np.float32)
        self.realHeight, self.realWidth = 19.97, 5.6#m
        self.transformImageHeight, self.transformImageWidth = 1000, 350
        transformPoints = np.array([[0, self.transformImageHeight], [0, 0], [self.transformImageWidth, 0], [self.transformImageWidth, self.transformImageHeight]], np.float32)
        self.social_distance_limit = 1#1m
        self.draw_socialDistanceInfo = draw_socialDistanceInfo
        
        # get transform matrix
        self.M = cv2.getPerspectiveTransform(self.socialDistanceArea, transformPoints)
        self.realHeightPerPixel, self.realWidthPerPixel = (self.realHeight / self.transformImageHeight), (self.realWidth / self.transformImageWidth)
        
        #fps calculate
        self.timer = Timer()
        
        
        # remove the exist video file
        video_path_original = os.path.join(self.output_dir_video, get_current_date_string(), get_current_hour_string())
        video_path_draw = os.path.join(self.output_dir_video_draw, get_current_date_string(), get_current_hour_string())
        
        if self.alias == "":
            self.video_output_name = 'output_original.mp4'
            self.video_output_draw_name = 'output_draw.mp4'           
            video_path_original = os.path.join(video_path_original, get_current_hour_string() + ".mp4")
            video_path_draw = os.path.join(video_path_draw, get_current_hour_string() + ".mp4")
        else:
            self.video_output_name = self.alias + '_output_original.mp4'
            self.video_output_draw_name = self.alias + '_output_draw.mp4'            
            video_path_original = os.path.join(video_path_original, get_current_hour_string() +"_"+ self.alias + ".mp4") 
            video_path_draw = os.path.join(video_path_draw, get_current_hour_string() +"_"+ self.alias + ".mp4")
        
        self.frame_original = cv2.VideoWriter(self.video_output_name, self.fourcc, self.FPS, (self.W, self.H))
        self.frame_draw = cv2.VideoWriter(self.video_output_draw_name, self.fourcc, self.FPS, (self.W, self.H))  
        if os.path.exists(video_path_original):
            os.remove(video_path_original)     
            
        if os.path.exists(video_path_original):
            os.remove(video_path_original)     
            
        if os.path.exists(video_path_draw):
            os.remove(video_path_draw)
            
        if os.path.exists(self.video_output_name):
            os.remove(self.video_output_name)            
            
        if os.path.exists(self.video_output_draw_name):
            os.remove(self.video_output_draw_name)
            
        print("[Info] Camera status {alias}:{s}".format(alias=self.alias, s=self.cap.isOpened()))
        
        
    def start(self):
        self.th = []
        self.write_log("[Info] Start the program.")
        if not self.testWithTxt:
            self.predictor = loadPredictor(self.args)
        
        if self.testMode:#test mode don't use multi threading
            # self.FPS.start()
            self.prediction()
            return
        
        if self.is_threading:
            self.th.append(threading.Thread(target = self.video_capture))
            self.th.append(threading.Thread(target = self.prediction))
        else:
            self.th.append(threading.Thread(target = self.prediction))
        
        for t in self.th:
            # self.FPS.start()
            t.start()

    
    def video_capture_wo_threading(self): 
        
        
        self.ret, self.frame = self.cap.read() 
        
            
        if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong
            
            if self.repeat:
                print(f"restart detection {datetime.datetime.now()}")
                self.cap = cv2.VideoCapture(self.video_url)#reread the video
                self.totalIn = 0
                self.currentIn = 0
                self.totalOut = 0
            else:
                print("[Info] Video detection is finished...")
                self.stop()            
        else:
            if self.save_video_original:
                self.save_video_frame(self.frame)
    
    
    def video_capture(self):
        t = threading.currentThread()   
        time.sleep(5) # waiting for loading yolo model
        
        while getattr(t, "do_run", True):
            self.ret, self.frame = self.cap.read() 
            
            if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong 
                print("[Error] Reconnecting:{group}:{alias}:{url} ".format(group=self.group, alias=self.alias, url=self.video_url))
                self.retry += 1
                self.cap.release()
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.video_url)
                time.sleep(5)
                
                if self.retry % 3 == 0:
                    self.write_log("[Error] Reconnecting to the camera.")
                    print("[Error] Restarting:{group}:{alias}:{url} ".format(group=self.group, alias=self.alias, url=self.video_url))
                    if self.auto_restart:
                        try:                            
                            self.restart()
                        except Exception as e:
                            self.write_log("[Error] Restart the program failed: {e}".format(e=e))
                            print("[Error] Can not restart:{group}:{alias}:{url} ".format(group=self.group, alias=self.alias, url=self.video_url))
                            time.sleep(10)
            else:
                if self.save_video_original:
                    self.save_video_frame(self.frame)

                    
    def prediction(self):        
        
        last_time = time.time() # to compute the fps
        cnt = 0  # to compute the fps
        predict_time_sum = 0  # to compute the fps        
        t = threading.currentThread() # get this function threading status
        
        while getattr(t, "do_run", True):
            if self.frame_id % 10000 == 0:
                logger.info('Processing frame {}, FPS={})'.format(self.frame_id, cnt / (time.time()-last_time))) 
            cnt+=1 
            
            if not self.is_threading:
                self.video_capture_wo_threading()

                
            if not self.cap.isOpened() or not self.ret:
                if self.testMode:
                    return#return and get new video                
                print("[Info] Waiting for reconnecting...\nVideo dead.")
                time.sleep(1)
                continue
                
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) # original image
            self.drawImage = self.frame.copy()
            
            if self.skip_frame != None and self.frame_id%self.skip_frame != 0:#skip frame
                self.frame_id += 1
                continue
            
            #reset web streaming image
            self.bboxImage = np.zeros((self.H,self.W,4))
            self.socialDistanceImage = np.zeros((self.H,self.W,4))
            self.infoImage = np.zeros((self.H,self.W,4))
                
            detections = []
            predict_time = time.time() # get start predict time
            
            if self.testWithTxt:#use txt detections:
                txtDets = getDets(self.txtDets, self.frame_id)
                for frameID, id, cx, cy, w, h, score in txtDets:
                    detections.append(['person', score, [cx, cy, w, h], id, [cx, cy + h / 2]])
                
                self.detect_target = detections
                self.detect_target = detect_filter(detections, self.target_classes, self.vertex, True)
                
            else:#use yolo detections
                outputs, img_info = self.predictor.inference(self.frame, self.timer)
                if outputs[0] is not None:
                    for x1, y1, x2, y2, obj_conf, class_conf, class_pred in outputs[0].cpu().detach().numpy():
                        x1 = x1 / 800 * self.H
                        y1 = y1 / 1440 * self.W
                        x2 = x2 / 800 * self.H
                        y2 = y2 / 1440 * self.W
                        w = x2 - x1
                        h = y2 - y1
                        cx = x1 + w / 2
                        cy = y1 + h / 2
                        detections.append([class_pred, obj_conf, (cx, cy, w, h)])
            
                
                # filter the scope and target class   
                self.detect_target = detect_filter(detections, self.target_classes, self.vertex, True)
                
                
                self.detect_target = self.object_tracker_BYTE()
                # print(self.detect_target)
            
            predict_time_sum +=  (time.time() - predict_time) # add sum predict time
            
            self.drawImage = drawTracks(self.drawImage, self.detect_target)
            
            save_path_img = None
            save_path_img_orig = None
            save_video_draw_path = None
            
            
            
            if self.draw_polygon: 
                self.drawImage = draw_polylines(self.drawImage, self.vertex)  # draw the polygon
                
            if self.draw_square:
                cv2.polylines(self.drawImage, pts=[self.countOutArea], isClosed=True, color=(255,0,0), thickness=3)#draw square area
                cv2.polylines(self.drawImage, pts=[self.countInArea_cal], isClosed=True, color=(0,0,255), thickness=3)#draw square area
                
                
            if self.draw_socialDistanceArea:
                socialDistanceArea_int = np.array(self.socialDistanceArea, np.int32)
                cv2.polylines(self.drawImage, pts=[socialDistanceArea_int], isClosed=True, color=(0,255,255), thickness=3)#draw square area

            
            if self.count_people and len(self.detect_target) > 0:
                self.people_counting()
                
            if self.social_distance and len(self.detect_target) > 0:
                self.drawImage = self.socialDistance(self.drawImage)
            
            # save draw bbox image
            if self.save_img and len(self.detect_target) > 0:                 
                save_path_img = self.save_img_draw(self.drawImage)
                
            self.drawImage = self.draw_info(self.drawImage)
            # self.saveDetectionsWithJson(self.detect_target)     
            
            # save oiginal image
            if self.save_img_original and len(self.detect_target) > 0:
                save_path_img_orig = self.save_img_orig(self.frame)
            
            # save video with draw            
            if self.save_video:
                save_video_draw_path = self.save_video_draw(self.drawImage)
            
            # callback function for user
            if len(self.detect_target) > 0:
                self.__trigger_callback(self.drawImage, self.group, self.alias, self.detect_target)
            
            # callback function for web streaming
            self.__trigger_web_callback(self.draw_info(self.frame), self.bboxImage, self.socialDistanceImage)            
                            
            
            # Compute FPS
            if time.time() - last_time  >= 5:
                fps = cnt / (time.time()-last_time)
                self.print_msg("[Info] FPS:{fps} , {alias}".format(alias=self.alias, fps=cnt / (time.time()-last_time)))
                self.print_msg("[Info] Predict FPS:{fps} , {alias}".format(alias=self.alias, fps=cnt / predict_time_sum))
                last_time = time.time()
                cnt = 0
                predict_time_sum = 0
                
            self.frame_id += 1
            
    def draw_info(self, image):
        #draw people counting info into image
        info = [
        ("Total Visitors", self.totalIn),
        ("Current", self.currentIn)
        ]

        info2 = [
        # ("Total people inside", x),
        ]

        # Display the output
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            # print(self.H)
            cv2.putText(image, text, (10, self.H - ((i * 50) + 100)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 5)
            cv2.putText(self.infoImage, text, (10, self.H - ((i * 50) + 100)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255, 255), 5)
            

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(image, text, (265, self.H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(self.infoImage, text, (265, self.H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255, 255), 2)
            
            
        return image
    
    def people_counting(self):
        
        if self.suspiciousArea is not None:
            self.__suspiciousAreaHandling()
        if 0 < len(self.detect_target) <= 10:
            # pass
            self.merge()
            self.split()
            
        for det in self.detect_target:
            if len(det) < 5 or None in det[4]:#center not None
                continue
            id = det[3]
            center_x, center_y = det[4]
            w, h = det[2][2:]
            countInAreaPolygon = Polygon(self.countInArea_cal)
            countOutAreaPolygon = Polygon(self.countOutArea)
            currentCentroid = Point((center_x, center_y))
            # if center_x <= 0 or center_x >= self.W or center_y <= 0 or center_y >= self.H:#out of boundary
            #     continue
            
            if self.lastCentroids.get(id, None) is None:#Don't have this id in last frame
                
                countIn = False
                countOut = False
                outIn = False
                
                if currentCentroid.within(countInAreaPolygon):#inside count in area that mean only can count in
                    countIn = True
                elif currentCentroid.within(countOutAreaPolygon):#inside count out area but not count in area
                    outIn = True
                    countOut = True
                else:#not in count in area and count out area
                    countOut = True
                    outIn = True
                    
                
                self.lastCentroids[id] = {"center":(center_x, center_y),#update id's center
                                          "wh":(w, h),
                                          "countIn":countIn,#set id not counted
                                          "countOut":countOut,
                                          "outIn":outIn,
                                          "InFrame":0,
                                          "OutFrame":0,
                                          }
                
                continue
            

            
            lastCentroid = self.lastCentroids[id]["center"]
            lastCentroid = Point(lastCentroid)
            
            # if the last centroid not in square and current centroid in square and non-counted
            # that mean the person get into the square from outside.
            # So count it
            isGetIn = not lastCentroid.within(countInAreaPolygon) and currentCentroid.within(countInAreaPolygon) and not self.lastCentroids[id]["countIn"]
            #last centroid in square and current centroid not in square
            isGetOut = lastCentroid.within(countOutAreaPolygon) and not currentCentroid.within(countOutAreaPolygon) and not self.lastCentroids[id]["countOut"]
            OutAndIn = self.lastCentroids[id]["OutFrame"] - self.lastCentroids[id]["InFrame"]  <= 30 and isGetIn
            
            if isGetIn:#get in and not counted
                print(f"Normal add:{id}, frameId:{self.frame_id}")
                self.totalIn += 1
                self.currentIn += 1
                self.lastCentroids[id]["countIn"] = True
                self.lastCentroids[id]["countOut"] = False
                self.lastCentroids[id]["InFrame"] = self.frame_id
                
                if id in self.unreliableID:
                    print("unreliableID in {i}")
                
                
            # if OutAndIn:
            #     print("Out and in", id)
            #     self.currentIn -= 1
            #     self.totalIn -= 1
            #     self.totalOut -= 1
                # self.lastCentroids[id]["outIn"] = True
                
                
            if isGetOut:
                if self.mergedIDs.get(id, None) is not None:
                    print("Normal merge out:", id, self.mergedIDs[id])
                    for i in self.mergedIDs[id]:
                        if not self.lastCentroids[i]["countOut"]:#id not count out
                            self.lastCentroids[i]["countOut"] = True
                            self.totalOut += 1
                            self.currentIn -= 1
                            
                        if i in self.unreliableID:
                            print("unreliableID out {i}")
                            
                            # print(f"id={i}, countOut={self.lastCentroids[i]}")
                        
                else:
                    print(f"Normal out:{id}, frameId:{self.frame_id}")
                    self.totalOut += 1
                    self.currentIn -= 1
                    
                    # self.lastCentroids[id]["countIn"] = False
                self.lastCentroids[id]["countOut"] = True
                self.lastCentroids[id]["outIn"] = True
                self.lastCentroids[id]["countIn"] = False 
                self.lastCentroids[id]["OutFrame"] = self.frame_id
                
                
                
            self.lastCentroids[id]["center"] = (center_x, center_y)#update id's center
        
        self.lastDetections = self.detect_target
                    
    def __suspiciousAreaHandling(self):
        '''
        This function can break down into three parts.
        occlusion handling
        ID switching
        object flash appearing
        '''
        IDsInCurrentSuspiciousArea = set()
        IDsInThisFrame = set()
        countedID = set()
        detections = dict()
        
        self.IDSwitch["frame"] += 1
        ############################
        #occlusion handling
        #assume that some people are occluded when 
        #they are getting in the square but will
        #appear in the suspicious area.
        #So count +1 when someone suddenly appears
        # n the suspicious area.
        ############################
        for det in self.detect_target:
            
            x, y = det[4]#use this xy not center of bbox
            id = det[3]
            w, h = det[2][2:]
            currentCentroid = Point((x, y))
            suspiciousAreaPolygon = Polygon(self.suspiciousArea)
            
            detections[id] = det#save as dict for later
            IDsInThisFrame.add(id)
            #this id is new and spawn in the suspicious area. That I can say it has occlusion
            if currentCentroid.within(suspiciousAreaPolygon):
                IDsInCurrentSuspiciousArea.add(id)
                if self.lastCentroids.get(id, None) is None:
                    print("Area add:", id)
                    self.totalIn += 1
                    self.currentIn += 1
                    countedID.add(id)
                    center_x, center_y = det[4]
                    self.lastCentroids[id] = {"center":(center_x, center_y),#update id's center
                                              "wh":(w,h),
                                          "countIn":True,
                                          "countOut":False,
                                          "outIn":False,
                                          "InFrame":self.frame_id,
                                          "OutFrame":0
                                          }#set id not counted
                
                    ############################
                    #ID switch happening
                    ############################
                    #FPS depends
                    if self.IDSwitch.get("frame", 20) < 10 and self.IDSwitch.get("amount", 0) > 0:
                        self.totalIn -= 1
                        self.totalOut += 1
                        self.currentIn -= 1
                        self.IDSwitch["amount"] -= 1
                        print(f"Id switch:{id}")
                
        ############################
        #ID switch handling
        #assume that 3 people are in the suspicious area but id switches happening and the id switch process is fast.
        #example:ID 1 -> ID 4
        #frame 1:
        #ID 1,2,3
        #frame 2:
        #ID 2,3
        #frame 3:
        #2, 3
        #frame 4:
        #2, 3, 4
        ############################
        if len(self.IDsInLastSuspiciousArea) > len(IDsInCurrentSuspiciousArea):#the amount of people in the last frame is larger than this frame, may have id switching in the future
            self.IDSwitch["frame"] = 0
            self.IDSwitch["amount"] += len(self.IDsInLastSuspiciousArea) - len(IDsInCurrentSuspiciousArea)
            
            
        
        ############################
        #object flash appearing
        #There has been some error detection of Yolo just flashing on the screen
        #when there have a lot of people. So just keep tracking the object.
        ############################
        # suddenly appear id
        self.unreliableID = []
        #FPS depends
        TRACK_FRAMES = 20  # const for amount of frames to track
        COUNTED_THRESHOLD = 16
        mode = "counted"  # ["counted", "continuous"]
        for old_ID in list(self.suspiciousAreaIDTracker.keys()):
            if self.suspiciousAreaIDTracker[old_ID]["tracked"] > TRACK_FRAMES:#checked
                continue
            
            
            if old_ID in IDsInThisFrame:#if id is in this frame
                # add counter and keep cont status if already not continuous
                old_ID_dict = self.suspiciousAreaIDTracker[old_ID]
                self.suspiciousAreaIDTracker[old_ID] = {"tracked": old_ID_dict["tracked"]+1, "counted": old_ID_dict["counted"]+1, "continuous": True if old_ID_dict["continuous"] else False}
                
            else:
                self.suspiciousAreaIDTracker[old_ID]["tracked"] += 1
                self.suspiciousAreaIDTracker[old_ID]["continuous"] = False
                
            if self.suspiciousAreaIDTracker[old_ID]["tracked"] == TRACK_FRAMES:
                if mode == "counted":
                    if self.suspiciousAreaIDTracker[old_ID]["counted"] < COUNTED_THRESHOLD:  # id appeared not enough times
                        print("Remove", old_ID, self.suspiciousAreaIDTracker[old_ID])
                        for i in self.mergedIDs:#remove flash id from merged id
                            if old_ID in self.mergedIDs[i]:
                                self.mergedIDs[i].remove(old_ID)
                                
                        self.totalIn -= 1
                        self.currentIn -= 1
                        
                else:  # "continuous" not using
                    if self.suspiciousAreaIDTracker[old_ID]["continuous"] == False:  # id appeared continuously
                        self.totalIn -= 1
                        self.currentIn -= 1
                        
                # self.suspiciousAreaIDTracker.pop(old_ID)#remove id
                
                
        # add new and counted ID to tracker
        # new_IDs = IDsInCurrentSuspiciousArea.difference(self.IDsInLastSuspiciousArea)
        for new_ID in countedID:
            if self.suspiciousAreaIDTracker.get(new_ID, None) is None :#new id in this frame and already +1 and self.lastCentroids[new_ID]["counted"]
                self.suspiciousAreaIDTracker[new_ID] = {"tracked": 1, "counted": 1, "continuous": True}
                
        self.IDsInLastSuspiciousArea = IDsInCurrentSuspiciousArea  # update id

    def merge(self):
        #find disappear person
        thisFrameDetections = {det[3]:det[2] for det in self.detect_target}#{id:center}
        lastFrameDetections = {det[3]:det[2] for det in self.lastDetections}#{id:center}
        thisFrameIDS = set(thisFrameDetections.keys())
        lastFrameIDS = set(lastFrameDetections.keys())
        disappearIDS = lastFrameIDS.difference(thisFrameIDS)
        
        for i in disappearIDS:
            x1, y1 = lastFrameDetections[i][:2]
            for j in lastFrameIDS:
                if i == j:#same id
                    continue
                x2, y2 = lastFrameDetections[j][:2]
                
                distance = ( (x1-x2)**2 + (y1-y2)**2 )**0.5
                if distance < self.mergeDistanceThreshold:#disappear person is very close to this person
                    if self.mergedIDs.get(j, None) is None:
                        self.mergedIDs[j] = set([j,i])
                    else:#already merged other ID
                        self.mergedIDs[j].add(i)
                    # print("ID merged:", self.mergedIDs)

    def split(self):
        #find new person
        thisFrameDetections = {det[3]:det[2] for det in self.detect_target}#{id:center}
        lastFrameDetections = {det[3]:det[2] for det in self.lastDetections}#{id:center}
        thisFrameIDS = set(thisFrameDetections.keys())
        lastFrameIDS = set(lastFrameDetections.keys())
        newIDS = thisFrameIDS.difference(lastFrameIDS)
        
        thisFrameIDsList = [det[3] for det in self.detect_target]
       
        for i in newIDS:
            x1, y1 = thisFrameDetections[i][:2]
            for j in thisFrameIDS:
                if i == j:#same id
                    continue
                x2, y2 = thisFrameDetections[j][:2]
                distance = ( (x1-x2)**2 + (y1-y2)**2 )**0.5
                if distance < self.splitDistanceThreshold:#new person is close to this person
                    if self.mergedIDs.get(j, None) is not None and len(self.mergedIDs[j]) > 1:#new id split
                        spiltID = list(self.mergedIDs[j])[1]
                        if spiltID == j:#same id
                            continue
                        self.mergedIDs[j].remove(spiltID)#remove spilt id from set
                        splitIDIndex = thisFrameIDsList.index(i)#find the new id's index of this frame
                        self.detect_target[splitIDIndex][3] = spiltID#recover id
                        # print(f"split ID {spiltID} from {j}, {self.mergedIDs[j]}")
        

    # #split unreliable id
    # for ID in self.mergedIDs:
    #     overlapID = self.mergedIDs[ID].intersection(set(self.unreliableID))#if merged IDS have unreliable ID
    #     if len(overlapID) != 0:
    #         for removeID in overlapID:#remove unreliable ID one by one
    #             if removeID == ID:#don't remove itself
    #                 continue
    #             self.mergedIDs[ID].remove(removeID)#remove spilt id from set
                
    #             print(f"split unreliable ID {removeID} from {ID}, {self.mergedIDs[ID]}")
        # self.lastDetections = self.detect_target
    

    def socialDistance(self, image):
        closePairs = []

        centroids = [det[4] for det in self.detect_target if len(det)>=5 and det[4] != [None, None]]#get 4th element which is (pose_center_x, pose_center_y)
        
        if len(centroids) < 2:#less than two people then no need to calculate social distance
            return image
            
        
        centroids = np.array(centroids)#change it to array type
        
        transformedCentroids = cv2.perspectiveTransform(np.array([centroids]), self.M)#perspective transform yolo center
        transformedCentroids = np.reshape(transformedCentroids, (-1, 2))#reshape to a list of point
        
        insidePointIndex = (transformedCentroids[:,0] >= 0) & (transformedCentroids[:,0] <= self.transformImageWidth) & (transformedCentroids[:,1] >= 0) & (transformedCentroids[:,1] <= self.transformImageHeight)
        transformedCentroids = transformedCentroids[insidePointIndex]#remove the transformed point outside the square
        centroids = centroids[insidePointIndex]#remove the non-transformed point outside the square
        
        
        for index, centroid in enumerate(transformedCentroids):
            
            x_Distance = (centroid[0] - transformedCentroids[:, 0]) * self.realWidthPerPixel
            y_Distance = (centroid[1] - transformedCentroids[:, 1]) * self.realHeightPerPixel

            distance = (x_Distance ** 2 + y_Distance**2) **0.5
            closePair = np.where( distance < self.social_distance_limit )[0]
            for pointIndex in closePair:
                if(index == pointIndex):#itself
                    continue
                closePairs.append([int(centroids[index][0]), int(centroids[index][1]), int(centroids[pointIndex][0]), int(centroids[pointIndex][1])])#add not transform point to the list for drawing line in the image
                
                
                if self.draw_socialDistanceInfo:
                    cv2.line(image, (int(centroids[index][0]), int(centroids[index][1]) ), (int(centroids[pointIndex][0]), int(centroids[pointIndex][1]) ), (255,0,0), 2)
                    
                    pairDistance = distance[pointIndex]
                    message = "%2.1f m" % pairDistance
                    pairCenterX = int( (centroids[index][0] + centroids[pointIndex][0]) / 2 )
                    pairCenterY = int( (centroids[index][1] + centroids[pointIndex][1]) / 2 ) 
                    
                    # cv2.putText(image, message, (pairCenterX, pairCenterY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)#draw distance below the line
                    
                    #draw web streaming image
                    self.socialDistanceImage = np.zeros((image.shape[0], image.shape[1], 4))
                    # cv2.line(self.socialDistanceImage, (int(centroids[index][0]), int(centroids[index][1]) ), (int(centroids[pointIndex][0]), int(centroids[pointIndex][1]) ), (255,0,0, 255), 2)
                    # cv2.putText(self.socialDistanceImage, message, (pairCenterX, pairCenterY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 255), 2)#draw distance below the line

        return image
 
    def object_tracker_BYTE(self):
        #[className, score, (cx, cy, W, H)] -> [x1, y1, x2, y2, score]
        
        dets = list()
        if len(self.detect_target) > 0:
            for det in self.detect_target:
                score = float(det[1])
                cx, cy, W, H = det[2]
                x1 = int(cx - W / 2)
                y1 = int(cy - H / 2)
                x2 = x1 + W
                y2 = y1 + H
                dets.append([x1, y1, x2, y2, score])
        
        else:
            dets = [[1, 1, 1, 1, 0]]
        
        dets = np.array(dets, dtype=float)
        online_targets = self.tracker.update(dets, [self.H, self.W], [self.H, self.W])
        
        detWithID = []
        for track in online_targets:
            t, l, w, h = track.tlwh
            id = track.track_id
            cx = int( (t + w / 2))
            cy = int( (l + h / 2))
            # assign each id with a color
            if self.bbox_colors.get(id) == None:
                self.bbox_colors[id] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
            # save results
            detWithID.append(['person', track.score, [cx, cy, w, h], id, [cx, cy + w / 2]])

        return detWithID

    def set_listener(self, on_detection):
        self.detection_listener = on_detection
        
        
    def __trigger_callback(self, image, group, alias, detect):        
        if self.detection_listener is None:
            return         
        self.detection_listener(image, group, alias, detect)
        
    def set_web_listener(self, on_detection):
        self.web_listener = on_detection
        
        
    def __trigger_web_callback(self, frame, bboxImage, distanceImage):        
        if self.web_listener is None:
            return         
        self.web_listener(frame, bboxImage, distanceImage)
  

    def get_current_frame(self):
        return self.draw_info(self.frame)
        
        
    def get_draw_image(self):
        
        return self.drawImage
    
    def get_bbox_image(self):
        
        return self.bboxImage
    
    def get_social_distance_image(self):
        
        return self.socialDistanceImage
        
    def get_info_image(self):
        return self.infoImage
        
    def stop(self):        
        if not self.testMode:
            for t in self.th:
                t.do_run = False
    #             t.join()
        
        self.write_log("[Info] Stop the program.")
        self.cap.release()
            
        print('[Info] Stop the program: Group:{group}, alias:{alias}, URL:{url}'\
              .format(group=self.group, alias=self.alias, url=self.video_url))    
        
        #save detections info to json file
        # with open(self.output_dir_json, "w") as outfile:
        #     json.dump(self.IDInfo, outfile)         
        
    def restart(self):
        self.stop()        
        self.write_log("[Info] Restart the program")
        restart()    
        
    def write_log(self, msg):     
        f= open('log.txt', "a")    
        f.write("{msg}, Time:{time}, Group:{group}, alias:{alias}, URL:{url} \n"\
                .format(time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), \
                 group=self.group, alias=self.alias, url=self.video_url, msg=msg))
        f.close()
       
    def print_msg(self, msg):
        if self.display_message == True:            
            print(msg)
             
    def save_video_frame(self, frame):
        
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video, get_current_date_string(), 
                        get_current_hour_string())
            if self.alias == "":                
                video_path_name = os.path.join(video_path, get_current_hour_string() + ".mp4")                
            else:
                video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".mp4")   
            
        else: # video input, so output wihtout time directory
            video_path_name = self.video_output_name
        
        if not os.path.exists(video_path_name):
            if self.is_threading: create_dir(video_path)                        
            self.frame_original = cv2.VideoWriter(video_path_name, self.fourcc, 30.0, (self.W, self.H))
            print("[Info] {alias} Set video frame writer. Height={H}, Width={W}".format(alias=self.alias, H=self.H, W=self.W))
            
        self.frame_original.write(frame)
        
        return video_path_name
    
    
    def save_video_draw(self, frame):
        
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video_draw, get_current_date_string(), 
                        get_current_hour_string())
            if self.alias == "":                
                video_path_name = os.path.join(video_path, get_current_hour_string() + ".mp4")                
            else:
                video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".mp4")   
            
        else: # video input, so output wihtout time directory
            video_path_name = self.video_output_draw_name            
            
        if not os.path.exists(video_path_name):
            if self.is_threading: create_dir(video_path)
            self.frame_draw = cv2.VideoWriter(video_path_name, self.fourcc, self.FPS, (self.W, self.H))
            print("[Info] {alias} Set video draw writer. Height={H}, Width={W}".format(alias=self.alias, H=self.H, W=self.W))  
            
        self.frame_draw.write(frame)
        
        return video_path_name
    
    
    def save_img_draw(self, image):        
        img_path = os.path.join(self.output_dir_img_draw, get_current_date_string(), get_current_hour_string())     
        
        if not os.path.exists(img_path):
            create_dir(img_path)      
            
        if self.alias == "":
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) + ".jpg")
        else:
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) +"_"+ self.alias + ".jpg")                
                  
        cv2.imwrite(img_path_name, image)
        
        return img_path_name
    
    
    def save_img_orig(self, image):
        if not self.is_threading:
            img_path = self.output_dir_img_draw
        else:
            img_path = os.path.join(self.output_dir_img, get_current_date_string(), get_current_hour_string())
        
        if not os.path.exists(img_path):
            create_dir(img_path)            
            
        if self.alias == "":
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) + ".jpg")
        else:
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) +"_"+ self.alias + ".jpg")            
                
        cv2.imwrite(img_path_name, image)
        
        return img_path_name

    def test(self, video, args, testWithTxt = True, txtFile = ""):#track not used
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.__testWithFile(video, args, testWithTxt, txtFile)
        
          
    def __testWithFile(self, video, args, testWithTxt = True, txtFile = ""):
        self.testWithTxt = testWithTxt
        if self.testWithTxt:
            self.txtDets = getNpData(txtFile)
        
        groundTruth = []#number of people get in square
        predictNumber = []
        error = []
        errorInfo = dict()
        
        print(f"Test {video}")
        self.alias = video.split("/")[-1].split(".")[0]
        self.cap = cv2.VideoCapture(video)
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.tracker = BYTETracker(args)
        print(f"thresh = {self.thresh} BYTE:age={args.track_buffer}  {self.FPS=}")
        
        
        self.totalIn = 0#reset counter
        self.currentIn = 0
        self.totalOut = 0
        self.lastCentroids = dict()#reset people counting info
        self.suspiciousAreaIDTracker = dict()
        
        # logger.info(video)
        GTNumberIn = int(video.split('.')[0].split('__')[-2])#get the ground truth number of the video
        GTNumberOut = int(video.split('.')[0].split('__')[-1])
        groundTruth.append(GTNumberIn)
        self.video_output_draw_name = os.path.join(self.output_dir, self.alias + '_output_draw.mp4')         
        
        
        self.start()#start video predict
        
        predictNumber.append(self.totalIn)
        
        
        errorInfo[video] = {
            "GT_TotalIn":GTNumberIn,
            "Pre__TotalIn":self.totalIn,
            "GT_Out":GTNumberOut,
            "Pred_Out":self.totalIn - self.currentIn,
            "out":self.totalOut,
        }
        error.append(abs(GTNumberIn - self.totalIn))
        #save evaluation result to json file
        with open(f"evaluation_BYTE_{self.args.exp}.json", 'a+') as outFile:
            outFile.seek(0)
            config = f"{args.yolo_thresh=},{args.track_thresh=},{args.track_buffer=},{args.match_thresh=}"
            data = dict()
            try:
                data = json.load(outFile)
                if(config not in data):#data do not have this config data
                    tmp = list()
                    tmp.append(errorInfo)
                    data[config] = tmp
                else:
                    data[config].append(errorInfo)
                
            except:#file is emtpy
                    tmp = list()
                    tmp.append(errorInfo)
                    data[config] = tmp
            # print(data)
            outFile.truncate(0)
            json.dump(data, outFile)
        print(errorInfo)
        
    def video2Label(self, videoFolder, outputDir):
        shutil.rmtree(outputDir, ignore_errors=True)#clear output dir
        os.mkdir(outputDir)
        
        videoFiles = os.listdir(videoFolder)
        
        for video in sorted(videoFiles):#loop each video
            print(f"Converting {video}")
            self.alias = video.split(".")[0]
            videoPath = os.path.join(videoFolder, video)
            self.cap = cv2.VideoCapture(videoPath)
            self.H = int(self.cap.get(4))
            self.W = int(self.cap.get(3))
            
            self.frame_id = 0
            totalFrame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = tqdm(total=totalFrame)
            
            while self.cap.isOpened():#loop each frame
                ret, frame = self.cap.read()
                self.frame_id += 1
                if ret == False:
                    self.cap.release()
                    break
                
                if self.skip_frame != None and self.frame_id%self.skip_frame != 0:#skip frame
                    continue
            
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # original image
                darknet_image = darknet.make_image(self.W, self.H, 3)
                darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
                detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
                
                darknet.free_image(darknet_image)
    
                # filter the scope and target class   
                self.detect_target = detect_filter(detections, self.target_classes, self.vertex)
        
                
                if len(self.detect_target) >= 3:
                    # print(self.detect_target)
                    file = open(os.path.join(outputDir, video+f"_{self.frame_id}.txt"), 'w')
                    for det in self.detect_target:#write each detection into txt and save this frame
                        classNum = 0#only have one class - person
                        x, y , w, h = det[2]
                        x /= self.W
                        y /= self.H
                        w /= self.W
                        h /= self.H 
                        text = f"{classNum} {x} {y} {w} {h}\n"
                        file.write(text)
                        
                    file.close()
                    
                    cv2.imwrite(os.path.join(outputDir, video+f"_{self.frame_id}.jpg"), frame)
                
                    
                
                progress.update(1) 
        
    def generateTrainTxt(self, dataFolder, valSplit=0.2, testSplit=0.1):
        print(dataFolder)
        labeledData = os.listdir(dataFolder)
        labeledData = [i for i in labeledData if i.endswith(".jpg")]
        random.shuffle(labeledData)
        length = len(labeledData)
        
        trainFile = open(os.path.join(dataFolder, "train.txt"), 'w')
        valFile = open(os.path.join(dataFolder, "val.txt"), 'w')
        testFile = open(os.path.join(dataFolder, "test.txt"), 'w')
        
        trainSplit = 1 - valSplit - testSplit
        trainIndex = int(length*trainSplit)
        #write train.txt
        for i in labeledData[:trainIndex]:
            trainFile.write(os.path.join(dataFolder, i)+'\n')
            
        trainFile.close()
        #write val.txt
        valIndex = int(trainIndex + length * valSplit)
        for i in labeledData[trainIndex:valIndex]:
            valFile.write(os.path.join(dataFolder, i)+'\n')
            
        valFile.close()
        
        #write test.txt
        for i in labeledData[valIndex:]:
            testFile.write(os.path.join(dataFolder, i)+'\n')
            
        testFile.close()  
        
    def saveDetectionsWithJson(self, detections):
        for det in detections:
            ID = int(det[3])
            x, y, w, h = det[2]
            data = {"frame":int(self.frame_id), "x":int(x), "y":int(y), "w":int(w), "h":int(h)}#change np.int32 to int
            if ID in self.IDInfo:
                self.IDInfo[ID].append(data)
            else:
                self.IDInfo[ID] = [data]
        
                    
            