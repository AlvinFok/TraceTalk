import cv2
from matplotlib.pyplot import draw
import numpy as np
import datetime
import time
import os
import shutil
import threading
import sys
import ctypes
import  random
#poise estimation
import mediapipe as mp
from tqdm import tqdm
import pathlib
import json

#people counting
from shapely.geometry import Point, Polygon

#FPS
from imutils.video import FPS


#BYTE tracker
# from yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from motpy.testing_viz import draw_detection, draw_track


# multiobjecttracker
sys.path.insert(1, 'multi-object-tracker')
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks

# SmartFence  lib
from libs.utils import *

from darknet import darknet


#config of BYTE tracker
class Arg():
    def __init__(self):
        self.track_thresh = 0.6
        self.track_buffer = 30
        self.match_thresh = 0.9
        # self.min-box-area = 100
        self.mot20 = False


class YoloDevice:
    def __init__(self, video_url="", output_dir="", run=True, auto_restart=False, repeat=False, obj_trace = False,
                 display_message=True, data_file="", config_file="", weights_file="", 
                 names_file="", thresh=0.5, vertex=None, target_classes=None, draw_bbox=True, draw_polygon=True, draw_square=True,
                 draw_socialDistanceArea=False, draw_socialDistanceInfo=False,  social_distance=False, draw_pose=False, count_people=False, draw_peopleCounting=False,
                 alias="", group="", place="", cam_info="", warning_level=None, is_threading=True, skip_frame=None,
                 schedule=[], save_img=True, save_original_img=False, save_video=False, save_video_original=False, testMode=False, gpu=0):
        

        
        
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
        
        #load model
        darknet.set_gpu(gpu)#set the gpu you want to use
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file = self.config_file,
            data_file = self.data_file,
            weights = self.weights_file)
        
        
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
#         self.tracker_motpy = MultiObjectTracker(
#                     dt=1 / 30,
#                     tracker_kwargs={'max_staleness': 5},
#                     model_spec={'order_pos': 1, 'dim_pos': 2,
#                                 'order_size': 0, 'dim_size': 2,
#                                 'q_var_pos': 5000., 'r_var_pos': 0.1},
# #                     matching_fn_kwargs={'min_iou': 0.25,
# #                                     'multi_match_min_iou': 0.93}
#                     )     
        # self.tracker = CentroidTracker(max_lost=7, tracker_output_format='mot_challenge', )
        self.tracker = CentroidKF_Tracker(max_lost=30, tracker_output_format='mot_challenge', centroid_distance_threshold=50)
#         self.tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.1)
        # self.tracker = IOUTracker(max_lost=120, iou_threshold=0.4, min_detection_confidence=0.4, max_detection_confidence=0.7, tracker_output_format='mot_challenge')
        
        args = Arg()
        self.tracker = BYTETracker(args)
        #SortOH Tracker
        # self.tracker = SortOHTracker.Sort_OH(max_age=30)  # create instance of the SORT with occlusion handling tracker
        # self.conf_trgt = 0.35
        # self.conf_objt = 0.75
        # self.tracker.conf_trgt = self.conf_trgt
        # self.tracker.conf_objt = self.conf_objt
        
        
        self.bbox_colors = {}
        
        # Video initilize
        self.frame = np.zeros((1080,1920,4))
        self.drawImage = None
        self.cap = cv2.VideoCapture(self.video_url)        
        self.ret = False
        self.H = int(self.cap.get(4))
        self.W = int(self.cap.get(3))      
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')              
        self.frame_id = 0
        self.obj_id = None
        self.retry = 0 # reconntecting counts of the video capture
        
        
        #web streaming image
        self.bboxImage = np.zeros((1080,1920,4))
        self.socialDistanceImage = np.zeros((1080,1920,4))
        self.infoImage = np.zeros((1080,1920,4))
        
        #people counting
        self.totalIn = 0#count how many people enter the area totally
        self.currentIn = 0#how many people are in the area right now
        self.draw_square = draw_square
        # self.countInArea_draw = np.array([[0, 1080],[0, 768],[557, 247],[983, 260], [993, 359],[1159, 493],[1137, 586],[1080, 590],[1425, 1007],[1525, 985],[1574, 814],[1920, 1080] ], np.int32)#The polygon of the area you want to count people inout
        # self.countInArea_cal = np.array([[0, 1090],[0, 768],[557, 247],[983, 260], [993, 359],[1159, 493],[1137, 586],[1090, 590],[1425, 1007],[1525, 985],[1574, 814],[1930, 1090] ])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
        self.countInArea_draw = np.array([[0, 1080],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1080, 590],[1425, 1007],[1525, 985],[1574, 814],[1920, 1080] ], np.int32)#The polygon of the area you want to count people inout
        self.countInArea_cal = np.array([[0, 1090],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1090, 590],[1425, 1007],[1525, 985],[1574, 814],[1930, 1090] ])#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
        self.countOutArea = np.array([[0, 1080],[0, 0],[877, 0],[1019, 257],[1007, 360],[1177, 501],[1165, 595],[1512, 962],[1575, 827],[1920, 1080]])
        self.suspiciousArea = np.array([[1080, 582],[850, 588],[981, 927],[1350, 921]])#This area use to handle occlusion when people grt in square
        self.suspiciousArea_L = np.array([[1080, 589],[846, 590],[890, 684],[1024, 732],[1129, 905],[1350, 927]])
        self.mergeIDArea = np.array([[144, 1074],[511, 365],[999, 385],[1643, 1080]])#only in this area id can merge
        self.lastCentroids = dict()
        self.IDsInLastSuspiciousArea = set()
        self.suspiciousAreaIDTracker = dict()
        self.IDSwith = {
                        "frame":1000,
                        "amount":0
                        }
        self.lastDetections = list()#for merge method
        self.mergedIDs = dict()
        self.AllIDtracker = dict()
        self.flashID = list()
        
        #social distance
        self.socialDistanceArea = np.array([ [378, 1080],[585, 345],[939, 339],[1590, 1080] ], np.float32)
        # self.realHeight, self.realWidth = 15.75, 5.6#m
        self.realHeight, self.realWidth = 19.97, 5.6#m
        self.transformImageHeight, self.transformImageWidth = 1000, 350
        transformPoints = np.array([[0, self.transformImageHeight], [0, 0], [self.transformImageWidth, 0], [self.transformImageWidth, self.transformImageHeight]], np.float32)
        self.social_distance_limit = 1#1m
        self.draw_socialDistanceInfo = draw_socialDistanceInfo
        
        # get transform matrix
        self.M = cv2.getPerspectiveTransform(self.socialDistanceArea, transformPoints)
        self.realHeightPerPixel, self.realWidthPerPixel = (self.realHeight / self.transformImageHeight), (self.realWidth / self.transformImageWidth)
        
        
        #pose estimation
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.detected_landmark = 0
        self.good_landmark = 0
        
        
        #face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        
        
        #fps calculate
        # self.FPS = FPS()
        
        
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
        
        self.frame_original = cv2.VideoWriter(self.video_output_name, self.fourcc, 20.0, (self.W, self.H))
        self.frame_draw = cv2.VideoWriter(self.video_output_draw_name, self.fourcc, 20.0, (self.W, self.H))  
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
        # network, class_names, class_colors = darknet.load_network(
        #     config_file = self.config_file,
        #     data_file = self.data_file,
        #     weights = self.weights_file)
        
        last_time = time.time() # to compute the fps
        cnt = 0  # to compute the fps
        predict_time_sum = 0  # to compute the fps        
        t = threading.currentThread() # get this function threading status
        
        while getattr(t, "do_run", True):
            # print(f"{self.alias} predecting")
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
            self.bboxImage = np.zeros((1080,1920,4))
            self.socialDistanceImage = np.zeros((1080,1920,4))
            self.infoImage = np.zeros((1080,1920,4))
                
            #do yolo prediction and tracking
            
            darknet_image = darknet.make_image(self.W, self.H, 3)
            darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
            
            predict_time = time.time() # get start predict time
            detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)#[className, score, (cx, cy, W, H)]
            predict_time_sum +=  (time.time() - predict_time) # add sum predict time
            
#             darknet.print_detections(detections, True) # print detection
            darknet.free_image(darknet_image)
    
            # filter the scope and target class   
            self.detect_target = detect_filter(detections, self.target_classes, self.vertex)
              

            
            if self.obj_trace and len(self.detect_target) > 0: # draw the image with object tracking           
                # self.drawImage = self.object_tracker(self.drawImage)
                self.drawImage = self.object_tracker_BYTE(self.drawImage)
                # self.drawImage = self.object_tracker_deep_sort(frame_rgb)
                # self.drawImage = self.object_tracker_motpy(frame_rgb)
            elif self.draw_bbox:
                self.drawImage = draw_boxes(detections, self.drawImage, self.class_colors, self.target_classes, self.vertex)
            
            save_path_img = None
            save_path_img_orig = None
            save_video_draw_path = None
            
            
            
            if self.draw_polygon: 
                self.drawImage = draw_polylines(self.drawImage, self.vertex)  # draw the polygon
                
            if self.draw_square:
                cv2.polylines(self.drawImage, pts=[self.countInArea_draw], isClosed=True, color=(0,0,255), thickness=3)#draw square area
                cv2.polylines(self.drawImage, pts=[self.countOutArea], isClosed=True, color=(255,0,0), thickness=3)#draw square area
                
                
            if self.draw_socialDistanceArea:
                socialDistanceArea_int = np.array(self.socialDistanceArea, np.int32)
                cv2.polylines(self.drawImage, pts=[socialDistanceArea_int], isClosed=True, color=(0,255,255), thickness=3)#draw square area
            
            # cv2.polylines(self.drawImage, pts=[self.suspiciousArea], isClosed=True, color=(0,255,0), thickness=3)#draw square area
            cv2.polylines(self.drawImage, pts=[self.suspiciousArea_L], isClosed=True, color=(0,255,0), thickness=3)#draw square area
            

            # if self.draw_pose and len(self.detect_target) > 0:
            #     image = self.pose_estimation(image)
                
            self.drawImage = self.face_detection(frame_rgb, self.drawImage)
            
            if self.count_people and len(self.detect_target) > 0:
                self.people_counting()
                
            if self.social_distance and len(self.detect_target) > 0:
                self.drawImage = self.socialDistance(self.drawImage)
            
            # save draw bbox image
            if self.save_img and len(self.detect_target) > 0:                 
                save_path_img = self.save_img_draw(self.drawImage)
                
            self.drawImage = self.draw_info(self.drawImage)
            self.saveDetectionsWithJson(self.detect_target)     
            
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
            # self.FPS.update()
            
    #https://www.youtube.com/watch?v=brwgBf6VB0I
    def pose_estimation(self, image):
        for index, detection in enumerate(self.detect_target):#for each detection
            
            left, top, right, bottom = darknet.bbox2points(detection[2])
            left, top, right, bottom = left-30, top-30, right+30, bottom+30
            frame = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)#change image from bgr to rgb
            result = self.pose.process(frame)#crop the detected area to do pose estimation
            
            if(result.pose_landmarks):
                self.detected_landmark += 1
                # print(result.pose_landmarks)
                self.mpDraw.draw_landmarks(image[top:bottom, left:right], result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                landmark = result.pose_landmarks.landmark
                twoHeelsDetected = 0<= landmark[29].x <=1 and 0<= landmark[29].y <=1 and 0<= landmark[30].x <=1 and 0<= landmark[30].y <=1
                twoFeetDetected = 0<= landmark[31].x <=1 and 0<= landmark[31].y <=1 and 0<= landmark[32].x <=1 and 0<= landmark[32].y <=1
                
                
                boxWidth, boxHeight = detection[2][2:4]
                
                centerX, centerY = None, None
                if(twoHeelsDetected):
                    self.good_landmark += 1
                    #change center to be the center of two heels
                    centerX = ( (landmark[29].x + landmark[30].x)/2 ) * boxWidth + left + 30
                    centerY = ( (landmark[29].y + landmark[30].y)/2 ) * boxHeight + top + 30
                    
                elif(twoFeetDetected):
                    self.good_landmark += 1
                    #change center to be the center of two feet
                    centerX = ( (landmark[31].x + landmark[32].x)/2 ) * boxWidth + left + 30
                    centerY = ( (landmark[31].y + landmark[32].y)/2 ) * boxHeight + top + 30
                
                
                self.detect_target[index].append([centerX, centerY])
                if self.draw_pose and centerX is not None and centerY is not None:#center not None
                    cv2.circle(image, (int(centerX), int(centerY)), 8, (255,0,0), -1)
        return image
    
    def draw_info(self, image):
        #draw people counting info into image
        info = [
        # ("Exit", totalUp),
        ("Total Visitors", self.totalIn),
        ("Current", self.currentIn)
        # ("Status", status),
        # ("total lanmarks", self.detected_landmark),
        # ("good lanmarks", self.good_landmark)
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
        
        self.__suspiciousAreaHandling()
        self.__checkFlashIDS()
        if 0 < len(self.detect_target) <= 5:
            self.__mergeID()
            self.__splitID()
        #     self.tracker.max_age = 10
            
        # else:
        #     self.tracker.max_age = 50
            
        for det in self.detect_target:
            if len(det) < 5 or None in det[4]:#center not None
                continue
            id = det[3]
            center_x, center_y = det[4]
            w, h = det[2][2:]
            countInAreaPolygon = Polygon(self.countInArea_cal)
            countOutAreaPolygon = Polygon(self.countOutArea)
            currentCentroid = Point((center_x, center_y))
            
            if self.lastCentroids.get(id, None) is None:#Don't have this id in last frame
                
                countIn = False
                countOut = False
                outIn = False
                
                if currentCentroid.within(countInAreaPolygon):#inside count in area that mean only can count in
                    countIn = True
                elif currentCentroid.within(countOutAreaPolygon):#inside count out area but not count in area
                    outIn = True
                
                self.lastCentroids[id] = {"center":(center_x, center_y),#update id's center
                                          "wh":(w, h),
                                          "countIn":countIn,#set id not counted
                                          "countOut":countOut,
                                          "outIn":outIn
                                          }
                
                continue
            
            # if self.lastCentroids[id]["counted"]:#already counted
            #     continue
            
            if center_x <= 0 or center_x >= self.W or center_y <= 0 or center_y >= self.H:#out of boundary
                continue
            
            lastCentroid = self.lastCentroids[id]["center"]
            lastCentroid = Point(lastCentroid)

            
            # inSquareWhenAppear = lastCentroid.within(countInAreaPolygon) and currentCentroid.within(countInAreaPolygon)
            # if inSquareWhenAppear:#The last position and current position are in the square but not counted. That means people show up in the square at the beginning.
            #     #mark the people counted but not plus 1
            #     count = True
            
            # if the last centroid not in square and current centroid in square and non-counted
            # that mean the person get into the square from outside.
            # So count it
            isGetIn = not lastCentroid.within(countInAreaPolygon) and currentCentroid.within(countInAreaPolygon) and not self.lastCentroids[id]["countIn"]
            #last centroid in square and current centroid not in square
            isGetOut = lastCentroid.within(countOutAreaPolygon) and not currentCentroid.within(countOutAreaPolygon) and not self.lastCentroids[id]["countOut"]
            
            if isGetIn:#get in and not counted
                # if self.mergedIDs.get(id, None) is not None:#this id represent multi ids
                #     print("Normal merge add:", self.mergedIDs[id])
                #     lenOfID = len(self.mergedIDs[id])
                #     self.totalIn += lenOfID
                #     self.currentIn += lenOfID
                #     for i in self.mergedIDs[id]:
                #         self.lastCentroids[i]["countIn"] = True
                # else:
                print("Normal add:", id)
                self.totalIn += 1
                self.currentIn += 1
                self.lastCentroids[id]["countIn"] = True
                # self.lastCentroids[id]["countOut"] = False
                
                
            if not self.lastCentroids[id]["countIn"] and self.lastCentroids[id]["countOut"] and not self.lastCentroids[id]["outIn"]:
                print("Out and in", id)
                # self.totalIn += 1
                self.currentIn += 1
                self.lastCentroids[id]["outIn"] = True
                
                
            if isGetOut:
                if self.mergedIDs.get(id, None) is not None:
                    print("Normal merge out:", id, self.mergedIDs[id])
                    for i in self.mergedIDs[id]:
                        if not self.lastCentroids[i]["countOut"]:#id not count out
                            self.lastCentroids[i]["countOut"] = True
                            self.currentIn -= 1
                            # print(f"id={i}, countOut={self.lastCentroids[i]}")
                        
                else:
                    print("Normal out:", id)
                    self.currentIn -= 1
                    # self.lastCentroids[id]["countIn"] = False
                    self.lastCentroids[id]["countOut"] = True
                
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
        
        self.IDSwith["frame"] += 1
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
            # suspiciousAreaPolygon = Polygon(self.suspiciousArea)
            suspiciousAreaPolygon = Polygon(self.suspiciousArea_L)
            
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
                                          "countOut":False
                                          }#set id not counted
                
                    ############################
                    #ID switch happening
                    ############################
                    if self.IDSwith.get("frame", 10) < 5 and self.IDSwith.get("amount", 0) > 0:
                        self.totalIn -= 1
                        self.currentIn -= 1
                        self.IDSwith["amount"] -= 1
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
            self.IDSwith["frame"] = 0
            self.IDSwith["amount"] += len(self.IDsInLastSuspiciousArea) - len(IDsInCurrentSuspiciousArea)
            
            
        
        ############################
        #object flash appearing
        #There has been some error detection of Yolo just flashing on the screen
        #when there have a lot of people. So just keep tracking the object.
        ############################
        # suddenly appear id
        self.flashID = []
        TRACK_FRAMES = 10  # const for amount of frames to track
        COUNTED_THRESHOLD = 8
        mode = "counted"  # ["counted", "continuous"]
        for old_ID in list(self.suspiciousAreaIDTracker.keys()):
            if self.suspiciousAreaIDTracker[old_ID]["tracked"] > TRACK_FRAMES:#checked
                continue
            
            
            if old_ID in IDsInThisFrame:#if id is in this frame
                # add counter and keep cont status if already not continuous
                old_ID_dict = self.suspiciousAreaIDTracker[old_ID]
                self.suspiciousAreaIDTracker[old_ID] = {"tracked": old_ID_dict["tracked"]+1, "counted": old_ID_dict["counted"]+1, "continuous": True if old_ID_dict["continuous"] else False}
                # print(old_ID, self.suspiciousAreaIDTracker[old_ID])
                # print(f"IDsInCurrentSuspiciousArea = {IDsInCurrentSuspiciousArea}")
                
            else:
                self.suspiciousAreaIDTracker[old_ID]["tracked"] += 1
                self.suspiciousAreaIDTracker[old_ID]["continuous"] = False
                
            if self.suspiciousAreaIDTracker[old_ID]["tracked"] == TRACK_FRAMES:
                if mode == "counted":
                    if self.suspiciousAreaIDTracker[old_ID]["counted"] < COUNTED_THRESHOLD:  # id appeared not enough times
                        print("Remove", old_ID, self.suspiciousAreaIDTracker[old_ID])
                        for i in self.mergedIDs:#remove flash id from merged id
                            if old_ID in self.mergedIDs[i]:
                                i.remove(old_ID)
                                
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

    def __mergeID(self):
        if len(self.detect_target) < len(self.lastDetections):#number of people in this frame is less than last frame. May be two person's merged
            #find disappear person
            thisFrameDetections = {det[3]:det[2] for det in self.detect_target}#{id:center}
            lastFrameDetections = {det[3]:det[2] for det in self.lastDetections}#{id:center}
            thisFrameIDS = set(thisFrameDetections.keys())
            lastFrameIDS = set(lastFrameDetections.keys())
            disappearIDS = lastFrameIDS.difference(thisFrameIDS)
            mergeDistanceThreshold = 80
            # print("disappear id", disappearIDS)
            mergeArea = Polygon(self.mergeIDArea)
            for i in disappearIDS:
                x1, y1 = lastFrameDetections[i][:2]
                id1 = Point((x1, y1))
                for j in lastFrameIDS:
                    if i == j:#same id
                        continue
                    x2, y2 = lastFrameDetections[j][:2]
                    id2 = Point((x2, y2))
                    
                    distance = ( (x1-x2)**2 + (y1-y2)**2 )**0.5
                    pointInArea = id1.within(mergeArea) and id2.within(mergeArea)
                    if distance < mergeDistanceThreshold and pointInArea:#disappear person is very close to this person and in merge id area. ID merged
                        if self.mergedIDs.get(j, None) is None:
                            self.mergedIDs[j] = set([j,i])
                        else:#already merged other ID
                            self.mergedIDs[j].add(i)
                        # print("ID merged:", self.mergedIDs)

    def __splitID(self):
        if len(self.detect_target) > len(self.lastDetections) and len(self.lastDetections) != 0:#number of people in this frame is more than last frame. May be two person's merged
            #find new person
            thisFrameDetections = {det[3]:det[2] for det in self.detect_target}#{id:center}
            lastFrameDetections = {det[3]:det[2] for det in self.lastDetections}#{id:center}
            thisFrameIDS = set(thisFrameDetections.keys())
            lastFrameIDS = set(lastFrameDetections.keys())
            newIDS = thisFrameIDS.difference(lastFrameIDS)
            mergeDistanceThreshold = 50
            
            thisFrameIDsList = [det[3] for det in self.detect_target]
            # print("disappear id", disappearIDS)
            for i in newIDS:
                x1, y1 = thisFrameDetections[i][:2]
                for j in thisFrameIDS:
                    if i == j:#same id
                        continue
                    x2, y2 = thisFrameDetections[j][:2]
                    distance = ( (x1-x2)**2 + (y1-y2)**2 )**0.5
                    
                    # spiltID = list(self.mergedIDs[j])[1]
    
                    # canSpilt = distance < mergeDistanceThreshold and self.mergedIDs.get(j, None) is not None and len(self.mergedIDs[j]) > 1 and spiltID != j
                    if distance < mergeDistanceThreshold:#disappear person is very close to this person. ID merged
                        if self.mergedIDs.get(j, None) is not None and len(self.mergedIDs[j]) > 1:#new id split from here
                            spiltID = list(self.mergedIDs[j])[1]
                            if spiltID == j:#same id
                                continue
                    # if canSpilt:
                            self.mergedIDs[j].remove(spiltID)#remove spilt id from set
                            splitIDIndex = thisFrameIDsList.index(i)#find the new id's index of this frame
                            self.detect_target[splitIDIndex][3] = spiltID#update this frame new id to split id
                            print(f"split ID {spiltID} from {j}, {self.mergedIDs[j]}")


        #split flash id
        for ID in self.mergedIDs:
            overlapID = self.mergedIDs[ID].intersection(set(self.flashID))#if merged IDS have flash ID
            if len(overlapID) != 0:
                for removeID in overlapID:#remove flash ID ine by one
                    if removeID == ID:#don't remove itself
                        continue
                    self.mergedIDs[ID].remove(removeID)#remove spilt id from set
                    
                    print(f"split flash ID {removeID} from {ID}, {self.mergedIDs[ID]}")
        # self.lastDetections = self.detect_target
    
    def __checkFlashIDS(self):
        '''
        input yolo detections
        check flashIDs with few frames
        '''
        self.flashID = list()
        #get IDS in this frame
        IDSThisFrame = [det[3] for det in self.detect_target]
        newIDs = set(IDSThisFrame).difference(set(self.AllIDtracker.keys()))
        
        TRACK_FRAMES = 15  # const for amount of frames to track
        COUNTED_THRESHOLD = 6
        for ID in list(self.AllIDtracker):#use list to copy the dict because will remove element in loop
            if ID in IDSThisFrame:#ID in this frame
                # if self.AllIDtracker.get(ID, None) is None :#new ID
                #     self.AllIDtracker[ID] = {"tracked": 1, "counted": 1, "continuous": True}
                self.AllIDtracker[ID]["tracked"] += 1
            self.AllIDtracker[ID]["counted"] += 1
            
            if self.AllIDtracker[ID]["counted"] >= TRACK_FRAMES:
                if self.AllIDtracker[ID]["tracked"] < COUNTED_THRESHOLD:#flash id
                    self.flashID.append(ID)
                    
                del self.AllIDtracker[ID]#del ID



        for new_id in newIDs:#add new id
            self.AllIDtracker[new_id] = {"tracked": 1, "counted": 1, "continuous": True}#continuous not using
                
        # if len(self.flashID) > 0:
        #     print("flash id:", self.flashID)
    
    
    def socialDistance(self, image):
        closePairs = []

        centroids = [det[4] for det in self.detect_target if len(det)>=5 and det[4] != [None, None]]#get 4th element which is (pose_center_x, pose_center_y)
        # centroids = [det[2][:2] for det in self.detect_target]#get 2rd element which is (center_x, center_y)
        
        if len(centroids) < 2:#less than two people then no need to calculate social distance
            return image
            
        
        centroids = np.array(centroids)#change it to array type
        
        transformedCentroids = cv2.perspectiveTransform(np.array([centroids]), self.M)#perspective transform yolo center
        transformedCentroids = np.reshape(transformedCentroids, (-1, 2))#reshape to a list of point
        
        insidePointIndex = (transformedCentroids[:,0] >= 0) & (transformedCentroids[:,0] <= self.transformImageWidth) & (transformedCentroids[:,1] >= 0) & (transformedCentroids[:,1] <= self.transformImageHeight)
        # print("in:",insidePointIndex.shape)
        # print("tran:", transformedCentroids.shape)
        transformedCentroids = transformedCentroids[insidePointIndex]#remove the transformed point outside the square
        
        # print("cen:", centroids.shape)
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
                    
                    cv2.putText(image, message, (pairCenterX, pairCenterY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)#draw distance below the line
                    
                    #draw web streaming image
                    self.socialDistanceImage = np.zeros((image.shape[0], image.shape[1], 4))
                    cv2.line(self.socialDistanceImage, (int(centroids[index][0]), int(centroids[index][1]) ), (int(centroids[pointIndex][0]), int(centroids[pointIndex][1]) ), (255,0,0, 255), 2)
                    cv2.putText(self.socialDistanceImage, message, (pairCenterX, pairCenterY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 255), 2)#draw distance below the line

        return image
    
    #https://google.github.io/mediapipe/solutions/face_detection.html
    def face_detection(self, detectImage, drawImage):
        #only use yolo center
        # for index, detection in enumerate(self.detect_target):#for each detection
        #     self.detect_target[index] = list(self.detect_target[index])
        #     left, top, right, bottom = darknet.bbox2points(detection[2])
        #     imageWidth, imageHeight = right - left, bottom - top
        #     centerX, centerY = (left + imageWidth/2), (top + imageHeight)#use the yolo bbox info to define center
        #     self.detect_target[index].append((centerX, centerY))
        #     cv2.circle(drawImage, (int(centerX), int(centerY)), 8, (255,0,0), -1)
        
        # return drawImage
        
        
        with self.mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
            for index, detection in enumerate(self.detect_target):#for each detection
            
                self.detect_target[index] = list(self.detect_target[index])
                left, top, right, bottom = darknet.bbox2points(detection[2])
                # left, top, right, bottom = left-50, top-50, right+50, bottom+50
            
                closestHeadIndex = None
                minDistance = right + bottom#you can define any value here
                imageWidth, imageHeight = right - left, bottom - top
                head_x, head_y = left + imageWidth/2, top + imageHeight / 7# self define head ideal location
                
                boundaryError = left < 0 or right > self.W or top < 0 or bottom > self.H
                if boundaryError:
                    centerX, centerY = (left + imageWidth/2), (top + imageHeight)#use the yolo bbox info to define center
                    self.detect_target[index].append((centerX, centerY))
                    cv2.circle(drawImage, (int(centerX), int(centerY)), 8, (255,0,0), -1)
                    continue
                
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = face_detection.process(detectImage[top:bottom, left:right])
                
                
                if not results.detections:#No face detected
                    centerX, centerY = (left + imageWidth/2), (top + imageHeight)#use the yolo bbox info to define center
                    self.detect_target[index].append((centerX, centerY))
                    cv2.circle(drawImage, (int(centerX), int(centerY)), 8, (0,255,0), -1)
                    continue
                
                for bboxIndex, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    xmin, ymin, width, height = bbox.xmin * imageWidth, bbox.ymin * imageHeight, bbox.width * imageWidth, bbox.height * imageHeight
                    centerX, centerY = xmin + width/2, ymin + height/2
                    distance = ( (head_x - centerX)**2 + (head_y - centerY)**2 )**0.5
                    if distance < minDistance:
                        minDistance = distance
                        closestHeadIndex = bboxIndex
                        
                # self.mp_drawing.draw_detection(drawImage[top:bottom, left:right], results.detections[closestHeadIndex])# draw the closest head
                
                bbox = results.detections[closestHeadIndex].location_data.relative_bounding_box
                xmin, ymin, width, height = bbox.xmin * imageWidth, bbox.ymin * imageHeight, bbox.width * imageWidth, bbox.height * imageHeight
                centerX, centerY = (left + xmin + width/2), bottom
                self.detect_target[index].append((centerX, centerY))
                try:
                    cv2.circle(drawImage, (int(centerX), int(centerY)), 8, (0,255,0), -1)
                
                except:
                    print("Draw image shape",drawImage.shape)
                    # print(drawImage)
                    #print(type(drawImage))
                
        return drawImage
    
    def object_tracker_BYTE(self, image):
        #[cx, cy, W, H, score] -> [x1, y1, x2, y2, score]
        dets = list()
        for det in self.detect_target:
            score = int(float(det[1]))
            cx, cy, W, H = det[2]
            x1 = int(cx - W / 2)
            y1 = int(cy - H / 2)
            x2 = x1 + W
            y2 = y1 + H
            dets.append([x1, y1, x2, y2, score])
        
        dets = np.array(dets)
        online_targets = self.tracker.update(dets, [1080, 1920], [800, 1440])
        detWithID = []
        for track in online_targets:
            t, l, w, h = track.tlwh
            id = track.track_id
            
            cx = int( (t + w / 2) / 800 * 1080 )
            cy = int( (l + h / 2) / 1440 * 1920 )
            # assign each id with a color
            if self.bbox_colors.get(id) == None:
                self.bbox_colors[id] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
            cv2.rectangle(image, (int(t), int(l)), (int(t + w), int(l + h)), self.bbox_colors[id], 3)
            cv2.putText(image, str(id), (cx, cy - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0,255,0), thickness=2)
            
            # save results
            detWithID.append(['person', track.score, [cx, cy, w, h], id])

        self.detect_target = detWithID
        return image
        
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
        # self.FPS.stop()
        
        self.write_log("[Info] Stop the program.")
        self.cap.release()
        # print(f"[Info] Avg FPS:{self.FPS.fps()}.")
            
        print('[Info] Stop the program: Group:{group}, alias:{alias}, URL:{url}'\
              .format(group=self.group, alias=self.alias, url=self.video_url))    
        
        #save detections info to json file
        with open(self.output_dir_json, "w") as outfile:
            json.dump(self.IDInfo, outfile)      
        
        
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
            self.frame_draw = cv2.VideoWriter(video_path_name, self.fourcc, 20.0, (self.W, self.H))
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

    def test(self, video, tracker="sortOH"):#track not used
        if os.path.isfile(video):
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.__testWithFile(video)
        else:#folder
            self.__testWithFolder(video)

    def __testWithFolder(self, videoFolder):
        shutil.rmtree(self.output_dir, ignore_errors=True)
        os.mkdir(self.output_dir)
        
        videoFiles = os.listdir(videoFolder)
        groundTruth = []#number of people get in square
        predictNumber = []
        error = []
        # print(sorted(videoFiles))
        print("Testing")
        errorInfo = dict()
        
        for video in sorted(videoFiles):
            print(f"Test {video}")
            self.alias = video.split(".")[0]
            videoPath = os.path.join(videoFolder, video)
            self.cap = cv2.VideoCapture(videoPath)
            FPS = self.cap.get(cv2.CAP_PROP_FPS)
            # print(FPS)
            # self.tracker = CentroidTracker(max_lost=30, tracker_output_format='mot_challenge', )#reset tracker
            # self.tracker = CentroidKF_Tracker(max_lost=30, tracker_output_format='mot_challenge', centroid_distance_threshold=50)#yolov4 seeting
            # self.tracker = SORT(max_lost=30, tracker_output_format='mot_challenge', iou_threshold=0.5)
            # self.tracker = IOUTracker(max_lost=20, iou_threshold=0.3, min_detection_confidence=0.2, max_detection_confidence=0.7, tracker_output_format='mot_challenge')
            
            #BYTE Tracker
            args = Arg()
            self.tracker = BYTETracker(args)
            print(f"thresh = {self.thresh} BYTE:age={30}")
            
            self.totalIn = 0#reset counter
            self.currentIn = 0
            self.lastCentroids = dict()#reset people counting info
            self.suspiciousAreaIDTracker = dict()
            
            
            GTNumberIn = int(video.split('.')[0].split('__')[-2])#get the ground truth number of the video
            GTNumberCurrent = int(video.split('.')[0].split('__')[-1])
            groundTruth.append(GTNumberIn)
            self.video_output_draw_name = os.path.join(self.output_dir, self.alias + '_output_draw.mp4')         
            
            
            self.start()#start video predict
            
            predictNumber.append(self.totalIn)
            
            if self.totalIn != GTNumberIn:
                errorInfo[video] = {
                    "GT_TotalIn":GTNumberIn,
                    "Pre__TotalIn":self.totalIn,
                    "GT_Current":GTNumberCurrent,
                    "Pred_Current":self.currentIn
                }
                error.append(abs(GTNumberIn - self.totalIn))
            
            # print(groundTruth)
            # print(predictNumber)
            
            msg = ('-----------------------------------------\n'
                    f'Ground truth:{sum(groundTruth)}\n'
                    f'Predict:{sum(predictNumber)}\n'
                    f'Error:{sum(error)/sum(groundTruth)}\n'
                    '-----------------------------------------'
            )
                    
            print(msg)
            print(errorInfo)
        
        
    def __testWithFile(self, video):
        
        groundTruth = []#number of people get in square
        predictNumber = []
        error = []
        errorInfo = dict()
        
        print(f"Test {video}")
        self.alias = video.split("/")[-1].split(".")[0]
        self.cap = cv2.VideoCapture(video)
        FPS = self.cap.get(cv2.CAP_PROP_FPS)
        # print(FPS)
        args = Arg()
        self.tracker = BYTETracker(args)
        print(f"thresh = {self.thresh} BYTE:age={30}")
        
        
        self.totalIn = 0#reset counter
        self.currentIn = 0
        self.lastCentroids = dict()#reset people counting info
        self.suspiciousAreaIDTracker = dict()
        
        
        GTNumberIn = int(video.split('.')[0].split('__')[-2])#get the ground truth number of the video
        GTNumberCurrent = int(video.split('.')[0].split('__')[-1])
        groundTruth.append(GTNumberIn)
        self.video_output_draw_name = os.path.join(self.output_dir, self.alias + '_output_draw.mp4')         
        
        
        self.start()#start video predict
        
        predictNumber.append(self.totalIn)
        
        
        errorInfo[video] = {
            "GT_TotalIn":GTNumberIn,
            "Pre__TotalIn":self.totalIn,
            "GT_Current":GTNumberCurrent,
            "Pred_Current":self.currentIn
        }
        error.append(abs(GTNumberIn - self.totalIn))
        
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
        
                    
            