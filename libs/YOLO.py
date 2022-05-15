from configparser import NoOptionError
from email.errors import BoundaryError
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

#people counting
from shapely.geometry import Point, Polygon

#FPS
from imutils.video import FPS

#object tracking with rgb image
import dlib


# motpy
from motpy.testing_viz import draw_detection, draw_track
from motpy import Detection, MultiObjectTracker

# multiobjecttracker
sys.path.insert(1, 'multi-object-tracker')
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks

# SmartFence  lib
from libs.utils import *

from darknet import darknet




class YoloDevice:
    def __init__(self, video_url="", output_dir="", run=True, auto_restart=False, repeat=False, obj_trace = False,
                 display_message=True, data_file="", config_file="", weights_file="", 
                 names_file="", thresh=0.5, vertex=None, target_classes=None, draw_bbox=True, draw_polygon=True, draw_square=True,
                 draw_socialDistanceArea=False, draw_socialDistanceInfo=False,  social_distance=False, draw_pose=False, count_people=False, draw_peopleCounting=False,
                 alias="", group="", place="", cam_info="", warning_level=None, is_threading=True, skip_frame=None,
                 schedule=[], save_img=True, save_original_img=False, save_video=False, save_video_original=False, testMode=False):
        self.video_url = video_url
        self.ourput_dir = output_dir
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
        
        
        #load model
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
        
        # Object Tracking
        self.id_storage = [] # save the trace id
        self.tracker_motpy = MultiObjectTracker(
                    dt=1 / 30,
                    tracker_kwargs={'max_staleness': 5},
                    model_spec={'order_pos': 1, 'dim_pos': 2,
                                'order_size': 0, 'dim_size': 2,
                                'q_var_pos': 5000., 'r_var_pos': 0.1},
#                     matching_fn_kwargs={'min_iou': 0.25,
#                                     'multi_match_min_iou': 0.93}
                    )     
        # self.tracker = CentroidTracker(max_lost=7, tracker_output_format='mot_challenge', )
        self.tracker = CentroidKF_Tracker(max_lost=10, tracker_output_format='mot_challenge', centroid_distance_threshold=30)
#         self.tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.1)
        # self.tracker = IOUTracker(max_lost=3, iou_threshold=0.1, min_detection_confidence=0.4, max_detection_confidence=0.7, tracker_output_format='mot_challenge')
        self.bbox_colors = {}
        
        # Video initilize
        self.frame = np.zeros((1080,1920,4))
        self.drawImage = None
        self.cap = cv2.VideoCapture(self.video_url)        
        self.ret = False
        self.H = int(self.cap.get(3))
        self.W = int(self.cap.get(4))      
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')              
        self.frame_id = 0
        self.obj_id = None
        self.retry = 0 # reconntecting counts of the video capture
        
        
        #web streaming image
        self.bboxImage = np.zeros((1080,1920,4))
        self.socialDistanceImage = np.zeros((1080,1920,4))
        self.infoImage = np.zeros((1080,1920,4))
        
        #people counting
        self.totalIn = 0#count how many people get in the area
        self.draw_square = draw_square
        self.squareArea = np.array([ [0, 1080],[0, 762],[547, 247],[1004, 260],[1160, 381],[1135, 588],[1066, 587],[1417, 990],[1533, 972],[1612, 645],[1920, 800],[1920, 1080] ], np.int32)#The polygon of the area you want to count people inout
        self.lastCentroids = dict()
        
        #social distance
        self.socialDistanceArea = np.array([ [378, 1080],[585, 345],[939, 339],[1590, 1080] ], np.float32)
        # self.realHeight, self.realWidth = 15.75, 5.6#m
        self.realHeight, self.realWidth = 19.97, 5.6#m
        self.transformImageHeight, self.transformImageWidth = 1000, 350
        tranfromPoints = np.array([[0, self.transformImageHeight], [0, 0], [self.transformImageWidth, 0], [self.transformImageWidth, self.transformImageHeight]], np.float32) # 这是变换之后的图上四个点的位置
        self.social_distance_limit = 1#1m
        self.draw_socialDistanceInfo = draw_socialDistanceInfo
        
        # get transform matrix
        self.M = cv2.getPerspectiveTransform(self.socialDistanceArea, tranfromPoints)
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
        
        
        #dlib rgb tracker
        self.RGBTrackers = []
        
        #fps calculate
        self.FPS = FPS()
        
        
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
        
        self.frame_original = cv2.VideoWriter(self.video_output_name, self.fourcc, 20.0, (self.H, self.W))
        self.frame_draw = cv2.VideoWriter(self.video_output_draw_name, self.fourcc, 20.0, (self.H, self.W))  
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
            self.FPS.start()
            self.prediction()
            return
        
        if self.is_threading:
            self.th.append(threading.Thread(target = self.video_capture))
            self.th.append(threading.Thread(target = self.prediction))
        else:
            self.th.append(threading.Thread(target = self.prediction))
        
        for t in self.th:
            self.FPS.start()
            t.start()

    
    def video_capture_wo_threading(self): 
        
        
        self.ret, self.frame = self.cap.read() 
        
            
        if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong
            
            if self.repeat:
                print(f"restart detection {datetime.datetime.now()}")
                self.cap = cv2.VideoCapture(self.video_url)#reread the video
                self.totalIn = 0
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
            # Get the latest pictures
            allcount = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,allcount)
            
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
            
            if self.skip_frame != None and self.frame_id%self.skip_frame != 0:#skip frame
                self.frame_id += 1
                continue
            
            
                
            #do yolo prediction and tracking
            self.RGBTrackers = []
            darknet_image = darknet.make_image(self.H, self.W, 3)
            darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
            
            predict_time = time.time() # get start predict time
            detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
            predict_time_sum +=  (time.time() - predict_time) # add sum predict time
            
#             darknet.print_detections(detections, True) # print detection
            darknet.free_image(darknet_image)
    
            # filter the scope and target class   
            self.detect_target = detect_filter(detections, self.target_classes, self.vertex)  
            # print(self.detect_target, "\n"*5)
            
            for index, det in enumerate(self.detect_target):
                self.detect_target[index] = list(self.detect_target[index])#change all detection to list
                (x1, y1, w, h) = det[2]#get bbox
                x2 = x1 + w
                y2 = y1 + h

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                tracker.start_track(frame_rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                self.RGBTrackers.append(tracker)
            
            #use dlib to track object not yolo detection
            # else:
            #     for index, tracker in enumerate(self.RGBTrackers):
	

            #         # update the tracker and grab the updated position
            #         tracker.update(frame_rgb)
            #         pos = tracker.get_position()

            #         # unpack the position object
            #         x1 = int(pos.left())
            #         y1 = int(pos.top())
            #         x2 = int(pos.right())
            #         y2 = int(pos.bottom())

            #         # add the bounding box coordinates to the rectangles list
            #         w = x2 - x1
            #         h = y2 - y1
            #         self.detect_target[index][2] = (x1, y1, w, h)
                    
            
            # convert to BGR image
            self.drawImage = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)            
            
            save_path_img = None
            save_path_img_orig = None
            save_video_draw_path = None
            
            
            
            if self.draw_polygon: 
                self.drawImage = draw_polylines(self.drawImage, self.vertex)  # draw the polygon
                
            if self.draw_square:
                cv2.polylines(self.drawImage, pts=[self.squareArea], isClosed=True, color=(0,0,255), thickness=3)#draw square area
                
            if self.draw_socialDistanceArea:
                socialDistanceArea_int = np.array(self.socialDistanceArea, np.int32)
                cv2.polylines(self.drawImage, pts=[socialDistanceArea_int], isClosed=True, color=(0,255,255), thickness=3)#draw square area
            
            
            if self.obj_trace: # draw the image with object tracking
#                 image = self.object_tracker(image)             
                self.drawImage = self.object_tracker(frame_rgb)    
            elif self.draw_bbox:
                self.drawImage = draw_boxes(detections, self.drawImage, self.class_colors, self.target_classes, self.vertex)
                
            
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
            
            self.FPS.update()
            
            
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
        ("Visitors", self.totalIn),
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
            cv2.putText(image, text, (10, self.W - ((i * 20) + 100)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 5)
            cv2.putText(self.infoImage, text, (10, self.W - ((i * 20) + 100)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255, 255), 5)
            

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(image, text, (265, self.W - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(self.infoImage, text, (265, self.W - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255, 255), 2)
            
            
        return image
    
    
    def people_counting(self):
        
        squareAreaPolygon = Polygon(self.squareArea)
        for det in self.detect_target:
            count = False
            id = det[3]
            if len(det) < 5 or None in det[4]:#center not None
                continue
            
            
            
            center_x, center_y = det[4]
            
            if(self.lastCentroids.get(id, None) == None):#Don't have this id in last frame
                self.lastCentroids[id] = {"center":(center_x, center_y),#update id's center
                                          "counted":False}#set id not counted
                continue
            
            if self.lastCentroids[id]["counted"]:#already counted
                continue
            
            lastCentroid = self.lastCentroids[id]["center"]
            
            # print(id, lastCentroid, (center_x, center_y))
            lastCentroid = Point(lastCentroid)
            currentCentorid = Point((center_x, center_y))
            
            # if the last centroid not in square and current centroid in square 
            # that mean the person get into the square from outside.
            # So count it
            if(not lastCentroid.within(squareAreaPolygon) and currentCentorid.within(squareAreaPolygon) and not self.lastCentroids[id]["counted"]):
                self.totalIn += 1
                count = True
                
            self.lastCentroids[id] = {"center":(center_x, center_y),#update id's center
                                          "counted":count}#set id's count
    
    
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
            
            # cv2.circle(image, (int(centroid[0]), int(centroid[1])), 3, (255,255,255), -1)
            
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
                    cv2.circle(drawImage, (int(centerX), int(centerY)), 8, (255,0,0), -1)
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
                    cv2.circle(drawImage, (int(centerX), int(centerY)), 8, (255,0,0), -1)
                
                except:
                    print("Draw image shape",drawImage.shape)
                    print(drawImage)
                    #print(type(drawImage))
                
        return drawImage
    
    # https://github.com/wmuron/motpy.git
    def object_tracker_motpy(self, image):           
        boxes = []
        scores = []
        class_ids = []
        
        # convert to the trace format
        for r in self.detect_target:
            boxes.append( darknet.bbox2points(r[2]) )
            scores.append( float(r[1]) )
            class_ids.append(r[0])    
            
        self.tracker_motpy.step(detections=[Detection(box=b, score=s, class_id=l) for b, s, l in zip(boxes, scores, class_ids)])
        tracks = self.tracker_motpy.active_tracks(min_steps_alive=3)
        
        self.detect_target = [] # re-assigned each bbox
        for track in tracks:
            # append the track.id to id_storage
            if track.id not in self.id_storage:
                self.id_storage.append(track.id)
                
            id_index = self.id_storage.index(track.id) #  the order of elements in the python list is persistent                
            self.detect_target.append((track.class_id, track.score, track.box, id_index)) # put the result to detect_target            
            draw_track(image, track, thickness=2, text_at_bottom=True, text_verbose=0) # draw the bbox
            
             # put the id to the image
            txt = track.class_id + " "+ str(track.score) +" ID=" + str(id_index)
            cv2.putText(image, txt, (int(track.box[0]), int(track.box[1])-7) ,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255,255,0))
            
        return image
    
    
    # https://github.com/adipandas/multi-object-tracker.git
    def object_tracker(self, image):
        boxes = []
        confidence = []
        class_ids = []
        
        self.bboxImage = np.zeros((image.shape[0], image.shape[1], 4))
        
        # convert to the trace format
        for r in self.detect_target:
            center_x, center_y, width, height = r[2]
            left, top, right, bottom = darknet.bbox2points(r[2])
            boxes.append([left, top, width, height])
            confidence.append(int(float(r[1])))
            class_ids.append(int(self.target_classes.index(r[0])))
#             cv2.rectangle(image, (int(left), int(top)), (int(left+width), int(top+height)), (0,0,255), 2) # draw the bbox   
        output_tracks = self.tracker.update(np.array(boxes), np.array(confidence), np.array(class_ids))
        
        self.detect_target = [] # re-assigned each bbox
        for track in output_tracks:
            frame, idx, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
            assert len(track) == 10
#             print(track)
            bbox = (bb_left+bb_width/2, bb_top+bb_height/2,  bb_width, bb_height)
            self.detect_target.append((self.target_classes[0], confidence, bbox, idx)) # put the result to detect_target 
                        
            # assigen each id for a color
            if self.bbox_colors.get(idx) == None:
                self.bbox_colors[idx] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
                
            if self.draw_bbox:
                cv2.rectangle(image, (int(bb_left), int(bb_top)), (int(bb_left+bb_width), int(bb_top+bb_height)), self.bbox_colors[idx], 2) # draw the bbox
                #draw web streaming image
                cv2.rectangle(self.bboxImage, (int(bb_left), int(bb_top)), (int(bb_left+bb_width), int(bb_top+bb_height)), (*self.bbox_colors[idx], 255), 2) # draw the bbox
                
                
            image = draw_tracks(image, output_tracks) # draw the id
            
            # put the score and class to the image
            txt = str(r[0]) + " "+ str(confidence)
            cv2.putText(image, txt, (int(bb_left), int(bb_top-7)) ,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=self.bbox_colors[idx])
            
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
        self.FPS.stop()
        
        self.write_log("[Info] Stop the program.")
        self.cap.release()
        print(f"[Info] Avg FPS:{self.FPS.fps()}.")
            
        print('[Info] Stop the program: Group:{group}, alias:{alias}, URL:{url}'\
              .format(group=self.group, alias=self.alias, url=self.video_url))      
        
        
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
            self.frame_original = cv2.VideoWriter(video_path_name, self.fourcc, 30.0, (self.H, self.W))
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
            self.frame_draw = cv2.VideoWriter(video_path_name, self.fourcc, 20.0, (self.H, self.W))
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

    def test(self, videoFolder):
        
        shutil.rmtree(self.ourput_dir, ignore_errors=True)
        os.mkdir(self.ourput_dir)
        videoFiles = os.listdir(videoFolder)
        groundTruth = []#number of people get in square
        predictNumber = []
        error = []
        # print(sorted(videoFiles))
        videoFiles = sorted(videoFiles)[1:]#remove first video which is demo video. The file name format don't math (split('.')[0].split('_')[1])
        print("Testing")
        errorInfo = dict()
        
        for video in sorted(videoFiles):
            print(f"Test {video}")
            self.alias = video.split(".")[0]
            videoPath = os.path.join(videoFolder, video)
            self.cap = cv2.VideoCapture(videoPath)
            FPS = self.cap.get(cv2.CAP_PROP_FPS)
            # print(FPS)
            self.totalIn = 0#reset counter
            # self.tracker = CentroidTracker(max_lost=30, tracker_output_format='mot_challenge', )#reset tracker
            # self.tracker = CentroidKF_Tracker(max_lost=30, tracker_output_format='mot_challenge', centroid_distance_threshold=30)#yolov4 seeting
            # self.tracker = CentroidKF_Tracker(max_lost=30, tracker_output_format='mot_challenge', centroid_distance_threshold=60)
            # self.tracker = SORT(max_lost=30, tracker_output_format='mot_challenge', iou_threshold=0.5)
            self.tracker = IOUTracker(max_lost=24, iou_threshold=0.5, min_detection_confidence=0.2, max_detection_confidence=0.7, tracker_output_format='mot_challenge')
            
            self.lastCentroids = dict()#rest people counting info
            GTNumber = int(video.split('.')[0].split('_')[1])#get the ground truth number of the video
            groundTruth.append(GTNumber)
            self.video_output_draw_name = os.path.join(self.ourput_dir, self.alias + '_output_draw.mp4')         
            
            
            self.start()#start video predict
            
            predictNumber.append(self.totalIn)
            
            if self.totalIn != GTNumber:
                errorInfo[video] = {
                    "GT":GTNumber,
                    "predict":self.totalIn
                }
                error.append(abs(GTNumber - self.totalIn))
            
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
            
            frameCount = 0
            totalFrame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = tqdm(total=totalFrame)
            
            while self.cap.isOpened():#loop each frame
                ret, frame = self.cap.read()
                
                if ret == False:
                    self.cap.release()
                    break
            
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # original image
                darknet_image = darknet.make_image(self.W, self.H, 3)
                darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
                detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
                
                darknet.free_image(darknet_image)
    
                # filter the scope and target class   
                self.detect_target = detect_filter(detections, self.target_classes, self.vertex)
        
                
                if len(self.detect_target) >= 2:
                    # print(self.detect_target)
                    file = open(os.path.join(outputDir, video+f"_{frameCount}.txt"), 'w')
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
                    
                    cv2.imwrite(os.path.join(outputDir, video+f"_{frameCount}.jpg"), frame)
                
                    frameCount += 1
                
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
        
        
        
                    
            