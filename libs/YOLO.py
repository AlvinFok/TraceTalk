from configparser import NoOptionError
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

#people counting
from shapely.geometry import Point, Polygon

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
    def __init__(self, video_url="", output_dir="", run=True, auto_restart=False, obj_trace = False,
                 display_message=True, data_file="", config_file="", weights_file="", 
                 names_file="", thresh=0.5, vertex=None, target_classes=None, draw_bbox=True, draw_polygon=True,
                 draw_socialDistance=False, social_distance=False, draw_pose=False, count_people=False, draw_peopleCounting=False,
                 alias="", group="", place="", cam_info="", warning_level=None, is_threading=True, skip_frame=None,
                 schedule=[], save_img=True, save_original_img=False, save_video=False, save_video_original=False):
        self.video_url = video_url
        self.ourput_dir = output_dir
        self.run = run
        self.auto_restart = auto_restart
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
        
        
        self.social_distance = social_distance
        self.draw_socialDistance=draw_socialDistance
        self.draw_pose=draw_pose
        self.count_people=count_people
        
        
        # callback function
        self.detection_listener = None
        
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
        self.tracker = CentroidTracker(max_lost=3, tracker_output_format='mot_challenge')
#         self.tracker = CentroidKF_Tracker(max_lost=3, tracker_output_format='mot_challenge')
#         self.tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.1)
#         self.tracker = IOUTracker(max_lost=3, iou_threshold=0.1, min_detection_confidence=0.4, max_detection_confidence=0.7,
#                          tracker_output_format='mot_challenge')
        self.bbox_colors = {}
        
        # Video initilize
        self.frame = None
        self.cap = cv2.VideoCapture(self.video_url)        
        self.ret = False
        self.H = int(self.cap.get(3))
        self.W = int(self.cap.get(4))      
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')              
        self.frame_id = 0
        self.obj_id = None
        self.retry = 0 # reconntecting counts of the video capture
        
        #people counting
        self.totalIn = 0#count how many people get in the area
        self.squareArea = np.array([ [21, 1080],[0, 767],[555, 241],[960, 230],[1203, 382],[1143, 587],[1527, 970],[1623, 646],[1920, 793],[1920, 1080] ], np.int32)#The polygon of the area you want to count people inout
        self.lastCentroids = dict()
        
        #social distance
        self.socialDistanceArea = np.array([[ 369, 1071],[ 564,407],[ 981, 402],[1689, 1074]], np.float32)
        self.realHeight, self.realWidth = 15.75, 5.6#m
        self.transformImageHeight, self.transformImageWidth = 1000, 350
        tranfromPoints = np.array([[0, self.transformImageHeight], [0, 0], [self.transformImageWidth, 0], [self.transformImageWidth, self.transformImageHeight]], np.float32) # 这是变换之后的图上四个点的位置
        self.social_distance_limit = 1#1m
        
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
        
        if self.is_threading:
            self.th.append(threading.Thread(target = self.video_capture))
            self.th.append(threading.Thread(target = self.prediction))
        else:
            self.th.append(threading.Thread(target = self.prediction))
        
        for t in self.th:
            t.start()

    
    def video_capture_wo_threading(self): 
        self.ret, self.frame = self.cap.read() 
            
        if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong
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
        network, class_names, class_colors = darknet.load_network(
            config_file = self.config_file,
            data_file = self.data_file,
            weights = self.weights_file)
        
        last_time = time.time() # to compute the fps
        cnt = 0  # to compute the fps
        predict_time_sum = 0  # to compute the fps        
        t = threading.currentThread() # get this function threading status
        
        while getattr(t, "do_run", True):
            self.frame_id += 1
            cnt+=1 
            
            if not self.is_threading:
                self.video_capture_wo_threading()
                
                if self.skip_frame != None and self.frame_id%self.skip_frame != 0:
                    continue
                
            if not self.cap.isOpened() or not self.ret:                
                #print("[Info] Waiting for reconnecting...")
                time.sleep(1)
                continue
                
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) # original image
            darknet_image = darknet.make_image(self.H, self.W, 3)
            darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
            
            predict_time = time.time() # get start predict time
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=self.thresh)
            predict_time_sum +=  (time.time() - predict_time) # add sum predict time
            
#             darknet.print_detections(detections, True) # print detection
            darknet.free_image(darknet_image)
    
            # filter the scope and target class   
            self.detect_target = detect_filter(detections, self.target_classes, self.vertex)  
            # convert to BGR image
            image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)            
            
            save_path_img = None
            save_path_img_orig = None
            save_video_draw_path = None
            
            
            
            if self.draw_polygon: 
                image = draw_polylines(image, self.vertex)  # draw the polygon
                
            # image = draw_polylines(image, self.squareArea)#draw square area
            cv2.polylines(image, pts=[self.squareArea], isClosed=True, color=(0,0,255), thickness=3)#draw square area
            
            
            if self.obj_trace: # draw the image with object tracking
#                 image = self.object_tracker(image)             
                image = self.object_tracker(image)    
            else:
                image = draw_boxes(detections, image, class_colors, self.target_classes, self.vertex)
                
            
            # if self.draw_pose and len(self.detect_target) > 0:
            #     image = self.pose_estimation(image)
                
            image = self.face_detection(frame_rgb, image)
            if self.count_people and len(self.detect_target) > 0:
                self.people_counting()
                
            if self.social_distance and len(self.detect_target) > 0:
                image = self.socialDistance(image)
            
            # save draw bbox image
            if self.save_img and len(self.detect_target) > 0:                 
                save_path_img = self.save_img_draw(image)
                
            image = self.draw_info(image)
                            
            
            # save oiginal image
            if self.save_img_original and len(self.detect_target) > 0:
                save_path_img_orig = self.save_img_orig(self.frame)
            
            # save video with draw            
            if self.save_video:
                save_video_draw_path = self.save_video_draw(image)
            
            # callback function for user
            if len(self.detect_target) > 0:
                self.__trigger_callback(save_path_img, self.group, self.alias, self.detect_target)            
            
            # Compute FPS
            if time.time() - last_time  >= 5:
                self.print_msg("[Info] FPS:{fps} , {alias}".format(alias=self.alias, fps=cnt / (time.time()-last_time)))
                self.print_msg("[Info] Predict FPS:{fps} , {alias}".format(alias=self.alias, fps=cnt / predict_time_sum))
                last_time = time.time()
                cnt = 0
                predict_time_sum = 0
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
                
                self.detect_target[index] = list(self.detect_target[index])
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
        ("total lanmarks", self.detected_landmark),
        ("good lanmarks", self.good_landmark)
        ]

        info2 = [
        # ("Total people inside", x),
        ]

        # Display the output
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            # print(self.H)
            cv2.putText(image, text, (10, self.W - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(image, text, (265, self.W - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        return image
    
    
    def people_counting(self):
        currentCentorids = dict()
        squareAreaPolygon = Polygon(self.squareArea)
        for det in self.detect_target:
            id = det[3]
            if len(det) < 5 or None in det[4]:#center not None
                continue
            
            center_x, center_y = det[4]
            
            if(self.lastCentroids.get(id, None) == None):#Don't have this id in last frame
                currentCentorids[id] = (center_x, center_y)
                continue
            
            
            lastCentroid = self.lastCentroids[id]
            
            # print(id, lastCentroid, (center_x, center_y))
            lastCentroid = Point(lastCentroid)
            currentCentorid = Point((center_x, center_y))
            currentCentorids[id] = (center_x, center_y)
            # if the last centroid not in square and current centroid in square 
            # that mean the person get into the square from outside.
            # So count it
            if(not lastCentroid.within(squareAreaPolygon) and currentCentorid.within(squareAreaPolygon)):
                self.totalIn += 1
        self.lastCentroids = currentCentorids
    
    
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

            closePair = np.where( (x_Distance ** 2 + y_Distance**2) **0.5 < self.social_distance_limit )[0]
            for pointIndex in closePair:
                if(index == pointIndex):#itself
                    continue
                closePairs.append([int(centroids[index][0]), int(centroids[index][1]), int(centroids[pointIndex][0]), int(centroids[pointIndex][1])])#add not transform point to the list for drawing line in the image
                cv2.line(image, (int(centroids[index][0]), int(centroids[index][1]) ), (int(centroids[pointIndex][0]), int(centroids[pointIndex][1]) ), (255,0,0), 2)

        return image
    
    #https://google.github.io/mediapipe/solutions/face_detection.html
    def face_detection(self, detectImage, drawImage):
        
        with self.mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
            for index, detection in enumerate(self.detect_target):#for each detection
                self.detect_target[index] = list(self.detect_target[index])
                
                left, top, right, bottom = darknet.bbox2points(detection[2])
                # left, top, right, bottom = left-50, top-50, right+50, bottom+50
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = face_detection.process(detectImage[top:bottom, left:right])

                # print(results.detections[0].location_data.relative_bounding_box)
                closestHeadIndex = None
                minDistance = right + bottom#you can define any value here
                imageWidth, imageHeight = right - left, bottom - top
                head_x, head_y = left + imageWidth/2, top + imageHeight / 7# self define head ideal location
                
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
                        
                self.mp_drawing.draw_detection(drawImage[top:bottom, left:right], results.detections[closestHeadIndex])# draw the closest head
                bbox = results.detections[closestHeadIndex].location_data.relative_bounding_box
                xmin, ymin, width, height = bbox.xmin * imageWidth, bbox.ymin * imageHeight, bbox.width * imageWidth, bbox.height * imageHeight
                centerX, centerY = (left + xmin + width/2), bottom
                self.detect_target[index].append((centerX, centerY))
                cv2.circle(drawImage, (int(centerX), int(centerY)), 8, (255,0,0), -1)
                
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
                
            cv2.rectangle(image, (int(bb_left), int(bb_top)), (int(bb_left+bb_width), int(bb_top+bb_height)), self.bbox_colors[idx], 2) # draw the bbox
            image = draw_tracks(image, output_tracks) # draw the id
            
            # put the score and class to the image
            txt = str(r[0]) + " "+ str(confidence)
            cv2.putText(image, txt, (int(bb_left), int(bb_top-7)) ,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=self.bbox_colors[idx])
            
        return image
        
        
    def set_listener(self, on_detection):
        self.detection_listener = on_detection
        
        
    def __trigger_callback(self, save_path_img, group, alias, detect):        
        if self.detection_listener is None:
            return         
        self.detection_listener(save_path_img, group, alias, detect)
  

    def get_current_frame(self):        
        return self.frame
        
        
    def stop(self):        
        for t in self.th:
            t.do_run = False
#             t.join()
            
        self.write_log("[Info] Stop the program.")
        self.cap.release()
        
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
