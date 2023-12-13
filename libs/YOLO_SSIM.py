#!/usr/bin/env python

import datetime
import json
import os
import random
import schedule
import sys
import threading
import time

from collections import Counter
from multiprocessing import Process

import cv2
import numpy as np
from motpy.testing_viz import draw_detection, draw_track
from skimage.metrics import structural_similarity
from sklearn import ensemble, metrics, preprocessing

sys.path.insert(1, 'multi-object-tracker')
from motpy import Detection, MultiObjectTracker
from darknet import darknet
from libs.utils import *
from motrackers import SORT, CentroidKF_Tracker, CentroidTracker, IOUTracker
from motrackers.utils import draw_tracks
from libs.SSIM_utils import SSIM_test_analyze


class YoloDevice():
    def __init__(self, data_file="", config_file="", weights_file="", data_file2=None, config_file2=None, weights_file2=None,
                 video_url="", output_dir="", target_classes=None, thresh=0.5, vertex=None, alias="demo", group="", obj_trace=True,
                 tracker_mode=1, is_threading=True, skip_frame=None, is_dir=False, is_count_obj=False, only_detect_center_bbox=False,
                 save_img=True, save_img_original=False, save_video=True, save_video_original=False, only_video_capture=False,
                 video_expire_day=None,  img_expire_day=None, display_message=True,  is_draw_polygon=True, auto_restart=True, 
                 restart_hour=None, set_W=None, set_H=None,
                 using_SSIM=False, SSIM_diff_area_size=3000, SSIM_optimal_thresh_path=None, SSIM_debug=False,
                 test_SSIM=False, save_test_SSIM_img=True, gt_dir_path=""
                 ):
        
        """
          Initialize input parameters.
        """
        self.video_url = video_url # Video url for detection
        self.output_dir = output_dir # Output dir for saving results
        self.alias = alias # Name the file and directory
        self.display_message = display_message # Show the message (FPS)
        self.data_file = data_file # darknet file
        self.config_file = config_file # darknet file
        self.weights_file = weights_file # darknet file
        self.data_file2 = data_file2 # Yolo ensemble detection
        self.config_file2 = config_file2 # Yolo ensemble detection
        self.weights_file2 = weights_file2 # Yolo ensemble detection
        self.thresh = thresh # Yolo threshold, float, range[0, 1]
        self.skip_frame = skip_frame # Skip the frame for video detect        
        self.target_classes = target_classes # Set None to detect all target
        self.is_draw_polygon = is_draw_polygon # Draw the polygon if Yolo detect the target object
        self.group = group # Name the folder and file
        self.obj_trace = obj_trace # Object tracking
        self.tracker_mode = tracker_mode # Different algorithm for object tracking (self.tracker[<index>])
        self.is_threading = is_threading # Set False if the input is video file
        self.is_dir = is_dir # Read the image from dictionary
        self.save_img = save_img # Save image when Yolo detect
        self.save_img_original = save_img_original # Save original image and results when Yolo detect
        self.save_video = save_video # Save video including Yolo detect results
        self.save_video_original = save_video_original # Save video
        self.video_expire_day = video_expire_day # Delete the video file if date over the `video_expire_day`
        self.img_expire_day = img_expire_day # Delete the img file if date over the `img_expire_day`
        self.only_video_capture = only_video_capture # Only video streaming, not load Yolo Network for prediction and not consume GPU memory
        self.set_W = set_W # Resize the image wi# Save video including Yolo detect resultsdth to speed up the predict and save storage space
        self.set_H = set_H # Resize the image height to speed up the predict and save storage space      
        self.only_detect_center_bbox = only_detect_center_bbox # Only judge center bbox
        self.auto_restart = auto_restart # Restart the program when RTSP video disconnection
        self.restart_hour = restart_hour # restart the program every `restart_hour` hour
        self.using_SSIM = using_SSIM # Using SSIM to find the moving object        
        self.SSIM_debug = SSIM_debug # Draw the SSIM image moving object even Yolo have detected object
        self.SSIM_diff_area_size = SSIM_diff_area_size # Filter too small moving object (noise)
        self.SSIM_optimal_thresh_path = SSIM_optimal_thresh_path # SSIM optimal threshold file path
        self.test_SSIM = test_SSIM # Search SSIM optimal threshold
        self.save_test_SSIM_img = save_test_SSIM_img # Saving img in each threshold results (Set False to save the storage)
        self.gt_dir_path = gt_dir_path # test_SSIM parameter to set ground_truth
        
        """
          Initialize YoloDevices parameters.
        """        
        # Create folder to save the detection results first.
        if not os.path.exists(os.path.join(self.output_dir, self.alias)):
            create_dir(os.path.join(self.output_dir, self.alias))        
        
        self.detection_listener = None # Callback function when Yolo detect target object
        self.frame_id = 0 # Used to name each frame Yolo detected        
        self.detections = [] # Save the Yolo detect results
        
        
        #  Args: self.is_count_obj:
        #      Return the detect results only when any object id count numbers is reach the certain frequency.
        #      This function is used to improve Yolo precision and reduce the repeat notification for same target object.
        self.is_count_obj = is_count_obj # Count each object appear frequency
        self.count_obj = {} # Save each object appear frequency
        self.count_obj_thresh = 3 #  Threshold of Each object appear frequency
        self.is_return_results = False # Return result when any object is reach the self.count_obj_thresh
        
        if is_count_obj:
            self.obj_trace = True
        
        # Initilize Yolo detect results saving path 
        self.output_dir_img = os.path.join(self.output_dir, self.alias, "img_original")
        self.output_dir_video = os.path.join(self.output_dir, self.alias, "video_original")
        self.output_dir_img_draw = os.path.join(self.output_dir, self.alias, "img_detect")
        self.output_dir_video_draw = os.path.join(self.output_dir, self.alias, "video_detect")  
        
        if self.save_video:
            create_dir(self.output_dir_video_draw)
        if self.save_video_original:
            create_dir(self.output_dir_video)
        
        
        # Initialize vertex    
        self.vertex = {}
        if isinstance(vertex, list) or vertex is None:
            self.vertex[alias] = vertex
        elif isinstance(vertex, dict):
            self.vertex = vertex
        else:
            raise AssertionError("[Error] Vertex type error. The vertex type should be a list or a dictionary or None.")
            
            
        # Check whether start Yolo ensemble model
        self.ensemble = False
        if self.data_file2 is not None and self.config_file2 is not None and self.weights_file2 is not None:            
            self.ensemble = True
            print("[Info] Ensemble model start.") 
        
        
        """
          SSIM parameters.
        """
        # Initialize SSIM parameters to find moviong object.
        self.is_compare_SSIM_bc = False # Set True to compare the background and current frame (set True will reduced processing speed)    
        self.remake_SSIM_dir = True # Delete old SSIM saving data directory and re-create        
        self.SSIM_optimal_thresh = {} # SSIM optimal threshold in each luminance        
        SSIM_default_thresh = 125 # SSIM default threshold if gt_dir_path not specify.
        self.count_after_detect = 0 # Number of frames processed after Yolo detect target object
        self.bc_img_list = [] # Store all different brightness background. structure: [{bc_name:{"brightness":brightness:, "img":img}}, ...]
        self.SSIM_tmp_bc = None # SSIM tmp background image
        self.SSIM_test_tmp_bc = None # SSIM_test tmp background image
        self.pre_frame = None
        
        # Initilize SSIM detect results saving path 
        self.SSIM_folder = os.path.join(self.output_dir, self.alias, "SSIM")
        self.output_dir_SSIM_img_draw = os.path.join(self.SSIM_folder, "SSIM_img_detect")
        self.output_dir_SSIM_img = os.path.join(self.SSIM_folder, "SSIM_img_gray")
        self.output_dir_SSIM_original_img = os.path.join(self.SSIM_folder, "SSIM_img_original")
        self.output_dir_SSIM_tmpbc = os.path.join(self.SSIM_folder, "SSIM_tmp_bc_img")
        self.output_dir_SSIM_bc = os.path.join(self.SSIM_folder, "SSIM_bc_img")        
        
        
        """
          SSIM_test parameters.
        """    
        # Initialize SSIM_test parameters to find SSIM optimal threshold.
        self.test_SSIM_thresh_start = 50 # Start optimal threshold
        self.test_SSIM_thresh_end = 200 # Final optimal threshold
        self.test_SSIM_thresh_interval = 25 # Interval optimal threshold        
        self.test_SSIM_TP = {} # Number of TP images at different threshold
        self.test_SSIM_TN = {} # Number of TN images at different threshold
        self.test_SSIM_FP = {} # Number of FP images at different threshold
        self.test_SSIM_FN = {} # Number of FN images at different threshold
        self.all_luminance_cnt = {} # Number of images at different luminance
        self.yolo_luminance_cnt = {} # Number of images Yolo deected at different luminance
        
        # Initilize SSIM_test detect results saving path 
        self.SSIM_test_folder = os.path.join(self.output_dir, self.alias, "SSIM_test")
        self.output_dir_SSIM_test_img = os.path.join(self.SSIM_test_folder, "SSIM_img_gray")
        self.output_dir_SSIM_test_tmpbc = os.path.join(self.SSIM_test_folder, "SSIM_tmp_bc_img")
        self.output_dir_test_SSIM_TP = os.path.join(self.SSIM_test_folder, "TP")
        self.output_dir_test_SSIM_TN = os.path.join(self.SSIM_test_folder, "TN")
        self.output_dir_test_SSIM_FP = os.path.join(self.SSIM_test_folder, "FP")        
        self.output_dir_test_SSIM_FN = os.path.join(self.SSIM_test_folder, "FN")
        
        
        """
          Initialize SSIM and SSIM_test parameters.
        """
        if self.using_SSIM and self.test_SSIM:
            raise AssertionError("[Error] using_SSIM or test_SSIM should only choose one to set True.")
        
        if self.using_SSIM or self.test_SSIM:
            for v in range(255):
                self.all_luminance_cnt[v] = 0
                self.yolo_luminance_cnt[v] = 0

            # Initialize SSIM TP, TN, FP, FN counter
            for i in range(self.test_SSIM_thresh_start, 
                           self.test_SSIM_thresh_end, 
                           self.test_SSIM_thresh_interval):
                tmp = {}
                for j in range(255):
                    tmp[j] = 0

                self.test_SSIM_TP[i] = tmp.copy()
                self.test_SSIM_TN[i] = tmp.copy() 
                self.test_SSIM_FP[i] = tmp.copy()
                self.test_SSIM_FN[i] = tmp.copy()     
        
        if self.test_SSIM:
            if self.gt_dir_path == "":                
                raise AssertionError("[Error] gt_dir_path should be specify if test_SSIM=True.")
            elif not os.path.exists(self.gt_dir_path):
                raise AssertionError("[Error] gt_dir_path folder not exists.")
            else:
                self.init_test_SSIM()
        
        if self.using_SSIM:            
            self.init_SSIM()
            
            # Set SSIM optimal threshold file path
            if self.SSIM_optimal_thresh_path is not None:
                if not os.path.exists(self.SSIM_optimal_thresh_path):
                    raise AssertionError("[Error] SSIM_optimal_thresh_path not exists.\
                                          Set None if you want to use default SSIM threshold.")
                
                self.SSIM_optimal_thresh_path = os.path.join(self.SSIM_optimal_thresh_path, 
                                                             "SSIM_optimal_threshold_recall.txt")
            else:
                self.SSIM_optimal_thresh_path = os.path.join(self.SSIM_test_folder, 
                                                             "SSIM_optimal_threshold_recall.txt")
                
            # Set SSIM optimal threshold
            if os.path.isfile(self.SSIM_optimal_thresh_path):                
                with open(self.SSIM_optimal_thresh_path) as file:
                    data = file.read() # Read SSIM optimal threshold file        
                    self.SSIM_optimal_thresh = json.loads(data)

                for i in range(255):
                    if self.SSIM_optimal_thresh[str(i)] == 0:
                        self.SSIM_optimal_thresh[str(i)] = SSIM_default_thresh
            else:
                print("[Warning] Using default value as SSIM threshold.")
                time.sleep(1)
                
                for i in range(255):
                    self.SSIM_optimal_thresh[str(i)] = SSIM_default_thresh
                    
                    
            # Using exist background image as SSIM background
            if os.path.exists(self.output_dir_SSIM_bc):                    
                for img_name in os.listdir(self.output_dir_SSIM_bc):
                    if img_name.split(".")[-1] == "jpg" or img_name.split(".")[-1] == "jpeg"\
                       or img_name.split(".")[-1] == "png":
                        img = cv2.imread(os.path.join(self.output_dir_SSIM_bc, img_name))
                        frame_lum = int(avg_color_img(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
                        self.bc_img_list.append({str(frame_lum)+".jpg": {"brightness": frame_lum, "img": img}}) 
                
        """
          Initialize object tracking parameter.
        """
        self.id_storage = [] # Save each object ID
        self.bbox_colors = {} # Store object tracking object bbox color        
        self.tracker = [MultiObjectTracker(
                            dt = 1 / 30,
                            tracker_kwargs = {'max_staleness': 5},
                            model_spec = {'order_pos': 1, 'dim_pos': 2,
                                        'order_size': 0, 'dim_size': 2,
                                        'q_var_pos': 5000., 'r_var_pos': 0.1},
                            ), 
                        CentroidTracker(max_lost=3, tracker_output_format='mot_challenge'),
                        CentroidKF_Tracker(max_lost=3, tracker_output_format='mot_challenge'),
                        SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.1),
                        IOUTracker(max_lost=3, iou_threshold=0.1, min_detection_confidence=0.4, max_detection_confidence=0.7,
                                   tracker_output_format='mot_challenge')
                       ]
        
            
        """
          Initialize cv2.VideoWriter() for saveing video.
        """
        if self.is_threading:
            self.video_output_name = os.path.join(self.output_dir_video, get_current_date_string(), get_current_hour_string(),
                                           get_current_hour_string() + ".avi")
            self.video_output_draw_detect_name = os.path.join(self.output_dir_video_draw, get_current_date_string(), get_current_hour_string(),
                                              get_current_hour_string() +"_"+ self.alias + ".avi")
        else:
            self.video_output_name = os.path.join(self.output_dir_video, self.alias + ".avi")
            self.video_output_draw_detect_name = os.path.join(self.output_dir_video_draw, self.alias + ".avi")
        
        if not self.is_dir:           
            self.cap = cv2.VideoCapture(self.video_url) 
            
            try:     
                if self.set_W is not None and self.set_H is not None:
                    self.W = self.set_W
                    self.H = self.set_H
                else:
                    self.W = int(self.cap.get(3))
                    self.H = int(self.cap.get(4))
                
                self.frame = None
                self.ret = False            
                self.reconnect_rtsp_num = 0 # count the rtsp video recoonecting frequency
                self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.frame_original = cv2.VideoWriter(self.video_output_name, self.fourcc, 30.0, (self.W, self.H))
                self.frame_draw = cv2.VideoWriter(self.video_output_draw_detect_name, self.fourcc, 30.0, (self.W, self.H))
                
            except Exception as e:
                print("[Error] cv2.VideoWriter() setup error ():", e)
                self.write_log("[Error] Video setup error:{url}".format(url=self.video_url)) 
        
        """
          Check the video_url path.
        """
        if self.is_dir:
            self.is_threading = False
            
            # Saveing image which Yolo not detect anything
            self.output_dir_no_detect = os.path.join(output_dir, alias, "img_no_detection") 
            
            if not os.path.exists(self.video_url):
                print('[Error] Image directory ', self.video_url, 'not exist')
                raise AssertionError("[Info] If your input is video, set is_dir = False")
                
            if self.only_video_capture:
                raise AssertionError("[Info] If your input is directory, set only_video_capture = False")
            
            re_make_dir(self.output_dir_no_detect)
                 
        elif self.is_threading:
            if not self.cap.isOpened():
                print("[Error] RTSP video can not connect:{url}, {alias}:{s}".format(
                      url=self.video_url, alias=self.alias, s=self.cap.isOpened()))
                
                if not self.auto_restart:
                    raise AssertionError("[Info] Vidoe connection falied. Please check your RTSP video url")            
        else:
            if not os.path.exists(self.video_url):
                print('[Error] Video file path:', self.video_url, 'not exist')
                raise AssertionError("[Info] If your input is rtsp video, set is_threading = False")            
        
        
        """
          Remove exist video file. The save_video function will re-create video file.
        """
        if os.path.exists(self.video_output_name):
            os.remove(self.video_output_name)   

        if os.path.exists(self.video_output_draw_detect_name):
            os.remove(self.video_output_draw_detect_name)
        
        
        """
          Initialize schedule parameters.
        """        
        if self.restart_hour is not None:
            if isinstance(self.restart_hour, int):
                schedule.every(self.restart_hour).hours.do(self.restart_schedule)
            else:
                raise AssertionError("[Error] restart_hour should be an integer.")
                
        
    def start(self):
        """
          Start the program.
        """
        self.th = []
        self.write_log("[Info] Start the program.")                
        
        if self.only_video_capture:
            self.th.append(threading.Thread(target = self.video_capture))  

        elif self.is_threading:
            self.th.append(threading.Thread(target = self.video_capture))
            self.th.append(threading.Thread(target = self.prediction))

        else:
            self.th.append(threading.Thread(target = self.prediction))

        for t in self.th:
            t.start()            
    
    
    def init_Yolo(self):
        """
          Initalize the darknet network.
        """
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file = self.config_file,
            data_file = self.data_file,
            weights = self.weights_file)
        
        if self.ensemble: 
            self.network2, self.class_names2, self.class_colors2 = darknet.load_network(
                config_file = self.config_file2,
                data_file = self.data_file2,
                weights = self.weights_file2)  
        
        if self.target_classes is None:
            self.target_classes = self.class_names
        
        self.non_target_classes = list(set(self.class_names).difference(self.target_classes)) # Non-target object class
    
    
    def init_SSIM(self):
        """
          Initialize SSIM folder.
        """
        if self.remake_SSIM_dir:            
            re_make_dir(self.output_dir_SSIM_img)
            re_make_dir(self.output_dir_SSIM_original_img)
            re_make_dir(self.output_dir_SSIM_bc)
            re_make_dir(self.output_dir_SSIM_tmpbc)
            if self.SSIM_debug:
                re_make_dir(self.output_dir_SSIM_img_draw)
        else:                
            create_dir(self.output_dir_SSIM_img)
            create_dir(self.output_dir_SSIM_original_img) 
            create_dir(self.output_dir_SSIM_bc)
            create_dir(self.output_dir_SSIM_tmpbc)
            if self.SSIM_debug:
                create_dir(self.output_dir_SSIM_img_draw)
    
    
    def init_test_SSIM(self):
        """
          Initialize SSIM_test folder.
        """        
        
        if self.remake_SSIM_dir:
            re_make_dir(self.output_dir_test_SSIM_TP)
            re_make_dir(self.output_dir_test_SSIM_TN)
            re_make_dir(self.output_dir_test_SSIM_FP)
            re_make_dir(self.output_dir_test_SSIM_FN)
        else:
            create_dir(self.output_dir_test_SSIM_TP)
            create_dir(self.output_dir_test_SSIM_TN)
            create_dir(self.output_dir_test_SSIM_FP)
            create_dir(self.output_dir_test_SSIM_FN)
        
        if self.remake_SSIM_dir:
                re_make_dir(self.output_dir_SSIM_test_img)
                re_make_dir(self.output_dir_SSIM_test_tmpbc)
        else:
            create_dir(self.output_dir_SSIM_test_img)
            create_dir(self.output_dir_SSIM_test_tmpbc)
        
        for v in range(255):           
            
            # Create each luminance dir to save SSIM each threshold results
            create_dir(os.path.join(self.output_dir_test_SSIM_TP, str(v).zfill(3)))
            create_dir(os.path.join(self.output_dir_test_SSIM_TN, str(v).zfill(3)))
            create_dir(os.path.join(self.output_dir_test_SSIM_FP, str(v).zfill(3)))
            create_dir(os.path.join(self.output_dir_test_SSIM_FN, str(v).zfill(3)))
            
        self.SSIM_results = SSIM_test_analyze(folder = self.SSIM_test_folder,                
                                              saving_path = self.SSIM_test_folder,
                                              start_thresh = self.test_SSIM_thresh_start,
                                              final_thresh = self.test_SSIM_thresh_end,
                                              interval_thresh = self.test_SSIM_thresh_interval,
                                              display_message = False,
                                              )
        
        
    def yolo_detect(self, image: np.ndarray) -> list: 
        """
          The image will be detect by Yolo netowrk.
        """
        if self.is_dir:
            self.H = image.shape[0]
            self.W = image.shape[1] 
        
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # original image
        darknet_image = darknet.make_image(self.W, self.H, 3)
        darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
        
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
        
        # Using assemble model
        if self.ensemble:
            detections2 = darknet.detect_image(self.network2, self.class_names2, darknet_image, thresh=self.thresh)
            all_class_names = set(self.class_names).union(self.class_names2)
            detections = nms(all_class_names, detections, detections2)     
        
        darknet.free_image(darknet_image)        
        
        if self.is_dir:
            darknet.print_detections(detections, True)
            print("="*10)
            
        return detections
                    
        
    def read_dir_img(self):
        """
          Only detect each image from directory 
          when variable is_dir=True        
        """
        for img_name in os.listdir(self.video_url):
            try:
                if img_name.split(".")[-1] == "jpg" or img_name.split(".")[-1] == "jpeg"\
                 or img_name.split(".")[-1] == "png":
                    image =  cv2.imread(os.path.join(self.video_url, img_name))                                     
                    
                    self.frame_id += 1 # record the current frame numbers                    
                      
                    print("[Info] Detection img:", img_name)  
                    self.detections = self.yolo_detect(image)
                    
                    # interate all vertex to filter the detect object
                    vertex_key = list(self.vertex.keys())
                    self.detect_target = []
                    for each_vertex_key in vertex_key:
                        for det in detect_filter(self.detections, self.target_classes, 
                                                 self.vertex[each_vertex_key], self.only_detect_center_bbox):
                            self.detect_target.append(det)            
                    
                    if self.is_draw_polygon: 
                        image_draw = draw_polylines(image.copy(), self.vertex)
                    else:
                        image_draw = image.copy()
                        
                    if len(self.detect_target) > 0:
                        image_draw = draw_boxes(self.detect_target, image_draw, self.class_colors, self.target_classes)
                        self.save_img_draw(image_draw)
                        
                        if self.save_img_original:
                            save_path_img_orig = self.save_img_original_fun(image, img_name.split(".")[0])
                            
                    else:
                        cv2.imwrite(os.path.join(self.output_dir_no_detect, img_name), image)
                        
            except Exception as e:
                print("[Error] Read the image from directory error:", e)
                
    
    def video_capture_wo_threading(self): 
        """
          Read each frame from video file.
        """
        self.pre_frame = self.frame
        self.ret, self.frame = self.cap.read() 
            
        if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong
            print("[Info] Video file detection finished ...")
            self.stop()            
        else:          
            if self.set_W is not None and self.set_H is not None:
                self.frame = cv2.resize(self.frame.copy(), (int(self.W), int(self.H)), 
                                        interpolation=cv2.INTER_AREA)
            
            if self.save_video_original:
                self.save_video_original_fun(self.frame)                
                
    
    def video_capture(self):
        """
          Read the current new frame from RTSP video.
        """
        t = threading.currentThread()
        
        while getattr(t, "do_run", True):
            self.pre_frame = self.frame
            self.ret, frame = self.cap.read() 
            
             # Camera connection error
            if not self.cap.isOpened() or not self.ret or frame is None:
                print("[Error] Reconnecting:{group}:{alias}:{url} ".format(
                      group=self.group, alias=self.alias, url=self.video_url))
                self.write_log("[Error] Reconnecting to the camera.")
                self.reconnect_rtsp_num += 1
                self.cap.release()
                time.sleep(3)
                self.cap = cv2.VideoCapture(self.video_url)
                time.sleep(3)                
                
                # Restart the program when RTSP video can not connect
                if self.reconnect_rtsp_num % 3 == 0:                    
                    print("[Error] Restarting program due to camera connecting failed:{group}:{alias}:{url}".format(
                          group=self.group, alias=self.alias, url=self.video_url))
                    if self.auto_restart:
                        try:                            
                            self.restart()
                        except Exception as e:
                            self.write_log("[Error] Restart the program failed: {e}".format(e=e))
                            print("[Error] Can not restart program:{group}:{alias}:{url} ".format(
                                  group=self.group, alias=self.alias, url=self.video_url))
                            time.sleep(10)
            else:
                self.frame = frame
                
                if self.set_W != None and self.set_H != None:
                    self.frame = cv2.resize(self.frame.copy(), (int(self.W), int(self.H)), 
                                            interpolation=cv2.INTER_AREA)
                if self.save_video_original:
                    self.save_video_original_fun(self.frame)
                    
                    
    def prediction(self):    
        """
          The main function for detection and saving results.
        """
        last_time = time.time() # Compute FPS
        cnt = 0  # Counting for skip_frame and FPS
        predict_time_sum = 0 # Compute predict FPS
        save_path_img = None # Image saving path with Yolo detected bounding bbox 
        save_path_img_orig = None # Image saving path without Yolo detected bounding bbox 
        t = threading.currentThread() # Get this function threading status        
        
        self.init_Yolo() # Initialize Yolo parameters
        
        
        """
          Only detect the image in directory.
          Stop the program when all image file detected.
        """
        if self.is_dir:
            read_dir_img = self.read_dir_img()
            print("[Info] Directory image detection is finished...")
            self.stop()
        
        
        """
          While loop for detect camera frame continuously .
        """
        while getattr(t, "do_run", True):
            try:
                cnt += 1 # Counting for skip_frame and FPS
                self.frame_id += 1 # Record the current frame number to name the file

                # Input is video file, get the frame w/o threading
                if not self.is_threading:
                    self.video_capture_wo_threading() 

                # RTSP read the camera frame error
                if not self.cap.isOpened() or not self.ret or self.frame is None:                
                    time.sleep(1)
                    continue

                # Skip frame
                if self.skip_frame is not None and cnt % self.skip_frame != 0:
                    continue    

                predict_time = time.time() # Get start predict time                      
                self.detections = self.yolo_detect(self.frame) # Yolo detect
                predict_time_sum += (time.time() - predict_time) # Add sum predict time

                # Interate all vertex to filter the detect object
                vertex_key = list(self.vertex.keys())
                self.detect_target = []

                for each_vertex_key in vertex_key:
                    for det in detect_filter(self.detections, self.target_classes, 
                                             self.vertex[each_vertex_key], self.only_detect_center_bbox):
                        self.detect_target.append(det)


                """
                  Draw the bounding box which Yolo detect.
                  If self.obj_trace=True, the object ID will be put on the target object.
                """
                if self.obj_trace:
                    if self.tracker_mode == 0:
                        image_detect,  self.detect_target = self.object_tracker_motpy(self.frame.copy(), self.detect_target)    
                    else:
                        image_detect, self.detect_target = self.object_tracker(self.frame.copy(), self.detect_target)    
                else:
                    image_detect = draw_boxes(self.detect_target, self.frame.copy(), 
                                              self.class_colors, self.target_classes)

                # Find the moving object using SSIM
                if self.using_SSIM or self.test_SSIM:
                    self.frame_lum = int(avg_color_img(cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2GRAY)))                
                    self.all_luminance_cnt[self.frame_lum] += 1

                    if self.count_after_detect > 0:                    
                        self.count_after_detect -= 1  

                        # Using SSIM to find moving object Yolo not detected
                        if self.using_SSIM:
                            if len(self.detect_target) == 0 or self.SSIM_debug:
                                SSIM_thresh = self.SSIM_optimal_thresh[str(self.frame_lum)]
                                image_detect, _, _ = self.find_moving_obj(image_detect, SSIM_thresh)

                        if self.test_SSIM:
                            self.test_SSIM_thresh()

                    # Count number of images yolo detect
                    if len(self.detect_target) > 0:
                        self.yolo_luminance_cnt[self.frame_lum] += 1

                    # Update tmp background when moving object detected
                    if len(self.detect_target) == 0 and self.frame_id % 30 == 0:                    
                        self.SSIM_tmp_bc = self.frame.copy()

                # Draw the polygon
                if self.is_draw_polygon: 
                    image_detect = draw_polylines(image_detect, self.vertex)      


                # Save the image with bounding box when target detected
                if self.save_img and len(self.detect_target) > 0:                
                    save_path_img = self.save_img_draw(image_detect)

                # Save original image
                if self.save_img_original and len(self.detect_target) > 0:
                    save_path_img_orig = self.save_img_original_fun(self.frame)

                # Set counter after Yolo detect
                if self.using_SSIM or self.test_SSIM:
                    if len(self.detect_target) > 0:
                        self.count_after_detect = 3

                # Save video with bounding box            
                if self.save_video:
                    self.save_video_draw(image_detect)

                # Check object appear frequency
                if self.is_count_obj and len(self.count_obj) > 0:
                    self.is_return_results = False                
                    max_id = max(self.count_obj, key=self.count_obj.get)

                    if self.count_obj[max_id] >= self.count_obj_thresh: 
                        self.is_return_results = True                   
                        self.count_obj[max_id] = 0                   

                # Callback function
                if len(self.detect_target) > 0:                
                    if not self.is_count_obj or \
                        (self.is_count_obj and self.is_return_results):
                         self.__trigger_callback(save_path_img, self.group, 
                                                 self.alias, self.detect_target)           
                
                if time.time() - last_time > 10:
                    self.print_msg("[Info] FPS (Overall):{fps} , {alias}".format(
                        alias=self.alias, fps=cnt/(time.time()-last_time)))
                    self.print_msg("[Info] FPS (Predict):{fps} , {alias}".format(
                        alias=self.alias, fps=cnt/predict_time_sum))
                    last_time = time.time()
                    cnt = 0
                    predict_time_sum = 0 

                    # Scheduled event
                    schedule.run_pending()                    
                    
                    # Delete too old directory
                    self.del_expire_day_dir()
                    
                    # Write each SSIM threshold results to text file
                    if self.test_SSIM:
                        self.write_test_SSIM_text()
                   
            except Exception as e:
                print(f"[Error] prediction() function error:{e}. This error should be solved.")
                print(e,e.__traceback__.tb_lineno)
                self.write_log(f"[Error] prediction() function error:{e}")
                
                        
    def find_moving_obj(self, image: np.ndarray, SSIM_thresh) -> tuple:
        """
          Detect the moving object based on SSIM.
        """        
        self.detect_target_wo_vertex = detect_filter(self.detections, self.target_classes, None, self.only_detect_center_bbox) # not using yet   
        
        if self.is_compare_SSIM_bc:
            
            # Select background image
            if len(self.bc_img_list) == 0:
                bc_img = self.frame.copy()
                self.bc_img_list.append({str(self.frame_lum)+".jpg": {"brightness": self.frame_lum, "img": bc_img}})                
                cv2.imwrite(os.path.join(self.output_dir_SSIM_bc, str(self.frame_lum) + ".jpg"), bc_img)                
                
            elif len(self.bc_img_list) == 1:
                bc_img_name = list(self.bc_img_list[0].keys())[0] 
                bc_img = self.bc_img_list[0][bc_img_name]["img"] 

            else:
                # Find the most similar birghtness between current frame and background
                min_diff_val = 255
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)            

                for bc_idx in range(len(self.bc_img_list)):
                    bc_img_name = list(self.bc_img_list[bc_idx].keys())[0] # get each image name in bc_list
                    bc_img_candidate = self.bc_img_list[bc_idx][bc_img_name]["img"] # get RGB image
                    bc_img_candidate_gray = cv2.cvtColor(bc_img_candidate, cv2.COLOR_BGR2GRAY)
                    bc_lum = int(avg_color_img(bc_img_candidate_gray))

                    if abs(self.frame_lum - bc_lum) < min_diff_val:
                        min_diff_val = abs(self.frame_lum - bc_lum)
                        bc_img = bc_img_candidate

            # Using SSIM to compare the background and current frame
            score, SSIM_img = SSIM(bc_img, self.frame.copy())
            is_find_moving_obj, image, _ = self.iterate_SSIM_result(SSIM_img, image, SSIM_thresh)

        else:
            is_find_moving_obj = True        
        
        
        # Using SSIM to compare the "tmp background" and current frame
        is_find_moving_obj2 = False   
        
        if is_find_moving_obj:
            if self.SSIM_tmp_bc is None:
                self.SSIM_tmp_bc = image.copy()
                is_find_moving_obj2 = True
                SSIM_img2 = self.frame.copy()                
                score = 100
            else:                
                score, SSIM_img2 = SSIM(self.SSIM_tmp_bc, self.frame.copy()) 
                is_find_moving_obj2, image, _ = self.iterate_SSIM_result(SSIM_img2, image, SSIM_thresh, 
                                                                         self.SSIM_diff_area_size)
        
        if is_find_moving_obj2:
#             self.SSIM_tmp_bc = self.frame.copy() # update tmp background if find moving object
            cv2.imwrite(os.path.join(self.output_dir_SSIM_tmpbc, 
                                     str(self.frame_id)+".jpg"), self.frame)
            cv2.imwrite(os.path.join(self.output_dir_SSIM_original_img, 
                                     str(self.frame_id) + ".jpg"), self.frame) 
            cv2.imwrite(os.path.join(self.output_dir_SSIM_img, 
                                     str(self.frame_id) + ".jpg"), SSIM_img2)
            if self.SSIM_debug:
                cv2.imwrite(os.path.join(self.output_dir_SSIM_img_draw, 
                                         str(self.frame_id) + ".jpg"), image)
    
        return image, score, is_find_moving_obj2
    
    
    def test_SSIM_thresh(self):
        """
          Find the SSIM optimal threshold.
        """
        
        # Check ground truth text file whether exist
        gt_file_path = os.path.join(self.gt_dir_path, str(self.frame_id).zfill(6) +"_"+ self.alias + ".txt") 
        obj_exist = False

        if os.path.exists(gt_file_path):            
            obj_exist = True
        
        # Check tmp background whether exist
        if self.SSIM_tmp_bc is None:
            self.SSIM_tmp_bc = self.frame.copy()
        
        # Compare current frame and tmp background
        score, SSIM_img2 = SSIM(self.SSIM_tmp_bc, self.frame.copy())
        cv2.imwrite(os.path.join(self.output_dir_SSIM_test_img, str(self.frame_id).zfill(6)+".jpg"), 
                                 SSIM_img2)
        cv2.imwrite(os.path.join(self.output_dir_SSIM_test_tmpbc, str(self.frame_id).zfill(6)+".jpg"), 
                                 self.SSIM_tmp_bc)
        
        # Interate all SSIM threshold
        for SSIM_thresh in range(self.test_SSIM_thresh_start, 
                                 self.test_SSIM_thresh_end, 
                                 self.test_SSIM_thresh_interval):
            
            is_find_moving_obj = False
            is_find_moving_obj, image, moving_obj_area = self.iterate_SSIM_result(SSIM_img2.copy(), self.frame.copy(),
                                                                                  SSIM_thresh, self.SSIM_diff_area_size) 
            
            if self.is_draw_polygon: 
                image = draw_polylines(image, self.vertex)  
                
            # SSIM find the moving object
            if is_find_moving_obj:                
                if obj_exist:
                    self.test_SSIM_TP[SSIM_thresh][self.frame_lum] += 1
                    self.print_msg("[TP Lum:{lum}] ,frame:{frame_id}, Threshold:{thresh}, Area:{area}".format(
                                   frame_id=str(self.frame_id).zfill(6), thresh=SSIM_thresh, lum=self.frame_lum, area=moving_obj_area))
                    if self.save_test_SSIM_img:
                        cv2.imwrite(os.path.join(self.output_dir_test_SSIM_TP, str(self.frame_lum).zfill(3), 
                                                 str(self.frame_id).zfill(6) + "_" + str(SSIM_thresh) + ".jpg"), image)
                else:
                    self.test_SSIM_FP[SSIM_thresh][self.frame_lum] += 1
                    self.print_msg("[FP Lum:{lum}] ,frame:{frame_id}, Threshold:{thresh}, Area:{area}".format(
                                   frame_id=str(self.frame_id).zfill(6), thresh=SSIM_thresh, lum=self.frame_lum, area=moving_obj_area))
                    if self.save_test_SSIM_img:
                        cv2.imwrite(os.path.join(self.output_dir_test_SSIM_FP, str(self.frame_lum).zfill(3), 
                                                 str(self.frame_id).zfill(6) + "_" + str(SSIM_thresh) + ".jpg"), image)                    
            else:
                if obj_exist:
                    self.test_SSIM_FN[SSIM_thresh][self.frame_lum] += 1
                    self.print_msg("[FN Lum:{lum}] ,frame:{frame_id}, Threshold:{thresh}, Area:{area}".format(
                                   frame_id=str(self.frame_id).zfill(6), thresh=SSIM_thresh, lum=self.frame_lum, area=moving_obj_area))
                    if self.save_test_SSIM_img:
                        cv2.imwrite(os.path.join(self.output_dir_test_SSIM_FN, str(self.frame_lum).zfill(3), 
                                                 str(self.frame_id).zfill(6) + "_" + str(SSIM_thresh) + ".jpg"), image)
                else:
                    self.test_SSIM_TN[SSIM_thresh][self.frame_lum] += 1
                    self.print_msg("[TN Lum:{lum}] ,frame:{frame_id}, Threshold:{thresh}, Area:{area}".format(
                                   frame_id=str(self.frame_id).zfill(6), thresh=SSIM_thresh, lum=self.frame_lum, area=moving_obj_area))
                    if self.save_test_SSIM_img:
                        cv2.imwrite(os.path.join(self.output_dir_test_SSIM_TN, str(self.frame_lum).zfill(3), 
                                                 str(self.frame_id).zfill(6) + "_" + str(SSIM_thresh) + ".jpg"), image)        
            
            
    def iterate_SSIM_result(self, SSIM_img: np.ndarray, image: np.ndarray, 
                            SSIM_thresh, SSIM_diff_area_size: int=3000) -> bool:
        """
          Binarize the SSIM Image for edge detection 
          to find different region.    
          
          Args:
              SSIM_img: Gray image. Darker regions having more disparity.
              image: the RGB image to draw the moving object.
              SSIM_thresh: SSIM_img binarization threhsold.
              SSIM_diff_area_size: Filter too small moving object (noise). 
                                   Moving object should bigger than SSIM_diff_area_size.
        """
        is_find_moving_obj = False
        moving_obj_area = [] # Record all moving object area
        
        thresh = cv2.threshold(SSIM_img, SSIM_thresh, 255, cv2.THRESH_BINARY_INV)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        # Iterate edge detecting results from SSIM image
        for c in contours:                 
            area = cv2.contourArea(c) # Moving object area
            
            if area > SSIM_diff_area_size:
                lx_diff, ly_diff, w_diff, h_diff = cv2.boundingRect(c)
                w_diff = int(w_diff) # width
                h_diff = int(h_diff) # height
                rx_diff = int(lx_diff + w_diff) # right x
                ry_diff = int(ly_diff + h_diff) # bottom y
                x_diff = int(lx_diff + w_diff/2) # center x
                y_diff = int(ly_diff + h_diff/2) # center y
                
                # Check the moving object bbox whether in vertex 
                vertex_key = list(self.vertex.keys())                
                for each_vertex_key in vertex_key:                                
                    if self.vertex[each_vertex_key] is not None:                        
                        if not(is_in_hull(self.vertex[each_vertex_key], (lx_diff, ly_diff))\
                               or is_in_hull(self.vertex[each_vertex_key], (lx_diff, ry_diff))\
                               or is_in_hull(self.vertex[each_vertex_key], (rx_diff, ly_diff))\
                               or is_in_hull(self.vertex[each_vertex_key], (rx_diff, ry_diff))\
                               or is_in_hull(self.vertex[each_vertex_key], (x_diff, y_diff))):  
                            continue
   
                    is_find_moving_obj = True
                    moving_obj_area.append(area)

                    # Draw the moving object bounding box
                    if self.SSIM_debug:
                        cv2.rectangle(image, (lx_diff, ly_diff), (rx_diff, ry_diff),
                                      (255,0,255), 4)
                
                # SSIM_debug mode will draw all moving object
                if is_find_moving_obj and not self.SSIM_debug:
                    break 
        
        return is_find_moving_obj, image, moving_obj_area         
        
    
    # https://github.com/adipandas/multi-object-tracker.git
    def object_tracker(self, image: np.ndarray, detect_target: list) -> tuple:
        """
          Implement the following object tracking: 
              1. CentroidTracker        
              2. CentroidKF_Tracker
              3. SORT
              4. IOUTracker
        """
        boxes = []
        confidence = []
        class_ids = []
        tracking_detect_target = []        
        
        # Convert the Yolo results to object tracking format
        for r in detect_target:
            center_x, center_y, width, height = r[2]
            left, top, right, bottom = darknet.bbox2points(r[2])
            boxes.append([left, top, width, height])
            confidence.append(int(float(r[1])))
            class_ids.append(int(self.target_classes.index(r[0])))
        
        # `output_tracks` is a list with each element containing tuple of
        # (<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>)
        output_tracks = self.tracker[self.tracker_mode].update(np.array(boxes), np.array(confidence), 
                                                               np.array(class_ids))        
        
        # Re-assigned each bbox        
        for track in output_tracks:
            frame, idx, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
            assert len(track) == 10
#             print(track)
            bbox = (bb_left+bb_width/2, bb_top+bb_height/2,  bb_width, bb_height)
    
            # Put the result to detect_target 
            for r in detect_target:
                center_x, center_y, width, height = r[2]
                left, top, right, bottom = darknet.bbox2points(r[2])
                
                # Match the object
                if int(left) == int(bb_left) and int(width) == int(bb_width)\
                   and int(height) == int(bb_height): 
                    obj_name = self.target_classes[int(self.target_classes.index(r[0]))]
                    tracking_detect_target.append((obj_name, confidence, bbox, idx))
                    break

            
            # Count object ID appear frequency
            if self.is_count_obj:
                if self.count_obj.get(idx) is None:
                    self.count_obj[idx] = 0
                else:
                    self.count_obj[idx] += 1                
            
            
            # Assigen each object ID a color for drawing bounding box
            if self.bbox_colors.get(idx) == None:
                self.bbox_colors[idx] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
                
            cv2.rectangle(image, (int(bb_left), int(bb_top)), 
                          (int(bb_left+bb_width), int(bb_top+bb_height)), self.bbox_colors[idx], 2)
            image = draw_tracks(image, output_tracks) # Draw the object ID
            
            
            # Put the score and class to the image
            txt = str(obj_name) + " "+ str(confidence)
            cv2.putText(image, txt, (int(bb_left), int(bb_top-7)) ,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=self.bbox_colors[idx])        
        
        return image, tracking_detect_target
    
    
    # https://github.com/wmuron/motpy.git
    def object_tracker_motpy(self, image: np.ndarray, detect_target: list) -> tuple:
        """
          Implement the following object tracking: 
              1. Kalman-based tracker
          This tracking results may including bounding box
          which object tracking algorithm predict instead of Yolo detection.
        """
        boxes = []
        scores = []
        class_ids = []
        tracking_detect_target = []
        
        # Convert the results to the object tracking format
        for r in detect_target:
            boxes.append(darknet.bbox2points(r[2]))
            scores.append(float(r[1]))
            class_ids.append(r[0])        
            
        self.tracker[self.tracker_mode].step(detections=[Detection(box=b, score=s, class_id=l)\
                                            for b, s, l in zip(boxes, scores, class_ids)])
        tracks = self.tracker[self.tracker_mode].active_tracks(min_steps_alive=3)
        
        
        # Re-assigned each bbox        
        for track in tracks:            
            if track.id not in self.id_storage:
                self.id_storage.append(track.id) # convert random number track.id to index
                
            id_index = self.id_storage.index(track.id) # the order of elements in the python list is persistent                
            tracking_detect_target.append((track.class_id, track.score, track.box, id_index)) # put the result to detect_target            
            draw_track(image, track, thickness=2, text_at_bottom=True, text_verbose=0) # draw the bbox
            
             # Put the object ID to image
            txt = track.class_id + " "+ str(track.score) +" ID:" + str(id_index)
            cv2.putText(image, txt, (int(track.box[0]), int(track.box[1])-7),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255,255,0))
            
        return image, tracking_detect_target
    
    
    def set_listener(self, on_detection: callable):
        self.detection_listener = on_detection
        
        
    def __trigger_callback(self, save_path_img: str, group: str, alias: str, detect: str):
        """
          Callback function when target object detected.
        """
        if self.detection_listener is None:
            return         
        self.detection_listener(save_path_img, group, alias, detect)
  

    def get_current_frame(self):
        """
          Get the current new frame.
        """
        return self.frame
    
    def get_current_SSIM_img(self):
        """
          Get the current SSIM gray image.
        """
        if self.frame is not None and self.pre_frame is not None:
            score, SSIM_img = SSIM(self.frame, self.pre_frame)
        
        return SSIM_img
        
    def stop(self):
        """
          Stop all threading and release the video resource.
        """
        for t in self.th:
            t.do_run = False
            
        if not self.is_dir:
            self.cap.release()
            
        self.write_log("[Info] Stop the program.")
        print('[Info] Stop the program: Group:{group}, alias:{alias}, URL:{url}'\
              .format(group=self.group, alias=self.alias, url=self.video_url))      
        
        
    def restart(self):
        """
          Restart this process.
        """
        self.stop()        
        self.write_log("[Info] Restart the program")
        restart()

        
    def restart_schedule(self):
        """
          Restart this process by scheduling.
        """
        self.stop()        
        self.write_log(f"[Info] Trigger restart program event: Every {self.restart_hour} hour.")
        restart()
        
        
    def write_log(self, msg: str):
        """
          Write the event to log file.
        """
        f = open('log.txt', "a")    
        f.write("Time:{time}, {msg}, Group:{group}, Alias:{alias}, URL:{url} \n"\
                .format(time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), \
                 group=self.group, alias=self.alias, url=self.video_url, msg=msg))
        f.close()
       
    
    def print_msg(self, msg: str):
        """
          Print the message.
        """
        if self.display_message:            
            print(msg)
            
    
    def del_expire_day_dir(self):
        """
          Delete the dir over than expire date.
        """


        if self.video_expire_day is not None:
            if os.path.isdir(self.output_dir_video_draw): 
                del_dir(self.output_dir_video_draw, self.video_expire_day)
            if os.path.isdir(self.output_dir_video): 
                del_dir(self.output_dir_video, self.video_expire_day)

        if self.img_expire_day is not None:
            if os.path.isdir(self.output_dir_img_draw): 
                del_dir(self.output_dir_img_draw, self.img_expire_day)
            if os.path.isdir(self.output_dir_img):     
                del_dir(self.output_dir_img, self.img_expire_day)
        
        
    def save_video_original_fun(self, frame: np.ndarray):
        """
          Save the original video frame.
        """
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video, get_current_date_string(), 
                                      get_current_hour_string())
            video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".avi")   
            
        else:
            video_path_name = self.video_output_name
        
        if not os.path.exists(video_path_name):
            if self.is_threading: 
                create_dir(video_path)                        
                
            self.frame_original = cv2.VideoWriter(video_path_name, self.fourcc, 30.0, (self.W, self.H))
            print("[Info] Set original cv2.VideoWriter(), Alias:{alias}, Width={W}, Height={H} ".format(
                  alias=self.alias, W=self.W, H=self.H))
            
        self.frame_original.write(frame)        
        
    
    def save_video_draw(self, frame: np.ndarray):
        """
          Save the video. The bounding box will draw when Yolo detect the target object.
        """
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video_draw, get_current_date_string(), 
                                      get_current_hour_string())
            video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".avi")
        else:
            video_path_name = self.video_output_draw_detect_name            
            
        if not os.path.exists(video_path_name):
            if self.is_threading: 
                create_dir(video_path)
                
            self.frame_draw = cv2.VideoWriter(video_path_name, self.fourcc, 30.0, (self.W, self.H))
            print("[Info] Set drawing cv2.VideoWriter(), Alias:{alias}, Width={W}, Height={H} ".format(
                  alias=self.alias, W=self.W, H=self.H))
            
        self.frame_draw.write(frame)
            
    
    def save_img_draw(self, image: np.ndarray) -> str:
        """
          Save the image with bounding box when Yolo detect the target object.
        """
        if self.is_threading:
            img_path = os.path.join(self.output_dir_img_draw, get_current_date_string(), 
                                    get_current_hour_string())                 
        else:
            img_path = self.output_dir_img_draw
        
        if not os.path.exists(img_path):
            create_dir(img_path)
        
        img_path_name = os.path.join(img_path, str(self.frame_id).zfill(6) +"_"+ self.alias + ".jpg")
        cv2.imwrite(img_path_name, image)
        
        return img_path_name
    
    
    def save_img_original_fun(self, image: np.ndarray, image_name: str=None) -> str:
        """
          Save the original image when Yolo detect target object.
          The detect results will also be saved as Yolo format to text file.        
        """
        if self.is_threading:
            img_path = os.path.join(self.output_dir_img, get_current_date_string(), 
                                    get_current_hour_string())            
        else:
            img_path = self.output_dir_img
        
        if not os.path.exists(img_path):
            create_dir(img_path)
        
        if image_name is None:
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(6) +"_"+ self.alias + ".jpg")            
            txt_path_name = os.path.join(img_path, str(self.frame_id).zfill(6) +"_"+ self.alias + ".txt")
        else:
            img_path_name = os.path.join(img_path, image_name + ".jpg")            
            txt_path_name = os.path.join(img_path, image_name + ".txt")
            
        cv2.imwrite(img_path_name, image)        
        
        # Save yolo format results to text file
        txt_list = []
        f= open(txt_path_name, "w")
        
        for bbox in self.detect_target:
            objclass = self.target_classes.index(bbox[0])
            x = bbox[2][0] / self.W
            y = bbox[2][1] / self.H
            w = bbox[2][2] / self.W
            h = bbox[2][3] / self.H
            txt_list.append(' '.join([str(objclass), str(x), str(y), str(w), str(h)]))
        f.write('\n'.join(txt_list))
        f.close()
        
        return img_path_name
    
    
    def write_test_SSIM_text(self):
        """
          Write each SSIM threshold results to text file.
        """
        
        with open(os.path.join(self.SSIM_test_folder, "TP.txt"), "w") as f:
            f.write(json.dumps(self.test_SSIM_TP))
            
        with open(os.path.join(self.SSIM_test_folder, "TN.txt"), "w") as f:
            f.write(json.dumps(self.test_SSIM_TN))
            
        with open(os.path.join(self.SSIM_test_folder, "FP.txt"), "w") as f:
            f.write(json.dumps(self.test_SSIM_FP))

        with open(os.path.join(self.SSIM_test_folder, "FN.txt"), "w") as f:
            f.write(json.dumps(self.test_SSIM_FN))
            
        with open(os.path.join(self.SSIM_test_folder, "yolo_luminance_cnt.txt"), "w")  as f:
            f.write(json.dumps(self.yolo_luminance_cnt))        
        
        with open(os.path.join(self.SSIM_test_folder, "all_luminance_cnt.txt"), "w") as f:
            f.write(json.dumps(self.all_luminance_cnt))            
        
        self.SSIM_results.analyze() # Generate SSIM optimal threshold in each luminance
        
        
