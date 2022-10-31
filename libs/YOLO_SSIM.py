import ctypes
import datetime
import json
import math
import os
import random
import shutil
import sys
import threading
import time

from collections import Counter
from multiprocessing import Process

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import cv2
import numpy as np
from motpy.testing_viz import draw_detection, draw_track
from skimage.metrics import structural_similarity
# from sklearn import ensemble, metrics, preprocessing

sys.path.insert(1, 'multi-object-tracker')
from motpy import Detection, MultiObjectTracker
from darknet import darknet
from libs.utils import *
from motrackers import SORT, CentroidKF_Tracker, CentroidTracker, IOUTracker
from motrackers.utils import draw_tracks


class YoloDevice():
    def __init__(self, video_url="", output_dir="", auto_restart=False, obj_trace=False, catch_miss=False,
                 display_message=True, data_file="", config_file="", weights_file="", is_count_obj=False,
                 thresh=0.5, vertex=None, target_classes=None, is_draw_polygon=True,
                 alias="", group="", is_threading=True, skip_frame=None, is_dir=False, 
                 save_img=True, save_img_original=False, save_video=True, save_video_original=False, is_test_SSIM_yolo_thresh=0.95,
                 is_save_img_after_detect=False, video_expire_day=None, SSIM_thresh=150,
                 data_file2=None, config_file2=None, weights_file2=None, is_test_SSIM=False,
                 set_W=None, set_H=None):
        
        self.video_url = video_url
        self.output_dir = output_dir
        self.video_output_name = ""
        self.video_output_draw_name = ""        
        self.auto_restart = auto_restart
        self.display_message = display_message
        self.data_file = data_file
        self.config_file = config_file
        self.weights_file = weights_file
        self.data_file2 = data_file2 # Yolo ensemble detection
        self.config_file2 = config_file2 # Yolo ensemble detection
        self.weights_file2 = weights_file2 # Yolo ensemble detection
        self.thresh = thresh # Yolo threshold
        self.skip_frame = skip_frame # skip the frame for video detect        
        self.target_classes = target_classes # set None to detect all target
        self.is_draw_polygon = is_draw_polygon
        self.alias = alias
        self.group = group
        self.obj_trace = obj_trace        
        self.is_test_SSIM = is_test_SSIM
        self.is_threading = is_threading # set False if the input is video file
        self.is_dir = is_dir # read the image in dictionary
        self.save_img = save_img
        self.save_img_original = save_img_original # set True to restore the original image
        self.save_video = save_video # set True to save the result to video
        self.save_video_original = save_video_original # set True to save the video stream
        self.is_save_img_after_detect = is_save_img_after_detect # set True to save a series of images after object is detected
        self.video_expire_day = video_expire_day # delete the video file if over the video_expire_day        
        self.bbox_colors = {} # use for object tracking object bbox color
        self.set_W = set_W # resize the image width 
        self.set_H = set_H # resize the image height
        self.detection_listener = None # callback function
        self.non_target_classes = None # non-target object class        
        self.run = True 
        self.frame_id = 0
        self.count_after_detect = 0 # count frame for save_after_image()
        
        
        '''
        Count the object appear frequency.
        When set self.is_count_obj True, 
        return the result only when any object id counting is reach the certain frequency.
        '''
        self.is_count_obj = is_count_obj
        self.count_obj = {} # save each object appear frequency
        self.is_return_results = False # check whether return Yolo detect result
        self.count_obj_thresh = 3 # each object appear frequency threshold
        if is_count_obj:     
            self.obj_trace = True        
        
        # SSIM variable      
        self.SSIM_thresh = SSIM_thresh
        self.catch_miss = catch_miss
        self.bc_list = [] # store all different brightness background. structure: [{bc_name:{"brightness":brightness:, "img":img}}, ...]
        self.bc_img_name = None # background image name
        self.bc_img_idx = None # background index of self.bc_list
        self.bc_img = None # SSIM background image       
        self.SSIM_tmp_bc = None # SSIM tmp background image       
        self.is_test_SSIM_yolo_thresh = is_test_SSIM_yolo_thresh       
        
                 
        # Yolo detect saving path initilize
        self.output_dir_img = os.path.join(output_dir, alias, "img_original")
        self.output_dir_video = os.path.join(output_dir, alias, "video_original")
        self.output_dir_img_draw = os.path.join(output_dir, alias, "img_detect")
        self.output_dir_video_draw = os.path.join(output_dir, alias, "video_detect")
        
        # SSIM detect saving path initilize
        self.output_dir_SSIM_draw_img = os.path.join(output_dir, alias, "SSIM_img_detect")
        self.output_dir_SSIM_img = os.path.join(output_dir, alias, "SSIM_img")
        self.output_dir_SSIM_original_img = os.path.join(output_dir, alias, "SSIM_img_original")
        self.output_dir_img_bc = os.path.join(output_dir, alias, "SSIM_bc_img")         

                
        # check video_url path 
        if self.is_dir:
            self.is_threading = False         
            self.output_dir_no_detect = os.path.join(output_dir, alias, "img_no_detection") # save the image which not detect anything (is_dir = True)
                 
            if not os.path.exists(self.video_url):
                print('Image directory ', self.video_url, 'not exist')
                raise AssertionError("If your input is rtsp, set is_dir = False")  
            re_make_dir(self.output_dir_no_detect)  
                 
        elif not self.is_threading:
            if not os.path.exists(self.video_url):
                print('Video path:', self.video_url, 'not exist')
                raise AssertionError("If your input is rtsp, set is_threading = False")
        
        # initialize vertex    
        self.vertex = {}
        if type(vertex) == list or vertex == None:
            self.vertex[alias] = vertex
        elif type(vertex) == dict:
            self.vertex = vertex
        else:
            raise AssertionError("[Error] Vertex type error. The vertex should be list or dictionary.")
            
        # check Yolo ensemble model
        self.ensemble = False
        if self.data_file2 != None and self.config_file2 != None and self.weights_file2 != None:            
            self.ensemble = True
            print("[Info] Ensemble model start.")       
        
        # initialize SSIM saving data directory
        if self.catch_miss or self.is_test_SSIM:
            create_dir(self.output_dir_img_bc)
            re_make_dir(self.output_dir_SSIM_draw_img)
            re_make_dir(self.output_dir_SSIM_img)
            re_make_dir(self.output_dir_SSIM_original_img)          
        
        
        
        # Object Tracking parameter initialize
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
        
        
        '''
        The following code is used for video initialize
        '''
        if self.is_dir:
            self.W = 1920
            self.H = 1080       
        else:            
            self.cap = cv2.VideoCapture(self.video_url)            
            try:     
                if self.set_W != None and self.set_H != None:
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
                self.frame_draw = cv2.VideoWriter(self.video_output_draw_name, self.fourcc, 30.0, (self.W, self.H))
                
            except Exception as e:
                print("[Error] Video setup error:",e)
                self.write_log("[Error] Video setup error:{url}".format(url=self.video_url))
            
        
               
        # remove the exist video file
        video_path_original = os.path.join(self.output_dir_video, get_current_date_string(), get_current_hour_string())
        video_path_draw = os.path.join(self.output_dir_video_draw, get_current_date_string(), get_current_hour_string())
        
        if self.alias == "":
            self.video_output_name = 'output_original.avi'
            self.video_output_draw_name = 'output_draw.avi'           
            video_path_original = os.path.join(video_path_original, get_current_hour_string() + ".avi")
            video_path_draw = os.path.join(video_path_draw, get_current_hour_string() + ".avi")
        else:
            self.video_output_name = self.alias + '_output_original.avi'
            self.video_output_draw_name = self.alias + '_output_draw.avi'            
            video_path_original = os.path.join(video_path_original, get_current_hour_string() +"_"+ self.alias + ".avi") 
            video_path_draw = os.path.join(video_path_draw, get_current_hour_string() +"_"+ self.alias + ".avi")        
        
        
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
            
        if not os.path.exists(os.path.join(self.output_dir, self.alias)):
            create_dir(os.path.join(self.output_dir, self.alias))
        
        if not self.is_dir:                
            print("[Info] Camera status {url}, {alias}:{s}".format(url=self.video_url,
                                                               alias=self.alias, s=self.cap.isOpened()))
           
        
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
            
    
    def init_Yolo(self):
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file = self.config_file,
            data_file = self.data_file,
            weights = self.weights_file)
        
        if self.ensemble: 
            self.network2, self.class_names2, self.class_colors2 = darknet.load_network(
                config_file = self.config_file2,
                data_file = self.data_file2,
                weights = self.weights_file2)  
        
        if self.target_classes == None:
            self.target_classes = self.class_names
        
        self.non_target_classes = list(set(self.class_names).difference(self.target_classes)) # Yolo detect non-target results
                 
                 
    def read_dir_img(self):        
        
        for img_name in os.listdir(self.video_url):
            try:
                if img_name.split(".")[-1] == "jpg" or img_name.split(".")[-1] == "jpeg"\
                 or img_name.split(".")[-1] == "png":
                    frame =  cv2.imread(os.path.join(self.video_url, img_name))                                     
                    
                    self.frame_id += 1 # record the current frame numbers
                    frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) # original image
                    darknet_image = darknet.make_image(self.W, self.H, 3)
                    darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
                    self.detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
                    darknet.free_image(darknet_image)
                    
                    print("Detection img:", img_name)   
                    darknet.print_detections(self.detections, True)
                    
                    # interate all vertex to filter the detect object
                    each_vertex_key = list(self.vertex.keys())
                    self.detect_target = []
                    for each_vertex in self.vertex[each_vertex_key]:
                        self.detect_target.append(detect_filter(self.detections, self.target_classes, each_vertex))
                
                    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    image = draw_boxes(self.detections, image, self.class_colors, self.target_classes, self.vertex)
                    cv2.imwrite(os.path.join(self.output_dir_no_detect, str(self.frame_id)+".jpg"), image)
                        
            except Exception as e:
                print("[Error] Read the image from directory error:", e)
            
                 
    def video_capture_wo_threading(self): 
        self.ret, self.frame = self.cap.read() 
            
        if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong
            print("[Info] Video detection is finished...")
            self.stop()            
        else:          
            if self.set_W != None and self.set_H != None:
                self.frame = cv2.resize(self.frame.copy(), (int(self.W), int(self.H)), interpolation=cv2.INTER_AREA)
            
            if self.save_video_original:
                self.save_video_frame(self.frame)
                
    
    def video_capture(self):
        t = threading.currentThread()   
        time.sleep(1) # waiting for loading yolo model
        
        while getattr(t, "do_run", True):
            self.ret, self.frame = self.cap.read() 
            
            if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong 
                print("[Error] Reconnecting:{group}:{alias}:{url} ".format(group=self.group, alias=self.alias, url=self.video_url))
                self.reconnect_rtsp_num += 1
                self.cap.release()
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.video_url)
                time.sleep(5)
                
                if self.reconnect_rtsp_num % 3 == 0:
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
                if self.set_W != None and self.set_H != None:
                    self.frame = cv2.resize(self.frame.copy(), (int(self.W), int(self.H)), interpolation=cv2.INTER_AREA)
                if self.save_video_original:
                    self.save_video_frame(self.frame)
                    
                    
    def prediction(self):        
        last_time = time.time() # to compute the fps
        cnt = 0  # to compute the fps
        predict_time_sum = 0  # to compute the fps  
        save_path_img = None # image saving path with Yolo detected bounding bbox 
        save_path_img_orig = None # image saving path without Yolo detected bounding bbox 
        t = threading.currentThread() # get this function threading status        
        
        self.init_Yolo()
        
        if self.is_dir:
            read_dir_img = self.read_dir_img()
            print("[Info] Directory image detection is finished...")
            self.stop() 
        
        while getattr(t, "do_run", True):
            cnt+=1 # for skip_frame
            
            if not self.is_threading and not self.is_dir:
                self.video_capture_wo_threading()
                
                if self.skip_frame != None and cnt % self.skip_frame != 0:
                    continue
            
            if not self.cap.isOpened() or not self.ret:                
                time.sleep(1)
                continue
    
            self.frame_id += 1 # record the current frame numbers
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) # original image
            darknet_image = darknet.make_image(self.W, self.H, 3)
            darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
            
            predict_time = time.time() # get start predict time
            self.detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
                
            if self.ensemble and not self.is_test_SSIM:
                detections2 = darknet.detect_image(self.network2, self.class_names2, darknet_image, thresh=self.thresh)
                all_class_names = set(self.class_names).union(self.class_names2)
                self.detections = nms(all_class_names, self.detections, detections2)                
            
            predict_time_sum += (time.time() - predict_time) # add sum predict time
            
            if self.is_dir: darknet.print_detections(self.detections, True) # print detection when read directory
            darknet.free_image(darknet_image)
    
            # interate all vertex to filter the detect object
            vertex_key = list(self.vertex.keys())
            self.detect_target = []
            for each_vertex_key in vertex_key:
                for det in detect_filter(self.detections, self.target_classes, self.vertex[each_vertex_key]):
                    self.detect_target.append(det)
            
            # convert to BGR image
            image_detect = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)            
            
            # draw the image bbox
            if self.obj_trace: 
                image_detect = self.object_tracker(image_detect)    
            else:
                image_detect = draw_boxes(self.detections, image_detect, self.class_colors, self.target_classes, self.vertex)            
            
            # save the difference image if occurred
            if self.catch_miss or self.is_test_SSIM:
                if self.count_after_detect > 0:                    
                    self.count_after_detect -= 1                    
                    image_detect, _, _ = self.find_moving_obj(self.bc_img, image_detect) # SSIM process                  
            
            # draw the polygon
            if self.is_draw_polygon: 
                image_detect = draw_polylines(image_detect, self.vertex)      
            
            # save draw bbox image
            if self.save_img and len(self.detect_target) > 0:                
                save_path_img = self.save_img_draw(image_detect)                 
            
            # save oiginal image
            if self.save_img_original and len(self.detect_target) > 0:
                save_path_img_orig = self.save_img_orig(self.frame)            
            
            # set counter after catch Yolo detect
            if self.is_save_img_after_detect or self.catch_miss or self.is_test_SSIM:
                if len(self.detect_target) > 0:
                    self.count_after_detect = 3
                    
            # save a series of image after Yolo detect
            if self.is_save_img_after_detect and self.count_after_detect > 0:
                save_path_after_img = self.save_after_img(self.frame)
                self.count_after_detect -= 1
            
            # save video with draw            
            if self.save_video:
                self.save_video_draw(image_detect)               
            
            # callback function for user            
            if self.is_count_obj and len(self.count_obj) > 0:
                self.is_return_results = False                
                max_id = max(self.count_obj, key=self.count_obj.get)
                
                if self.count_obj[max_id] >= self.count_obj_thresh: 
                    self.is_return_results = True                   
                    self.count_obj[max_id] = 0                   
                
            if len(self.detect_target) > 0:
                if self.is_count_obj == False or \
                    (self.is_count_obj == True and self.is_return_results == True):
                    self.__trigger_callback(save_path_img, self.group, self.alias, self.detect_target)
            
            # Compute FPS
            if time.time() - last_time > 30:
                self.print_msg("[Info] FPS:{fps} , {alias}".format(alias=self.alias, fps=cnt / (time.time()-last_time)))
                self.print_msg("[Info] Predict FPS:{fps} , {alias}".format(alias=self.alias, fps=cnt / predict_time_sum))
                last_time = time.time()
                cnt = 0
                predict_time_sum = 0 
                
                        
    def find_moving_obj(self, bc_img: np.ndarray, image: np.ndarray) -> tuple:
        is_find_moving_obj2 = False    
        
        self.frame_lum = avg_color_img(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY))                    
        self.detect_target_wo_vertex = detect_filter(self.detections, self.target_classes, None)    
        
        # get all Yolo detect non-target results
        vertex_key = list(self.vertex.keys())
        self.diff_detect_target = []
        for each_vertex_key in vertex_key:
            for det in detect_filter(self.detections, self.target_classes, self.vertex[each_vertex_key]):
                self.diff_detect_target = [].append(det)
        
        # if no background 
        if len(self.bc_list) == 0:
            self.bc_list.append({str(self.frame_lum)+".jpg":{"brightness":self.frame_lum, "img":self.frame}})
            cv2.imwrite(os.path.join(self.output_dir_img_bc, str(self.frame_lum) + ".jpg"), self.frame)           
        
        # find the most similar image and background
        if len(self.bc_list) > 1:
            max_diff_val = 0                   
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                    

            for bc_idx in range(len(self.bc_list)):
                bc_img_name = list(self.bc_list[bc_idx].keys())[0] # get each image name in bc_list
                bc_img = self.bc_list[bc_idx][bc_img_name]["img"] # get RGB image
                bc_gray = cv2.cvtColor(bc_img, cv2.COLOR_BGR2GRAY)
                score = structural_similarity(image_gray, bc_gray)

                if score > max_diff_val:
                    self.bc_img_idx = bc_idx
                    self.bc_img_name = bc_img_name
                    self.bc_img = bc_img
                    max_diff_val = score
        else:
            self.bc_img_name = list(self.bc_list[0].keys())[0] 
            self.bc_img_idx = 0
            self.bc_img = self.bc_list[0][self.bc_img_name]["img"]        
        
        
        # Using SSIM to compare the background and current frame
        score, SSIM_img = SSIM(self.bc_img, self.frame)
        is_find_moving_obj = self.iterate_SSIM_result(SSIM_img)
        
        # Using SSIM to compare the tmp background and current frame
        if is_find_moving_obj:
            score, SSIM_img2 = SSIM(self.SSIM_tmp_bc, self.frame)
            is_find_moving_obj2 = self.iterate_SSIM_result(SSIM_img2)        
        
        if is_find_moving_obj2:
            self.SSIM_tmp_bc = self.frame # update tmp background
            cv2.imwrite(os.path.join(self.output_dir_SSIM_original_img, str(self.frame_id) + ".jpg"), self.frame)    
            cv2.imwrite(os.path.join(self.output_dir_SSIM_draw_img, str(self.frame_id) + ".jpg"), image)
            cv2.imwrite(os.path.join(self.output_dir_SSIM_img, str(self.frame_id) + ".jpg"), diff)
    
        return image, score, is_find_moving_obj2
    
    
    def iterate_SSIM_result(self, SSIM_img: np.ndarray, SSIM_diff_area_size: int = 3000) -> bool:
        is_find_moving_obj = False
        
        thresh = cv2.threshold(SSIM_img, self.SSIM_thresh, 255, cv2.THRESH_BINARY_INV)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        for c in contours:                 
            area = cv2.contourArea(c)
            
            if area > SSIM_diff_area_size:
                lx_diff, ly_diff, w_diff, h_diff = cv2.boundingRect(c)
                w_diff = int(w_diff)
                h_diff = int(h_diff)
                rx_diff = int(lx_diff + w_diff)
                ry_diff = int(ly_diff + h_diff)
                x_diff = int(lx_diff + w_diff/2)
                y_diff = int(ly_diff + h_diff/2)                
                
                # check the moving object whether in the vertex 
                vertex_key = list(self.vertex.keys())
                self.diff_detect_target = []
                for each_vertex_key in vertex_key:                                
                    if self.vertex != None:                    
                        if not(is_in_hull(self.vertex, (lx_diff, ly_diff)) or is_in_hull(self.vertex, (lx_diff, ry_diff))\
                            or is_in_hull(self.vertex, (rx_diff, ly_diff)) or is_in_hull(self.vertex, (rx_diff, ry_diff))):  
                            continue
                            
                is_find_moving_obj = True
                break
        
        return is_find_moving_obj
        
        
    # https://github.com/adipandas/multi-object-tracker.git
    def object_tracker(self, image: np.ndarray) -> np.ndarray:
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
            
        output_tracks = self.tracker.update(np.array(boxes), np.array(confidence), np.array(class_ids))
        
        self.detect_target = [] # re-assigned each bbox
        for track in output_tracks:
            frame, idx, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
            assert len(track) == 10
        #   print(track)
            bbox = (bb_left+bb_width/2, bb_top+bb_height/2,  bb_width, bb_height)
            self.detect_target.append((self.target_classes[0], confidence, bbox, idx)) # put the result to detect_target 
            
            # count id number and determine if post results            
            if self.is_count_obj == True:
                if self.count_obj.get(idx) == None:
                    self.count_obj[idx] = 0
                else:
                    self.count_obj[idx] += 1
                
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
    
    
    # https://github.com/wmuron/motpy.git
    def object_tracker_motpy(self, image: np.ndarray) -> np.ndarray:           
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
    
    
    def set_listener(self, on_detection: callable):
        self.detection_listener = on_detection
        
        
    def __trigger_callback(self, save_path_img: str, group: str, alias: str, detect: str):        
        if self.detection_listener is None:
            return         
        self.detection_listener(save_path_img, group, alias, detect)
  

    def get_current_frame(self):        
        return self.frame
        
        
    def stop(self): 
        for t in self.th:
            t.do_run = False
            
        self.write_log("[Info] Stop the program.")
        self.cap.release()
        
        print('[Info] Stop the program: Group:{group}, alias:{alias}, URL:{url}'\
              .format(group=self.group, alias=self.alias, url=self.video_url))      
        
        
    def restart(self):
        self.stop()        
        self.write_log("[Info] Restart the program")
        restart()
        
        
    def write_log(self, msg: str):     
        f= open('log.txt', "a")    
        f.write("Time:{time}, {msg}, Group:{group}, alias:{alias}, URL:{url} \n"\
                .format(time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), \
                 group=self.group, alias=self.alias, url=self.video_url, msg=msg))
        f.close()
       
    
    def print_msg(self, msg: str):
        if self.display_message == True:            
            print(msg)
            
            
    def save_video_frame(self, frame: np.ndarray):        
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video, get_current_date_string(), 
                        get_current_hour_string())
            if self.alias == "":                
                video_path_name = os.path.join(video_path, get_current_hour_string() + ".avi")                
            else:
                video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".avi")   
            
        else: # video input, so output wihtout time directory
            video_path_name = self.video_output_name
        
        if not os.path.exists(video_path_name):
            if self.is_threading: create_dir(video_path)                        
            self.frame_original = cv2.VideoWriter(video_path_name, self.fourcc, 30.0, (self.W, self.H))
            print("[Info] {alias} Set video frame writer. Width={W}, Height={H} ".format(alias=self.alias, W=self.W, H=self.H))
            
        self.frame_original.write(frame)
        
        if self.video_expire_day != None:
            del_dir(self.output_dir_video, self.video_expire_day)
                
    
    def save_video_draw(self, frame: np.ndarray):        
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video_draw, get_current_date_string(), 
                        get_current_hour_string())
            if self.alias == "":                
                video_path_name = os.path.join(video_path, get_current_hour_string() + ".avi")                
            else:
                video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".avi")   
            
        else: # video input, so output wihtout time directory
            video_path_name = self.video_output_draw_name            
            
        if not os.path.exists(video_path_name):
            if self.is_threading: create_dir(video_path)
            self.frame_draw = cv2.VideoWriter(video_path_name, self.fourcc, 20.0, (self.W, self.H))
            print("[Info] {alias} Set video draw writer. Width={W}, Height={H} ".format(alias=self.alias, W=self.W, H=self.H))
            
        self.frame_draw.write(frame)
        
        if self.video_expire_day != None:
            del_dir(self.output_dir_video_draw, self.video_expire_day)
            
    
    def save_img_draw(self, image: np.ndarray) -> str:              
        # if input is video, dir will not separate into the time directory
        if not self.is_threading:
            img_path = self.output_dir_img_draw
        else:
            img_path = os.path.join(self.output_dir_img_draw, get_current_date_string(), get_current_hour_string())     
        
        if not os.path.exists(img_path):
            create_dir(img_path)      
            
        if self.alias == "":
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) + ".jpg")
        else:
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) +"_"+ self.alias + ".jpg")               
                  
        cv2.imwrite(img_path_name, image)
        
        return img_path_name
    
    
    def save_img_orig(self, image: np.ndarray) -> str:        
        # if input is video, dir will not separate into the time directory
        if not self.is_threading:
            img_path = self.output_dir_img
        else:
            img_path = os.path.join(self.output_dir_img, get_current_date_string(), get_current_hour_string())
        
        if not os.path.exists(img_path):
            create_dir(img_path)            
            
        if self.alias == "":
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) + ".jpg")
            txt_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) + ".txt")
        else:
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) +"_"+ self.alias + ".jpg")            
            txt_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) +"_"+ self.alias + ".txt")            
                
        cv2.imwrite(img_path_name, image)
        
        # save class bbox to txt file
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
    