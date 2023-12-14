import cv2
import sys
import numpy as np

from collections import namedtuple
from shapely.geometry import Point, Polygon
from darknet import darknet
from libs.utils import *
from loguru import logger


sys.path.insert(1, 'multi-object-tracker')
sys.path.insert(1, 'ByteTrack')

from motpy import Detection, MultiObjectTracker
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from motrackers import SORT, CentroidKF_Tracker, CentroidTracker, IOUTracker
from motrackers.utils import draw_tracks

class Tracker():
    def __init__(self, tracker_mode="BYTE", track_buffer=30, #object tracking setting
                 enable_people_counting=False, in_area=None, out_area=None, #people counting setting
                 ) -> None:
        self.H = None
        self.W = None
        """
          Initialize object tracking parameter.
        """
        self.tracker_buffer = track_buffer
        self.id_storage = [] # Save each object ID
        self.bbox_colors = {} # Store object tracking object bbox color       
        BYTEArgs = namedtuple('BYTEArgs', 'track_thresh, track_buffer, match_thresh, mot20')
        args = BYTEArgs(track_buffer=self.tracker_buffer, track_thresh=0.6, match_thresh=0.9, mot20=False)
        self.tracker = BYTETracker(args)#default tracker
        self.tracker_mode = tracker_mode.lower()
        if tracker_mode == "center":
            self.tracker = CentroidTracker(max_lost=3, tracker_output_format='mot_challenge')
        elif tracker_mode == "sort":
            SORT(max_lost=self.tracker_buffer, tracker_output_format='mot_challenge', iou_threshold=0.1)
        elif tracker_mode == "iou":
            IOUTracker(max_lost=self.tracker_buffer, iou_threshold=0.1, min_detection_confidence=0.4, max_detection_confidence=0.7, tracker_output_format='mot_challenge'),
        else:
            ValueError(f"There have no tracker mode {tracker_mode}")
            
        
        """
          Initialize people counting parameter.
        """
        
        self.totalIn = 0#count how many people enter the area totally
        self.currentIn = 0#how many people are in the area right now
        self.enable_people_counting = enable_people_counting
        self.mergeDistanceThreshold = 70
        self.splitDistanceThreshold = 100
        # self.draw_square = draw_square
        # self.countInArea_draw = np.array([[0, 1080],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1080, 590],[1425, 1007],[1525, 985],[1574, 814],[1920, 1080] ], np.int32)#The polygon of the area you want to count people inout
        self.in_area = np.array(in_area)#Make the area of bottom lower because some people walk in from there. If not making lower, system will count those person
        self.out_area = np.array(out_area)
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
        self.unreliableID = list()

    
    def update(self, detections, frame=np.zeros((1,1,1))) -> None:
        self.H, self.W = frame.shape[:2]
        self.detect_target = detections
        if self.tracker_mode == "byte":
            self.drawed_image, self.detect_target = self.object_tracker_BYTE(frame, self.detect_target)
        else:
            self.drawed_image, self.detect_target = self.object_tracker(frame, self.detect_target)
        # elif self.tracker_mode == 5:
        #     self.drawed_image,  self.detect_target = self.object_tracker_motpy(frame, self.detect_target)
            
            
        if self.enable_people_counting:
            self.drawed_image = self.draw_info(self.drawed_image)
            if len(self.detect_target) > 0:
                self.drawed_image = self.face_detection(frame, self.drawed_image)
                self.people_counting()
                
        # Draw people counting polygon
        if self.in_area is not None:
            cv2.polylines(self.drawed_image, pts=[self.in_area], isClosed=True, color=(0,0,255), thickness=3)#draw count in area
            
        if self.out_area is not None:
            cv2.polylines(self.drawed_image, pts=[self.out_area], isClosed=True, color=(255,0,0), thickness=3)#draw count out area
            
        
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
            left, top, right, bottom = bbox2points(r[2])
            boxes.append([left, top, width, height])
            confidence.append(int(float(r[1])))
            class_ids.append(int(self.target_classes.index(r[0])))
        
        # `output_tracks` is a list with each element containing tuple of
        # (<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>)
        output_tracks = self.tracker.update(np.array(boxes), np.array(confidence), 
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
            
            color = getColor(id)
                
            cv2.rectangle(image, (int(bb_left), int(bb_top)), 
                          (int(bb_left+bb_width), int(bb_top+bb_height)), color, 2)
            # image = draw_tracks(image, output_tracks) # Draw the object ID
            
            
            # Put the score and class to the image
            txt = str(obj_name) + " "+ str(confidence)
            cv2.putText(image, txt, (int(bb_left), int(bb_top-7)) ,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0,255,0))        
        
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
            
        self.tracker.step(detections=[Detection(box=b, score=s, class_id=l)\
                                            for b, s, l in zip(boxes, scores, class_ids)])
        tracks = self.tracker.active_tracks(min_steps_alive=3)
        
        
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
    
    def object_tracker_BYTE(self, image: np.ndarray, detect_target: list) -> tuple:
        #[className, score, (cx, cy, W, H)] -> [x1, y1, x2, y2, score]
        dets = list()
        if len(detect_target) > 0:
            for det in detect_target:
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
        
        online_targets = self.tracker.update(dets, [1080, 1920], [1080, 1920])
        
        detWithID = []
        for track in online_targets:
            t, l, w, h = track.tlwh
            id = track.track_id
            cx = int( (t + w / 2))
            cy = int( (l + h / 2))
            color = getColor(id)
            cv2.rectangle(image, (int(t), int(l)), (int(t + w), int(l + h)), color, 3)
            cv2.putText(image, str(id), (cx, cy - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0,255,0), thickness=2)
            
            # save results
            detWithID.append(['person', track.score, [cx, cy, w, h], id])

        
        return image, detWithID
    
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
            # cv2.putText(self.infoImage, text, (10, self.H - ((i * 50) + 100)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255, 255), 5)
            

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(image, text, (265, self.H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # cv2.putText(self.infoImage, text, (265, self.H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255, 255), 2)
            
            
        return image
    
    def people_counting(self):
        
        if 0 < len(self.detect_target) <= 5:
            self.merge()
            self.split()
        
            
        for det in self.detect_target:
            if len(det) < 5 or None in det[4]:#center not None
                continue
            id = det[3]
            center_x, center_y = det[4]
            w, h = det[2][2:]
            countInAreaPolygon = Polygon(self.in_area)
            countOutAreaPolygon = Polygon(self.out_area)
            currentCentroid = Point((center_x, center_y))
            if center_x <= 0 or center_x >= self.W or center_y <= 0 or center_y >= self.H:#out of boundary
                continue
            
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
                                          "outIn":outIn
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
            OutAndIn = not self.lastCentroids[id]["countIn"] and self.lastCentroids[id]["countOut"] and not self.lastCentroids[id]["outIn"]
            
            if isGetIn:#get in and not counted
                print("Normal add:", id)
                self.totalIn += 1
                self.currentIn += 1
                self.lastCentroids[id]["countIn"] = True
                self.lastCentroids[id]["countOut"] = False
                
                
            if OutAndIn:
                print("Out and in", id)
                self.currentIn += 1
                self.lastCentroids[id]["outIn"] = True
                
                
            if isGetOut:
                if self.mergedIDs.get(id, None) is not None:
                    print("Normal merge out:", id, self.mergedIDs[id])
                    for i in self.mergedIDs[id]:
                        if not self.lastCentroids[i]["countOut"]:#id not count out
                            self.lastCentroids[i]["countOut"] = True
                            self.currentIn -= 1
                        
                else:
                    print("Normal out:", id)
                    self.currentIn -= 1
    
                self.lastCentroids[id]["countOut"] = True
                self.lastCentroids[id]["outIn"] = True
                self.lastCentroids[id]["countIn"] = False 
                
                
            self.lastCentroids[id]["center"] = (center_x, center_y)#update id's center
        
        self.lastDetections = self.detect_target                   
    
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
                    
                    cv2.putText(image, message, (pairCenterX, pairCenterY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)#draw distance below the line
                    
                    #draw web streaming image
                    self.socialDistanceImage = np.zeros((image.shape[0], image.shape[1], 4))
                    cv2.line(self.socialDistanceImage, (int(centroids[index][0]), int(centroids[index][1]) ), (int(centroids[pointIndex][0]), int(centroids[pointIndex][1]) ), (255,0,0, 255), 2)
                    cv2.putText(self.socialDistanceImage, message, (pairCenterX, pairCenterY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 255), 2)#draw distance below the line

        return image
    
    #https://google.github.io/mediapipe/solutions/face_detection.html
    def face_detection(self, detectImage, drawImage):
        #only use yolo center
        for index, detection in enumerate(self.detect_target):#for each detection
            self.detect_target[index] = list(self.detect_target[index])
            left, top, right, bottom = darknet.bbox2points(detection[2])
            imageWidth, imageHeight = right - left, bottom - top
            centerX, centerY = (left + imageWidth/2), (top + imageHeight)#use the yolo bbox info to define center
            self.detect_target[index].append((centerX, centerY))
            cv2.circle(drawImage, (int(centerX), int(centerY)), 8, (255,0,0), -1)
        
        return drawImage
        
        
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
    