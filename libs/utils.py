from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import datetime
import os
import shutil
import sys

import cv2
import matplotlib.path as mplPath
import numpy as np

from skimage.metrics import structural_similarity


def detect_filter(detections, target_classes, vertex, only_detect_center_bbox=False):
    """
      Filter the non-target objects and target objects which outside the vertex.
      
      Args:
        detections: [(left, top, right, bottom), (left2, top2, right2, bottom2), ...]
        target_classes: ["target1", "target2", ...]
        vertex: [(x1,y1), (x2,y2), (x3,y3), ...]
        only_detect_center_bbox: only detect center point of bbox or containing four corner of bbox.
    """
    results = []
    
    for label, confidence, bbox in detections:        
        left, top, right, bottom = bbox2points(bbox)
        
        # Filter the target class
        if target_classes != None:
            if label not in target_classes:
                continue            
        
        # Filter the bbox outside the vertex
        if vertex == None:            
            results.append((label, confidence, bbox, None))
        else:
            center_x = (left + right)/2
            center_y = (top + bottom)/2
            
            # Only detect bounding box center point
            if only_detect_center_bbox:
                if is_in_hull(vertex,(center_x, center_y)):          
                    results.append((label, confidence, bbox, None))
            else:
                if is_in_hull(vertex,(left, top))\
                    or is_in_hull(vertex,(left, bottom))\
                    or is_in_hull(vertex,(right, top))\
                    or is_in_hull(vertex,(right, bottom))\
                    or is_in_hull(vertex,(center_x, center_y)):          
                    results.append((label, confidence, bbox, None))

    return  results
    

def bbox2points(bbox):
    """
      From bounding box yolo format to corner points cv2 rectangle.
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def draw_boxes(detections, image, colors, target_classes):
    """
      Draw the detections results bounding box to image.
    """    
    
    for label, confidence, bbox, _ in detections:        
        left, top, right, bottom = bbox2points(bbox)        
        
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
        
    return image


def draw_polylines(img: np.ndarray, vertex: list):
    """
      Draw the vertex to image.
    """
    vertex_key = list(vertex.keys())
    
    for each_vertex_key in vertex_key:
        if vertex[each_vertex_key] is not None:
            pts = np.array(vertex[each_vertex_key], np.int32)
            red_color = (0, 0, 255) # BGR
            cv2.polylines(img, [pts], isClosed=True, color=red_color, thickness=3)
    
    return img


def get_current_date_string():
    now_dt = datetime.datetime.now()
    return "{:04d}-{:02d}-{:02d}".format(now_dt.year, now_dt.month, now_dt.day)


def get_current_hour_string():
    now_dt = datetime.datetime.now()
    return "{:02d}".format(now_dt.hour)


def create_dir(output_dir: str):
    """
      Create the `output_dir` folder.
    """
    try:        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    except OSError as e:
        print("[Error] Create dirictory error:", e)
        

def del_dir(output_dir :str, expire_day: int):   
    """
      Delete all files in the <output_dir> older than <expire_day> days.
    """
    date_dir = []
    expire_date = (datetime.datetime.today() - datetime.timedelta(days=expire_day)).strftime("%Y-%m-%d")
    expire_date = datetime.datetime.strptime(expire_date, '%Y-%m-%d').date()
    
    # Get all output_dir
    for directory in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, directory)):
            try:
                directory_date = datetime.datetime.strptime(directory, '%Y-%m-%d').date()            
                date_dir.append(directory_date)
            except:
                pass
    
    # Delete all expire day directory
    for each_date_dir in date_dir:
        if each_date_dir < expire_date :
            try:
                del_date_path = os.path.join(output_dir, each_date_dir.strftime("%Y-%m-%d"))
                shutil.rmtree(del_date_path)
                print("[Info] Delete ", del_date_path, " successful.")
            except OSError as e:
                pass

                
def re_make_dir(path: str):    
    """
      Delete all file in the folder.
    """
    try:
        shutil.rmtree(path)        
    except OSError as e:
        print ("[Warning]: %s - %s." % (e.filename, e.strerror))
        
    os.makedirs(path)
    
    
def restart():    
    """
      Restart the program.
    """
    os.execv(sys.executable, ['python'] + sys.argv)

      
# https://www.tutorialspoint.com/what-s-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
def is_in_hull(vertex:list, p:tuple) -> bool:    
    """
      Returns true if the point p inside the polygon
    """
    
    poly_path = mplPath.Path(np.array(vertex))
    
    return poly_path.contains_point(p)


def bb_intersection_over_union(boxA, boxB):
    """
      Return two bounding box IOU.
    """
    boxA = [int(x) for x in bbox2points(boxA)]
    boxB = [int(x) for x in bbox2points(boxB)]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou


def avg_color_img(value: int):
    """
      Calculate the average value of image.
    """
    average_color_row = np.average(value, axis=0) # average each row first
    average_color = np.average(average_color_row, axis=0)
    
    return average_color


def get_img_brightness(img: np.ndarray):
    """
      Get image brightness. (HSV)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    average_color = avg_color_img(v)   

    return average_color


# https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
def nms(all_class_names, detections, detections2, threshold=0.5) -> list:
    """
      Compute ensemble model NMS detection results.
    """    
    
    if len(detections) == 0 and len(detections2) == 0:
        return []
    
    for det in detections2:
        detections.append(det)        
    
    nms_detections = []     
    picked_boxes = []
    picked_score = []
    
    for name in all_class_names:
        bounding_boxes  = [] # return the original results
        boxes = [] # compute NMS
        confidence_score = []   
        picked_boxes = []
        picked_score = []
        
        for det in detections:
            if det[0] == name:                
                boxes.append(bbox2points(det[2])) 
                bounding_boxes.append(det[2]) 
                confidence_score.append(round(float(det[1])/100, 4))
                
        if len(bounding_boxes) == 0:
            continue
            
        boxes = np.array(boxes)
        score = np.array(confidence_score)
        
        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]
        
        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h
            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
            
            left = np.where(ratio < threshold)
            order = order[left]
        
        for i in range(len(picked_boxes)):
            nms_detections.append((name, str(picked_score[i]*100), picked_boxes[i]))
    
    return nms_detections

    
# https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
def SSIM(bc_img: np.ndarray, image: np.ndarray):        
    """
      Implement SSIM function.
    """
    before_gray = cv2.cvtColor(bc_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, win_size=25, full=True)
    SSIM_img = (diff * 255).astype("uint8")
    
    return score, SSIM_img


def parse_args():
    """
      Get the user input parameters in conmand line.
    """
    parser = ArgumentParser(description='YoloTalk args',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-url", "--video_url", type=str, default="output.avi", 
                        help="The video path or rtsp URL.")
    parser.add_argument("-threading", "--is_threading", dest="is_threading", action='store_true', 
                        help="If input is RTSP URL, set True. If input is video file, set False.")
    parser.add_argument("-alias", "--alias", type=str, default="demo", 
                        help="Named for output file.")
    parser.add_argument("-thresh", "--thresh", type=float, default=0.5, 
                        help="The threshold of Yolo.")
    parser.add_argument("-SSIM", "--using_SSIM", dest="using_SSIM", action='store_true',
                        help="Set True to using SSIM.")

    parser.set_defaults(is_threading=False)
    parser.set_defaults(using_SSIM=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    pass
    
