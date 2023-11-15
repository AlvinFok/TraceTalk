import cv2
import numpy as np
import datetime
import os
import shutil
import sys
import matplotlib.path as mplPath

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
    
    for det in detections:        
        label, confidence, bbox = det[0], det[1], det[2]
        left, top, right, bottom = bbox2points(bbox)
        
        # Filter the target class
        if target_classes != None:
            if label not in target_classes:
                continue            
        
        # Filter the bbox outside the vertex
        if vertex == None:            
            results.append(det)
        else:
            center_x = (left + right)/2
            center_y = (top + bottom)/2
            
            # Only detect bounding box center point
            if only_detect_center_bbox:
                if is_in_hull(vertex,(center_x, center_y)):          
                    results.append(det)
            else:
                if is_in_hull(vertex,(left, top))\
                    or is_in_hull(vertex,(left, bottom))\
                    or is_in_hull(vertex,(right, top))\
                    or is_in_hull(vertex,(right, bottom))\
                    or is_in_hull(vertex,(center_x, center_y)):          
                    results.append(det)

    return  results
    
def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def draw_boxes(detections, image, colors, target_classes, vertex):
    """
    Target will be drawed if vertex is None or bounding box in vertex
    """
    detections = detect_filter(detections, target_classes, vertex)
    
    for label, confidence, bbox, _ in detections:        
        left, top, right, bottom = bbox2points(bbox)        
        
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
        
    return image

def draw_polylines(img, vertex):
    if vertex != None:
        pts = np.array([vertex], np.int32)
        red_color = (0, 0, 255) # BGR
        cv2.polylines(img, [pts], isClosed=True, color=red_color, thickness=3) # BGR
    
    return img

def get_current_date_string():
    now_dt = datetime.datetime.now()
    return "{:04d}-{:02d}-{:02d}".format(now_dt.year, now_dt.month, now_dt.day)

def get_current_hour_string():
    now_dt = datetime.datetime.now()
    return "{:02d}".format(now_dt.hour)

def create_dir(output_dir):
    try:
#         path = os.path.join(output_dir, get_current_date_string(), get_current_hour_string())
        path = os.path.join(output_dir)
        os.makedirs(path, exist_ok=True)
        return path
    
    except OSError as e:
        print("Create dirictory error:",e)
        

def del_dir(output_dir, expire_day):    
    for i in range(expire_day, expire_day+10):
        yesterday_dir =  os.path.join(output_dir, ((datetime.datetime.today() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")))
        if os.path.exists(yesterday_dir):
            try:
                shutil.rmtree(yesterday_dir)
                print("[Info] Delete ", yesterday_dir, " successful.")
            except OSError as e:
                print("[Error] Delete video error:",e)

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
    os.execv(sys.executable, ['python'] + sys.argv)


# https://www.tutorialspoint.com/what-s-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
# Returns true if the point p lies 
# inside the polygon[] with n verticesdef is_in_hull(vertex:list, p:tuple) -> bool:    
    poly_path = mplPath.Path(np.array(vertex))
    
    return poly_path.contains_point(p)


def getDets(dets:np.ndarray, frameID:int) -> np.ndarray:
    indices = np.where(dets[:, 0] == frameID)
    return dets[indices]

def getNpData(filePath:str) -> np.ndarray:
    data = open(filePath).readlines()
    data = [i.split(',') for i in data]
    data = np.array(data, dtype=float)
    return data

def getColor(id):
    return ( (id * 5 + 20) %255, (id * 66 + 3) %255, (id * 50 - 1) %255)

def drawTracks(image, tracks):
    for i in tracks:
        cx, cy, w, h = i[2]
        id = i[3]
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(x1 + w), int(y1 + h)
        color = getColor(id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, str(id), (int(cx), int(cy) - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0,255,0), thickness=2)
            
    return image

# https://www.tutorialspoint.com/what-s-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
def is_in_hull(vertex:list, p:tuple) -> bool:    
    """
      Returns true if the point p inside the polygon
    """
    
    poly_path = mplPath.Path(np.array(vertex))
    
    return poly_path.contains_point(p)