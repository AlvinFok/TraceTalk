# coding:utf-8

import libs.DAN as DAN
import LineNotify
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import json
import cv2
import numpy as np
import argparse

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")



#config of BYTE tracker
class Arg():
    def __init__(self):
        self.track_thresh = 0.6
        self.track_buffer = 30
        self.match_thresh = 0.9
        # self.min-box-area = 100
        self.mot20 = False
        
def xy2tlrb(detections):
    dets = list()
    for cx, cy, W, H, score in detections:
        x1 = int(cx - W / 2)
        y1 = int(cy - H / 2)
        x2 = x1 + W
        y2 = y1 + H
        dets.append([x1, y1, x2, y2, score])
    
    return dets

def getColor(id):
    return ( (id * 2 +5)%255, (id * 5 + 20) %255, (id * 3 +66)%255 )

if __name__ == '__main__': 
    
    ServerURL = 'https://6.iottalk.tw'    
    Reg_addr = '567890' #if None, Reg_addr = MAC address
    
    DAN.profile['dm_name']='Yolo'
    DAN.profile['df_list']=['yperson(json)-I', 'yperson(json)-O']
    DAN.profile['d_name']= 'Yolo(56789)'

    DAN.device_registration_with_retry(ServerURL, Reg_addr)
    # DAN.deregister()  #if you want to deregister this device, uncomment this line
    # exit()            #if you want to deregister this device, uncomment this line
    
    args = Arg()
    tracker = BYTETracker(args)
    while(True):
        data = DAN.pull('yperson(json)-O')
        if data == None:
            continue
        
        
        data = json.loads(data[0])
        dataArray = np.array(data["detections"], dtype=object)
        detections = dataArray[:,2]#(cx, cy, w, h)
        detections = np.array(detections.tolist())
        detections = np.hstack((detections, dataArray[:,1].reshape(-1,1)))#score
        
        detections_tlrb = xy2tlrb(detections)
        detections_tlrb = np.array(detections_tlrb, dtype=float)
        online_targets = tracker.update(detections_tlrb, [1080, 1920], [1080, 1920])
        # logger.info(f"{online_targets}")
        imagePath = data['imgPath']
        
        detWithID = []
        for track in online_targets:
            t, l, w, h = track.tlwh
            id = track.track_id
            cx = int( (t + w / 2))
            cy = int( (l + h / 2))
            # assign each id with a color
            color = getColor(id)
            # cv2.rectangle(image, (int(t), int(l)), (int(t + w), int(l + h)), self.bbox_colors[id], 3)
            # cv2.putText(image, str(id), (cx, cy - 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0,255,0), thickness=2)
            
            # save results
            detWithID.append(['person', track.score, [cx, cy, w, h], id])
        
        print(detWithID)
       

