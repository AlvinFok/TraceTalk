from libs.YOLO_SSIM import YoloDevice
from libs.tracker import Tracker
import libs.DAN as DAN
# import LineNotify
import json




if __name__ == '__main__': 
    ServerURL = 'https://demo.iottalk.tw/'    
    Reg_addr = '567891' #if None, Reg_addr = MAC address
    DAN.profile['u_name'] = 'Alvin'
    DAN.profile['dm_name']='Trace'
    DAN.profile['df_list']=['Trace-I', 'SFDB-O']
    DAN.profile['d_name']= 'Trace(567891)'

    DAN.device_registration_with_retry(ServerURL, Reg_addr)

    
        
            
    
    tracker = Tracker(
                enable_people_counting=True,
                track_buffer=30,
                tracker_mode="BYTE",
                in_area=[[0, 1100],[0, 100],[557, 100],[983, 260], [993, 359],[1159, 493],[1137, 586],[1100, 590],[1425, 1007],[1525, 985],[1574, 814],[1930, 1100] ],#count in vertex
                out_area=[[0, 1080],[0, 0],[877, 0],[1019, 257],[1007, 360],[1177, 501],[1165, 595],[1512, 962],[1609, 578], [1980, 728], [1980, 1080]],#count out vertex
                )
    
    
    while(True):
        detections = DAN.pull("SFDB-O")
        if detections is None:
            continue
        detections = json.loads(detections[0])['detections']
        print(detections)
        tracker.update(detections)
        data = {
            "detections":tracker.detect_target,
            "TotalIn":tracker.totalIn,
            "Current":tracker.currentIn,
        }
        data = json.dumps(data)
        print(DAN.push('Trace-I', data))