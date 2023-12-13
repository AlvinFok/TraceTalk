from libs.YOLO_SSIM import YoloDevice
import libs.DAN as DAN
# import LineNotify
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__': 
    ServerURL = 'https://demo.iottalk.tw/'    
    Reg_addr = '567890' #if None, Reg_addr = MAC address
    DAN.profile['u_name'] = 'Alvin'
    DAN.profile['dm_name']='Yolo'
    DAN.profile['df_list']=[ 'SFDB-I']
    DAN.profile['d_name']= 'Yolo(567890)'

    DAN.device_registration_with_retry(ServerURL, Reg_addr)

    def on_data(save_path_img: str, group: str, alias: str, detect: str):
        data = {
            "detections":detect,
        }
        data = json.dumps(data, cls=NpEncoder)
        print(DAN.push('SFDB-I', data))
            
    video = "./0325__12__1.mp4"
    alias = "test"
    yolo1 = YoloDevice(
                # darknet file, preset is coco.data(80 classes)
                data_file="./cfg_person/coco.data",
                config_file="./cfg_person/yolov4-tiny.cfg",  # darknet file, preset is yolov4
                weights_file="./weights/yolov4-tiny.weights",  # darknet file, preset is yolov4
                thresh=0.3,  # Yolo threshold, float, range[0, 1]
                output_dir="record/",  # Output dir for saving results
                video_url=video,  # Video url for detection
                is_threading=False,  # Set False if the input is video file
                vertex=None,  # vertex of fence, None -> get all image
                alias=alias,    # Name the file and directory
                display_message=False,  # Show the message (FPS)
                save_img=False,  # Save image when Yolo detect
                save_img_original=True,    # Save original image and results when Yolo detect
                img_expire_day=1,   # Delete the img file if date over the `img_expire_day`
                save_video=False,   # Save video including Yolo detect results
                video_expire_day=1,  # Delete the video file if date over the `video_expire_day`
                target_classes=["person"],  # Set None to detect all target
                auto_restart=False,  # Restart the program when RTSP video disconnection
                using_SSIM=False,    # Using SSIM to find the moving object
                SSIM_debug=False,    # Draw the SSIM image moving object even Yolo have detected object
                
            )
    print(f"\n======== Activating YOLO , alias:{alias}========\n")
    yolo1.set_listener(on_data)
    yolo1.start()