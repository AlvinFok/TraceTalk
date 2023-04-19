from libs.YOLOX_BYTE import YoloDevice
import argparse
from pathlib import Path

# parser = argparse.ArgumentParser()
# parser.add_argument('--video', type=str, required=True,  help='video file or folder')
# parser.add_argument('--yolo_thresh', type=float, default=0.4,  help='yolo threshold')
# parser.add_argument('--track_thresh', type=float, default=0.6,  help='BYTETracker parameter')
# parser.add_argument('--track_buffer', type=int, default=30,  help='BYTETracker parameter')
# parser.add_argument('--match_thresh', type=float, default=0.9,  help='BYTETracker parameter')
# parser.add_argument('--no_save_video',action='store_false' , help='save video')

# args = parser.parse_args()


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    
    parser.add_argument('--video', type=str, required=True,  help='video file or folder')
    parser.add_argument('--yolo_thresh', type=float, default=0.4,  help='yolo threshold')
    parser.add_argument('--no_save_video',action='store_false' , help='save video')
    
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


args = make_parser().parse_args()


# print(Path(args.video).name)
#yolov4 tiny
yolo1 = YoloDevice(
        config_file = './cfg_person/yolov4-tiny-person.cfg',
        weights_file = './weights/yolov4-tiny-person_final.weights',
        # config_file = './cfg_person/yolov4.cfg',
        # weights_file = 'darknet/backup/yolov4-preson_last.weights',
        data_file = './cfg_person/person.data',
        thresh = args.yolo_thresh,
        output_dir = 'videoTest_BYTE',
        video_url = args.video,
        is_threading = False,
        vertex = [[0, 1080],[0, 764],[544, 225],[1014, 229],[1920, 809],[1920, 1080]],
        draw_polygon=False,
        alias=Path(args.video).name,
        display_message = False,
        obj_trace = True,        
        save_img = False,
        save_video = args.no_save_video,        
        target_classes=["person"],
        auto_restart = True,
        skip_frame=None,
        count_people=True,
        draw_peopleCounting=True,
        draw_pose=True,
        social_distance=True,
        draw_socialDistanceInfo=True,
        testMode=True,
        repeat=False,
        gpu=0,
        args=args,
    )    

    
#config of BYTE tracker
class Arg():
    def __init__(self, track_thresh=0.6, track_buffer=30, match_thresh=0.9, yolo_thresh=0.4):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.yolo_thresh = yolo_thresh
        # self.min-box-area = 100
        self.mot20 = False


# yolo1.test("oneVideo43")
# yolo1.test("videoClips")
BYTE_args = Arg(track_thresh=args.track_thresh, track_buffer=args.track_buffer, match_thresh=args.match_thresh, yolo_thresh=args.yolo_thresh)
yolo1.test(args.video, BYTE_args)
