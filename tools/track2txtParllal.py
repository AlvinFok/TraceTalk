import subprocess
import time
import argparse
from pathlib import Path

def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_thresh', type=float, default=0.01,  help='yolo threshold')
parser.add_argument('--videos', type=str, default="/home/alvin/Square_videos")
parser.add_argument('--savePath', type=str, default="/home/alvin/Square_videos")
parser.add_argument('--thread', type=int, default=2)
args = parser.parse_args()


processes = list()
N = args.thread#how many process running at the same time
queue = list()


#generate command
for video in sorted(Path(args.videos).glob("*.mp4")):
    queue.append(['python3', 'tools/track2txt.py', '-f', 'exps/example/mot/yolox_x_mix_det.py', '-c', 'pretrained/bytetrack_x_mot17.pth.tar', '--fp16', '--fuse', '--path', str(video), '--conf', str(args.yolo_thresh), '--savePath', str(args.savePath)])

#log file
file = open("track2txt.log", 'w')




for process in queue:
    p = subprocess.Popen(process, stdout=file)
    processes.append(p)
    if len(processes) == N:
        wait = True
        while wait:
            done, num = check_for_done(processes)

            if done:
                processes.pop(num)
                wait = False
            else:
                time.sleep(60) # set this so the CPU does not go crazy

file.close()