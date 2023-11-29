import subprocess
import time
import argparse
import os
from pathlib import Path

def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_thresh', type=float, default=0.01,  help='yolo threshold')
parser.add_argument('--track_thresh', type=float, default=0.6,  help='BYTETracker parameter')
parser.add_argument('--track_buffer', type=int, default=30,  help='BYTETracker parameter')
parser.add_argument('--match_thresh', type=float, default=0.9,  help='BYTETracker parameter')
parser.add_argument('--thread', type=int, default=10)
parser.add_argument("--txt", action="store_true", help="test with txt detections")
parser.add_argument("--txtFolder", default="", help="txt detections folder")
parser.add_argument("--videoFolder", default="", help="video folder")
parser.add_argument("--exp", default="", help="video save folder")
args = parser.parse_args()


processes = list()
N = args.thread#how many process running at the same time
queue = list()

txt = '--txt' if args.txt else ''

#generate command
for video in sorted(list(Path(args.videoFolder).glob("*.mkv"))):
    txtFile = Path(args.txtFolder) / (str(video.stem) + '.txt')
    queue.append(['python', 'test_BYTE_YOLOX.py', '--video', video, '--track_thresh', str(args.track_thresh), '--track_buffer', str(args.track_buffer) ,'--match_thresh', str(args.match_thresh), '-f', 'ByteTrack/exps/example/mot/yolox_x_mix_det.py', '-c', 'ByteTrack/pretrained/bytetrack_x_mot17.pth.tar', '--fp16', '--fuse', txt, '--txtFile', str(txtFile), '--exp', args.exp])


#log file
file = open(f"testResult_{args.exp}.log", 'w')

subprocess.run(f'rm videoTest_{args.exp}/*', shell=True)#remove old videos
subprocess.run(f'rm evaluation_BYTE_{args.exp}.json', shell=True)



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