import subprocess
import time
import argparse
import os

def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_thresh', type=float, nargs="+", default=0.4,  help='yolo threshold')
parser.add_argument('--track_thresh', type=float,nargs="+", default=0.6,  help='BYTETracker parameter')
parser.add_argument('--track_buffer', type=int,nargs="+", default=30,  help='BYTETracker parameter')
parser.add_argument('--match_thresh', type=float,nargs="+", default=0.9,  help='BYTETracker parameter')
args = parser.parse_args()


processes = list()
N = 13#how many process running at the same time
queue = list()

folder = "usedVideos"

#for each parameter setting
for yolo_t in args.yolo_thresh:
    for track_t in args.track_thresh:
        for track_b in args.track_buffer:
            for match_t in args.match_thresh:
                #generate command
                for video in sorted(os.listdir(folder)):
                    queue.append(['python', 'test_BYTE.py', '--video', os.path.join(folder, video), '--yolo_thresh', str(yolo_t), '--track_thresh', str(track_t), '--track_buffer', str(track_b) ,'--match_thresh', str(match_t), "--no_save_video"])

#log file
file = open("testResult_BYTE.log", 'w')

# subprocess.run('rm videoTest_BYTE/*', shell=True)#remove old videos
subprocess.run('rm evaluation_BYTE.json', shell=True)



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